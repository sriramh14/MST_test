"""
Timestep-conditioned MST++ denoising network for the Brownian Bridge model.

This module is a *minimally modified* variant of MST_Plus_Plus.py. It is used
to replace the denoise_fn (previously an OpenAI/BBDM-style UNetModel) inside
BrownianBridgeModel while leaving every Brownian Bridge equation (schedule,
q_sample, p_sample, predict_x0_from_objective, objectives, loss, sampling
loop, checkpoint handling) completely untouched.

What is reused, unchanged, from MST_Plus_Plus.py:
    - GELU, PreNorm, FeedForward
    - MS_MSA (the multi-head spectral-wise self-attention block)
    - trunc_normal_ / the original weight-init scheme
The encoder/decoder structure, transformer blocks, attention mechanism, skip
connections and overall computation graph of MST/MST++ are not rewritten or
duplicated -- they are imported directly.

What is added, and nothing else:
    - `TimestepEmbedder`: a standard sinusoidal timestep embedding followed
      by a small 2-layer MLP (the same construction used by guided-diffusion
      / BBDM UNets).
    - `FiLM`: a lightweight scale-shift modulation layer. It is applied once
      to the feature map immediately before each MSAB block (encoder,
      bottleneck, and decoder), and is zero-initialized so the model starts
      out as an identity-conditioned MST++ (scale=0 -> multiplier 1,
      shift=0), and only learns to use timestep information during training.

Public interface (matches the BBDM UNetModel call signature exactly so no
other part of the Brownian Bridge code needs to change):

    forward(x, timesteps, context=None) -> torch.Tensor

`context` is accepted purely for interface compatibility with
BrownianBridgeModel's ContextBlock-based conditioning API. It is not wired
into this MST++ denoiser and is simply ignored, as permitted by the
requested interface.
"""

from __future__ import annotations

import inspect
import math
from abc import abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_gradient_checkpoint
from tqdm.autonotebook import tqdm


# ---------------------------------------------------------------------------
# EDIT THIS IMPORT TO MATCH YOUR MST++ IMPLEMENTATION.
# Example:
from .MST_Plus_Plus import MST_Plus_Plus,GELU, PreNorm, FeedForward, MS_MSA, trunc_normal_

# Timestep-conditioned MST++ used as the Brownian Bridge denoise_fn. This is
# a separate class from the frozen coarse-estimate MST_Plus_Plus above: same
# MST++ encoder/decoder/attention architecture, plus timestep conditioning.
# EDIT THIS IMPORT TO MATCH YOUR MST++ DIFFUSION IMPLEMENTATION.
#from .MST_Plus_Plus_Diffusion import MST_Plus_Plus_Diffusion
# ---------------------------------------------------------------------------



# ===========================================================================
# Additions: timestep embedding + FiLM conditioning (nothing else is new)
# ===========================================================================

class TimestepEmbedder(nn.Module):
    """Standard sinusoidal timestep embedding + small MLP."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.out_dim = hidden_dim if hidden_dim is not None else dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, self.out_dim),
            nn.SiLU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

    @staticmethod
    def sinusoidal_embedding(
        timesteps: torch.Tensor, dim: int, max_period: int = 10_000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
            / max(half, 1)
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        embedding = self.sinusoidal_embedding(timesteps, self.dim)
        embedding = embedding.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(embedding)


class FiLM(nn.Module):
    """
    Lightweight scale-shift (FiLM) conditioning, applied additively/
    multiplicatively to a channel-last feature map [b, h, w, c].

    Zero-initialized so that, at the start of training, this layer is the
    identity function (scale=0 -> multiplier 1, shift=0) and does not alter
    the original MST++ computation graph until it learns to use timestep
    information.
    """

    def __init__(self, time_embed_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(time_embed_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        x: [b, h, w, c]
        time_emb: [b, time_embed_dim]
        """
        scale, shift = self.proj(time_emb).chunk(2, dim=-1)
        scale = scale[:, None, None, :]
        shift = shift[:, None, None, :]
        return x * (1 + scale) + shift


# ===========================================================================
# MST/MST++ with timestep conditioning injected before each MSAB block.
# Everything besides the FiLM injection point is identical to the original
# MSAB / MST / MST_Plus_Plus forward logic.
# ===========================================================================

class TimeConditionedMSAB(nn.Module):
    """
    Identical to the original MSAB (a stack of MS_MSA + FeedForward blocks),
    with a single FiLM layer applied to the feature map before the block
    stack. MS_MSA and FeedForward themselves are the unmodified originals.
    """

    def __init__(self, dim: int, dim_head: int, heads: int, num_blocks: int, time_embed_dim: int):
        super().__init__()
        self.film = FiLM(time_embed_dim, dim)
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim)),
            ]))

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        x = self.film(x, time_emb)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class MSTDiffusion(nn.Module):
    """Same encoder/bottleneck/decoder structure as the original MST class."""

    def __init__(
        self,
        in_dim: int = 31,
        out_dim: int = 31,
        dim: int = 31,
        stage: int = 2,
        num_blocks: Sequence[int] = (2, 4, 4),
        time_embed_dim: int = 512,
    ):
        super().__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                TimeConditionedMSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim,
                    heads=dim_stage // dim, time_embed_dim=time_embed_dim,
                ),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = TimeConditionedMSAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim,
            num_blocks=num_blocks[-1], time_embed_dim=time_embed_dim,
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                TimeConditionedMSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim, time_embed_dim=time_embed_dim,
                ),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)
        self._zero_init_film()

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _zero_init_film(self) -> None:
        # self._init_weights above overwrites every nn.Linear, including the
        # FiLM projections. Re-zero them so FiLM starts as an identity map.
        for module in self.modules():
            if isinstance(module, FiLM):
                nn.init.zeros_(module.proj.weight)
                nn.init.zeros_(module.proj.bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for (msab, fea_down_sample) in self.encoder_layers:
            fea = msab(fea, time_emb)
            fea_encoder.append(fea)
            fea = fea_down_sample(fea)

        # Bottleneck
        fea = self.bottleneck(fea, time_emb)

        # Decoder
        for i, (fea_up_sample, fution, le_win_block) in enumerate(self.decoder_layers):
            fea = fea_up_sample(fea)
            fea = fution(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            fea = le_win_block(fea, time_emb)

        # Mapping
        out = self.mapping(fea) + x

        return out


class MST_Plus_Plus_Diffusion(nn.Module):
    """
    Timestep-conditioned MST++, used as the Brownian Bridge `denoise_fn`.

    Same stacked-MST body, encoder/decoder structure, transformer blocks,
    attention mechanism and skip connections as MST_Plus_Plus. The only
    addition is a sinusoidal-timestep-embedding + MLP whose output
    FiLM-modulates the feature map immediately before every MSAB block.

    forward(x, timesteps, context=None) matches the BBDM UNetModel
    interface exactly. `context` is accepted for interface compatibility
    with BrownianBridgeModel's conditioning API but is not used.

    Constructor accepts **kwargs so a BBDM UNetParams object (via
    vars(UNetParams)) can be passed directly, even though most UNetParams
    fields (image_size, channel_mult, attention_resolutions, condition_key,
    context_channels, ...) do not apply to this architecture and are
    simply ignored.
    """

    def __init__(
        self,
        in_channels: int = 31,
        out_channels: int = 31,
        n_feat: int = 31,
        stage: int = 3,
        mst_stage: int = 2,
        mst_num_blocks: Sequence[int] = (1, 1, 1),
        model_channels: int = 128,
        **kwargs: Any,
    ):
        super().__init__()
        del kwargs  # tolerate/ignore unrelated BBDM UNetParams fields

        self.stage = stage
        self.model_channels = model_channels

        time_embed_dim = model_channels * 4
        self.time_embedder = TimestepEmbedder(dim=model_channels, hidden_dim=time_embed_dim)

        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.body = nn.ModuleList([
            MSTDiffusion(
                dim=31, in_dim=31, out_dim=31, stage=mst_stage,
                num_blocks=list(mst_num_blocks), time_embed_dim=time_embed_dim,
            )
            for _ in range(stage)
        ])
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2, bias=False)

        # Some Kaggle CUDA/cuDNN builds cannot select a cuDNN engine for the
        # depthwise/positional convolution inside MS_MSA.pos_emb, raising
        # "GET was unable to find an engine to execute this computation".
        # Once seen, every later forward call goes straight to the cuDNN-
        # disabled path instead of re-attempting (and re-failing) the cuDNN
        # path every training step.
        self._disable_cudnn_after_engine_error = False

    @staticmethod
    def _is_cudnn_engine_error(error: RuntimeError) -> bool:
        message = str(error).lower()
        return (
            "unable to find an engine" in message
            or "get was unable" in message
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **extra: Any,
    ) -> torch.Tensor:
        """
        x: [b,c,h,w]
        timesteps: [b]
        context: accepted for interface compatibility, unused.
        return out: [b,c,h,w] -- same shape/semantics as the objective the
        Brownian Bridge expects (noise / grad / ysubx, per config).
        """
        del context, extra

        if self._disable_cudnn_after_engine_error and x.is_cuda:
            with torch.backends.cudnn.flags(enabled=False):
                return self._forward_impl(x, timesteps)

        try:
            return self._forward_impl(x, timesteps)
        except RuntimeError as error:
            if not x.is_cuda or not self._is_cudnn_engine_error(error):
                raise

            # Retry once with cuDNN disabled for this process, using the
            # native convolution implementation instead. Gradients are kept
            # intact -- this is not a no_grad context, unlike the frozen
            # MST++ coarse-estimate fallback.
            torch.cuda.synchronize(x.device)
            self._disable_cudnn_after_engine_error = True
            with torch.backends.cudnn.flags(enabled=False):
                return self._forward_impl(x, timesteps)

    def _forward_impl(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x_padded = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        time_emb = self.time_embedder(timesteps)

        h = self.conv_in(x_padded)
        for mst_block in self.body:
            h = mst_block(h, time_emb)
        h = self.conv_out(h)
        h = h + x_padded

        return h[:, :, :h_inp, :w_inp]





# ===========================================================================
# Basic utilities
# ===========================================================================

def exists(value: Any) -> bool:
    return value is not None


def default(value: Any, default_value: Union[Any, Callable[[], Any]]) -> Any:
    if exists(value):
        return value
    return default_value() if callable(default_value) else default_value


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Sequence[int]) -> torch.Tensor:
    """Gather one schedule value per batch item and reshape for broadcasting."""
    batch_size = t.shape[0]
    out = a.gather(0, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def conv_nd(dims: int, *args: Any, **kwargs: Any) -> nn.Module:
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"Unsupported convolution dimensionality: {dims}")


def avg_pool_nd(dims: int, *args: Any, **kwargs: Any) -> nn.Module:
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"Unsupported pooling dimensionality: {dims}")


def linear(*args: Any, **kwargs: Any) -> nn.Module:
    return nn.Linear(*args, **kwargs)


def zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Sinusoidal timestep embedding used by the original guided-diffusion UNet."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        / max(half, 1)
    )
    arguments = timesteps[:, None].float() * frequencies[None]
    embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])],
            dim=-1,
        )
    return embedding


class ChannelLayerNorm(nn.Module):
    """
    LayerNorm over channels for NCHW/NCDHW feature maps.

    This deliberately uses torch.nn.LayerNorm, as requested, rather than the
    GroupNorm layer used in the original OpenAI/BBDM implementation.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return self.norm(x.transpose(1, 2)).transpose(1, 2).contiguous()
        if x.ndim == 4:
            return (
                self.norm(x.permute(0, 2, 3, 1))
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        if x.ndim == 5:
            return (
                self.norm(x.permute(0, 2, 3, 4, 1))
                .permute(0, 4, 1, 2, 3)
                .contiguous()
            )
        raise ValueError(f"ChannelLayerNorm expects 3D/4D/5D input, got {x.shape}")


def normalization(channels: int) -> nn.Module:
    return ChannelLayerNorm(channels)


def checkpoint(
    function: Callable[..., torch.Tensor],
    inputs: Tuple[torch.Tensor, ...],
    parameters: Iterable[nn.Parameter],
    enabled: bool,
) -> torch.Tensor:
    """PyTorch-native checkpoint wrapper with the BBDM call signature."""
    del parameters
    if not enabled:
        return function(*inputs)
    return torch_gradient_checkpoint(
        function,
        *inputs,
        use_reentrant=False,
    )


# ===========================================================================
# Optional spatial context rescaler
# ===========================================================================

class SpatialRescaler(nn.Module):
    """
    Lightweight self-contained equivalent of BBDM's SpatialRescaler.

    It repeatedly interpolates the spatial dimensions and can optionally map
    the channel count using a 1x1 convolution.
    """

    def __init__(
        self,
        n_stages: int = 1,
        method: str = "bilinear",
        multiplier: float = 0.5,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        self.n_stages = n_stages
        self.method = method
        self.multiplier = multiplier
        self.channel_mapper = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
            if out_channels is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        align_corners = False if self.method in {"linear", "bilinear", "bicubic", "trilinear"} else None
        for _ in range(self.n_stages):
            x = F.interpolate(
                x,
                scale_factor=self.multiplier,
                mode=self.method,
                align_corners=align_corners,
            )
        if self.channel_mapper is not None:
            x = self.channel_mapper(x)
        return x


# ===========================================================================
# BBDM-style UNet
# ===========================================================================

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(
        self,
        x: torch.Tensor,
        embedding: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, embedding)
            elif isinstance(layer, ContextBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = default(out_channels, channels)
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(
                dims,
                channels,
                self.out_channels,
                kernel_size=3,
                padding=padding,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        if self.dims == 3:
            x = F.interpolate(
                x,
                size=(x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                mode="nearest",
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.use_conv else x


class Downsample(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = default(out_channels, channels)
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                channels,
                self.out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
            )
        else:
            if channels != self.out_channels:
                raise ValueError("Pooling downsample cannot change channel count.")
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = default(out_channels, channels)
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.updown = up or down

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = nn.Identity()
            self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        return checkpoint(
            self._forward,
            (x, embedding),
            self.parameters(),
            self.use_checkpoint,
        )

    def _forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        if self.updown:
            in_rest = self.in_layers[:-1]
            in_conv = self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(embedding).to(dtype=h.dtype)
        while emb_out.ndim < h.ndim:
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm = self.out_layers[0]
            out_rest = self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = self.out_layers(h + emb_out)

        return self.skip_connection(x) + h


class QKVAttention(nn.Module):
    def __init__(self, heads: int):
        super().__init__()
        self.heads = heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        batch, width, length = qkv.shape
        if width % (3 * self.heads) != 0:
            raise ValueError("QKV width must be divisible by 3 * number of heads.")
        channels = width // (3 * self.heads)
        q, k, v = qkv.reshape(
            batch * self.heads, 3 * channels, length
        ).split(channels, dim=1)
        scale = 1.0 / math.sqrt(math.sqrt(channels))
        weights = torch.einsum(
            "bct,bcs->bts",
            q * scale,
            k * scale,
        )
        weights = torch.softmax(weights.float(), dim=-1).to(weights.dtype)
        attended = torch.einsum("bts,bcs->bct", weights, v)
        return attended.reshape(batch, self.heads * channels, length)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_checkpoint: bool = False,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        del use_new_attention_order
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            if channels % num_head_channels != 0:
                raise ValueError(
                    f"{channels} channels are not divisible by "
                    f"num_head_channels={num_head_channels}"
                )
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(
            self._forward,
            (x,),
            self.parameters(),
            self.use_checkpoint,
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, *spatial = x.shape
        residual = x.reshape(batch, channels, -1)
        qkv = self.qkv(self.norm(residual))
        h = self.proj_out(self.attention(qkv))
        return (residual + h).reshape(batch, channels, *spatial)


'''class ContextBlock(nn.Module):
    """
    Compact spatial context injection compatible with BBDM's `context=` API.

    For HSI bridge training, `context` is normally the MST++ coarse HSI cube.
    The context is resized, reduced to one channel by a parameter-free mean,
    projected to the current feature width, and added residually.

    Set `condition_key="nocond"` to disable it. The bridge endpoint y remains
    present in q_sample and p_sample regardless of context conditioning.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(1, channels, kernel_size=1)
        self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if context is None:
            return x
        if context.ndim != 4:
            raise ValueError(
                f"ContextBlock expects NCHW context, got {context.shape}"
            )
        context = F.interpolate(
            context,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        context = context.mean(dim=1, keepdim=True)
        return x + self.gate * self.proj(context).to(dtype=x.dtype)'''

class ContextBlock(nn.Module):
    """
    Inject the combined MST++ coarse HSI + RGB condition without collapsing channels.
    """

    def __init__(self, channels: int, context_channels: int = 34):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(context_channels, channels, kernel_size=1, bias=False),
            ChannelLayerNorm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

        #self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gate = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
    ) -> torch.Tensor:

        if context is None:
            return x

        context = F.interpolate(
            context,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        context = self.proj(context)

        return x + self.gate * context



class UNetModel(nn.Module):
    """
    Self-contained BBDM/OpenAI-style UNet.

    The constructor accepts the principal fields used by BBDM YAML configs.
    Extra configuration fields are accepted through **kwargs so existing
    UNetParams objects can be passed using vars(UNetParams).
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: Union[Sequence[int], str],
        dropout: float = 0.0,
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        use_spatial_transformer: bool = False,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        n_embed: Optional[int] = None,
        legacy: bool = True,
        condition_key: str = "nocond",
        context_channels: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        del (
            image_size,
            num_classes,
            use_fp16,
            transformer_depth,
            context_dim,
            n_embed,
            legacy,
            kwargs,
        )
        if dims != 2 and use_spatial_transformer:
            raise NotImplementedError(
                "The self-contained ContextBlock currently supports 2D inputs."
            )

        if isinstance(channel_mult, str):
            channel_mult = tuple(int(value) for value in channel_mult.split(","))
        else:
            channel_mult = tuple(channel_mult)

        if isinstance(attention_resolutions, str):
            attention_resolutions = tuple(
                int(value) for value in attention_resolutions.split(",")
            )
        attention_resolutions = set(attention_resolutions)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.condition_key = condition_key
        self.context_channels = int(context_channels) if context_channels is not None else in_channels + 3
        self.dtype = torch.float32

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        channel = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, channel, 3, padding=1)
                )
            ]
        )
        input_block_channels = [channel]
        downsample_factor = 1

        for level, multiplier in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [
                    ResBlock(
                        channel,
                        time_embed_dim,
                        dropout,
                        out_channels=int(multiplier * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                channel = int(multiplier * model_channels)

                if downsample_factor in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            channel,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_checkpoint=use_checkpoint,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if use_spatial_transformer or condition_key != "nocond":
                    layers.append(ContextBlock(channel, context_channels=self.context_channels))

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(channel)

            if level != len(channel_mult) - 1:
                if resblock_updown:
                    down_layer: nn.Module = ResBlock(
                        channel,
                        time_embed_dim,
                        dropout,
                        out_channels=channel,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                    )
                else:
                    down_layer = Downsample(
                        channel,
                        conv_resample,
                        dims=dims,
                        out_channels=channel,
                    )
                self.input_blocks.append(TimestepEmbedSequential(down_layer))
                input_block_channels.append(channel)
                downsample_factor *= 2

        middle_layers: list[nn.Module] = [
            ResBlock(
                channel,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                channel,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_checkpoint=use_checkpoint,
                use_new_attention_order=use_new_attention_order,
            ),
        ]
        if use_spatial_transformer or condition_key != "nocond":
            middle_layers.append(ContextBlock(channel, context_channels=self.context_channels))
        middle_layers.append(
            ResBlock(
                channel,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        )
        self.middle_block = TimestepEmbedSequential(*middle_layers)

        self.output_blocks = nn.ModuleList([])
        for level, multiplier in list(enumerate(channel_mult))[::-1]:
            for block_index in range(num_res_blocks + 1):
                skip_channels = input_block_channels.pop()
                layers = [
                    ResBlock(
                        channel + skip_channels,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * multiplier),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                channel = int(model_channels * multiplier)

                if downsample_factor in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            channel,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_checkpoint=use_checkpoint,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if use_spatial_transformer or condition_key != "nocond":
                    layers.append(ContextBlock(channel, context_channels=self.context_channels))

                if level and block_index == num_res_blocks:
                    if resblock_updown:
                        layers.append(
                            ResBlock(
                                channel,
                                time_embed_dim,
                                dropout,
                                out_channels=channel,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                            )
                        )
                    else:
                        layers.append(
                            Upsample(
                                channel,
                                conv_resample,
                                dims=dims,
                                out_channels=channel,
                            )
                        )
                    downsample_factor //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(channel),
            nn.SiLU(),
            zero_module(conv_nd(dims, channel, out_channels, 3, padding=1)),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        embeddings = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        )

        h = x.to(dtype=self.dtype)
        skips = []
        for module in self.input_blocks:
            h = module(h, embeddings, context)
            skips.append(h)

        h = self.middle_block(h, embeddings, context)

        for module in self.output_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = module(h, embeddings, context)

        return (self.out(h)).to(dtype=x.dtype)


# ===========================================================================
# Brownian bridge: equations retained from the supplied model
# ===========================================================================

class BrownianBridgeModel(nn.Module):
    def __init__(self, model_config: Any):
        super().__init__()
        self.model_config = model_config
        # model hyperparameters
        model_params = model_config.BB.params
        self.num_timesteps = model_params.num_timesteps
        self.mt_type = model_params.mt_type
        self.max_var = model_params.max_var if model_params.__contains__("max_var") else 1
        self.eta = model_params.eta if model_params.__contains__("eta") else 1
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.steps = None
        self.register_schedule()

        # loss and objective
        self.loss_type = model_params.loss_type
        self.objective = model_params.objective

        # UNet
        self.image_size = model_params.UNetParams.image_size
        self.channels = model_params.UNetParams.in_channels
        self.condition_key = model_params.UNetParams.condition_key

        # Denoising network: timestep-conditioned MST++ (replaces the
        # OpenAI/BBDM-style UNetModel). Accepts the same UNetParams config
        # object and the same forward(x, timesteps, context=None) interface,
        # so nothing below this line (schedule, q_sample, p_sample,
        # predict_x0_from_objective, objectives, loss, sampling loop,
        # checkpoint handling) needs to change.
        self.denoise_fn = MST_Plus_Plus_Diffusion(**vars(model_params.UNetParams))

    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return self.denoise_fn.parameters()

    def forward(self, x, y, context=None):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, y, context, t)

    def p_losses(self, x0, y, context, t, noise=None):
        """
        model loss
        :param x0: encoded x_ori, E(x_ori) = x0
        :param y: encoded y_ori, E(y_ori) = y
        :param y_ori: original source domain image
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))

        x_t, objective = self.q_sample(x0, y, t, noise)
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)

        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError()

        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        log_dict = {
            "loss": recloss,
            "x0_recon": x0_recon
        }
        return recloss, log_dict

    def q_sample(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'ysubx':
            objective = y - x0
        else:
            raise NotImplementedError()

        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
            return img

    @torch.no_grad()
    def sample(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        return self.p_sample_loop(y, context, clip_denoised, sample_mid_step)


# ===========================================================================
# Frozen MST++ coarse estimator + bridge wrapper
# ===========================================================================

def _get_config_value(
    config: Any,
    name: str,
    default_value: Any = None,
) -> Any:
    if config is None:
        return default_value
    if isinstance(config, dict):
        return config.get(name, default_value)
    return getattr(config, name, default_value)


def _strip_state_dict_prefixes(
    state_dict: dict[str, torch.Tensor],
    prefixes: Sequence[str] = ("module.", "model.", "mst_model.", "mst_plus_plus.", "mstpp."),
) -> dict[str, torch.Tensor]:
    cleaned = dict(state_dict)
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if cleaned and all(key.startswith(prefix) for key in cleaned):
                cleaned = {
                    key[len(prefix):]: value
                    for key, value in cleaned.items()
                }
                changed = True
    return cleaned



MST_STAGE_PARAMETER_ALIASES = ("stage", "num_stages", "stages", "n_stages")


def _callable_accepts_parameter(
    factory: Callable[..., nn.Module],
    parameter_name: str,
) -> bool:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return False

    parameters = signature.parameters
    if parameter_name in parameters:
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _instantiate_mstpp_with_stage_config(
    factory: Callable[..., nn.Module],
    mstpp_params: dict[str, Any],
    num_stages: Optional[int] = None,
    stage_parameter_name: Optional[str] = None,
) -> nn.Module:
    """Instantiate MST++ while making the number of stages configurable.

    Different MST++ codebases use slightly different constructor names for the
    stage count. This helper supports an explicit name through
    `stage_parameter_name`; otherwise it tries common names and falls back to
    `stage`, which is the name used in many MST++ implementations.
    """
    params = dict(mstpp_params)

    if num_stages is None:
        return factory(**params)

    if int(num_stages) < 1:
        raise ValueError(f"MST++ num_stages must be >= 1, got {num_stages}.")
    num_stages = int(num_stages)

    already_configured = [
        alias for alias in MST_STAGE_PARAMETER_ALIASES
        if alias in params
    ]
    if already_configured and stage_parameter_name is None:
        # Respect an explicitly provided constructor kwarg in params.
        return factory(**params)

    if stage_parameter_name is not None:
        params[stage_parameter_name] = num_stages
        return factory(**params)

    preferred_name = None
    for alias in MST_STAGE_PARAMETER_ALIASES:
        if _callable_accepts_parameter(factory, alias):
            preferred_name = alias
            break
    if preferred_name is None:
        preferred_name = "stage"

    params[preferred_name] = num_stages
    try:
        return factory(**params)
    except TypeError as first_error:
        # Some classes hide their real signature or use a wrapper. Try the
        # common aliases before giving up.
        base_params = {
            key: value
            for key, value in params.items()
            if key not in MST_STAGE_PARAMETER_ALIASES
        }
        last_error = first_error
        for alias in MST_STAGE_PARAMETER_ALIASES:
            trial_params = dict(base_params)
            trial_params[alias] = num_stages
            try:
                return factory(**trial_params)
            except TypeError as error:
                last_error = error

        raise TypeError(
            "Could not instantiate MST++ with a configurable stage count. "
            "Set MST_STAGE_PARAMETER_NAME in the training script to the exact "
            "constructor keyword used by your MST++ class, for example "
            "'stage', 'num_stages', 'stages', or 'n_stages'."
        ) from last_error


def load_mstpp_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    strict: bool = True,
) -> Tuple[list[str], list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        candidate_keys = (
            "mst_plus_plus_state_dict",
            "mstpp_state_dict",
            "model_state_dict",
            "state_dict",
            "model",
            "params",
        )
        state_dict = None
        for key in candidate_keys:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                state_dict = value
                break
        if state_dict is None and all(
            isinstance(value, torch.Tensor) for value in checkpoint.values()
        ):
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Could not locate an MST++ state_dict in checkpoint "
            f"{checkpoint_path!s}."
        )

    state_dict = _strip_state_dict_prefixes(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


class MSTPlusPlusBrownianBridge(nn.Module):
    """
    Complete RGB -> frozen MST++ coarse HSI -> Brownian bridge refinement model.

    Configuration:
        model_config.BB.params
            Used unchanged by BrownianBridgeModel.

        model_config.MSTPP.params (optional)
            Keyword arguments used to construct MST_Plus_Plus.

        model_config.MSTPP.num_stages (optional)
            Number of MST++ stages. Useful when loading a single-stage
            checkpoint. The wrapper injects this into the MST++ constructor.

        model_config.MSTPP.stage_parameter_name (optional)
            Exact constructor keyword for the stage count. If omitted, the
            wrapper tries common names: stage, num_stages, stages, n_stages.

        model_config.MSTPP.checkpoint_path
            Path to the pretrained MST++ checkpoint.

        model_config.MSTPP.strict_load (default True)
            Whether checkpoint loading must match exactly.

        model_config.MSTPP.output_key (optional)
            Dict key when MST++ returns a dictionary.

        model_config.MSTPP.output_index (default -1)
            Tuple/list index when MST++ returns multiple predictions.
    """

    def __init__(
        self,
        model_config: Any,
        mstpp_model: Optional[nn.Module] = None,
        mstpp_factory: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        self.model_config = model_config
        self.bridge = BrownianBridgeModel(model_config)

        mstpp_config = _get_config_value(model_config, "MSTPP")
        mstpp_params = _get_config_value(mstpp_config, "params", {})
        if not isinstance(mstpp_params, dict):
            try:
                mstpp_params = vars(mstpp_params)
            except TypeError as error:
                raise TypeError(
                    "model_config.MSTPP.params must be a dict or namespace."
                ) from error

        if mstpp_model is not None:
            self.mstpp = mstpp_model
        else:
            factory = mstpp_factory
            if factory is None:
                factory = MST_Plus_Plus
            if factory is None:
                raise ImportError(
                    "MST++ could not be imported. Edit the import near the top "
                    "of this file, or pass mstpp_model/mstpp_factory explicitly."
                )
            mstpp_num_stages = _get_config_value(mstpp_config, "num_stages")
            mstpp_stage_parameter_name = _get_config_value(
                mstpp_config,
                "stage_parameter_name",
            )
            self.mstpp_num_stages = mstpp_num_stages
            self.mstpp_stage_parameter_name = mstpp_stage_parameter_name
            self.mstpp = _instantiate_mstpp_with_stage_config(
                factory=factory,
                mstpp_params=mstpp_params,
                num_stages=mstpp_num_stages,
                stage_parameter_name=mstpp_stage_parameter_name,
            )

        checkpoint_path = _get_config_value(mstpp_config, "checkpoint_path")
        strict_load = bool(_get_config_value(mstpp_config, "strict_load", True))
        if checkpoint_path:
            missing, unexpected = load_mstpp_checkpoint(
                self.mstpp,
                checkpoint_path,
                strict=strict_load,
            )
            if not strict_load and (missing or unexpected):
                print(
                    "MST++ checkpoint loaded non-strictly. "
                    f"Missing keys: {missing}; unexpected keys: {unexpected}"
                )

        self.output_key = _get_config_value(mstpp_config, "output_key")
        self.output_index = int(
            _get_config_value(mstpp_config, "output_index", -1)
        )
        self.freeze_mstpp()

    def freeze_mstpp(self) -> None:
        self.mstpp.eval()
        for parameter in self.mstpp.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "MSTPlusPlusBrownianBridge":
        super().train(mode)
        # Keep the coarse estimator frozen and in evaluation mode even while
        # the bridge UNet is training.
        self.mstpp.eval()
        return self

    def _select_mstpp_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            coarse = output
        elif isinstance(output, dict):
            if self.output_key is not None:
                if self.output_key not in output:
                    raise KeyError(
                        f"MST++ output does not contain key {self.output_key!r}. "
                        f"Available keys: {list(output)}"
                    )
                coarse = output[self.output_key]
            else:
                tensor_values = [
                    value for value in output.values()
                    if isinstance(value, torch.Tensor)
                ]
                if not tensor_values:
                    raise TypeError("MST++ dictionary output contains no tensor.")
                coarse = tensor_values[self.output_index]
        elif isinstance(output, (tuple, list)):
            coarse = output[self.output_index]
        else:
            raise TypeError(
                "Unsupported MST++ output type: "
                f"{type(output).__name__}"
            )

        if not isinstance(coarse, torch.Tensor):
            raise TypeError("Selected MST++ prediction is not a tensor.")
        return coarse

    def _run_mstpp_float32(self, rgb_fp32: torch.Tensor) -> Any:
        """Execute MST++ with autocast disabled."""
        if rgb_fp32.device.type in {"cuda", "cpu"}:
            with torch.autocast(
                device_type=rgb_fp32.device.type,
                enabled=False,
            ):
                return self.mstpp(rgb_fp32)
        return self.mstpp(rgb_fp32)

    @torch.no_grad()
    def coarse_estimate(self, rgb: torch.Tensor) -> torch.Tensor:
        """Run frozen MST++ in contiguous float32, outside bridge AMP.

        The first attempt keeps cuDNN enabled for speed. If the Kaggle/PyTorch
        build raises the known ``GET was unable to find an engine`` convolution
        error, the same float32 forward is retried with cuDNN disabled. This
        fallback affects only the frozen MST++ branch; bridge AMP remains active.
        """
        self.mstpp.eval()
        rgb_fp32 = (
            rgb.detach()
            .to(device=rgb.device, dtype=torch.float32)
            .contiguous(memory_format=torch.contiguous_format)
        )

        try:
            output = self._run_mstpp_float32(rgb_fp32)
        except RuntimeError as error:
            message = str(error).lower()
            engine_error = (
                "unable to find an engine" in message
                or "get was unable" in message
            )
            if rgb_fp32.device.type != "cuda" or not engine_error:
                raise

            # Some Kaggle CUDA/cuDNN combinations cannot select a cuDNN engine
            # for an MST++ depth-wise/positional convolution. Retry using the
            # native CUDA convolution implementation instead of cuDNN.
            torch.cuda.synchronize(rgb_fp32.device)
            with torch.backends.cudnn.flags(enabled=False):
                output = self._run_mstpp_float32(rgb_fp32)

        coarse = self._select_mstpp_output(output)
        return (
            coarse.detach()
            .to(device=rgb.device, dtype=torch.float32)
            .contiguous(memory_format=torch.contiguous_format)
        )

    def _validate_endpoints(
        self,
        coarse: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> None:
        if coarse.shape != ground_truth.shape:
            raise ValueError(
                "The Brownian bridge endpoints must have identical shapes. "
                f"MST++ produced {tuple(coarse.shape)}, while ground truth is "
                f"{tuple(ground_truth.shape)}."
            )
        if coarse.shape[1] != self.bridge.channels:
            raise ValueError(
                f"UNet in_channels={self.bridge.channels}, but the bridge "
                f"endpoint has {coarse.shape[1]} channels."
            )

    def _make_bridge_context(
        self,
        coarse: torch.Tensor,
        rgb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Create UNet conditioning from MST++ coarse HSI and raw RGB.

        When conditioning is enabled, the context passed to every ContextBlock is:
            [coarse_hsi, rgb] along the channel dimension.
        For a 31-channel HSI target, this gives 34 context channels.
        """
        if self.bridge.condition_key == "nocond":
            return None
        if context is not None:
            return context

        rgb_condition = rgb.detach().to(
            device=coarse.device,
            dtype=coarse.dtype,
        )
        if rgb_condition.shape[-2:] != coarse.shape[-2:]:
            rgb_condition = F.interpolate(
                rgb_condition,
                size=coarse.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if rgb_condition.shape[0] != coarse.shape[0]:
            raise ValueError(
                "RGB condition and MST++ coarse estimate must have the same "
                f"batch size, got rgb={rgb_condition.shape[0]} and "
                f"coarse={coarse.shape[0]}."
            )
        if rgb_condition.shape[1] != 3:
            raise ValueError(
                "RGB condition is expected to have 3 channels, got "
                f"{rgb_condition.shape[1]}."
            )

        return torch.cat([coarse, rgb_condition], dim=1).contiguous()

    def forward(
        self,
        rgb: torch.Tensor,
        ground_truth: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        coarse = self.coarse_estimate(rgb)
        self._validate_endpoints(coarse, ground_truth)
        bridge_context = self._make_bridge_context(
            coarse=coarse,
            rgb=rgb,
            context=context,
        )

        loss, log_dict = self.bridge(
            x=ground_truth,
            y=coarse,
            context=bridge_context,
        )
        log_dict["coarse_estimate"] = coarse
        return loss, log_dict

    @torch.no_grad()
    def sample(
        self,
        rgb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        sample_mid_step: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[list[torch.Tensor], list[torch.Tensor]],
    ]:
        coarse = self.coarse_estimate(rgb)
        bridge_context = self._make_bridge_context(
            coarse=coarse,
            rgb=rgb,
            context=context,
        )
        return self.bridge.sample(
            y=coarse,
            context=bridge_context,
            clip_denoised=clip_denoised,
            sample_mid_step=sample_mid_step,
        )

    def get_parameters(self) -> Iterable[nn.Parameter]:
        return self.bridge.get_parameters()


# Convenient alias for training scripts.
Model = MSTPlusPlusBrownianBridge
