"""
Dual-Approximator Brownian Bridge for MST++ HSI refinement.

Purpose
-------
Refine the output of a frozen, trained single-stage MST++ model toward the
ground-truth hyperspectral cube using the deterministic Dual-Approximator
Brownian Bridge methodology.

Notation
--------
x0 : ground-truth HSI
y  : frozen MST++ prediction
xt : Brownian-bridge state

Forward bridge:
    xt = (1 - m_t) * x0 + m_t * y + B_t * z
    m_t = t / T
    B_t = sqrt(m_t * (1 - m_t))

Forward approximator:
    eps_theta(xt, t, y) ~= xt - x0
    x0_hat = xt - eps_theta(xt, t, y)

Reverse approximator:
    z_phi(xt, t, y) ~= z

The implementation is standalone PyTorch and does not depend on the original
repository's config system. It keeps the paper/repository objectives while
adapting the input/output channels to hyperspectral data.

Expected tensor format:
    [batch, spectral_channels, height, width]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from os import PathLike
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# -------------------------------------------------------------------------
# MST++ architecture import
# -------------------------------------------------------------------------
#
# Change only this import to match your project structure and class name.
# Example:
from .MST_Plus_Plus import MST_Plus_Plus




# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def _extract_scalar_to_image(value: Tensor, reference: Tensor) -> Tensor:
    """Reshape [B] values to [B, 1, 1, 1] for image broadcasting."""
    return value.reshape(reference.shape[0], *((1,) * (reference.ndim - 1)))


def _valid_group_count(channels: int, requested_groups: int = 8) -> int:
    """Return the largest valid GroupNorm group count <= requested_groups."""
    groups = min(requested_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


def _match_spatial(x: Tensor, reference: Tensor) -> Tensor:
    """Resize x to reference spatial size when odd dimensions cause mismatch."""
    if x.shape[-2:] != reference.shape[-2:]:
        x = F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return x


# -------------------------------------------------------------------------
# Time embedding
# -------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding for integer diffusion timesteps."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 4:
            raise ValueError("embedding_dim must be at least 4")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: Tensor) -> Tensor:
        if timesteps.ndim != 1:
            timesteps = timesteps.reshape(-1)

        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )
            * exponent
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)

        if embedding.shape[1] < self.embedding_dim:
            embedding = F.pad(embedding, (0, self.embedding_dim - embedding.shape[1]))
        return embedding


# -------------------------------------------------------------------------
# HSI denoising network
# -------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with timestep conditioning and spectral channel mixing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.0,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()

        self.norm1 = nn.GroupNorm(
            _valid_group_count(in_channels, norm_groups),
            in_channels,
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(
            _valid_group_count(out_channels, norm_groups),
            out_channels,
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # 1x1 convolution explicitly mixes spectral/channel information.
        self.spectral_mix = nn.Conv2d(out_channels, out_channels, 1)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        residual = self.skip(x)

        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_embedding).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        h = h + self.spectral_mix(h)

        return h + residual


class SpectralAttention(nn.Module):
    """
    Lightweight channel attention.

    For HSI, channels correspond to spectral bands or learned spectral features,
    so channel attention is a natural lightweight spectral interaction module.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.net(x) + x


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class HSIConditionedUNet(nn.Module):
    """
    Compact U-Net used for either approximator.

    By default, the current bridge state xt and the frozen MST++ prediction y
    are concatenated along the channel dimension. This corresponds to explicit
    source conditioning, which the paper describes as an optional conditioning
    design and is appropriate for deterministic HSI refinement.
    """

    def __init__(
        self,
        spectral_channels: int = 31,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_dim: int = 256,
        dropout: float = 0.0,
        condition_on_mst: bool = True,
        use_spectral_attention: bool = True,
    ) -> None:
        super().__init__()

        if len(channel_multipliers) < 2:
            raise ValueError("channel_multipliers must contain at least two levels")

        self.spectral_channels = spectral_channels
        self.condition_on_mst = condition_on_mst
        input_channels = spectral_channels * (2 if condition_on_mst else 1)

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        level_channels = [base_channels * mult for mult in channel_multipliers]

        self.input_conv = nn.Conv2d(input_channels, level_channels[0], 3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        current_channels = level_channels[0]
        for level_index, output_channels in enumerate(level_channels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ResBlock(
                        current_channels,
                        output_channels,
                        time_dim,
                        dropout=dropout,
                    )
                )
                current_channels = output_channels

            attention = (
                SpectralAttention(current_channels)
                if use_spectral_attention
                else nn.Identity()
            )
            self.encoder_blocks.append(nn.ModuleDict({
                "blocks": blocks,
                "attention": attention,
            }))

            if level_index < len(level_channels) - 1:
                self.downsamples.append(Downsample(current_channels))

        self.mid_block1 = ResBlock(
            current_channels,
            current_channels,
            time_dim,
            dropout=dropout,
        )
        self.mid_attention = (
            SpectralAttention(current_channels)
            if use_spectral_attention
            else nn.Identity()
        )
        self.mid_block2 = ResBlock(
            current_channels,
            current_channels,
            time_dim,
            dropout=dropout,
        )

        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level_index in reversed(range(len(level_channels))):
            skip_channels = level_channels[level_index]
            output_channels = level_channels[level_index]

            blocks = nn.ModuleList()
            blocks.append(
                ResBlock(
                    current_channels + skip_channels,
                    output_channels,
                    time_dim,
                    dropout=dropout,
                )
            )
            current_channels = output_channels

            for _ in range(num_res_blocks - 1):
                blocks.append(
                    ResBlock(
                        current_channels,
                        output_channels,
                        time_dim,
                        dropout=dropout,
                    )
                )

            attention = (
                SpectralAttention(current_channels)
                if use_spectral_attention
                else nn.Identity()
            )
            self.decoder_blocks.append(nn.ModuleDict({
                "blocks": blocks,
                "attention": attention,
            }))

            if level_index > 0:
                self.upsamples.append(Upsample(current_channels))

        self.output_norm = nn.GroupNorm(
            _valid_group_count(current_channels),
            current_channels,
        )
        self.output_conv = nn.Conv2d(
            current_channels,
            spectral_channels,
            3,
            padding=1,
        )

        # Match common diffusion U-Net initialization: initially predict zero.
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        condition: Optional[Tensor] = None,
    ) -> Tensor:
        if x_t.ndim != 4:
            raise ValueError(f"x_t must be [B,C,H,W], got {tuple(x_t.shape)}")

        if self.condition_on_mst:
            if condition is None:
                raise ValueError("condition_on_mst=True requires the MST++ output")
            if condition.shape != x_t.shape:
                raise ValueError(
                    "condition and x_t must have identical shapes; "
                    f"got {tuple(condition.shape)} and {tuple(x_t.shape)}"
                )
            network_input = torch.cat([x_t, condition], dim=1)
        else:
            network_input = x_t

        time_embedding = self.time_embedding(timesteps)
        h = self.input_conv(network_input)

        skips: List[Tensor] = []
        for level_index, encoder in enumerate(self.encoder_blocks):
            for block in encoder["blocks"]:
                h = block(h, time_embedding)
            h = encoder["attention"](h)
            skips.append(h)

            if level_index < len(self.downsamples):
                h = self.downsamples[level_index](h)

        h = self.mid_block1(h, time_embedding)
        h = self.mid_attention(h)
        h = self.mid_block2(h, time_embedding)

        upsample_index = 0
        for decoder_index, decoder in enumerate(self.decoder_blocks):
            skip = skips.pop()
            h = _match_spatial(h, skip)
            h = torch.cat([h, skip], dim=1)

            for block in decoder["blocks"]:
                h = block(h, time_embedding)
            h = decoder["attention"](h)

            if decoder_index < len(self.upsamples):
                h = self.upsamples[upsample_index](h)
                upsample_index += 1

        return self.output_conv(F.silu(self.output_norm(h)))


# -------------------------------------------------------------------------
# Bridge outputs
# -------------------------------------------------------------------------

@dataclass
class BridgeTrainingOutput:
    total_loss: Tensor
    forward_loss: Tensor
    reverse_loss: Tensor
    x_t: Tensor
    x0_prediction: Tensor
    forward_prediction: Tensor
    reverse_prediction: Tensor
    forward_target: Tensor
    reverse_target: Tensor
    timesteps: Tensor

    def as_dict(self) -> Dict[str, Tensor]:
        return {
            "loss": self.total_loss,
            "forward_loss": self.forward_loss,
            "reverse_loss": self.reverse_loss,
            "x_t": self.x_t,
            "x0_prediction": self.x0_prediction,
            "forward_prediction": self.forward_prediction,
            "reverse_prediction": self.reverse_prediction,
            "forward_target": self.forward_target,
            "reverse_target": self.reverse_target,
            "timesteps": self.timesteps,
        }


# -------------------------------------------------------------------------
# Dual-approximator Brownian bridge
# -------------------------------------------------------------------------

class DualApproximatorHSIBridge(nn.Module):
    """
    Brownian bridge from frozen MST++ output y to ground-truth HSI x0.

    This module contains:
      1. forward_approximator: predicts x_t - x0
      2. reverse_approximator: predicts the standardized Gaussian z

    Both approximators use the same architecture but do not share weights.
    """

    def __init__(
        self,
        spectral_channels: int = 31,
        num_timesteps: int = 1000,
        sampling_steps: Union[int, Sequence[int]] = 3,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_dim: int = 256,
        dropout: float = 0.0,
        condition_on_mst: bool = True,
        use_spectral_attention: bool = True,
        loss_type: str = "l1",
        forward_loss_weight: float = 1.0,
        reverse_loss_weight: float = 1.0,
        bridge_noise_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if num_timesteps < 2:
            raise ValueError("num_timesteps must be at least 2")
        if loss_type not in {"l1", "l2"}:
            raise ValueError("loss_type must be 'l1' or 'l2'")
        if bridge_noise_scale <= 0:
            raise ValueError("bridge_noise_scale must be positive")

        self.spectral_channels = spectral_channels
        self.num_timesteps = int(num_timesteps)
        self.loss_type = loss_type
        self.forward_loss_weight = float(forward_loss_weight)
        self.reverse_loss_weight = float(reverse_loss_weight)
        self.bridge_noise_scale = float(bridge_noise_scale)
        self.condition_on_mst = condition_on_mst

        network_kwargs = dict(
            spectral_channels=spectral_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            time_dim=time_dim,
            dropout=dropout,
            condition_on_mst=condition_on_mst,
            use_spectral_attention=use_spectral_attention,
        )

        self.forward_approximator = HSIConditionedUNet(**network_kwargs)
        self.reverse_approximator = HSIConditionedUNet(**network_kwargs)

        steps = self._build_sampling_schedule(sampling_steps)
        self.register_buffer(
            "sampling_schedule",
            torch.tensor(steps, dtype=torch.long),
            persistent=True,
        )

    # ------------------------------------------------------------------
    # Schedule and bridge coefficients
    # ------------------------------------------------------------------

    def _build_sampling_schedule(
        self,
        sampling_steps: Union[int, Sequence[int]],
    ) -> List[int]:
        """
        Build descending integer timesteps ending at 1.

        A value of 3 gives approximately [T, T/2, 1]. The paper reports that
        very few steps can improve faithfulness; the full training horizon T
        remains independent from the number of reverse sampling evaluations.
        """
        if isinstance(sampling_steps, int):
            if sampling_steps < 2:
                raise ValueError("sampling_steps must be at least 2")
            raw = torch.linspace(
                self.num_timesteps,
                1,
                sampling_steps,
            ).round().long().tolist()
        else:
            raw = [int(value) for value in sampling_steps]

        steps = sorted(set(raw), reverse=True)

        if not steps or steps[0] != self.num_timesteps:
            steps.insert(0, self.num_timesteps)
        if steps[-1] != 1:
            steps.append(1)

        if any(step < 1 or step > self.num_timesteps for step in steps):
            raise ValueError(
                f"All sampling timesteps must lie in [1, {self.num_timesteps}]"
            )
        return steps

    def bridge_coefficients(
        self,
        timesteps: Tensor,
        reference: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Return m_t and B_t.

        The original repository computes:
            B_t = (1 - m_t) * sqrt(m_t / (1 - m_t))
        which is algebraically sqrt(m_t * (1 - m_t)).
        """
        m_t = timesteps.to(reference.dtype) / float(self.num_timesteps)
        m_t = _extract_scalar_to_image(m_t, reference)

        # Clamping only prevents roundoff at the endpoints.
        variance = torch.clamp(m_t * (1.0 - m_t), min=0.0)
        b_t = self.bridge_noise_scale * torch.sqrt(variance)
        return m_t, b_t

    # ------------------------------------------------------------------
    # Forward bridge and objectives
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: Tensor,
        y: Tensor,
        timesteps: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample x_t and return both repository/paper training targets.

        Returns
        -------
        x_t
        forward_target = x_t - x0
        reverse_target = noise
        """
        self._validate_pair(x0, y)

        if noise is None:
            noise = torch.randn_like(x0)
        if noise.shape != x0.shape:
            raise ValueError("noise must have the same shape as x0")

        m_t, b_t = self.bridge_coefficients(timesteps, x0)
        x_t = (1.0 - m_t) * x0 + m_t * y + b_t * noise

        forward_target = x_t - x0
        reverse_target = noise
        return x_t, forward_target, reverse_target

    def predict_x0(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        y: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Predict the clean target using x0_hat = x_t - eps_theta(...)."""
        forward_prediction = self.forward_approximator(
            x_t,
            timesteps,
            y if self.condition_on_mst else None,
        )
        x0_prediction = x_t - forward_prediction
        return x0_prediction, forward_prediction

    def predict_noise(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Predict the standardized bridge noise z."""
        return self.reverse_approximator(
            x_t,
            timesteps,
            y if self.condition_on_mst else None,
        )

    def _loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self.loss_type == "l1":
            return F.l1_loss(prediction, target)
        return F.mse_loss(prediction, target)

    def training_predictions(
        self,
        ground_truth_hsi: Tensor,
        mst_hsi: Tensor,
        timesteps: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
    ) -> BridgeTrainingOutput:
        """
        Compute both paper objectives at one randomly sampled timestep.

        The same x_t and same sampled z are used for the two approximators.
        """
        self._validate_pair(ground_truth_hsi, mst_hsi)
        batch_size = ground_truth_hsi.shape[0]

        if timesteps is None:
            # Use 1..T-1 in training so B_t is nonzero and both objectives
            # remain well-conditioned. Endpoint T is still used in sampling.
            timesteps = torch.randint(
                low=1,
                high=self.num_timesteps,
                size=(batch_size,),
                device=ground_truth_hsi.device,
                dtype=torch.long,
            )
        else:
            timesteps = timesteps.to(
                device=ground_truth_hsi.device,
                dtype=torch.long,
            )
            if timesteps.shape != (batch_size,):
                raise ValueError(
                    f"timesteps must have shape ({batch_size},), "
                    f"got {tuple(timesteps.shape)}"
                )

        if noise is None:
            noise = torch.randn_like(ground_truth_hsi)

        x_t, forward_target, reverse_target = self.q_sample(
            ground_truth_hsi,
            mst_hsi,
            timesteps,
            noise,
        )

        x0_prediction, forward_prediction = self.predict_x0(
            x_t,
            timesteps,
            mst_hsi,
        )
        reverse_prediction = self.predict_noise(
            x_t,
            timesteps,
            mst_hsi,
        )

        forward_loss = self._loss(forward_prediction, forward_target)
        reverse_loss = self._loss(reverse_prediction, reverse_target)
        total_loss = (
            self.forward_loss_weight * forward_loss
            + self.reverse_loss_weight * reverse_loss
        )

        return BridgeTrainingOutput(
            total_loss=total_loss,
            forward_loss=forward_loss,
            reverse_loss=reverse_loss,
            x_t=x_t,
            x0_prediction=x0_prediction,
            forward_prediction=forward_prediction,
            reverse_prediction=reverse_prediction,
            forward_target=forward_target,
            reverse_target=reverse_target,
            timesteps=timesteps,
        )

    def forward(
        self,
        ground_truth_hsi: Tensor,
        mst_hsi: Tensor,
        timesteps: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        return self.training_predictions(
            ground_truth_hsi=ground_truth_hsi,
            mst_hsi=mst_hsi,
            timesteps=timesteps,
            noise=noise,
        ).as_dict()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        mst_hsi: Tensor,
        *,
        initial_noise: Optional[Tensor] = None,
        deterministic_initialization: bool = False,
        clip_output: Optional[Tuple[float, float]] = None,
        return_intermediates: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        Refine a frozen MST++ prediction using the dual approximators.

        This implementation follows the paper's core sampler:
          - start at x_T = y;
          - estimate x0 with the forward approximator;
          - estimate z with the reverse approximator after initialization;
          - reconstruct bridge states at adjacent requested timesteps;
          - move deterministically along the learned bridge.

        For the first transition from T, the paper samples a single Gaussian
        perturbation scaled by 1/sqrt(T). Set deterministic_initialization=True
        to replace that perturbation with zero for exactly repeatable inference.
        """
        if mst_hsi.ndim != 4:
            raise ValueError(
                f"mst_hsi must be [B,C,H,W], got {tuple(mst_hsi.shape)}"
            )
        if mst_hsi.shape[1] != self.spectral_channels:
            raise ValueError(
                f"Expected {self.spectral_channels} spectral channels, "
                f"got {mst_hsi.shape[1]}"
            )

        y = mst_hsi
        x_t = y.clone()
        intermediates: List[Tensor] = [x_t.clone()]

        steps = self.sampling_schedule.tolist()

        # Paper-style first transition:
        # X_{T-1} = Y - (1/T) eps_theta(Y,T) - z/sqrt(T)
        first_t = torch.full(
            (y.shape[0],),
            self.num_timesteps,
            device=y.device,
            dtype=torch.long,
        )
        _, forward_prediction = self.predict_x0(x_t, first_t, y)

        next_step = steps[1]
        # Scale the one-time initialization by the actual schedule gap.
        delta = (self.num_timesteps - next_step) / float(self.num_timesteps)

        if deterministic_initialization:
            init_z = torch.zeros_like(y)
        elif initial_noise is not None:
            if initial_noise.shape != y.shape:
                raise ValueError("initial_noise must match mst_hsi shape")
            init_z = initial_noise
        else:
            init_z = torch.randn_like(y)

        x_t = (
            y
            - delta * forward_prediction
            - math.sqrt(max(delta, 0.0)) * self.bridge_noise_scale * init_z
        )
        if clip_output is not None:
            x_t = x_t.clamp(*clip_output)
        intermediates.append(x_t.clone())

        # Remaining transitions use the learned reverse-noise approximator.
        # For an arbitrary reduced schedule, we reconstruct the state at the
        # desired next timestep from the same predicted x0 and z. This retains
        # the dual-approximator mechanism while supporting the few-step regime.
        for schedule_index in range(1, len(steps) - 1):
            current_step = steps[schedule_index]
            next_step = steps[schedule_index + 1]

            t = torch.full(
                (y.shape[0],),
                current_step,
                device=y.device,
                dtype=torch.long,
            )
            next_t = torch.full(
                (y.shape[0],),
                next_step,
                device=y.device,
                dtype=torch.long,
            )

            x0_prediction, _ = self.predict_x0(x_t, t, y)
            z_prediction = self.predict_noise(x_t, t, y)

            if next_step == 1:
                # At the final network evaluation, return the clean estimate,
                # consistent with x0 = x1 - eps_theta(x1, 1).
                final_t = torch.ones(
                    y.shape[0],
                    device=y.device,
                    dtype=torch.long,
                )

                # First move to the t=1 bridge state using the current x0/z.
                m_next, b_next = self.bridge_coefficients(next_t, y)
                x_t = (
                    (1.0 - m_next) * x0_prediction
                    + m_next * y
                    + b_next * z_prediction
                )
                x0_prediction, _ = self.predict_x0(x_t, final_t, y)
                x_t = x0_prediction
            else:
                m_next, b_next = self.bridge_coefficients(next_t, y)
                x_t = (
                    (1.0 - m_next) * x0_prediction
                    + m_next * y
                    + b_next * z_prediction
                )

            if clip_output is not None:
                x_t = x_t.clamp(*clip_output)
            intermediates.append(x_t.clone())

        # Defensive handling in case a custom schedule has only [T, 1].
        if steps[-1] == 1 and len(intermediates) < len(steps):
            final_t = torch.ones(
                y.shape[0],
                device=y.device,
                dtype=torch.long,
            )
            x_t, _ = self.predict_x0(x_t, final_t, y)
            if clip_output is not None:
                x_t = x_t.clamp(*clip_output)
            intermediates.append(x_t.clone())

        if return_intermediates:
            return x_t, intermediates
        return x_t

    # ------------------------------------------------------------------
    # Parameter groups and validation
    # ------------------------------------------------------------------

    def get_parameter_groups(self) -> Tuple[nn.ParameterList, nn.ParameterList]:
        """
        Return the two independent parameter groups, mirroring the repository's
        use of separate optimizers for the two approximators.
        """
        return (
            nn.ParameterList(self.forward_approximator.parameters()),
            nn.ParameterList(self.reverse_approximator.parameters()),
        )

    def _validate_pair(self, x0: Tensor, y: Tensor) -> None:
        if x0.ndim != 4 or y.ndim != 4:
            raise ValueError("x0 and y must both be [B,C,H,W]")
        if x0.shape != y.shape:
            raise ValueError(
                "Ground-truth HSI and MST++ output must have identical shapes; "
                f"got {tuple(x0.shape)} and {tuple(y.shape)}"
            )
        if x0.shape[1] != self.spectral_channels:
            raise ValueError(
                f"Expected {self.spectral_channels} channels, got {x0.shape[1]}"
            )


# -------------------------------------------------------------------------
# Optional wrapper around a trained MST++ model
# -------------------------------------------------------------------------

class FrozenMSTDualBridge(nn.Module):
    """
    End-to-end wrapper:
        RGB -> imported/frozen MST++ -> Dual-Approximator HSI Bridge

    Edit the import near the top of this file:

        from mst_architecture import MSTPlusPlus

    so that it points to your MST++ architecture file and class.

    The wrapper instantiates the imported architecture internally. The MST++
    output may be:
      - a tensor;
      - a tuple/list whose first item is the HSI tensor; or
      - a dictionary containing 'out', 'output', 'hsi', or 'prediction'.
    """

    def __init__(
        self,
        bridge: DualApproximatorHSIBridge,
        mst_checkpoint_path: Optional[Union[str, PathLike[str]]] = None,
        mst_model_kwargs: Optional[Dict[str, object]] = None,
        freeze_mst: bool = True,
        strict_checkpoint_loading: bool = True,
        checkpoint_state_key: Optional[str] = None,
    ) -> None:
        super().__init__()

        if MSTPlusPlus is None:
            raise ImportError(
                "The MST++ architecture could not be imported. Edit the "
                "'from mst_architecture import MSTPlusPlus' line near the "
                "top of this file to match your architecture module and "
                "class name."
            ) from _MST_IMPORT_ERROR

        mst_model_kwargs = {} if mst_model_kwargs is None else mst_model_kwargs

        self.mst_model = MSTPlusPlus(**mst_model_kwargs)
        self.bridge = bridge
        self.freeze_mst = freeze_mst

        if mst_checkpoint_path is not None:
            self.load_mst_checkpoint(
                checkpoint_path=mst_checkpoint_path,
                strict=strict_checkpoint_loading,
                state_key=checkpoint_state_key,
            )

        if self.freeze_mst:
            self._freeze_mst()

    def _freeze_mst(self) -> None:
        self.mst_model.eval()
        for parameter in self.mst_model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenMSTDualBridge":
        super().train(mode)

        if self.freeze_mst:
            # Prevent BatchNorm/Dropout state changes in the pretrained model.
            self.mst_model.eval()

        return self

    @staticmethod
    def _select_state_dict(
        checkpoint: object,
        state_key: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        """
        Extract a state dictionary from common checkpoint formats.

        Supplying state_key is best when the checkpoint has a known layout.
        Otherwise common keys are checked automatically.
        """
        if not isinstance(checkpoint, dict):
            raise TypeError(
                "The MST++ checkpoint must be a state dict or a dictionary "
                "containing a state dict."
            )

        if state_key is not None:
            state = checkpoint.get(state_key)
            if not isinstance(state, dict):
                raise KeyError(
                    f"Checkpoint key {state_key!r} does not contain a state dict"
                )
            checkpoint = state
        else:
            for key in (
                "state_dict",
                "model_state_dict",
                "model",
                "net",
                "network",
                "generator",
                "mst_model",
                "mst_state_dict",
            ):
                candidate = checkpoint.get(key)
                if isinstance(candidate, dict):
                    checkpoint = candidate
                    break

        if not checkpoint:
            raise ValueError("The selected MST++ state dictionary is empty")

        if not all(isinstance(key, str) for key in checkpoint):
            raise TypeError("MST++ state-dict keys must be strings")

        tensor_state = {
            key: value
            for key, value in checkpoint.items()
            if torch.is_tensor(value)
        }

        if not tensor_state:
            raise ValueError(
                "No tensor parameters were found in the selected checkpoint"
            )

        return tensor_state

    @staticmethod
    def _remove_common_prefixes(
        state_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Remove wrappers commonly introduced by DDP or parent model classes.
        """
        prefixes = (
            "module.",
            "model.",
            "net.",
            "network.",
            "mst_model.",
        )

        cleaned = dict(state_dict)

        # Remove a prefix only when every key has that prefix. Repeating this
        # handles combinations such as "module.mst_model.layer.weight".
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
                    break

        return cleaned

    def load_mst_checkpoint(
        self,
        checkpoint_path: Union[str, PathLike[str]],
        strict: bool = True,
        state_key: Optional[str] = None,
    ) -> None:
        """
        Load the trained MST++ weights.

        Parameters
        ----------
        checkpoint_path:
            Path to the MST++ checkpoint.
        strict:
            Passed to load_state_dict.
        state_key:
            Optional key containing the state dictionary inside the checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = self._select_state_dict(checkpoint, state_key=state_key)
        state_dict = self._remove_common_prefixes(state_dict)

        incompatible = self.mst_model.load_state_dict(
            state_dict,
            strict=strict,
        )

        if not strict:
            if incompatible.missing_keys:
                print(
                    "MST++ missing checkpoint keys:",
                    incompatible.missing_keys,
                )
            if incompatible.unexpected_keys:
                print(
                    "MST++ unexpected checkpoint keys:",
                    incompatible.unexpected_keys,
                )

    def _extract_mst_output(self, output: object) -> Tensor:
        if torch.is_tensor(output):
            return output

        if isinstance(output, (tuple, list)) and output:
            if torch.is_tensor(output[0]):
                return output[0]

        if isinstance(output, dict):
            for key in ("out", "output", "hsi", "prediction"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value

        raise TypeError(
            "Could not extract an HSI tensor from the MST++ model output. "
            "Adjust _extract_mst_output() to match the architecture's return "
            "format."
        )

    def run_mst(self, rgb: Tensor) -> Tensor:
        if self.freeze_mst:
            with torch.no_grad():
                output = self.mst_model(rgb)
        else:
            output = self.mst_model(rgb)

        mst_hsi = self._extract_mst_output(output)

        if mst_hsi.ndim != 4:
            raise ValueError(
                "MST++ must return [B,C,H,W], "
                f"but returned {tuple(mst_hsi.shape)}"
            )

        if mst_hsi.shape[1] != self.bridge.spectral_channels:
            raise ValueError(
                "MST++ output and bridge spectral-channel counts differ: "
                f"{mst_hsi.shape[1]} versus "
                f"{self.bridge.spectral_channels}"
            )

        return mst_hsi

    def forward(
        self,
        rgb: Tensor,
        ground_truth_hsi: Tensor,
        timesteps: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        mst_hsi = self.run_mst(rgb)

        outputs = self.bridge(
            ground_truth_hsi=ground_truth_hsi,
            mst_hsi=mst_hsi,
            timesteps=timesteps,
            noise=noise,
        )
        outputs["mst_hsi"] = mst_hsi
        return outputs

    @torch.no_grad()
    def reconstruct(
        self,
        rgb: Tensor,
        **sample_kwargs,
    ) -> Tensor:
        mst_hsi = self.run_mst(rgb)
        return self.bridge.sample(mst_hsi, **sample_kwargs)


# -------------------------------------------------------------------------
# Minimal construction example
# -------------------------------------------------------------------------

def build_default_hsi_bridge(
    spectral_channels: int = 31,
) -> DualApproximatorHSIBridge:
    return DualApproximatorHSIBridge(
        spectral_channels=spectral_channels,
        num_timesteps=1000,
        sampling_steps=3,
        base_channels=64,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=2,
        time_dim=256,
        dropout=0.0,
        condition_on_mst=True,
        use_spectral_attention=True,
        loss_type="l1",
        forward_loss_weight=1.0,
        reverse_loss_weight=1.0,
        bridge_noise_scale=1.0,
    )


