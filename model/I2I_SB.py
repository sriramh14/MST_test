"""
i2sb_hsi_model.py
==================

PyTorch implementation of an Image-to-Image Schrodinger Bridge (I2SB)
[Liu et al., 2023, "I2SB: Image-to-Image Schrodinger Bridge", ICML 2023]
specialized for RGB -> Hyperspectral Image (HSI) reconstruction.

Pipeline
--------
    RGB image
        |
        v
    MST++ (pretrained, frozen)  ---->  X1  (degraded / coarse HSI estimate)
        |
        v
    I2SB bridge network  ---->  X0  (refined / clean HSI estimate)

Rather than diffusing from Gaussian noise (as in a standard DDPM), I2SB
builds a *tractable Schrodinger Bridge* directly between the clean HSI
distribution p_A (boundary at t=0, ground-truth X0) and the degraded HSI
distribution p_B(.|X0) (boundary at t=1, the MST++ prediction X1). This
file implements only the model side of that framework:

    * A symmetric noise schedule beta(t) (Fig. 6 of the paper) together
      with the closed-form accumulated variances sigma_t^2 and
      sigma_bar_t^2 (I2SBScheduler).
    * The analytic bridge posterior q(Xt | X0, X1) of Proposition 3.3
      (I2SBScheduler.bridge_posterior_params / I2SBModel.bridge_posterior_sample).
    * The rescaled denoising objective of Eq. (12)
      (I2SBModel.compute_loss).
    * The DDPM-style discrete reverse process of Algorithm 2, built from
      the analytic posterior p(X_{n-1} | X0_pred, X_n) derived in the
      Appendix (proof of Proposition 3.3)
      (I2SBScheduler.ddpm_posterior_params / I2SBModel.reverse_sample).
    * A U-Net epsilon-predictor using LayerNorm instead of GroupNorm,
      optionally conditioned on the MST++ estimate X1 (Section 5.3,
      "General image-to-image translation").

No training loop, dataloader, or inference script is included: this file
only defines the `nn.Module`s and the mathematical machinery, exposing
the primitives requested (bridge posterior sampling, reverse sampling,
prediction, loss computation).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# MST++ import
# ---------------------------------------------------------------------------
# MST++ ("Multi-stage Spectral-wise Transformer for Efficient Spectral
# Reconstruction", Cai et al., CVPRW 2022) is used as the frozen backbone
# that produces the degraded/coarse HSI boundary X1 from an RGB input.
#
# This assumes the standard MST-plus-plus repository layout, where the
# model class is importable as:
#
#     from mst_plus_plus import MST_Plus_Plus
#
# (see https://github.com/caiyuanhao1998/MST-plus-plus,
#  predict_code/architecture/MST_Plus_Plus.py). Pretrained weights are
# assumed to be loaded externally by the caller (e.g. via
# `model.load_state_dict(torch.load(ckpt_path))`) before being wrapped by
# `MSTPPBackbone` / `I2SBModel` below. If the package is not present on
# the path, we fall back to `None` so that this file can still be
# imported/inspected; a `RuntimeError` is raised only if the user
# actually tries to auto-construct MST++ without providing their own
# instance.
try:
    from .MST_Plus_Plus import MST_Plus_Plus
except ImportError:
    MST_Plus_Plus = None


# ---------------------------------------------------------------------------
# 1. Noise schedule / Schrodinger-Bridge scheduler
# ---------------------------------------------------------------------------

class I2SBScheduler:
    """Implements the symmetric noise schedule beta(t) and the analytic
    variances (sigma_t^2, sigma_bar_t^2) used throughout the I2SB paper.

    We adopt a symmetric, parabolic schedule

        beta(t) = beta_min + (beta_max - beta_min) * 4 * t * (1 - t)

    which is small near the two boundaries t=0 and t=1 and peaks at
    t=0.5 (Figure 6 of the paper: "the diffusion shrinks at both
    boundaries"). Because this schedule is a simple polynomial, the
    accumulated variances

        sigma_t^2      = int_0^t beta(tau) d tau   (accumulated from the
                                                     X0 / clean side)
        sigma_bar_t^2  = int_t^1 beta(tau) d tau   (accumulated from the
                                                     X1 / degraded side)

    admit closed forms, which we use directly instead of numerical
    integration.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_min: float = 1.0e-6,
        beta_max: float = 1.2e-4,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max

    # -- schedule primitives -------------------------------------------------

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """beta(t), t in [0, 1]."""
        return self.beta_min + (self.beta_max - self.beta_min) * 4.0 * t * (1.0 - t)

    def sigma2(self, t: torch.Tensor) -> torch.Tensor:
        """sigma_t^2 = int_0^t beta(tau) d tau (closed form)."""
        return self.beta_min * t + (self.beta_max - self.beta_min) * (
            2.0 * t.pow(2) - (4.0 / 3.0) * t.pow(3)
        )

    def total_var(self, device=None, dtype=None) -> torch.Tensor:
        """int_0^1 beta(tau) d tau."""
        one = torch.tensor(1.0, device=device, dtype=dtype)
        return self.sigma2(one)

    def sigma_bar2(self, t: torch.Tensor) -> torch.Tensor:
        """sigma_bar_t^2 = int_t^1 beta(tau) d tau = total_var - sigma_t^2."""
        return self.total_var(device=t.device, dtype=t.dtype) - self.sigma2(t)

    # -- Proposition 3.3: analytic bridge posterior q(Xt|X0,X1) --------------

    def bridge_posterior_params(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (w0, w1, var) such that

            q(Xt | X0, X1) = N(Xt; w0 * X0 + w1 * X1, var * I)

        following Proposition 3.3:
            mu_t = sigma_bar_t^2 / (sigma_bar_t^2 + sigma_t^2) * X0
                 + sigma_t^2     / (sigma_bar_t^2 + sigma_t^2) * X1
            Sigma_t = sigma_t^2 * sigma_bar_t^2 / (sigma_bar_t^2 + sigma_t^2)
        """
        sigma_t2 = self.sigma2(t)
        sigma_bar_t2 = self.sigma_bar2(t)
        denom = sigma_t2 + sigma_bar_t2
        w0 = sigma_bar_t2 / denom
        w1 = sigma_t2 / denom
        var = (sigma_t2 * sigma_bar_t2) / denom
        return w0, w1, var

    # -- discretization for the reverse (generation) process -----------------

    def get_discrete_timesteps(
        self, num_steps: int, device=None
    ) -> torch.Tensor:
        """Quadratic discretization of [0, 1] as used in the paper
        (Song et al., 2020a), returned in *descending* order
        (t_N = 1 ... t_0 = 0) so it can be iterated directly as in
        Algorithm 2.
        """
        i = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        t = i.pow(2)  # quadratic spacing, denser near t=0
        t = torch.flip(t, dims=[0])  # descending: 1 -> 0
        return t

    # -- Appendix (proof of Prop. 3.3): discrete DDPM posterior --------------

    def ddpm_posterior_params(
        self, t_prev: torch.Tensor, t_cur: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Analytic parameters of p(X_{n-1} | X0_pred, X_n) for two
        consecutive discretization times t_prev < t_cur (t_prev is the
        earlier / cleaner time step we are sampling *towards*).

        Following the recursion used to prove Proposition 3.3:
            alpha_n^2  = sigma_{t_cur}^2 - sigma_{t_prev}^2
            mean       = (alpha_n^2 / sigma_{t_cur}^2) * X0_pred
                       + (sigma_{t_prev}^2 / sigma_{t_cur}^2) * X_n
            var        = sigma_{t_prev}^2 * alpha_n^2 / sigma_{t_cur}^2

        Returns (mean_x0_coef, mean_xn_coef, var).
        """
        sigma_prev2 = self.sigma2(t_prev)
        sigma_cur2 = self.sigma2(t_cur)
        alpha2 = (sigma_cur2 - sigma_prev2).clamp_min(0.0)
        denom = (alpha2 + sigma_prev2).clamp_min(1e-12)  # == sigma_cur2
        mean_x0_coef = alpha2 / denom
        mean_xn_coef = sigma_prev2 / denom
        var = (sigma_prev2 * alpha2) / denom
        return mean_x0_coef, mean_xn_coef, var


# ---------------------------------------------------------------------------
# 2. Building blocks (LayerNorm-based U-Net)
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for (B, C, H, W) feature maps, used in place
    of GroupNorm as requested. Normalizes across the channel dimension
    independently at every spatial location.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding of a continuous scalar t in [0, 1]."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # scale t up so the embedding has useful frequency resolution
        t = t.float() * 1000.0
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.embed = SinusoidalTimeEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(t))


class ResBlock(nn.Module):
    """Residual block: Conv -> LayerNorm -> SiLU -> (+time) -> Conv ->
    LayerNorm -> SiLU, with a residual (1x1-projected if needed) skip.
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = LayerNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = LayerNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Lightweight spatial self-attention block (used at the bottleneck)."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = c // self.num_heads

        def reshape(t):
            return t.reshape(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(head_dim), dim=-1)
        out = attn @ v
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.op(x)


# ---------------------------------------------------------------------------
# 3. Epsilon-prediction network: eps(Xt, t; theta)
# ---------------------------------------------------------------------------

class UNetEpsilonNet(nn.Module):
    """U-Net that parameterizes epsilon(Xt, t; theta), the network trained
    with the rescaled objective of Eq. (12). Optionally concatenates the
    MST++ prediction X1 as conditioning, following the "general
    image-to-image translation" recipe of Section 5.3
    (eps(Xt, t, X1 | theta)).
    """

    def __init__(
        self,
        hsi_channels: int = 31,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_dim: int = 256,
        condition_on_x1: bool = True,
        attn_at_bottleneck: bool = True,
    ):
        super().__init__()
        self.condition_on_x1 = condition_on_x1
        in_ch = hsi_channels * (2 if condition_on_x1 else 1)

        self.time_mlp = TimeMLP(dim=time_dim, hidden_dim=time_dim)

        self.in_conv = nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1)

        # ---- encoder ----
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch = base_channels
        enc_channels = [ch]
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList(
                [ResBlock(ch if j == 0 else out_ch, out_ch, time_dim) for j in range(num_res_blocks)]
            )
            self.down_blocks.append(blocks)
            ch = out_ch
            enc_channels.extend([ch] * num_res_blocks)
            if i < len(channel_mults) - 1:
                self.downsamplers.append(Downsample(ch))
                enc_channels.append(ch)
            else:
                self.downsamplers.append(None)

        # ---- bottleneck ----
        self.mid_block1 = ResBlock(ch, ch, time_dim)
        self.mid_attn = SelfAttention2d(ch) if attn_at_bottleneck else nn.Identity()
        self.mid_block2 = ResBlock(ch, ch, time_dim)

        # ---- decoder ----
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for j in range(num_res_blocks + 1):
                skip_ch = enc_channels.pop()
                blocks.append(ResBlock(ch + skip_ch, out_ch, time_dim))
                ch = out_ch
            self.up_blocks.append(blocks)
            if i > 0:
                self.upsamplers.append(Upsample(ch))
            else:
                self.upsamplers.append(None)

        self.out_norm = LayerNorm2d(ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, hsi_channels, kernel_size=3, padding=1)

    def forward(
        self, xt: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.condition_on_x1:
            assert cond is not None, "condition_on_x1=True requires cond=X1"
            h = torch.cat([xt, cond], dim=1)
        else:
            h = xt

        t_emb = self.time_mlp(t)
        h = self.in_conv(h)

        skips = [h]
        for blocks, down in zip(self.down_blocks, self.downsamplers):
            for block in blocks:
                h = block(h, t_emb)
                skips.append(h)
            if down is not None:
                h = down(h)
                skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for blocks, up in zip(self.up_blocks, self.upsamplers):
            for block in blocks:
                skip = skips.pop()
                h = block(torch.cat([h, skip], dim=1), t_emb)
            if up is not None:
                h = up(h)

        h = self.out_act(self.out_norm(h))
        return self.out_conv(h)


# ---------------------------------------------------------------------------
# 4. MST++ backbone wrapper
# ---------------------------------------------------------------------------

class MSTPPBackbone(nn.Module):
    """Thin, frozen wrapper around a (pretrained) MST++ model. It produces
    the degraded boundary X1 of the bridge (the "prior" that I2SB refines
    towards ground-truth HSI X0).

    The caller is expected to construct `MST_Plus_Plus` (imported above)
    and load its pretrained weights before passing the resulting module
    in as `mstpp_model`, e.g.:

        from mst_plus_plus import MST_Plus_Plus
        mstpp = MST_Plus_Plus()
        mstpp.load_state_dict(torch.load("mst_plus_plus.pth")["state_dict"])
        backbone = MSTPPBackbone(mstpp)

    If `mstpp_model` is left as `None`, this wrapper will attempt to
    auto-construct a default `MST_Plus_Plus()` instance (with *randomly
    initialized* weights -- the caller must still load a checkpoint via
    `self.mstpp_model.load_state_dict(...)` afterwards). This wrapper
    only freezes the network and standardizes its call signature; it
    does not load any weights itself.
    """

    def __init__(self, mstpp_model: Optional[nn.Module] = None, freeze: bool = True):
        super().__init__()
        if mstpp_model is None:
            if MST_Plus_Plus is None:
                raise RuntimeError(
                    "MST_Plus_Plus could not be imported (module "
                    "'mst_plus_plus' not found on the path) and no "
                    "`mstpp_model` instance was provided. Either install/"
                    "expose the MST-plus-plus package, or construct the "
                    "model yourself and pass it in as `mstpp_model`."
                )
            mstpp_model = MST_Plus_Plus()
        self.mstpp_model = mstpp_model
        if freeze:
            for p in self.mstpp_model.parameters():
                p.requires_grad_(False)
            self.mstpp_model.eval()
        self._freeze = freeze

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if self._freeze:
            self.mstpp_model.eval()
        return self.mstpp_model(rgb)


# ---------------------------------------------------------------------------
# 5. I2SB model: ties everything together
# ---------------------------------------------------------------------------

class I2SBModel(nn.Module):
    """Image-to-Image Schrodinger Bridge for RGB -> HSI refinement.

    Boundary distributions:
        X0 ~ p_A          : ground-truth (clean) hyperspectral image
        X1 ~ p_B(.|X0)     : MST++ prediction (degraded / coarse HSI),
                              treated as the tractable Dirac-delta-conditioned
                              boundary of Corollary 3.2.

    `mstpp_model` should be an instance of `MST_Plus_Plus` (imported at
    the top of this file from `mst_plus_plus`) with pretrained weights
    already loaded, e.g.:

        from mst_plus_plus import MST_Plus_Plus
        mstpp = MST_Plus_Plus()
        mstpp.load_state_dict(torch.load("mst_plus_plus.pth")["state_dict"])
        model = I2SBModel(mstpp_model=mstpp, hsi_channels=31)

    If `mstpp_model` is omitted, a default (randomly-initialized)
    `MST_Plus_Plus()` is constructed automatically -- the caller is then
    responsible for loading pretrained weights into
    `model.backbone.mstpp_model` before use.

    Exposed API
    ------------
        get_degraded_prediction(rgb)          -> X1 from frozen MST++
        bridge_posterior_sample(x0, x1, t)     -> Xt ~ q(Xt|X0,X1), target eps
        predict_x0_from_eps(xt, t, eps)        -> X0 estimate from eps
        compute_loss(x0, rgb=None, x1=None)    -> training loss (Eq. 12)
        reverse_sample(x1, cond=None, ...)     -> full DDPM-style generation
                                                   (Algorithm 2)
        predict(rgb, num_steps=None)           -> end-to-end RGB -> refined HSI
    """

    def __init__(
        self,
        mstpp_model: Optional[nn.Module] = None,
        hsi_channels: int = 31,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_dim: int = 256,
        condition_on_x1: bool = True,
        num_train_timesteps: int = 1000,
        beta_min: float = 1.0e-6,
        beta_max: float = 1.2e-4,
        freeze_mstpp: bool = True,
    ):
        super().__init__()
        self.hsi_channels = hsi_channels
        self.condition_on_x1 = condition_on_x1

        self.backbone = MSTPPBackbone(mstpp_model, freeze=freeze_mstpp)
        self.eps_net = UNetEpsilonNet(
            hsi_channels=hsi_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            time_dim=time_dim,
            condition_on_x1=condition_on_x1,
        )
        self.scheduler = I2SBScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_min=beta_min,
            beta_max=beta_max,
        )

    # -- boundary construction -------------------------------------------

    @torch.no_grad()
    def get_degraded_prediction(self, rgb: torch.Tensor) -> torch.Tensor:
        """Runs the frozen MST++ backbone to obtain X1, the degraded /
        coarse HSI boundary of the bridge."""
        return self.backbone(rgb)

    # -- Proposition 3.3: bridge posterior sampling ------------------------

    def bridge_posterior_sample(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draws Xt ~ q(Xt | X0, X1) = N(mu_t, Sigma_t) (Eq. 11) and
        returns (Xt, target_eps) where target_eps = (Xt - X0) / sigma_t
        is the regression target of Eq. (12).

        Args:
            x0: (B, C, H, W) ground-truth HSI.
            x1: (B, C, H, W) degraded HSI (MST++ prediction).
            t:  (B,) timesteps in [0, 1].
        """
        w0, w1, var = self.scheduler.bridge_posterior_params(t)
        w0 = w0.view(-1, 1, 1, 1)
        w1 = w1.view(-1, 1, 1, 1)
        std = var.clamp_min(0.0).sqrt().view(-1, 1, 1, 1)

        mean = w0 * x0 + w1 * x1
        noise = torch.randn_like(x0)
        xt = mean + std * noise

        sigma_t = self.scheduler.sigma2(t).clamp_min(1e-12).sqrt().view(-1, 1, 1, 1)
        target_eps = (xt - x0) / sigma_t
        return xt, target_eps

    # -- Eq. (12): training loss -------------------------------------------

    def compute_loss(
        self,
        x0: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        x1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the I2SB training loss (Algorithm 1):

            t  ~ U([0, 1])
            X1 ~ p_B(.|X0)              (obtained from MST++ if not given)
            Xt ~ q(Xt|X0, X1)           (bridge_posterior_sample)
            L  = || eps(Xt, t; theta) - (Xt - X0) / sigma_t ||^2

        Either `rgb` (from which X1 is derived via MST++) or a
        precomputed `x1` must be supplied.
        """
        assert (rgb is not None) or (x1 is not None), "Provide rgb or x1"
        if x1 is None:
            x1 = self.get_degraded_prediction(rgb)

        b = x0.shape[0]
        t = torch.rand(b, device=x0.device, dtype=x0.dtype)

        xt, target_eps = self.bridge_posterior_sample(x0, x1, t)
        cond = x1 if self.condition_on_x1 else None
        pred_eps = self.eps_net(xt, t, cond=cond)

        return F.mse_loss(pred_eps, target_eps)

    # -- eps -> X0 mapping (footnote 1 of the paper) ------------------------

    def predict_x0_from_eps(
        self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """X0_eps := Xt - sigma_t * eps (since Xt = X0 + sigma_t * eps,
        f := 0)."""
        sigma_t = self.scheduler.sigma2(t).clamp_min(1e-12).sqrt().view(-1, 1, 1, 1)
        return xt - sigma_t * eps

    # -- Algorithm 2: single reverse (DDPM-style) step -----------------------

    def reverse_step(
        self,
        x_cur: torch.Tensor,
        t_cur: torch.Tensor,
        t_prev: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One iteration of Algorithm 2's loop body:
            1. predict X0 via the network output at (x_cur, t_cur)
            2. sample x_prev ~ p(x_prev | X0_pred, x_cur) using the
               analytic DDPM posterior of the Appendix.

        Args:
            x_cur:  current bridge state X_n.
            t_cur:  current (later) time, scalar tensor broadcast to batch.
            t_prev: target (earlier) time to sample towards, t_prev < t_cur.
            cond:   optional conditioning (X1), used if condition_on_x1.
        Returns:
            (x_prev, x0_pred)
        """
        b = x_cur.shape[0]
        t_cur_b = t_cur.expand(b) if t_cur.dim() == 0 else t_cur
        t_prev_b = t_prev.expand(b) if t_prev.dim() == 0 else t_prev

        eps = self.eps_net(x_cur, t_cur_b, cond=cond)
        x0_pred = self.predict_x0_from_eps(x_cur, t_cur_b, eps)

        mean_x0_coef, mean_xn_coef, var = self.scheduler.ddpm_posterior_params(
            t_prev_b, t_cur_b
        )
        mean_x0_coef = mean_x0_coef.view(-1, 1, 1, 1)
        mean_xn_coef = mean_xn_coef.view(-1, 1, 1, 1)
        std = var.clamp_min(0.0).sqrt().view(-1, 1, 1, 1)

        mean = mean_x0_coef * x0_pred + mean_xn_coef * x_cur

        # No noise injection on the final step (t_prev == 0).
        is_final = (t_prev_b <= 0).view(-1, 1, 1, 1)
        noise = torch.randn_like(x_cur)
        x_prev = mean + torch.where(is_final, torch.zeros_like(std), std) * noise
        return x_prev, x0_pred

    # -- Algorithm 2: full reverse sampling ----------------------------------

    @torch.no_grad()
    def reverse_sample(
        self,
        x1: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Runs the full DDPM-style reverse process of Algorithm 2,
        starting from X_N ~ p_B (here, the MST++ prediction x1) down to
        X_0 (the refined HSI estimate).

        Args:
            x1: (B, C, H, W) initial (degraded) boundary sample.
            cond: conditioning tensor fed to the network at every step
                  (defaults to x1 itself if condition_on_x1 is True).
            num_steps: number of discretization steps; defaults to the
                       scheduler's num_train_timesteps.
        Returns:
            x0: (B, C, H, W) refined HSI prediction.
        """
        if cond is None and self.condition_on_x1:
            cond = x1
        num_steps = num_steps or self.scheduler.num_train_timesteps
        timesteps = self.scheduler.get_discrete_timesteps(num_steps, device=x1.device)

        x_cur = x1
        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_prev = timesteps[i + 1]
            x_cur, _ = self.reverse_step(x_cur, t_cur, t_prev, cond=cond)
        return x_cur

    # -- End-to-end convenience API ------------------------------------------

    @torch.no_grad()
    def predict(self, rgb: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """End-to-end inference: RGB -> MST++ (X1) -> I2SB refinement -> X0."""
        x1 = self.get_degraded_prediction(rgb)
        cond = x1 if self.condition_on_x1 else None
        return self.reverse_sample(x1, cond=cond, num_steps=num_steps)

    def forward(
        self,
        x0: Optional[torch.Tensor] = None,
        rgb: Optional[torch.Tensor] = None,
        mode: str = "loss",
        num_steps: Optional[int] = None,
    ):
        """Convenience dispatch.
            mode="loss":    requires x0 and rgb -> returns scalar loss.
            mode="predict": requires rgb        -> returns refined HSI.
        """
        if mode == "loss":
            assert x0 is not None and rgb is not None
            return self.compute_loss(x0, rgb=rgb)
        elif mode == "predict":
            assert rgb is not None
            return self.predict(rgb, num_steps=num_steps)
        else:
            raise ValueError(f"Unknown mode: {mode}")
