"""
residual_diffusion_rgb2hsi.py

Residual Diffusion model for RGB -> Hyperspectral Image (HSI) reconstruction,
adapted from:

    Selective Hourglass Mapping for Universal Image Restoration Based on
    Diffusion Model (DiffUIR), CVPR 2024.
    https://github.com/iSEE-Laboratory/DiffUIR

--------------------------------------------------------------------------
Pipeline
--------------------------------------------------------------------------

    RGB Image
        -> Frozen MST++                         (coarse HSI prediction)
        -> Residual Diffusion Model             (predicts the residual)
        -> Final HSI = Coarse Prediction + Predicted Residual

Only the residual between the coarse MST++ prediction and the ground-truth
HSI is diffused/denoised. MST++ is always run under `torch.no_grad()` and
all of its parameters are frozen -- gradients never flow into it.

--------------------------------------------------------------------------
Mapping from the original DiffUIR notation to this adaptation
--------------------------------------------------------------------------

    Original DiffUIR                         This adaptation
    ---------------------------------------  ----------------------------
    I_0  (clean target image)                GroundTruthHSI
    I_in (degraded image, the condition)     MSTPrediction (coarse HSI)
    I_res = I_in - I_0 (residual)            MSTPrediction - GroundTruthHSI
    I_t (forward-diffused sample)            noisy residual representation
    Predicted I_res                          predicted internal residual

--------------------------------------------------------------------------
IMPORTANT: residual sign convention
--------------------------------------------------------------------------
The DiffUIR forward process

    I_t = I_0 + alpha_t * I_res + beta_t * eps - delta_t * I_in

is specifically constructed so that, as alpha_t -> 1 and delta_t -> delta_max
(t -> T), the I_0 term cancels against I_res = I_in - I_0, i.e.

    I_T = I_0 + I_res - delta_T * I_in
        = I_0 + (I_in - I_0) - delta_T * I_in
        = (1 - delta_T) * I_in + beta_T * eps

This is the entire point of the "selective hourglass mapping" / shared
distribution term (SDT): the diffusion endpoint depends only on the
condition image I_in (here, MSTPrediction) and noise, never on the
unknown target I_0. This is what makes the reverse process well-posed
at sampling time (I_0 / GroundTruthHSI is obviously not available then).

The task specification defines:

    Residual := GroundTruthHSI - MSTPrediction   (i.e. I_0 - I_in)

which is the *negative* of DiffUIR's I_res = I_in - I_0. Plugging that
sign directly into the forward equation above would break the
cancellation property and make I_T depend on the (unknown-at-inference)
ground truth -- i.e. it would silently break the diffusion math.

To stay faithful to the original DiffUIR derivation (schedules, SDT,
posterior, DDPM/DDIM sampling all preserved exactly) while still
honoring the requested public semantics, this file:

  1. Performs all internal diffusion computation using DiffUIR's original
     convention: `internal_residual = MSTPrediction - GroundTruthHSI`.
  2. Exposes a `predicted_residual` to the outside world defined as
     `-internal_residual`, i.e. `GroundTruthHSI - MSTPrediction`, exactly
     matching the spec.
  3. Reconstructs `FinalHSI = MSTPrediction + predicted_residual`, which
     is algebraically identical to DiffUIR's own reconstruction
     `I_0 = I_in - internal_residual` -- so the externally visible
     behaviour matches the spec exactly, and the internal math matches
     the paper exactly. Nothing is silently changed; this file only
     performs a sign flip at the public boundary.

--------------------------------------------------------------------------
Scope of this file
--------------------------------------------------------------------------
This file defines ONLY the model architecture and diffusion machinery:

    - ResidualDiffusionRGB2HSI   (top-level model)
    - ResidualDiffusionScheduler (schedules + forward/reverse diffusion math)
    - ResidualUNet               (conditional denoiser / residual predictor)
    - DiffusionEmbedding         (sinusoidal timestep embedding)
    - forward diffusion utilities
    - reverse diffusion utilities (DDPM + DDIM samplers)

It intentionally does NOT implement: loss functions, optimizers, the
training loop, dataset / dataloader code, metrics, or inference scripts.
Those are assumed to live elsewhere in your codebase.

Imports of MST++, and any other project-specific utilities, are given as
placeholders -- adjust the import paths to match your project structure.
"""

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Placeholder imports -- adjust these to match your project structure.
# --------------------------------------------------------------------------
# MST++ is assumed to already exist, fully implemented, in your codebase.
# We only import and instantiate it here; its architecture is NOT
# reimplemented in this file.
from MST_Plus_Plus import MST_Plus_Plus  # noqa: F401  (placeholder import)


# ==========================================================================
# Timestep embedding
# ==========================================================================

class DiffusionEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding followed by a small MLP, in the style of
    the original DDPM / DiffUIR U-Net time embedding.
    """

    def __init__(self, embedding_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim * 4
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """Standard transformer-style sinusoidal embedding of integer timesteps."""
        half_dim = dim // 2
        device = timesteps.device
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=device).float() / half_dim
        )
        args = timesteps.float()[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self._sinusoidal_embedding(timesteps, self.embedding_dim)
        return self.mlp(emb)


# ==========================================================================
# U-Net building blocks
# ==========================================================================

class ResBlock(nn.Module):
    """Residual block with time-embedding conditioning (DDPM-style)."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 dropout: float = 0.0, groups: int = 8):
        super().__init__()
        groups_in = min(groups, in_channels)
        groups_out = min(groups, out_channels)

        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Simple single-head self-attention block, used at low spatial resolutions."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        groups = min(groups, channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(b, 3, c, h * w).unbind(1)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) / math.sqrt(c), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj_out(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.op(x)


# ==========================================================================
# Residual U-Net (the conditional residual denoiser)
# ==========================================================================

class ResidualUNet(nn.Module):
    """
    Conditional U-Net that predicts the (internal-convention) residual
    given the current noisy residual representation and the MST++ coarse
    prediction as conditioning.

    Implicit condition (per DiffUIR): the conditioning image (MSTPrediction)
    is concatenated channel-wise with the noisy residual input before being
    passed into the network -- this is the "implicit condition" mechanism
    described in the paper, preserved here unchanged.

    Explicit condition (per DiffUIR): the conditioning image is additionally
    injected into the diffusion algorithm itself (via the forward/reverse
    equations in `ResidualDiffusionScheduler`), not inside this network.
    """

    def __init__(
        self,
        hsi_channels: int,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.0,
        time_emb_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hsi_channels = hsi_channels
        # Input = noisy residual (hsi_channels) concatenated with the
        # MST++ coarse prediction used as the conditioning image
        # (implicit condition, hsi_channels).
        in_channels = hsi_channels * 2
        time_emb_dim = time_emb_dim or base_channels * 4

        self.time_embedding = DiffusionEmbedding(base_channels, time_emb_dim)

        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ---------------- Encoder ----------------
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        current_res = 64  # nominal resolution tracker for attention placement (relative)

        for level, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            stage_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                block = ResBlock(now_channels, out_ch, time_emb_dim, dropout)
                stage_blocks.append(block)
                now_channels = out_ch
                channels.append(now_channels)
                if current_res in attention_resolutions:
                    stage_blocks.append(AttentionBlock(now_channels))
            self.down_blocks.append(stage_blocks)

            if level != len(channel_multipliers) - 1:
                self.downsamples.append(Downsample(now_channels))
                channels.append(now_channels)
                current_res //= 2
            else:
                self.downsamples.append(None)

        # ---------------- Bottleneck ----------------
        self.middle_block1 = ResBlock(now_channels, now_channels, time_emb_dim, dropout)
        self.middle_attn = AttentionBlock(now_channels)
        self.middle_block2 = ResBlock(now_channels, now_channels, time_emb_dim, dropout)

        # ---------------- Decoder ----------------
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            stage_blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                block = ResBlock(now_channels + skip_ch, out_ch, time_emb_dim, dropout)
                stage_blocks.append(block)
                now_channels = out_ch
                if current_res in attention_resolutions:
                    stage_blocks.append(AttentionBlock(now_channels))
            self.up_blocks.append(stage_blocks)

            if level != 0:
                self.upsamples.append(Upsample(now_channels))
                current_res *= 2
            else:
                self.upsamples.append(None)

        self.out_norm = nn.GroupNorm(min(8, now_channels), now_channels)
        self.out_conv = nn.Conv2d(now_channels, hsi_channels, kernel_size=3, padding=1)

    def forward(
        self,
        noisy_residual: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_residual: I_t, the current noisy residual representation.
                             Shape (B, hsi_channels, H, W).
            condition:       MSTPrediction (coarse HSI), used as the
                             implicit conditioning image.
                             Shape (B, hsi_channels, H, W).
            timesteps:       Integer diffusion timesteps, shape (B,).

        Returns:
            Predicted internal residual (I_res, in DiffUIR's original
            sign convention: MSTPrediction - GroundTruthHSI), shape
            (B, hsi_channels, H, W).
        """
        time_emb = self.time_embedding(timesteps)

        x = torch.cat([noisy_residual, condition], dim=1)
        h = self.input_conv(x)

        skips = [h]
        for stage_blocks, downsample in zip(self.down_blocks, self.downsamples):
            for block in stage_blocks:
                h = block(h, time_emb) if isinstance(block, ResBlock) else block(h)
            skips.append(h)
            if downsample is not None:
                h = downsample(h)
                skips.append(h)

        h = self.middle_block1(h, time_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, time_emb)

        for stage_blocks, upsample in zip(self.up_blocks, self.upsamples):
            for block in stage_blocks:
                if isinstance(block, ResBlock):
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = block(h, time_emb)
                else:
                    h = block(h)
            if upsample is not None:
                h = upsample(h)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h


# ==========================================================================
# Schedule construction utilities (forward diffusion utilities)
# ==========================================================================

def make_cumulative_schedule(
    num_timesteps: int,
    start_value: float,
    end_value: float,
    schedule_type: str = "linear",
) -> torch.Tensor:
    """
    Builds a monotonically increasing cumulative schedule
    (e.g. alpha_bar_t, delta_bar_t) of length `num_timesteps`, running
    from `start_value` at t=1 to `end_value` at t=T.

    This mirrors the increasing cumulative-coefficient schedules used in
    DiffUIR / RDDM (alpha_bar_t: 0 -> 1, delta_bar_t: 0 -> 0.9).
    """
    t = torch.linspace(0, 1, num_timesteps)
    if schedule_type == "linear":
        curve = t
    elif schedule_type == "cosine":
        curve = 1 - torch.cos(t * math.pi / 2)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")
    return start_value + (end_value - start_value) * curve


def make_beta_bar_schedule(
    num_timesteps: int,
    beta_bar_min: float = 1e-4,
    beta_bar_max: float = 1.0,
    schedule_type: str = "linear",
) -> torch.Tensor:
    """
    Builds the cumulative Gaussian-noise-scale schedule beta_bar_t
    (monotonically increasing from beta_bar_min at t=1 to beta_bar_max
    at t=T), matching the role of `beta_bar_t` in Eq. (2)/(4) of DiffUIR.
    """
    return make_cumulative_schedule(num_timesteps, beta_bar_min, beta_bar_max, schedule_type)


# ==========================================================================
# Residual Diffusion Scheduler
# ==========================================================================

class ResidualDiffusionScheduler(nn.Module):
    """
    Holds the alpha / beta / delta schedules (both cumulative and per-step
    forms) and implements the forward diffusion process, the DDPM
    posterior, and the DDIM sampling update from DiffUIR, unchanged in
    form from the paper.

    All internal computation uses DiffUIR's original residual convention:

        I_res_internal = I_in - I_0 = MSTPrediction - GroundTruthHSI

    Notation (matches the paper):
        alpha_bar_t, beta_bar_t, delta_bar_t : cumulative coefficients
        alpha_t, beta_t, delta_t             : per-step coefficients
                                                (used directly in the
                                                 reverse-process update,
                                                 Eq. 7 / Eq. 8)
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        alpha_bar_max: float = 1.0,
        beta_bar_min: float = 1e-4,
        beta_bar_max: float = 1.0,
        delta_bar_max: float = 0.9,
        schedule_type: str = "linear",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        # Cumulative schedules, index 0 corresponds to t=1 ... index T-1 to t=T.
        alpha_bar = make_cumulative_schedule(num_timesteps, 0.0, alpha_bar_max, schedule_type)
        beta_bar = make_beta_bar_schedule(num_timesteps, beta_bar_min, beta_bar_max, schedule_type)
        delta_bar = make_cumulative_schedule(num_timesteps, 0.0, delta_bar_max, schedule_type)

        # Per-step coefficients, derived from the cumulative schedules:
        #   alpha_t = alpha_bar_t - alpha_bar_{t-1}
        #   delta_t = delta_bar_t - delta_bar_{t-1}
        #   beta_t^2 = beta_bar_t^2 - beta_bar_{t-1}^2   (independent-noise
        #              accumulation, as in Eq. (1) -> Eq. (2))
        alpha = alpha_bar.clone()
        alpha[1:] = alpha_bar[1:] - alpha_bar[:-1]

        delta = delta_bar.clone()
        delta[1:] = delta_bar[1:] - delta_bar[:-1]

        beta_sq = beta_bar.clone() ** 2
        beta_sq[1:] = beta_bar[1:] ** 2 - beta_bar[:-1] ** 2
        beta = torch.sqrt(beta_sq.clamp(min=0.0))

        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("beta_bar", beta_bar)
        self.register_buffer("delta_bar", delta_bar)
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta)
        self.register_buffer("delta", delta)

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    def _extract(self, schedule: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Gathers schedule values for a batch of timesteps and reshapes for broadcasting."""
        out = schedule.gather(0, t)
        return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))

    # ---------------------------------------------------------------
    # Forward diffusion utilities
    # ---------------------------------------------------------------

    def q_sample(
        self,
        i0: torch.Tensor,
        i_res: torch.Tensor,
        i_in: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion (Algorithm 1, line 5 / Eq. 4):

            I_t = I_0 + alpha_bar_t * I_res + beta_bar_t * eps - delta_bar_t * I_in

        Args:
            i0:    GroundTruthHSI (I_0), shape (B, C, H, W)
            i_res: internal residual = MSTPrediction - GroundTruthHSI
            i_in:  conditioning image = MSTPrediction (I_in)
            t:     integer timesteps, shape (B,), 0-indexed into the schedule
            noise: optional pre-sampled standard Gaussian noise

        Returns:
            (i_t, noise) tuple.
        """
        if noise is None:
            noise = torch.randn_like(i0)

        alpha_bar_t = self._extract(self.alpha_bar, t, i0.shape)
        beta_bar_t = self._extract(self.beta_bar, t, i0.shape)
        delta_bar_t = self._extract(self.delta_bar, t, i0.shape)

        i_t = i0 + alpha_bar_t * i_res + beta_bar_t * noise - delta_bar_t * i_in
        return i_t, noise

    def get_terminal_distribution(
        self, i_in: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Shared-distribution diffusion endpoint (t -> T):

            I_T = (1 - delta_T) * I_in + beta_T * eps

        This is the impure-Gaussian shared distribution described in the
        paper: it depends only on the condition image (MSTPrediction) and
        noise, which is why sampling can start here without access to the
        ground truth.
        """
        if noise is None:
            noise = torch.randn_like(i_in)
        delta_T = self.delta_bar[-1]
        beta_T = self.beta_bar[-1]
        return (1.0 - delta_T) * i_in + beta_T * noise

    # ---------------------------------------------------------------
    # Reverse diffusion utilities
    # ---------------------------------------------------------------

    def epsilon_from_residual(
        self,
        i_t: torch.Tensor,
        i_in: torch.Tensor,
        i_res_pred: torch.Tensor,
        i0_pred: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recovers the implied noise eps^theta from the predicted internal
        residual and predicted I_0, via reparameterization of Eq. (4):

            eps = (I_t - I_0 + delta_bar_t * I_in - alpha_bar_t * I_res) / beta_bar_t
        """
        alpha_bar_t = self._extract(self.alpha_bar, t, i_t.shape)
        beta_bar_t = self._extract(self.beta_bar, t, i_t.shape)
        delta_bar_t = self._extract(self.delta_bar, t, i_t.shape)

        eps = (i_t - i0_pred + delta_bar_t * i_in - alpha_bar_t * i_res_pred) / beta_bar_t.clamp(min=1e-8)
        return eps

    def p_sample_ddpm(
        self,
        i_t: torch.Tensor,
        i_in: torch.Tensor,
        i_res_pred: torch.Tensor,
        t: torch.Tensor,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        DDPM reverse step (Eq. 6 / Eq. 7):

            I_{t-1} = I_t - alpha_t * I_res^theta + delta_t * I_in
                      - (beta_t^2 / beta_bar_t) * eps^theta
                      + (beta_t * beta_bar_{t-1} / beta_bar_t) * eps_*

        `i_res_pred` is the network's predicted internal residual
        (I_in - I_0 convention). `eps^theta` is derived from it via
        reparameterization, matching the paper's approach of directly
        predicting the residual and deriving the noise term from it.
        """
        alpha_t = self._extract(self.alpha, t, i_t.shape)
        beta_t = self._extract(self.beta, t, i_t.shape)
        delta_t = self._extract(self.delta, t, i_t.shape)
        beta_bar_t = self._extract(self.beta_bar, t, i_t.shape)

        t_prev = (t - 1).clamp(min=0)
        beta_bar_prev = self._extract(self.beta_bar, t_prev, i_t.shape)
        # At t == 0 there is no previous step; beta_bar_prev is unused there.
        is_first_step = (t == 0).reshape(t.shape[0], *([1] * (len(i_t.shape) - 1)))

        # I_0^theta is implicitly I_in - i_res_pred (paper's own reconstruction),
        # used only to recover eps^theta via reparameterization.
        i0_pred = i_in - i_res_pred
        eps_theta = self.epsilon_from_residual(i_t, i_in, i_res_pred, i0_pred, t)

        mean = (
            i_t
            - alpha_t * i_res_pred
            + delta_t * i_in
            - (beta_t ** 2 / beta_bar_t.clamp(min=1e-8)) * eps_theta
        )

        if add_noise:
            noise = torch.randn_like(i_t)
            sigma = beta_t * beta_bar_prev / beta_bar_t.clamp(min=1e-8)
            sample = mean + sigma * noise
        else:
            sample = mean

        # At the very last reverse step (t == 0) there is no stochastic term.
        sample = torch.where(is_first_step, mean, sample)
        return sample

    def p_sample_ddim(
        self,
        i_t: torch.Tensor,
        i_in: torch.Tensor,
        i_res_pred: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        DDIM reverse step (Eq. 8 / Algorithm 2), used for fast sampling:

            I_{t-1} = I_t - alpha_t * I_res^theta + delta_t * I_in     (t > 1)
            I_0     = I_in - I_res^theta                                (t == 1)
        """
        alpha_t = self._extract(self.alpha, t, i_t.shape)
        delta_t = self._extract(self.delta, t, i_t.shape)

        is_last_step = (t == 0).reshape(t.shape[0], *([1] * (len(i_t.shape) - 1)))

        deterministic_update = i_t - alpha_t * i_res_pred + delta_t * i_in
        final_reconstruction = i_in - i_res_pred

        return torch.where(is_last_step, final_reconstruction, deterministic_update)


# ==========================================================================
# Top-level model: Residual Diffusion for RGB -> HSI
# ==========================================================================

class ResidualDiffusionRGB2HSI(nn.Module):
    """
    Top-level residual diffusion model for RGB-to-Hyperspectral-Image
    reconstruction, adapted from DiffUIR.

    Components:
        - Frozen MST++            : RGB -> coarse HSI prediction (no_grad)
        - ResidualDiffusionScheduler : forward/reverse diffusion math
        - ResidualUNet             : predicts the (internal) residual

    The diffusion model never sees or predicts the full HSI directly --
    it only ever predicts the residual between the MST++ coarse
    prediction and the ground truth.
    """

    def __init__(
        self,
        rgb_channels: int = 3,
        hsi_channels: int = 31,
        num_timesteps: int = 1000,
        unet_base_channels: int = 64,
        unet_channel_multipliers: Tuple[int, ...] = (1, 2, 2, 4),
        unet_num_res_blocks: int = 2,
        unet_attention_resolutions: Tuple[int, ...] = (16,),
        unet_dropout: float = 0.0,
        alpha_bar_max: float = 1.0,
        beta_bar_min: float = 1e-4,
        beta_bar_max: float = 1.0,
        delta_bar_max: float = 0.9,
        schedule_type: str = "linear",
        mst_plus_plus_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.hsi_channels = hsi_channels
        self.num_timesteps = num_timesteps

        # ---------------- Frozen MST++ ----------------
        # Assumed fully implemented elsewhere; only instantiated here.
        # Pretrained weights are assumed to be loaded externally
        # (e.g. via `model.mst_plus_plus.load_state_dict(...)`) before
        # training/inference of this residual diffusion model.
        mst_plus_plus_kwargs = mst_plus_plus_kwargs or {}
        self.mst_plus_plus = MST_Plus_Plus(
            in_channels=rgb_channels, out_channels=hsi_channels, **mst_plus_plus_kwargs
        )
        self._freeze_mst_plus_plus()

        # ---------------- Diffusion scheduler ----------------
        self.scheduler = ResidualDiffusionScheduler(
            num_timesteps=num_timesteps,
            alpha_bar_max=alpha_bar_max,
            beta_bar_min=beta_bar_min,
            beta_bar_max=beta_bar_max,
            delta_bar_max=delta_bar_max,
            schedule_type=schedule_type,
        )

        # ---------------- Residual-predicting U-Net ----------------
        self.residual_unet = ResidualUNet(
            hsi_channels=hsi_channels,
            base_channels=unet_base_channels,
            channel_multipliers=unet_channel_multipliers,
            num_res_blocks=unet_num_res_blocks,
            attention_resolutions=unet_attention_resolutions,
            dropout=unet_dropout,
        )

    # ---------------------------------------------------------------
    # MST++ handling
    # ---------------------------------------------------------------

    def _freeze_mst_plus_plus(self) -> None:
        """Freezes all MST++ parameters so gradients never flow through it."""
        for param in self.mst_plus_plus.parameters():
            param.requires_grad_(False)
        self.mst_plus_plus.eval()

    def train(self, mode: bool = True):
        """
        Overridden so that calling `.train()` on the full model does not
        accidentally put the frozen MST++ submodule into training mode
        (e.g. re-enabling BatchNorm running-stat updates or Dropout).
        """
        super().train(mode)
        self.mst_plus_plus.eval()
        return self

    @torch.no_grad()
    def get_coarse_prediction(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Runs the frozen MST++ model to obtain the coarse HSI prediction.
        Always executed under `torch.no_grad()`; MST++ parameters are
        also frozen via `requires_grad_(False)`, so this is doubly safe
        against accidental gradient flow.
        """
        self.mst_plus_plus.eval()
        return self.mst_plus_plus(rgb)

    # ---------------------------------------------------------------
    # Training-time forward pass (no loss computed here)
    # ---------------------------------------------------------------

    def forward(
        self,
        rgb: torch.Tensor,
        gt_hsi: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Single training forward pass. Computes everything needed for an
        external loss function to be applied, but does NOT compute any
        loss itself.

        Args:
            rgb:    input RGB image, shape (B, rgb_channels, H, W)
            gt_hsi: ground-truth hyperspectral image, shape (B, hsi_channels, H, W)
            t:      optional integer timesteps, shape (B,). If None, sampled
                    uniformly at random per Algorithm 1.

        Returns a dict with:
            mst_prediction:      frozen MST++ coarse HSI prediction (I_in)
            target_residual:     GroundTruthHSI - MSTPrediction (public convention)
            predicted_residual:  network's predicted residual, same public
                                  convention as `target_residual`
            noisy_input:         I_t, the diffused residual representation
                                  fed into the U-Net
            timesteps:           the timesteps used, shape (B,)
        """
        mst_prediction = self.get_coarse_prediction(rgb)  # frozen, no_grad

        # Internal DiffUIR convention: I_res = I_in - I_0 = MSTPred - GT.
        internal_target_residual = mst_prediction - gt_hsi

        batch_size = rgb.shape[0]
        device = rgb.device
        if t is None:
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

        i_t, _ = self.scheduler.q_sample(
            i0=gt_hsi, i_res=internal_target_residual, i_in=mst_prediction, t=t
        )

        internal_predicted_residual = self.residual_unet(
            noisy_residual=i_t, condition=mst_prediction, timesteps=t
        )

        # Public-facing residual convention, as specified: GT - MSTPred.
        predicted_residual = -internal_predicted_residual
        target_residual = -internal_target_residual

        return {
            "mst_prediction": mst_prediction,
            "target_residual": target_residual,
            "predicted_residual": predicted_residual,
            "noisy_input": i_t,
            "timesteps": t,
        }

    # ---------------------------------------------------------------
    # Inference-time sampling (Algorithm 2)
    # ---------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        rgb: torch.Tensor,
        num_sampling_steps: Optional[int] = None,
        use_ddim: bool = True,
        return_intermediates: bool = False,
    ):
        """
        Full reverse-diffusion sampling pipeline (Algorithm 2), producing
        the final reconstructed hyperspectral image.

        Args:
            rgb:                 input RGB image, shape (B, rgb_channels, H, W)
            num_sampling_steps:  number of reverse steps to take (with
                                  uniform striding over the training
                                  timestep range, as in DDIM). Defaults to
                                  the full `num_timesteps`.
            use_ddim:             whether to use the DDIM update (Eq. 8,
                                  deterministic, fast) or the DDPM update
                                  (Eq. 7, stochastic).
            return_intermediates: if True, also returns the list of all
                                  intermediate noisy-residual states.

        Returns:
            final_hsi, and optionally the list of intermediate states.
        """
        mst_prediction = self.get_coarse_prediction(rgb)  # frozen, no_grad
        device = rgb.device
        batch_size = rgb.shape[0]

        num_sampling_steps = num_sampling_steps or self.num_timesteps
        # Uniformly strided timestep indices from T-1 down to 0.
        step_indices = torch.linspace(
            self.num_timesteps - 1, 0, num_sampling_steps, device=device
        ).long()
        step_indices = torch.unique_consecutive(step_indices)

        # Shared-distribution diffusion endpoint I_T = (1 - delta_T) I_in + beta_T * eps.
        i_t = self.scheduler.get_terminal_distribution(mst_prediction)

        intermediates = [i_t] if return_intermediates else None

        for step_idx in step_indices:
            t = step_idx.repeat(batch_size)
            internal_residual_pred = self.residual_unet(
                noisy_residual=i_t, condition=mst_prediction, timesteps=t
            )
            if use_ddim:
                i_t = self.scheduler.p_sample_ddim(
                    i_t=i_t, i_in=mst_prediction, i_res_pred=internal_residual_pred, t=t
                )
            else:
                add_noise = bool(step_idx.item() > 0)
                i_t = self.scheduler.p_sample_ddpm(
                    i_t=i_t, i_in=mst_prediction, i_res_pred=internal_residual_pred, t=t,
                    add_noise=add_noise,
                )
            if return_intermediates:
                intermediates.append(i_t)

        # After the final reverse step, i_t already equals I_0 (see
        # `p_sample_ddim` / `p_sample_ddpm` handling of t == 0). We also
        # expose the explicit reconstruction below for clarity, matching
        # the spec's `FinalHSI = MSTPrediction + PredictedResidual`:
        #
        #     final_predicted_residual (public convention, GT - MSTPred)
        #         = -(mst_prediction - i_t) = i_t - mst_prediction
        #   => FinalHSI = mst_prediction + final_predicted_residual = i_t
        final_hsi = i_t

        if return_intermediates:
            return final_hsi, intermediates
        return final_hsi
