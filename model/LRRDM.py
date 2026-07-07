"""Low-rank residual diffusion refinement for a frozen MST++ reconstructor.

The model learns the residual between an MST++ hyperspectral estimate and the
paired ground-truth HSI cube. The forward process uses a truncated-SVD residual,
while the reverse denoiser remains a full-resolution/full-rank neural network.

Expected tensors
----------------
rgb:          [B, 3, H, W]
ground_truth: [B, C, H, W]
MST++ output: [B, C, H, W]

MST++ is imported from the local project and instantiated inside
``MSTPlusPlusLRDM``. Change only the import line below if your repository uses a
different module path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

# Adjust this single import path if your repository stores MST++ elsewhere.
from .MST_Plus_Plus import MST_Plus_Plus


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


class LayerNorm2d(nn.Module):
    """Apply PyTorch ``nn.LayerNorm`` over channels of an NCHW feature map."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"LayerNorm2d expects [B,C,H,W], got {tuple(x.shape)}")
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


def _extract_tensor(output: object) -> Tensor:
    """Extract a reconstruction tensor from common model output containers."""
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item

    if isinstance(output, dict):
        preferred_keys = (
            "out",
            "output",
            "reconstruction",
            "recon",
            "prediction",
            "pred",
            "hsi",
        )
        for key in preferred_keys:
            value = output.get(key)
            if torch.is_tensor(value):
                return value
        for value in output.values():
            if torch.is_tensor(value):
                return value

    raise TypeError(
        "Could not extract a tensor from the MST++ output. "
        "Modify _extract_tensor() for your repository's return format."
    )


def extract(values: Tensor, timesteps: Tensor, reference: Tensor) -> Tensor:
    """Gather a 1-D diffusion schedule and reshape it for image broadcasting."""
    gathered = values.gather(0, timesteps.long())
    return gathered.view(reference.shape[0], *([1] * (reference.ndim - 1)))


# -----------------------------------------------------------------------------
# Adaptive rank and low-rank projection
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RankConfig:
    """Configuration for the deterministic rank schedule k(t)."""

    schedule: str = "poly_decrease"
    min_rank: int = 1
    max_rank: int = 31
    polynomial_order: int = 3


class AdaptiveRankScheduler(nn.Module):
    """Adaptive rank schedules described in LRDM.

    Supported schedules:
        fixed, linear_increase, linear_decrease,
        poly_increase, poly_decrease
    """

    def __init__(self, num_timesteps: int, config: RankConfig) -> None:
        super().__init__()
        if num_timesteps < 1:
            raise ValueError("num_timesteps must be positive.")
        if config.min_rank < 1:
            raise ValueError("min_rank must be at least 1.")
        if config.max_rank < config.min_rank:
            raise ValueError("max_rank must be >= min_rank.")
        if config.polynomial_order < 1:
            raise ValueError("polynomial_order must be >= 1.")

        valid = {
            "fixed",
            "linear_increase",
            "linear_decrease",
            "poly_increase",
            "poly_decrease",
        }
        if config.schedule not in valid:
            raise ValueError(f"Unknown rank schedule {config.schedule!r}. Valid: {sorted(valid)}")

        self.num_timesteps = int(num_timesteps)
        self.config = config

    def forward(self, timesteps: Tensor) -> Tensor:
        t = timesteps.float().clamp(0, self.num_timesteps)
        d = t / float(self.num_timesteps)

        lo = float(self.config.min_rank)
        hi = float(self.config.max_rank)
        span = hi - lo
        schedule = self.config.schedule

        if schedule == "fixed":
            fraction = torch.ones_like(d)
        elif schedule == "linear_increase":
            fraction = d
        elif schedule == "linear_decrease":
            fraction = 1.0 - d
        else:
            p = float(self.config.polynomial_order)
            a = -((p + 1.0) * (p + 2.0)) / 2.0
            b = p * (p + 2.0)
            c = -(p * (p + 1.0)) / 2.0
            envelope = 1.0 + a * d.pow(p) + b * d.pow(p + 1.0) + c * d.pow(p + 2.0)
            envelope = envelope.clamp(0.0, 1.0)
            fraction = 1.0 - envelope if schedule == "poly_increase" else envelope

        ranks = torch.ceil(lo + span * fraction)
        return ranks.long().clamp(self.config.min_rank, self.config.max_rank)


class TruncatedSVDProjector(nn.Module):
    """Project HSI residuals into a low-rank subspace using truncated SVD.

    Modes
    -----
    spectral:
        Reshape each cube from [C,H,W] to [C,H*W]. This constrains the
        spectral-spatial residual rank and is usually the practical choice for
        RGB-to-HSI reconstruction because C is small (for example, 31).

    spatial:
        Apply an independent HxW truncated SVD to every spectral band. This is
        closer to treating each image band as the 2-D residual matrix.
    """

    def __init__(self, mode: str = "spectral") -> None:
        super().__init__()
        if mode not in {"spectral", "spatial"}:
            raise ValueError("projection mode must be 'spectral' or 'spatial'.")
        self.mode = mode

    @torch.no_grad()
    def forward(self, residual: Tensor, ranks: Tensor) -> Tensor:
        if residual.ndim != 4:
            raise ValueError(f"Expected residual [B,C,H,W], got {tuple(residual.shape)}")
        if ranks.ndim != 1 or ranks.shape[0] != residual.shape[0]:
            raise ValueError("ranks must have shape [B].")

        original_dtype = residual.dtype
        # torch.linalg.svd is more reliable in float32 than float16/bfloat16.
        matrix_input = residual.float()

        if self.mode == "spectral":
            b, c, h, w = matrix_input.shape
            matrix = matrix_input.reshape(b, c, h * w)
            projected = self._batched_truncated_svd(matrix, ranks)
            return projected.reshape(b, c, h, w).to(original_dtype)

        b, c, h, w = matrix_input.shape
        matrix = matrix_input.reshape(b * c, h, w)
        repeated_ranks = ranks.repeat_interleave(c)
        projected = self._batched_truncated_svd(matrix, repeated_ranks)
        return projected.reshape(b, c, h, w).to(original_dtype)

    @staticmethod
    def _batched_truncated_svd(matrix: Tensor, ranks: Tensor) -> Tensor:
        max_possible_rank = min(matrix.shape[-2], matrix.shape[-1])
        ranks = ranks.clamp(1, max_possible_rank)

        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        component_ids = torch.arange(s.shape[-1], device=s.device).unsqueeze(0)
        keep = component_ids < ranks.unsqueeze(1)
        s = s * keep.to(s.dtype)
        return (u * s.unsqueeze(-2)) @ vh


# -----------------------------------------------------------------------------
# Simple full-rank conditional U-Net denoiser
# -----------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: Tensor) -> Tensor:
        half_dim = self.embedding_dim // 2
        if half_dim < 1:
            raise ValueError("embedding_dim must be at least 2.")

        exponent = -math.log(10_000.0) * torch.arange(
            half_dim, device=timesteps.device, dtype=torch.float32
        ) / max(half_dim - 1, 1)
        frequencies = exponent.exp()
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class TimeResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 2 * out_channels),
        )
        self.norm2 = LayerNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_projection(time_embedding).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class SimpleConditionalUNet(nn.Module):
    """Predict both the low-rank residual and Gaussian noise.

    The network is deliberately full-rank: no SVD or low-rank factorization is
    imposed on its reverse-process output.
    """

    def __init__(
        self,
        hsi_channels: int = 31,
        base_channels: int = 64,
        time_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if hsi_channels < 1 or base_channels < 8:
            raise ValueError("Invalid channel configuration.")

        time_dim = time_dim or 4 * base_channels
        self.hsi_channels = hsi_channels

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # x_t and MST++ estimate are concatenated as the image condition.
        self.input_conv = nn.Conv2d(2 * hsi_channels, base_channels, 3, padding=1)

        self.down_block1 = TimeResidualBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(base_channels, 2 * base_channels, 4, stride=2, padding=1)

        self.down_block2 = TimeResidualBlock(2 * base_channels, 2 * base_channels, time_dim)
        self.downsample2 = nn.Conv2d(2 * base_channels, 4 * base_channels, 4, stride=2, padding=1)

        self.middle_block1 = TimeResidualBlock(4 * base_channels, 4 * base_channels, time_dim)
        self.middle_block2 = TimeResidualBlock(4 * base_channels, 4 * base_channels, time_dim)

        self.up_reduce2 = nn.Conv2d(4 * base_channels, 2 * base_channels, 3, padding=1)
        self.up_block2 = TimeResidualBlock(4 * base_channels, 2 * base_channels, time_dim)

        self.up_reduce1 = nn.Conv2d(2 * base_channels, base_channels, 3, padding=1)
        self.up_block1 = TimeResidualBlock(2 * base_channels, base_channels, time_dim)

        self.output_norm = LayerNorm2d(base_channels)
        self.output_conv = nn.Conv2d(base_channels, 2 * hsi_channels, 3, padding=1)

        # Start close to zero prediction, which is often stable for diffusion heads.
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x_t: Tensor, timesteps: Tensor, condition: Tensor) -> Tuple[Tensor, Tensor]:
        if x_t.shape != condition.shape:
            raise ValueError(
                f"x_t and condition must match, got {tuple(x_t.shape)} and {tuple(condition.shape)}"
            )

        time_embedding = self.time_mlp(timesteps)
        x = self.input_conv(torch.cat((x_t, condition), dim=1))

        skip1 = self.down_block1(x, time_embedding)
        skip2 = self.down_block2(self.downsample1(skip1), time_embedding)

        middle = self.downsample2(skip2)
        middle = self.middle_block1(middle, time_embedding)
        middle = self.middle_block2(middle, time_embedding)

        up2 = F.interpolate(middle, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.up_reduce2(up2)
        up2 = self.up_block2(torch.cat((up2, skip2), dim=1), time_embedding)

        up1 = F.interpolate(up2, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.up_reduce1(up1)
        up1 = self.up_block1(torch.cat((up1, skip1), dim=1), time_embedding)

        output = self.output_conv(F.silu(self.output_norm(up1)))
        predicted_residual, predicted_noise = output.chunk(2, dim=1)
        return predicted_residual, predicted_noise


# -----------------------------------------------------------------------------
# MST++ + LRDM wrapper
# -----------------------------------------------------------------------------


class MSTPlusPlusLRDM(nn.Module):
    """Refine a frozen MST++ estimate using low-rank residual diffusion.

    Residual convention
    -------------------
    residual = mst_estimate - ground_truth
    ground_truth = mst_estimate - residual

    Forward training state
    ----------------------
    x_t = ground_truth
          + alpha_bar[t] * Q_k(t)(residual)
          + beta_bar[t] * noise
    """

    def __init__(
        self,
        hsi_channels: int = 31,
        mst_model_kwargs: Optional[Dict[str, object]] = None,
        num_timesteps: int = 10,
        noise_scale: float = 0.1,
        rank_config: Optional[RankConfig] = None,
        projection_mode: str = "spectral",
        base_channels: int = 64,
        residual_loss_weight: float = 1.0,
        noise_loss_weight: float = 1.0,
        x0_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if num_timesteps < 1:
            raise ValueError("num_timesteps must be positive.")
        if noise_scale <= 0:
            raise ValueError("noise_scale must be positive.")

        self.mst_model = MST_Plus_Plus(**(mst_model_kwargs or {}))
        self.hsi_channels = int(hsi_channels)
        self.num_timesteps = int(num_timesteps)
        self.residual_loss_weight = float(residual_loss_weight)
        self.noise_loss_weight = float(noise_loss_weight)
        self.x0_loss_weight = float(x0_loss_weight)

        self._freeze_mst_model()

        rank_config = rank_config or RankConfig(max_rank=hsi_channels)
        self.rank_scheduler = AdaptiveRankScheduler(num_timesteps, rank_config)
        self.projector = TruncatedSVDProjector(mode=projection_mode)
        self.denoiser = SimpleConditionalUNet(
            hsi_channels=hsi_channels,
            base_channels=base_channels,
        )

        # RDDM-style cumulative schedules:
        # alpha_bar[t] = sum_i alpha_i = t/T
        # beta_bar[t]^2 = sum_i beta_i^2 = noise_scale^2 * t/T
        steps = torch.arange(num_timesteps + 1, dtype=torch.float32)
        alpha_bar = steps / float(num_timesteps)
        beta_bar = noise_scale * torch.sqrt(steps / float(num_timesteps))
        beta_step_sq = torch.zeros_like(beta_bar)
        beta_step_sq[1:] = beta_bar[1:].square() - beta_bar[:-1].square()

        self.register_buffer("alpha_bar", alpha_bar, persistent=True)
        self.register_buffer("beta_bar", beta_bar, persistent=True)
        self.register_buffer("beta_step_sq", beta_step_sq, persistent=True)

    def _freeze_mst_model(self) -> None:
        self.mst_model.requires_grad_(False)
        self.mst_model.eval()

    def train(self, mode: bool = True) -> "MSTPlusPlusLRDM":
        # Train the diffusion model, but never put frozen MST++ into training mode.
        super().train(mode)
        self.mst_model.eval()
        return self

    @torch.no_grad()
    def mst_reconstruction(self, rgb: Tensor) -> Tensor:
        estimate = _extract_tensor(self.mst_model(rgb))
        if estimate.ndim != 4:
            raise ValueError(f"MST++ must return [B,C,H,W], got {tuple(estimate.shape)}")
        if estimate.shape[1] != self.hsi_channels:
            raise ValueError(
                f"Expected {self.hsi_channels} HSI channels, got {estimate.shape[1]}."
            )
        return estimate

    def q_sample(
        self,
        ground_truth: Tensor,
        mst_estimate: Tensor,
        timesteps: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Construct x_t using the timestep-dependent low-rank residual."""
        if ground_truth.shape != mst_estimate.shape:
            raise ValueError(
                "Ground truth and MST++ estimate must have identical shapes, got "
                f"{tuple(ground_truth.shape)} and {tuple(mst_estimate.shape)}."
            )
        if timesteps.shape != (ground_truth.shape[0],):
            raise ValueError("timesteps must have shape [B].")

        full_residual = mst_estimate - ground_truth
        ranks = self.rank_scheduler(timesteps)
        low_rank_residual = self.projector(full_residual, ranks)

        noise = torch.randn_like(ground_truth) if noise is None else noise
        if noise.shape != ground_truth.shape:
            raise ValueError("noise must match ground_truth shape.")

        alpha_t = extract(self.alpha_bar, timesteps, ground_truth)
        beta_t = extract(self.beta_bar, timesteps, ground_truth)
        x_t = ground_truth + alpha_t * low_rank_residual + beta_t * noise

        return {
            "x_t": x_t,
            "noise": noise,
            "full_residual": full_residual,
            "low_rank_residual": low_rank_residual,
            "ranks": ranks,
        }

    def training_predictions(
        self,
        rgb: Tensor,
        ground_truth: Tensor,
        timesteps: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Return predictions, targets, and LRDM losses for one training batch."""
        mst_estimate = self.mst_reconstruction(rgb)
        if mst_estimate.shape != ground_truth.shape:
            raise ValueError(
                "MST++ output and ground truth must match exactly. "
                f"Got {tuple(mst_estimate.shape)} and {tuple(ground_truth.shape)}."
            )

        batch_size = ground_truth.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                1,
                self.num_timesteps + 1,
                (batch_size,),
                device=ground_truth.device,
            )
        else:
            timesteps = timesteps.to(device=ground_truth.device, dtype=torch.long)
            if torch.any((timesteps < 1) | (timesteps > self.num_timesteps)):
                raise ValueError(f"Training timesteps must lie in [1, {self.num_timesteps}].")

        diffusion = self.q_sample(
            ground_truth=ground_truth,
            mst_estimate=mst_estimate,
            timesteps=timesteps,
            noise=noise,
        )

        predicted_residual, predicted_noise = self.denoiser(
            diffusion["x_t"],
            timesteps,
            mst_estimate,
        )

        alpha_t = extract(self.alpha_bar, timesteps, ground_truth)
        beta_t = extract(self.beta_bar, timesteps, ground_truth)
        predicted_ground_truth = (
            diffusion["x_t"]
            - alpha_t * predicted_residual
            - beta_t * predicted_noise
        )

        residual_loss = F.mse_loss(predicted_residual, diffusion["low_rank_residual"])
        noise_loss = F.mse_loss(predicted_noise, diffusion["noise"])
        x0_loss = F.l1_loss(predicted_ground_truth, ground_truth)

        total_loss = (
            self.residual_loss_weight * residual_loss
            + self.noise_loss_weight * noise_loss
            + self.x0_loss_weight * x0_loss
        )

        return {
            "loss": total_loss,
            "residual_loss": residual_loss,
            "noise_loss": noise_loss,
            "x0_loss": x0_loss,
            "mst_estimate": mst_estimate,
            "x_t": diffusion["x_t"],
            "target_residual": diffusion["low_rank_residual"],
            "full_residual": diffusion["full_residual"],
            "target_noise": diffusion["noise"],
            "predicted_residual": predicted_residual,
            "predicted_noise": predicted_noise,
            "predicted_ground_truth": predicted_ground_truth,
            "timesteps": timesteps,
            "ranks": diffusion["ranks"],
        }

    def forward(
        self,
        rgb: Tensor,
        ground_truth: Tensor,
        timesteps: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        return self.training_predictions(rgb, ground_truth, timesteps, noise)

    @torch.no_grad()
    def sample(
        self,
        rgb: Tensor,
        eta: float = 0.0,
        initial_noise: Optional[Tensor] = None,
        clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
        return_trajectory: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """Run the full reverse LRDM chain.

        Args:
            rgb: RGB condition [B,3,H,W].
            eta: 0 gives deterministic sampling; values in [0,1] add stochasticity.
            initial_noise: Optional noise matching the MST++ output.
            clamp: Optional output range. Use None when training data are standardized.
            return_trajectory: Also return states from T down to 0.
        """
        if not 0.0 <= eta <= 1.0:
            raise ValueError("eta must lie in [0, 1].")

        mst_estimate = self.mst_reconstruction(rgb)
        noise = torch.randn_like(mst_estimate) if initial_noise is None else initial_noise
        if noise.shape != mst_estimate.shape:
            raise ValueError("initial_noise must match the MST++ output shape.")

        # Practical RDDM initialization: begin from the source estimate plus noise.
        x_t = mst_estimate + self.beta_bar[self.num_timesteps] * noise
        trajectory: List[Tensor] = [x_t.detach().clone()] if return_trajectory else []

        for current_t in range(self.num_timesteps, 0, -1):
            previous_t = current_t - 1
            timesteps = torch.full(
                (x_t.shape[0],),
                current_t,
                device=x_t.device,
                dtype=torch.long,
            )

            predicted_residual, predicted_noise = self.denoiser(
                x_t,
                timesteps,
                mst_estimate,
            )

            alpha_current = self.alpha_bar[current_t]
            alpha_previous = self.alpha_bar[previous_t]
            beta_current = self.beta_bar[current_t]
            beta_previous = self.beta_bar[previous_t]

            sigma_sq = (
                eta
                * self.beta_step_sq[current_t]
                * beta_previous.square()
                / beta_current.square().clamp_min(1e-12)
            )
            sigma = sigma_sq.clamp_min(0.0).sqrt()
            noise_coefficient = (
                beta_current
                - (beta_previous.square() - sigma_sq).clamp_min(0.0).sqrt()
            )

            stochastic_noise = (
                torch.randn_like(x_t)
                if previous_t > 0 and float(sigma) > 0.0
                else torch.zeros_like(x_t)
            )

            x_t = (
                x_t
                - (alpha_current - alpha_previous) * predicted_residual
                - noise_coefficient * predicted_noise
                + sigma * stochastic_noise
            )

            if return_trajectory:
                trajectory.append(x_t.detach().clone())

        if clamp is not None:
            x_t = x_t.clamp(*clamp)

        if return_trajectory:
            trajectory[-1] = x_t.detach().clone()
            return x_t, trajectory
        return x_t


__all__ = [
    "LayerNorm2d",
    "RankConfig",
    "AdaptiveRankScheduler",
    "TruncatedSVDProjector",
    "SimpleConditionalUNet",
    "MSTPlusPlusLRDM",
]
