"""Residual diffusion refinement for a frozen MST++ RGB-to-HSI model.

The MST++ output is treated as the degraded HSI estimate. The diffusion model
learns the correction residual

    residual = ground_truth_hsi - mst_prediction

and reconstructs the final HSI as

    refined_hsi = mst_prediction + predicted_residual.

Edit the MST++ import below to match your project structure.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Change this import to match the file and class name used in your project.
# Example alternatives:
#   from models.mst_plus_plus import MST_Plus_Plus
#   from architecture import MST_Plus_Plus
# -----------------------------------------------------------------------------
try:
    from .MST_Plus_Plus import MST_Plus_Plus
except ImportError:
    MST_Plus_Plus = None


class LayerNorm2d(nn.Module):
    """PyTorch LayerNorm applied over the channel dimension of an NCHW tensor."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for normalized diffusion timesteps."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even")
        self.embedding_dim = embedding_dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        scale = math.log(10_000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            -scale
            * torch.arange(
                half_dim,
                device=timestep.device,
                dtype=timestep.dtype,
            )
        )
        angles = timestep[:, None] * frequencies[None, :] * 1000.0
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class ResidualBlock(nn.Module):
    """Residual convolution block conditioned on the diffusion timestep."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_projection = nn.Linear(time_dim, out_channels)

        self.norm2 = LayerNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(time_embedding)[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(x)


class ResidualUNet(nn.Module):
    """Small U-Net that predicts the clean HSI correction residual."""

    def __init__(
        self,
        hsi_channels: int = 31,
        rgb_channels: int = 3,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        input_channels = (2 * hsi_channels) + rgb_channels
        time_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(input_channels, base_channels, 3, padding=1)

        self.encoder1 = ResidualBlock(base_channels, base_channels, time_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)

        self.encoder2 = ResidualBlock(
            base_channels * 2,
            base_channels * 2,
            time_dim,
        )
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)

        self.middle = ResidualBlock(
            base_channels * 4,
            base_channels * 4,
            time_dim,
        )

        self.up2 = nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1)
        self.decoder2 = ResidualBlock(
            base_channels * 4,
            base_channels * 2,
            time_dim,
        )

        self.up1 = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.decoder1 = ResidualBlock(
            base_channels * 2,
            base_channels,
            time_dim,
        )

        self.output_norm = LayerNorm2d(base_channels)
        self.output_conv = nn.Conv2d(base_channels, hsi_channels, 3, padding=1)

    def forward(
        self,
        noisy_residual: torch.Tensor,
        mst_prediction: torch.Tensor,
        rgb: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if rgb.shape[-2:] != mst_prediction.shape[-2:]:
            rgb = F.interpolate(
                rgb,
                size=mst_prediction.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        time_embedding = self.time_mlp(timestep)
        x = torch.cat((noisy_residual, mst_prediction, rgb), dim=1)

        x = self.input_conv(x)
        skip1 = self.encoder1(x, time_embedding)

        x = self.down1(skip1)
        skip2 = self.encoder2(x, time_embedding)

        x = self.down2(skip2)
        x = self.middle(x, time_embedding)

        x = F.interpolate(x, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(x)
        x = self.decoder2(torch.cat((x, skip2), dim=1), time_embedding)

        x = F.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(x)
        x = self.decoder1(torch.cat((x, skip1), dim=1), time_embedding)

        return self.output_conv(F.silu(self.output_norm(x)))


def _find_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Extract a state dictionary from common checkpoint formats."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("The MST++ checkpoint must contain a state dictionary.")

    common_keys = (
        "mst_state_dict",
        "model_state_dict",
        "state_dict",
        "model",
        "network",
        "params",
    )
    for key in common_keys:
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint

    raise KeyError("Could not locate the MST++ state dictionary in the checkpoint.")


def _remove_prefix(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return dict(state_dict)


class MSTResidualDiffusion(nn.Module):
    """Frozen MST++ followed by residual-space diffusion refinement.

    You can either let this class construct MST++ from the imported architecture,
    or pass an already-created MST++ instance through ``mst_model``.
    """

    def __init__(
        self,
        mst_checkpoint: Optional[str | Path] = None,
        mst_model: Optional[nn.Module] = None,
        mst_model_kwargs: Optional[Dict[str, Any]] = None,
        hsi_channels: int = 31,
        rgb_channels: int = 3,
        base_channels: int = 64,
        num_steps: int = 15,
        kappa: float = 2.0,
        strict_checkpoint: bool = True,
    ) -> None:
        super().__init__()

        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        if kappa <= 0:
            raise ValueError("kappa must be positive")

        if mst_model is None:
            if MST_Plus_Plus is None:
                raise ImportError(
                    "MST++ could not be imported. Edit the MST_Plus_Plus import "
                    "near the top of this file, or pass mst_model explicitly."
                )
            mst_model = MST_Plus_Plus(**(mst_model_kwargs or {}))

        self.mst_model = mst_model
        self.num_steps = int(num_steps)
        self.kappa = float(kappa)

        if mst_checkpoint is not None:
            checkpoint = torch.load(mst_checkpoint, map_location="cpu")
            state_dict = _find_state_dict(checkpoint)
            state_dict = _remove_prefix(state_dict, "module.")
            state_dict = _remove_prefix(state_dict, "mst_model.")
            self.mst_model.load_state_dict(
                state_dict,
                strict=strict_checkpoint,
            )

        for parameter in self.mst_model.parameters():
            parameter.requires_grad = False
        self.mst_model.eval()

        self.residual_net = ResidualUNet(
            hsi_channels=hsi_channels,
            rgb_channels=rgb_channels,
            base_channels=base_channels,
        )

        # beta[0] = 0 gives the clean residual; beta[T] = 1 gives noise.
        beta = torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float32)
        self.register_buffer("beta", beta, persistent=True)

    def train(self, mode: bool = True) -> "MSTResidualDiffusion":
        super().train(mode)
        self.mst_model.eval()
        return self

    @staticmethod
    def _extract_mst_output(output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            return output

        if isinstance(output, (tuple, list)):
            if output and torch.is_tensor(output[0]):
                return output[0]
            raise TypeError("The first MST++ output must be a tensor.")

        if isinstance(output, Mapping):
            for key in ("output", "prediction", "pred", "hsi", "reconstruction"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
            raise KeyError("No HSI tensor was found in the MST++ output dictionary.")

        raise TypeError(f"Unsupported MST++ output type: {type(output)!r}")

    @torch.no_grad()
    def get_mst_prediction(self, rgb: torch.Tensor) -> torch.Tensor:
        prediction = self._extract_mst_output(self.mst_model(rgb))
        return prediction.detach()

    def _beta_at(
        self,
        timestep: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return self.beta[timestep].to(reference.dtype).view(-1, 1, 1, 1)

    def q_sample(
        self,
        clean_residual: torch.Tensor,
        timestep: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diffuse the clean residual toward Gaussian noise."""
        if noise is None:
            noise = torch.randn_like(clean_residual)

        beta_t = self._beta_at(timestep, clean_residual)
        noisy_residual = (
            (1.0 - beta_t) * clean_residual
            + self.kappa * torch.sqrt(beta_t) * noise
        )
        return noisy_residual, noise

    def predict_residual(
        self,
        noisy_residual: torch.Tensor,
        mst_prediction: torch.Tensor,
        rgb: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        normalized_timestep = timestep.to(noisy_residual.dtype) / self.num_steps
        return self.residual_net(
            noisy_residual,
            mst_prediction,
            rgb,
            normalized_timestep,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        ground_truth_hsi: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run one training-time diffusion pass."""
        mst_prediction = self.get_mst_prediction(rgb)

        if mst_prediction.shape != ground_truth_hsi.shape:
            raise ValueError(
                "MST++ output and ground-truth HSI must have identical shapes. "
                f"Received {tuple(mst_prediction.shape)} and "
                f"{tuple(ground_truth_hsi.shape)}."
            )

        batch_size = ground_truth_hsi.shape[0]
        if timestep is None:
            timestep = torch.randint(
                1,
                self.num_steps + 1,
                (batch_size,),
                device=ground_truth_hsi.device,
                dtype=torch.long,
            )
        else:
            timestep = timestep.to(ground_truth_hsi.device, dtype=torch.long)

        target_residual = ground_truth_hsi - mst_prediction
        noisy_residual, used_noise = self.q_sample(
            target_residual,
            timestep,
            noise,
        )
        predicted_residual = self.predict_residual(
            noisy_residual,
            mst_prediction,
            rgb,
            timestep,
        )

        return {
            "predicted_residual": predicted_residual,
            "target_residual": target_residual,
            "reconstruction": mst_prediction + predicted_residual,
            "mst_prediction": mst_prediction,
            "noisy_residual": noisy_residual,
            "noise": used_noise,
            "timestep": timestep,
        }

    @torch.no_grad()
    def sample(
        self,
        rgb: torch.Tensor,
        clamp_range: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Refine an MST++ prediction using the learned reverse process."""
        mst_prediction = self.get_mst_prediction(rgb)
        residual_state = self.kappa * torch.randn_like(mst_prediction)
        batch_size = rgb.shape[0]

        for step in range(self.num_steps, 0, -1):
            timestep = torch.full(
                (batch_size,),
                step,
                device=rgb.device,
                dtype=torch.long,
            )
            predicted_residual = self.predict_residual(
                residual_state,
                mst_prediction,
                rgb,
                timestep,
            )

            beta_t = self.beta[step].to(residual_state.dtype)
            beta_previous = self.beta[step - 1].to(residual_state.dtype)
            alpha_t = beta_t - beta_previous

            posterior_mean = (
                (beta_previous / beta_t) * residual_state
                + (alpha_t / beta_t) * predicted_residual
            )

            if step > 1:
                posterior_variance = (
                    self.kappa**2
                    * (beta_previous / beta_t)
                    * alpha_t
                )
                residual_state = posterior_mean + torch.sqrt(
                    posterior_variance
                ) * torch.randn_like(residual_state)
            else:
                residual_state = posterior_mean

        reconstruction = mst_prediction + residual_state
        if clamp_range is not None:
            reconstruction = reconstruction.clamp(*clamp_range)

        return {
            "reconstruction": reconstruction,
            "predicted_residual": residual_state,
            "mst_prediction": mst_prediction,
        }
