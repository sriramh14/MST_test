\from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# Adjust this import path to match your project structure.
from model.MST_Plus_Plus import MST, MST_Plus_Plus


# ============================================================================
# Checkpoint helpers
# ============================================================================

def _load_checkpoint(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strip_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {
            key[len(prefix):]: value
            for key, value in state_dict.items()
        }
    return state_dict


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint)}"
        )

    state_dict = None

    for key in (
        "model_state_dict",
        "state_dict",
        "mst_state_dict",
        "mst_model_state_dict",
        "model",
        "params",
    ):
        value = checkpoint.get(key)
        if (
            isinstance(value, dict)
            and value
            and all(torch.is_tensor(item) for item in value.values())
        ):
            state_dict = value
            break

    if state_dict is None:
        if (
            checkpoint
            and all(torch.is_tensor(item) for item in checkpoint.values())
        ):
            state_dict = checkpoint
        else:
            raise KeyError(
                "Could not find an MST++ state_dict in the checkpoint."
            )

    for prefix in (
        "module.",
        "mst_model.",
        "coarse_model.",
    ):
        state_dict = _strip_prefix(
            state_dict,
            prefix,
        )

    return state_dict


def _extract_tensor_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        for item in reversed(output):
            if torch.is_tensor(item):
                return item

    if isinstance(output, dict):
        for key in (
            "prediction",
            "reconstruction",
            "output",
            "out",
            "hsi",
        ):
            value = output.get(key)
            if torch.is_tensor(value):
                return value

        for value in output.values():
            if torch.is_tensor(value):
                return value

    raise TypeError(
        "MST++ must return a tensor, a sequence containing a tensor, "
        "or a dictionary containing a tensor."
    )


# ============================================================================
# Time embedding
# ============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        if dim < 2:
            raise ValueError(
                "Time-embedding dimension must be at least 2."
            )

        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2

        if half == 1:
            frequencies = torch.ones(
                1,
                device=timestep.device,
                dtype=timestep.dtype,
            )
        else:
            frequencies = torch.exp(
                -math.log(10000.0)
                * torch.arange(
                    half,
                    device=timestep.device,
                    dtype=timestep.dtype,
                )
                / (half - 1)
            )

        angles = timestep[:, None] * frequencies[None, :]

        embedding = torch.cat(
            [
                angles.sin(),
                angles.cos(),
            ],
            dim=-1,
        )

        if embedding.shape[-1] < self.dim:
            embedding = F.pad(
                embedding,
                (0, self.dim - embedding.shape[-1]),
            )

        return embedding


# ============================================================================
# MST denoiser
# ============================================================================

class MSTBBDMDenoiser(nn.Module):
    """
    Predict the standard BBDM bridge objective.

    For the default ``grad`` objective:

        target_objective = x_t - x_0

    Therefore the clean HSI estimate is recovered as:

        x_0_hat = x_t - predicted_objective
    """

    def __init__(
        self,
        hsi_channels: int = 31,
        rgb_channels: int = 3,
        n_feat: int = 31,
        body_depth: int = 3,
        mst_stage: int = 2,
        num_blocks: Sequence[int] = (1, 1, 1),
    ):
        super().__init__()

        self.hsi_channels = hsi_channels
        self.rgb_channels = rgb_channels
        self.pad_multiple = 2 ** mst_stage

        input_channels = (
            hsi_channels
            + hsi_channels
            + rgb_channels
        )

        self.conv_in = nn.Conv2d(
            input_channels,
            n_feat,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(n_feat),
            nn.Linear(n_feat, n_feat * 4),
            nn.GELU(),
            nn.Linear(n_feat * 4, n_feat),
        )

        self.body = nn.Sequential(
            *[
                MST(
                    in_dim=n_feat,
                    out_dim=n_feat,
                    dim=n_feat,
                    stage=mst_stage,
                    num_blocks=list(num_blocks),
                )
                for _ in range(body_depth)
            ]
        )

        self.conv_out = nn.Conv2d(
            n_feat,
            hsi_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        coarse_hsi: torch.Tensor,
        rgb: torch.Tensor,
        t: torch.Tensor,
        total_steps: int,
    ) -> torch.Tensor:
        if x_t.shape != coarse_hsi.shape:
            raise ValueError(
                "x_t and coarse_hsi must have identical shapes, "
                f"but received {x_t.shape} and {coarse_hsi.shape}."
            )

        if rgb.shape[0] != x_t.shape[0]:
            raise ValueError(
                "RGB and HSI tensors must have the same batch size."
            )

        if rgb.shape[-2:] != x_t.shape[-2:]:
            raise ValueError(
                "RGB and HSI tensors must have the same spatial size."
            )

        _, _, height, width = x_t.shape

        pad_height = (
            self.pad_multiple - height % self.pad_multiple
        ) % self.pad_multiple

        pad_width = (
            self.pad_multiple - width % self.pad_multiple
        ) % self.pad_multiple

        # The imported MST architecture contains depthwise positional
        # convolutions that can fail under FP16/BF16 autocast on some Kaggle
        # GPU/cuDNN combinations. Run the trainable MST denoiser in FP32.
        #
        # Autocast is disabled only for this denoiser. Gradients are still
        # calculated normally, so the MST denoiser remains fully trainable.
        with torch.autocast(
            device_type=x_t.device.type,
            enabled=False,
        ):
            inputs = torch.cat(
                [
                    x_t.float(),
                    coarse_hsi.float(),
                    rgb.float(),
                ],
                dim=1,
            ).contiguous()

            if pad_height or pad_width:
                padding_mode = (
                    "reflect"
                    if height > pad_height and width > pad_width
                    else "replicate"
                )

                inputs = F.pad(
                    inputs,
                    (0, pad_width, 0, pad_height),
                    mode=padding_mode,
                )

            features = self.conv_in(inputs)

            normalized_timestep = (
                t.float() / float(total_steps)
            )

            time_features = self.time_mlp(
                normalized_timestep
            ).float()

            features = (
                features
                + time_features[:, :, None, None]
            )

            features = self.body(
                features.contiguous()
            )

            predicted_objective = self.conv_out(
                features
            )

        return predicted_objective[
            :,
            :,
            :height,
            :width,
        ].contiguous()


# ============================================================================
# Standard Brownian Bridge Diffusion Model
# ============================================================================

class BrownianBridgeDiffusion(nn.Module):
    """
    Brownian bridge from the clean HSI x_0 to the frozen MST++ endpoint y.

    Forward process:

        x_t = (1 - m_t) x_0 + m_t y + sqrt(delta_t) epsilon

    Standard ``grad`` training objective:

        objective_t
            = m_t (y - x_0) + sqrt(delta_t) epsilon
            = x_t - x_0

    The denoiser does not predict ``x_0 - y``. It predicts the timestep-
    dependent Brownian-bridge objective.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        num_timesteps: int = 50,
        midpoint_variance: float = 0.05,
    ):
        super().__init__()

        if num_timesteps < 2:
            raise ValueError(
                "num_timesteps must be at least 2."
            )

        if midpoint_variance <= 0:
            raise ValueError(
                "midpoint_variance must be positive."
            )

        self.denoiser = denoiser
        self.num_timesteps = num_timesteps

        m_schedule = torch.linspace(
            0.0,
            1.0,
            num_timesteps + 1,
        )

        delta_schedule = (
            4.0
            * midpoint_variance
            * m_schedule
            * (1.0 - m_schedule)
        )

        delta_schedule[0] = 0.0
        delta_schedule[-1] = 0.0

        self.register_buffer(
            "m_schedule",
            m_schedule,
        )
        self.register_buffer(
            "delta_schedule",
            delta_schedule,
        )

    @staticmethod
    def _extract(
        values: torch.Tensor,
        t: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        selected = values.gather(0, t)

        return selected.reshape(
            t.shape[0],
            *((1,) * (reference.ndim - 1)),
        ).to(reference.dtype)

    def q_sample(
        self,
        ground_truth: torch.Tensor,
        coarse_hsi: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if ground_truth.shape != coarse_hsi.shape:
            raise ValueError(
                "ground_truth and coarse_hsi must have identical shapes."
            )

        if noise is None:
            noise = torch.randn_like(ground_truth)

        m_t = self._extract(
            self.m_schedule,
            t,
            ground_truth,
        )
        delta_t = self._extract(
            self.delta_schedule,
            t,
            ground_truth,
        )
        sigma_t = torch.sqrt(
            delta_t.clamp_min(0.0)
        )

        target_objective = (
            m_t * (coarse_hsi - ground_truth)
            + sigma_t * noise
        )

        x_t = ground_truth + target_objective

        return (
            x_t,
            target_objective,
            noise,
        )

    @staticmethod
    def predict_x0_from_objective(
        x_t: torch.Tensor,
        predicted_objective: torch.Tensor,
    ) -> torch.Tensor:
        return x_t - predicted_objective

    def training_predictions(
        self,
        rgb: torch.Tensor,
        coarse_hsi: torch.Tensor,
        ground_truth: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = ground_truth.shape[0]

        if t is None:
            t = torch.randint(
                1,
                self.num_timesteps + 1,
                (batch_size,),
                device=ground_truth.device,
                dtype=torch.long,
            )

        (
            x_t,
            target_objective,
            used_noise,
        ) = self.q_sample(
            ground_truth=ground_truth,
            coarse_hsi=coarse_hsi,
            t=t,
            noise=noise,
        )

        predicted_objective = self.denoiser(
            x_t=x_t,
            coarse_hsi=coarse_hsi,
            rgb=rgb,
            t=t,
            total_steps=self.num_timesteps,
        )

        reconstruction = (
            self.predict_x0_from_objective(
                x_t=x_t,
                predicted_objective=predicted_objective,
            )
        )

        return {
            "t": t,
            "x_t": x_t,
            "noise": used_noise,
            "target_objective": target_objective,
            "predicted_objective": predicted_objective,
            "reconstruction": reconstruction,
        }

    @torch.no_grad()
    def sample(
        self,
        rgb: torch.Tensor,
        coarse_hsi: torch.Tensor,
        clip_denoised: bool = True,
        stochastic: bool = False,
    ) -> torch.Tensor:
        x_t = coarse_hsi.clone()
        endpoint = coarse_hsi
        batch_size = coarse_hsi.shape[0]

        for step in range(
            self.num_timesteps,
            0,
            -1,
        ):
            t = torch.full(
                (batch_size,),
                step,
                device=coarse_hsi.device,
                dtype=torch.long,
            )

            predicted_objective = self.denoiser(
                x_t=x_t,
                coarse_hsi=coarse_hsi,
                rgb=rgb,
                t=t,
                total_steps=self.num_timesteps,
            )

            x0_hat = self.predict_x0_from_objective(
                x_t=x_t,
                predicted_objective=predicted_objective,
            )

            if clip_denoised:
                x0_hat = x0_hat.clamp(
                    0.0,
                    1.0,
                )

            if step == 1:
                x_t = x0_hat
                break

            previous_t = torch.full_like(
                t,
                step - 1,
            )

            m_t = self._extract(
                self.m_schedule,
                t,
                x_t,
            )
            delta_t = self._extract(
                self.delta_schedule,
                t,
                x_t,
            )
            m_previous = self._extract(
                self.m_schedule,
                previous_t,
                x_t,
            )
            delta_previous = self._extract(
                self.delta_schedule,
                previous_t,
                x_t,
            )

            previous_bridge_mean = (
                (1.0 - m_previous) * x0_hat
                + m_previous * endpoint
            )

            # The endpoint at t=T is deterministic because delta_T=0.
            if step == self.num_timesteps:
                posterior_mean = previous_bridge_mean
                posterior_variance = delta_previous
            else:
                denominator = (
                    1.0 - m_previous
                ).clamp_min(1e-8)

                transition_scale = (
                    (1.0 - m_t) / denominator
                )

                transition_variance = (
                    delta_t
                    - transition_scale.square()
                    * delta_previous
                ).clamp_min(1e-12)

                posterior_mean = (
                    (
                        transition_variance
                        / delta_t.clamp_min(1e-12)
                    )
                    * previous_bridge_mean
                    + (
                        transition_scale
                        * delta_previous
                        / delta_t.clamp_min(1e-12)
                    )
                    * (
                        x_t
                        - (1.0 - transition_scale)
                        * endpoint
                    )
                )

                posterior_variance = (
                    delta_previous
                    * transition_variance
                    / delta_t.clamp_min(1e-12)
                ).clamp_min(0.0)

            if stochastic:
                x_t = (
                    posterior_mean
                    + torch.sqrt(posterior_variance)
                    * torch.randn_like(x_t)
                )
            else:
                x_t = posterior_mean

        return x_t


# ============================================================================
# Complete MST++ + BBDM model
# ============================================================================

class MSTPlusPlusBBDM(nn.Module):
    def __init__(
        self,
        bridge: BrownianBridgeDiffusion,
        coarse_model: Optional[nn.Module] = None,
        coarse_checkpoint: Optional[str | Path] = None,
        coarse_model_kwargs: Optional[dict[str, Any]] = None,
        freeze_coarse_model: bool = True,
        strict_checkpoint_loading: bool = True,
    ):
        super().__init__()

        if coarse_model is None:
            coarse_model = MST_Plus_Plus(
                **(coarse_model_kwargs or {})
            )

        self.coarse_model = coarse_model
        self.bridge = bridge
        self.freeze_coarse_model = freeze_coarse_model

        if coarse_checkpoint is not None:
            checkpoint = _load_checkpoint(
                coarse_checkpoint
            )
            state_dict = _extract_state_dict(
                checkpoint
            )

            self.coarse_model.load_state_dict(
                state_dict,
                strict=strict_checkpoint_loading,
            )

        if self.freeze_coarse_model:
            self.coarse_model.float()

            for parameter in self.coarse_model.parameters():
                parameter.requires_grad_(False)

            self.coarse_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_coarse_model:
            self.coarse_model.eval()

        return self

    def get_coarse(
        self,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        if self.freeze_coarse_model:
            self.coarse_model.eval()

            with torch.no_grad():
                with torch.autocast(
                    device_type=rgb.device.type,
                    enabled=False,
                ):
                    coarse_output = self.coarse_model(
                        rgb.detach()
                        .float()
                        .contiguous()
                    )

            coarse_hsi = _extract_tensor_output(
                coarse_output
            ).detach().float()
        else:
            coarse_hsi = _extract_tensor_output(
                self.coarse_model(rgb)
            )

        coarse_hsi = coarse_hsi[
            :,
            :,
            :rgb.shape[-2],
            :rgb.shape[-1],
        ]

        return coarse_hsi.contiguous()

    def forward(
        self,
        rgb: torch.Tensor,
        ground_truth: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        coarse_hsi = self.get_coarse(rgb)

        if coarse_hsi.shape != ground_truth.shape:
            raise ValueError(
                "The MST++ output and ground-truth HSI must have "
                f"the same shape, but received {coarse_hsi.shape} "
                f"and {ground_truth.shape}."
            )

        outputs = self.bridge.training_predictions(
            rgb=rgb,
            coarse_hsi=coarse_hsi,
            ground_truth=ground_truth,
            t=t,
        )
        outputs["coarse_hsi"] = coarse_hsi

        return outputs

    @torch.no_grad()
    def reconstruct(
        self,
        rgb: torch.Tensor,
        clip_denoised: bool = True,
        stochastic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coarse_hsi = self.get_coarse(rgb)

        refined_hsi = self.bridge.sample(
            rgb=rgb,
            coarse_hsi=coarse_hsi,
            clip_denoised=clip_denoised,
            stochastic=stochastic,
        )

        return coarse_hsi, refined_hsi


def build_mst_bbdm(
    coarse_checkpoint: Optional[str | Path] = None,
    coarse_model_kwargs: Optional[dict[str, Any]] = None,
    hsi_channels: int = 31,
    rgb_channels: int = 3,
    n_feat: int = 31,
    body_depth: int = 3,
    mst_stage: int = 2,
    num_blocks: Sequence[int] = (1, 1, 1),
    num_timesteps: int = 50,
    midpoint_variance: float = 0.05,
    freeze_coarse_model: bool = True,
) -> MSTPlusPlusBBDM:
    denoiser = MSTBBDMDenoiser(
        hsi_channels=hsi_channels,
        rgb_channels=rgb_channels,
        n_feat=n_feat,
        body_depth=body_depth,
        mst_stage=mst_stage,
        num_blocks=num_blocks,
    )

    bridge = BrownianBridgeDiffusion(
        denoiser=denoiser,
        num_timesteps=num_timesteps,
        midpoint_variance=midpoint_variance,
    )

    return MSTPlusPlusBBDM(
        bridge=bridge,
        coarse_model=None,
        coarse_checkpoint=coarse_checkpoint,
        coarse_model_kwargs=coarse_model_kwargs,
        freeze_coarse_model=freeze_coarse_model,
    )
