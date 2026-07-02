import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        if half == 1:
            freqs = torch.ones(1, device=t.device, dtype=t.dtype)
        else:
            freqs = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, device=t.device, dtype=t.dtype)
                / (half - 1)
            )
        angles = t[:, None] * freqs[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class MSTResidualDenoiser(nn.Module):
    def __init__(
        self,
        hsi_channels: int = 31,
        rgb_channels: int = 3,
        n_feat: int = 31,
        body_depth: int = 3,
        mst_stage: int = 2,
        num_blocks=(1, 1, 1),
    ):
        super().__init__()
        self.hsi_channels = hsi_channels
        self.pad_multiple = 2 ** mst_stage

        input_channels = hsi_channels + hsi_channels + rgb_channels
        self.conv_in = nn.Conv2d(input_channels, n_feat, 3, 1, 1, bias=False)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(n_feat),
            nn.Linear(n_feat, n_feat * 4),
            nn.GELU(),
            nn.Linear(n_feat * 4, n_feat),
        )

        self.body = nn.Sequential(*[
            MST(
                in_dim=n_feat,
                out_dim=n_feat,
                dim=n_feat,
                stage=mst_stage,
                num_blocks=list(num_blocks),
            )
            for _ in range(body_depth)
        ])
        self.conv_out = nn.Conv2d(n_feat, hsi_channels, 3, 1, 1, bias=False)

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
                f"x_t and coarse_hsi must have the same shape, got "
                f"{x_t.shape} and {coarse_hsi.shape}."
            )
        if rgb.shape[0] != x_t.shape[0] or rgb.shape[-2:] != x_t.shape[-2:]:
            raise ValueError("RGB and HSI tensors must share batch and spatial dimensions.")

        _, _, h, w = x_t.shape
        pad_h = (self.pad_multiple - h % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - w % self.pad_multiple) % self.pad_multiple

        inputs = torch.cat([x_t, coarse_hsi, rgb], dim=1)
        if pad_h or pad_w:
            mode = "reflect" if h > pad_h and w > pad_w else "replicate"
            inputs = F.pad(inputs, (0, pad_w, 0, pad_h), mode=mode)

        features = self.conv_in(inputs)
        t_normalized = t.float() / float(total_steps)
        time_features = self.time_mlp(t_normalized).to(features.dtype)
        features = features + time_features[:, :, None, None]

        features = self.body(features)
        residual = self.conv_out(features)
        return residual[:, :, :h, :w]


class ResidualBBDM(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        num_timesteps: int = 50,
        midpoint_variance: float = 0.05,
    ):
        super().__init__()
        if num_timesteps < 2:
            raise ValueError("num_timesteps must be at least 2.")
        if midpoint_variance <= 0:
            raise ValueError("midpoint_variance must be positive.")

        self.denoiser = denoiser
        self.num_timesteps = num_timesteps

        m = torch.linspace(0.0, 1.0, num_timesteps + 1)
        delta = 4.0 * midpoint_variance * m * (1.0 - m)
        delta[0] = 0.0
        delta[-1] = 0.0

        self.register_buffer("m_schedule", m)
        self.register_buffer("delta_schedule", delta)

    @staticmethod
    def _extract(values: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = values.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (x.ndim - 1))).to(dtype=x.dtype)

    def q_sample(
        self,
        ground_truth: torch.Tensor,
        coarse_hsi: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(ground_truth)

        m_t = self._extract(self.m_schedule, t, ground_truth)
        delta_t = self._extract(self.delta_schedule, t, ground_truth)

        x_t = (
            (1.0 - m_t) * ground_truth
            + m_t * coarse_hsi
            + torch.sqrt(delta_t.clamp_min(0.0)) * noise
        )
        return x_t, noise

    def training_predictions(
        self,
        rgb: torch.Tensor,
        coarse_hsi: torch.Tensor,
        ground_truth: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = ground_truth.shape[0]
        if t is None:
            t = torch.randint(
                1,
                self.num_timesteps + 1,
                (batch,),
                device=ground_truth.device,
                dtype=torch.long,
            )

        x_t, noise = self.q_sample(ground_truth, coarse_hsi, t)
        predicted_residual = self.denoiser(
            x_t=x_t,
            coarse_hsi=coarse_hsi,
            rgb=rgb,
            t=t,
            total_steps=self.num_timesteps,
        )

        target_residual = ground_truth - coarse_hsi
        reconstruction = coarse_hsi + predicted_residual

        return {
            "t": t,
            "x_t": x_t,
            "noise": noise,
            "target_residual": target_residual,
            "predicted_residual": predicted_residual,
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
        """Reverse Brownian-bridge sampling from coarse HSI to refined HSI."""
        x_t = coarse_hsi.clone()
        endpoint = coarse_hsi
        batch = coarse_hsi.shape[0]

        for step in range(self.num_timesteps, 0, -1):
            t = torch.full(
                (batch,), step, device=coarse_hsi.device, dtype=torch.long
            )
            predicted_residual = self.denoiser(
                x_t=x_t,
                coarse_hsi=coarse_hsi,
                rgb=rgb,
                t=t,
                total_steps=self.num_timesteps,
            )
            x0_hat = coarse_hsi + predicted_residual
            if clip_denoised:
                x0_hat = x0_hat.clamp(0.0, 1.0)

            if step == 1:
                x_t = x0_hat
                break

            t_prev = torch.full_like(t, step - 1)
            m_t = self._extract(self.m_schedule, t, x_t)
            delta_t = self._extract(self.delta_schedule, t, x_t)
            m_prev = self._extract(self.m_schedule, t_prev, x_t)
            delta_prev = self._extract(self.delta_schedule, t_prev, x_t)

            previous_bridge_mean = (1.0 - m_prev) * x0_hat + m_prev * endpoint

            if step == self.num_timesteps:
                posterior_mean = previous_bridge_mean
                posterior_variance = delta_prev
            else:
                transition_scale = (1.0 - m_t) / (1.0 - m_prev).clamp_min(1e-8)
                transition_variance = (
                    delta_t - transition_scale.square() * delta_prev
                ).clamp_min(1e-12)

                posterior_mean = (
                    (transition_variance / delta_t.clamp_min(1e-12))
                    * previous_bridge_mean
                    + (transition_scale * delta_prev / delta_t.clamp_min(1e-12))
                    * (x_t - (1.0 - transition_scale) * endpoint)
                )
                posterior_variance = (
                    delta_prev * transition_variance / delta_t.clamp_min(1e-12)
                ).clamp_min(0.0)

            if stochastic and step > 1:
                x_t = posterior_mean + torch.sqrt(posterior_variance) * torch.randn_like(x_t)
            else:
                x_t = posterior_mean

        return x_t



class MSTPlusPlusResidualBBDM(nn.Module):
    def __init__(
        self,
        coarse_model: nn.Module,
        bridge: ResidualBBDM,
        freeze_coarse_model: bool = True,
    ):
        super().__init__()
        self.coarse_model = coarse_model
        self.bridge = bridge
        self.freeze_coarse_model = freeze_coarse_model

        if freeze_coarse_model:
            for parameter in self.coarse_model.parameters():
                parameter.requires_grad_(False)
            self.coarse_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_coarse_model:
            self.coarse_model.eval()
        return self

    def get_coarse(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.freeze_coarse_model:
            with torch.no_grad():
                return self.coarse_model(rgb)
        return self.coarse_model(rgb)

    def forward(
        self,
        rgb: torch.Tensor,
        ground_truth: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        coarse_hsi = self.get_coarse(rgb)
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
    ):
        coarse_hsi = self.get_coarse(rgb)
        refined_hsi = self.bridge.sample(
            rgb=rgb,
            coarse_hsi=coarse_hsi,
            clip_denoised=clip_denoised,
            stochastic=stochastic,
        )
        return coarse_hsi, refined_hsi
