import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the base MST block used to build the Denoiser body.
# Ensure this matches your project's folder structure.
from model.MST_Plus_Plus import MST

class ResShiftNoiseSchedule:
    """
    Implements the non-uniform geometric shifting schedule from the ResShift paper.
    """
    def __init__(self, T=15, p=0.3, kappa=2.0):
        self.T = T
        self.p = p
        self.kappa = kappa
        
        self.eta_1 = min((0.04 / kappa) ** 2, 0.001)
        self.eta_T = 0.999
        
        self.eta = np.zeros(T + 1)
        self.eta[1] = self.eta_1
        
        # Compute geometric intermediate steps (Eq 9 & 10)
        if T > 2:
            b0 = np.exp(1.0 / (2 * (T - 1)) * np.log(self.eta_T / self.eta_1))
            for t in range(2, T):
                beta_t = ((t - 1) / (T - 1)) ** p * (T - 1)
                self.eta[t] = self.eta_1 * (b0 ** (2 * beta_t)) # sqrt_eta squared
        self.eta[T] = self.eta_T
        
        self.alpha = np.zeros(T + 1)
        self.alpha[1] = self.eta[1]
        for t in range(2, T + 1):
            self.alpha[t] = self.eta[t] - self.eta[t - 1]

class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard Diffusion Time Embedding.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(1, half - 1)
        )
        angles = t[:, None] * freqs[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[-1]))
        return embedding

class ResShiftDenoiser(nn.Module):
    """
    The main network f_theta that predicts the target x_0.
    Conditioned on the current diffused state x_t, the coarse image y_0, and the RGB guide.
    """
    def __init__(self, hsi_channels=31, rgb_channels=3, n_feat=31, body_depth=3, mst_stage=2, num_blocks=(1, 1, 1)):
        super().__init__()
        self.pad_multiple = 2 ** mst_stage
        
        # Inputs: x_t (HSI_CHANNELS), y_0 (HSI_CHANNELS), rgb (RGB_CHANNELS)
        in_channels = (hsi_channels * 2) + rgb_channels
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Time conditioning MLP
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(n_feat),
            nn.Linear(n_feat, n_feat * 4),
            nn.GELU(),
            nn.Linear(n_feat * 4, n_feat)
        )
        
        # Deep network body using MST blocks
        self.body = nn.Sequential(*[
            MST(in_dim=n_feat, out_dim=n_feat, dim=n_feat, stage=mst_stage, num_blocks=list(num_blocks))
            for _ in range(body_depth)
        ])
        
        # Output mapping back to HSI channels
        self.conv_out = nn.Conv2d(n_feat, hsi_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x_t, y_0, rgb, t, total_steps):
        _, _, h, w = x_t.shape
        
        # Calculate padding to ensure spatial dimensions are cleanly divisible by the MST stage requirement
        pad_h = (self.pad_multiple - h % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - w % self.pad_multiple) % self.pad_multiple
        
        inputs = torch.cat([x_t, y_0, rgb], dim=1)
        if pad_h or pad_w:
            inputs = F.pad(inputs, (0, pad_w, 0, pad_h), mode="replicate")
            
        # Initial feature extraction
        feat = self.conv_in(inputs)
        
        # Add time embedding
        normalized_t = t.float() / float(total_steps)
        t_feat = self.time_mlp(normalized_t).to(feat.dtype)
        feat = feat + t_feat[:, :, None, None]
        
        # Pass through MST blocks
        feat = self.body(feat)
        out = self.conv_out(feat)
        
        # Crop back to original dimensions
        return out[:, :, :h, :w]


class MSTPlusPlusResShift(nn.Module):
    """
    Wrapper model that houses the frozen MST++ model and the ResShift model,
    coordinating the forward training diffusion and reverse inference steps.
    """
    def __init__(self, coarse_model: nn.Module, denoiser: nn.Module, T=15, p=0.3, kappa=2.0, metric_data_range=1.0):
        super().__init__()
        self.coarse_model = coarse_model
        self.denoiser = denoiser
        self.metric_data_range = metric_data_range
        
        self.schedule = ResShiftNoiseSchedule(T, p, kappa)
        
        # Register the schedule parameters as buffers so they move to the correct device automatically
        self.register_buffer('eta', torch.tensor(self.schedule.eta, dtype=torch.float32))
        self.register_buffer('alpha', torch.tensor(self.schedule.alpha, dtype=torch.float32))
        
        # Lock the MST++ model
        self.coarse_model.requires_grad_(False)
        self.coarse_model.eval()

    def train(self, mode=True):
        """Override train to ensure coarse model ALWAYS stays in eval mode."""
        super().train(mode)
        self.coarse_model.eval()
        return self

    def get_coarse(self, rgb: torch.Tensor) -> torch.Tensor:
        """Helper to get the initial coarse prediction safely."""
        with torch.no_grad():
            out = self.coarse_model(rgb)
            # Handle if the specific MST++ implementation returns a list of intermediate outputs
            return out[-1] if isinstance(out, (list, tuple)) else out

    def forward(self, rgb, ground_truth, t=None):
        """
        Forward Diffusion Step for Training (Eq. 1 & Eq. 8).
        Diffuses the ground truth towards the coarse model's output.
        """
        y_0 = self.get_coarse(rgb)
        b = ground_truth.shape[0]
        
        # Randomly sample t if not provided
        if t is None:
            t = torch.randint(1, self.schedule.T + 1, (b,), device=ground_truth.device, dtype=torch.long)
            
        e_0 = y_0 - ground_truth
        eta_t = self.eta[t].view(b, 1, 1, 1)
        
        # Diffuse state x_t
        noise = torch.randn_like(ground_truth)
        x_t = ground_truth + (eta_t * e_0) + (self.schedule.kappa * torch.sqrt(eta_t) * noise)
        
        # Predict the target x_0
        predicted_x0 = self.denoiser(x_t, y_0, rgb, t, self.schedule.T)
        
        return {
            "coarse_hsi": y_0,
            "x_t": x_t,
            "predicted_x0": predicted_x0,
            "ground_truth": ground_truth
        }

    @torch.no_grad()
    def sample(self, rgb, clip_denoised=True):
        """
        Reverse Diffusion Process for Inference (Section 2.1, Eq. 4, 6 & 7).
        Iteratively refines the coarse image back into a high-quality HSI.
        """
        y_0 = self.get_coarse(rgb)
        b, c, h, w = y_0.shape
        device = y_0.device
        
        # Initialize x_T at the coarse prediction with some initial noise
        x_t = y_0 + self.schedule.kappa * torch.randn_like(y_0)
        
        for step in range(self.schedule.T, 0, -1):
            t_tensor = torch.full((b,), step, device=device, dtype=torch.long)
            
            # 1. Predict x_0 using f_theta network
            pred_x0 = self.denoiser(x_t, y_0, rgb, t_tensor, self.schedule.T)
            
            if clip_denoised: 
                pred_x0 = pred_x0.clamp(0.0, self.metric_data_range)
            
            eta_t = self.eta[step]
            eta_t_prev = self.eta[step - 1]
            alpha_t = self.alpha[step]
            
            # 2. Compute posterior mean parameter adjustment (Eq. 7)
            mu_t = (eta_t_prev / eta_t) * x_t + (alpha_t / eta_t) * pred_x0
            
            # 3. Add transition noise variance (Eq. 6)
            if step > 1:
                var_coef = (self.schedule.kappa ** 2) * (eta_t_prev / eta_t) * alpha_t
                noise = torch.randn_like(x_t)
                x_t = mu_t + torch.sqrt(torch.tensor(var_coef, device=device)) * noise
            else:
                x_t = mu_t
                
        return y_0, x_t
