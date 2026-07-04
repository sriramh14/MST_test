import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class ResidualShiftingHSI(nn.Module):
    def __init__(self, coarse_net, residual_net, channels=31, max_timesteps=15, kappa=1.0):
        """
        Two-stage HSI Reconstruction:
        1. coarse_net (MST++): Predicts deterministic coarse HSI from LR input y_0.
        2. residual_net: Diffusion model predicting x_GT from noisy residual state x_t.
        """
        super(ResidualShiftingHSI, self).__init__()
        self.coarse_net = coarse_net      # e.g., MST_Plus_Plus(in_channels=3, out_channels=31)
        self.residual_net = residual_net  # Diffusion UNet or Transformer taking [x_t, x_coarse, t]
        self.channels = channels
        self.max_timesteps = max_timesteps
        self.kappa = kappa
        
        self.register_buffer('eta', self._build_noise_schedule(max_timesteps, kappa))

    def _build_noise_schedule(self, T, kappa, p=0.3):
        eta = torch.zeros(T + 1)
        eta[1] = min((0.04 / kappa)**2, 0.001)
        eta[T] = 0.999
        b_0 = math.exp((1 / (T - 1)) * math.log(math.sqrt(eta[T]) / math.sqrt(eta[1])))
        for t in range(2, T):
            beta_t = ((t - 1) / (T - 1))**p * (T - 1)
            eta[t] = (math.sqrt(eta[1]) * (b_0 ** beta_t)) ** 2
        return eta

    def forward_loss(self, y_0, x_gt):
        """
        Training forward pass.
        """
        b, c, h, w = x_gt.shape
        
        # 1. Generate coarse prediction from MST++
        x_coarse = self.coarse_net(y_0)
        
        # 2. Sample random timestep t in [1, T]
        t = torch.randint(1, self.max_timesteps + 1, (b,), device=x_gt.device)
        
        # 3. Construct noisy shifted state x_t
        eta_t = self.eta[t].view(b, 1, 1, 1)
        noise = torch.randn_like(x_gt)
        x_t = torch.sqrt(eta_t) * x_gt + (1.0 - torch.sqrt(eta_t)) * x_coarse + self.kappa * torch.sqrt(eta_t) * noise
        
        # 4. Predict ground truth anchor
        net_input = torch.cat([x_t, x_coarse], dim=1)
        x_pred = self.residual_net(net_input, t)
        
        # Loss: Coarse L1 + Residual Diffusion L1
        loss_coarse = nn.functional.l1_loss(x_coarse, x_gt)
        loss_diff = nn.functional.l1_loss(x_pred, x_gt)
        
        return loss_coarse + loss_diff

    @torch.no_grad()
    def sample(self, y_0):
        """
        Inference sampling process.
        """
        # 1. Coarse anchor from MST++
        x_coarse = self.coarse_net(y_0)
        
        # 2. Initialize x_T at coarse prediction + variance
        x_t = x_coarse + self.kappa * torch.randn_like(x_coarse)
        
        # 3. Iterative reverse residual shift
        for t in reversed(range(1, self.max_timesteps + 1)):
            t_tensor = torch.full((y_0.shape[0],), t, device=y_0.device, dtype=torch.long)
            net_input = torch.cat([x_t, x_coarse], dim=1)
            x_0_pred = self.residual_net(net_input, t_tensor)
            
            eta_t = self.eta[t]
            eta_prev = self.eta[t-1] if t > 1 else torch.tensor(0.0, device=y_0.device)
            alpha_t = eta_t - eta_prev
            
            mu = (eta_prev / eta_t) * x_t + (alpha_t / eta_t) * x_0_pred
            
            if t > 1:
                sigma = self.kappa * math.sqrt((eta_prev / eta_t) * alpha_t)
                x_t = mu + sigma * torch.randn_like(x_t)
            else:
                x_t = mu
                
        return x_t, x_coarse

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. Timestep Embedding & Conditional U-Net
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SimpleConditionalUNet(nn.Module):
    """
    A simplified conditional U-Net for the diffusion backbone.
    In practice, you might want to scale this up with attention blocks (like SR3).
    """
    def __init__(self, in_channels=62, out_channels=31, time_emb_dim=256):
        super().__init__()
        
        # Timestep embedding module
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Initial projection
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
        # Downsampling path (simplified)
        self.down1 = nn.Conv2d(64, 128, 4, 2, 1)
        self.down2 = nn.Conv2d(128, 256, 4, 2, 1)
        
        # Time embedding projections for each resolution
        self.time_emb1 = nn.Linear(time_emb_dim, 128)
        self.time_emb2 = nn.Linear(time_emb_dim, 256)
        
        # Upsampling path
        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(128 * 2, 64, 4, 2, 1) # *2 for skip connection
        
        # Final projection to target HSI channels (31)
        self.conv_out = nn.Conv2d(64 * 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x, time):
        # 1. Time embedding
        t_emb = self.time_mlp(time)
        
        # 2. Forward pass with skip connections
        x1 = self.conv_in(x) # [B, 64, H, W]
        
        x2 = self.down1(x1)  # [B, 128, H/2, W/2]
        x2 = x2 + self.time_emb1(t_emb)[:, :, None, None]
        x2 = F.relu(x2)
        
        x3 = self.down2(x2)  # [B, 256, H/4, W/4]
        x3 = x3 + self.time_emb2(t_emb)[:, :, None, None]
        x3 = F.relu(x3)
        
        # 3. Upsample and concatenate
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1) # Skip connection
        
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1) # Skip connection
        
        return self.conv_out(x)


# ==========================================
# 2. ResShift Diffusion Module
# ==========================================
class ResShiftDiffusion(nn.Module):
    def __init__(self, model, num_timesteps=15, kappa=2.0):
        super().__init__()
        self.model = model 
        self.num_timesteps = num_timesteps
        self.kappa = kappa
        
        sqrt_eta = torch.linspace(0.01, 0.999**0.5, num_timesteps)
        eta = sqrt_eta ** 2
        eta_prev = F.pad(eta[:-1], (1, 0), value=0.0) 
        alpha = eta - eta_prev
        
        self.register_buffer('eta', eta)
        self.register_buffer('eta_prev', eta_prev)
        self.register_buffer('alpha', alpha)
        
    def _extract(self, val, t, x_shape):
        batch_size = t.shape[0]
        out = val.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def forward(self, x0, y0):
        b, c, h, w = x0.shape
        device = x0.device
        
        t = torch.randint(0, self.num_timesteps, (b,), device=device, dtype=torch.long)
        eta_t = self._extract(self.eta, t, x0.shape)
        
        noise = torch.randn_like(x0)
        xt = (1 - eta_t) * x0 + eta_t * y0 + self.kappa * torch.sqrt(eta_t) * noise
        
        # Concatenate noisy state (31) and coarse prediction (31) -> 62 channels
        model_input = torch.cat([xt, y0], dim=1) 
        x0_pred = self.model(model_input, t)
        
        return F.mse_loss(x0_pred, x0)

    @torch.no_grad()
    def sample(self, y0):
        b, c, h, w = y0.shape
        device = y0.device
        
        xt = y0 + self.kappa * torch.randn_like(y0)
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            
            model_input = torch.cat([xt, y0], dim=1)
            x0_pred = self.model(model_input, t)
            
            eta_t = self._extract(self.eta, t, xt.shape)
            eta_prev_t = self._extract(self.eta_prev, t, xt.shape)
            alpha_t = self._extract(self.alpha, t, xt.shape)
            
            mu = (eta_prev_t / eta_t) * xt + (alpha_t / eta_t) * x0_pred
            
            variance = (self.kappa ** 2) * (eta_prev_t / eta_t) * alpha_t
            sigma = torch.sqrt(variance)
            
            noise = torch.randn_like(xt) if i > 0 else torch.zeros_like(xt)
            xt = mu + sigma * noise
            
        return xt


# ==========================================
# 3. End-to-End SSR Pipeline Wrapper
# ==========================================
class EndToEndSSR(nn.Module):
    def __init__(self, mst_model_class, mst_ckpt_path, num_timesteps=15, kappa=2.0):
        super().__init__()
        
        # 1. Initialize and load MST++ (Coarse Predictor)
        # Assuming mst_model_class is your imported MST++ model definition
        self.coarse_predictor = mst_model_class(in_channels=3, out_channels=31)
        
        # Load the checkpoint
        checkpoint = torch.load(mst_ckpt_path, map_location='cpu')
        # Handle state_dict keys if they were saved with DataParallel/DDP
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        self.coarse_predictor.load_state_dict(state_dict)
        
        # Freeze the coarse predictor (Recommended for Stage-2 training)
        for param in self.coarse_predictor.parameters():
            param.requires_grad = False
        self.coarse_predictor.eval()
            
        # 2. Initialize the Denoising Backbone and Diffusion Module
        # in_channels = 31 (noisy HSI) + 31 (coarse HSI from MST++) = 62
        unet = SimpleConditionalUNet(in_channels=62, out_channels=31)
        self.diffusion = ResShiftDiffusion(unet, num_timesteps=num_timesteps, kappa=kappa)

    def forward(self, rgb, hsi_gt):
        """
        Training forward pass.
        rgb: (B, 3, H, W) 
        hsi_gt: Ground truth high-fidelity HSI (B, 31, H, W)
        """
        # Generate the coarse prediction (y0) without tracking gradients for MST++
        with torch.no_grad():
            coarse_hsi = self.coarse_predictor(rgb)
            
        # The diffusion model calculates the noise/prediction loss
        loss = self.diffusion(x0=hsi_gt, y0=coarse_hsi)
        return loss

    @torch.no_grad()
    def sample(self, rgb):
        """
        Inference pass to generate high-fidelity HSI from an RGB image.
        """
        # 1. Get coarse prediction from MST++
        coarse_hsi = self.coarse_predictor(rgb)
        
        # 2. Refine using ResShift
        fine_hsi = self.diffusion.sample(y0=coarse_hsi)
        
        return fine_hsi, coarse_hsi # Returning both allows for visualization/comparison
