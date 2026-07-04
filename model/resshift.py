import torch
import torch.nn as nn
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
