import torch
import torch.nn as nn
import numpy as np

# Assuming mst++ can be imported as requested
# from mst_plus_plus import MSTPlusPlus 

class ResShiftNoiseSchedule:
    def __init__(self, T=15, p=0.3, kappa=2.0):
        """
        Implements the non-uniform geometric schedule from Section 2.2 of the paper.
        """
        self.T = T
        self.p = p
        self.kappa = kappa
        
        # Determine eta_1 and eta_T based on the paper's constraints
        self.eta_1 = min((0.04 / kappa) ** 2, 0.001)
        self.eta_T = 0.999
        
        # Setup sequence
        self.sqrt_eta = np.zeros(T + 1)
        self.sqrt_eta[0] = 0.0  # Base boundary
        self.sqrt_eta[1] = np.sqrt(self.eta_1)
        self.sqrt_eta[T] = np.sqrt(self.eta_T)
        
        # Compute geometric intermediate steps (Eq 9 & 10)
        if T > 2:
            b0 = np.exp(1.0 / (2 * (T - 1)) * np.log(self.eta_T / self.eta_1))
            for t in range(2, T):
                beta_t = ((t - 1) / (T - 1)) ** p * (T - 1)
                self.sqrt_eta[t] = self.sqrt_eta[1] * (b0 ** beta_t)
                
        self.eta = self.sqrt_eta ** 2
        self.alpha = np.zeros(T + 1)
        self.alpha[1] = self.eta[1]
        for t in range(2, T + 1):
            self.alpha[t] = self.eta[t] - self.eta[t - 1]

class ResShiftModelWrapper(nn.Module):
    def __init__(self, mst_model, T=15, p=0.3, kappa=2.0, latent_channels=3):
        super().__init__()
        self.mst_model = mst_model  # Pre-trained/imported MST++ model
        self.schedule = ResShiftNoiseSchedule(T=T, p=p, kappa=kappa)
        
        # Time-embedding layer to feed the current step 't' into the network
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, latent_channels)
        )
        
        # Registering non-trainable schedule coefficients as buffers
        self.register_buffer('eta', torch.tensor(self.schedule.eta, dtype=torch.float32))
        self.register_buffer('alpha', torch.tensor(self.schedule.alpha, dtype=torch.float32))

    def forward(self, x_t, y_0, t):
        """
        Corresponds to f_theta(x_t, y_0, t) aiming to predict x_0 (Eq. 7 & 8)
        t: tensor of shape (batch_size,) indicating current timestep (1 to T)
        """
        # 1. Obtain anchored guidance from the MST++ model using the LR counterpart
        with torch.no_grad():
            mst_output = self.mst_model(y_0)
            
        # 2. Compute time context embedding
        t_input = t.to(x_t.device).float().unsqueeze(-1)
        t_emb = self.time_embed(t_input).utils.unsqueeze(-1).unsqueeze(-1) # Match BCHW
        
        # 3. Predict residual map/target clean estimate x_0
        # Feeding structural target, noisy state, and time adjustments
        net_input = x_t + mst_output + t_emb
        
        # Predict target refinement (x_0 prediction)
        predicted_x0 = net_input 
        
        return predicted_x0

    @torch.no_grad()
    def sample(self, y_0):
        """
        Reverse sampling path (Section 2.1, Eq. 4, 6 & 7)
        Transforms from LR state configuration to fine HR output
        """
        batch_size = y_0.shape[0]
        device = y_0.device
        
        # Initialize x_T matching Equation 4: N(x_T; y_0, kappa^2 * I)
        kappa = self.schedule.kappa
        x_t = y_0 + kappa * torch.randn_like(y_0)
        
        # Iteratively shift residual steps backward from T down to 1
        for t_idx in range(self.schedule.T, 0, -1):
            t_tensor = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            
            # Predict x_0 using f_theta network
            pred_x0 = self.forward(x_t, y_0, t_tensor)
            
            # Compute posterior mean parameter adjustment (Eq. 7)
            eta_t = self.eta[t_idx]
            eta_t_minus_1 = self.eta[t_idx - 1]
            alpha_t = self.alpha[t_idx]
            
            mu_t = (eta_t_minus_1 / eta_t) * x_t + (alpha_t / eta_t) * pred_x0
            
            # Apply transition update step
            if t_idx > 1:
                # Add transition noise variance sequence element (Eq. 6)
                var_coef = (kappa ** 2) * (eta_t_minus_1 / eta_t) * alpha_t
                noise = torch.randn_like(x_t)
                x_t = mu_t + torch.sqrt(var_coef) * noise
            else:
                x_t = mu_t
                
        return x_t
