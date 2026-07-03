"""
ResShiftMSTPP: Residual-Shifting Diffusion Model conditioned on an MST++ rough prediction.

Faithful implementation of "ResShift: Efficient Diffusion Model for Image
Super-resolution by Residual Shifting" (Yue, Wang & Loy, NeurIPS 2023), adapted so
that the conditioning signal y0 (which is the LR image in the original paper) is
instead the rough prediction produced by an already-trained MST++ model, and x0
(the HR image in the original paper) is the ground-truth target. The diffusion
process shifts the residual e0 = y0 - x0 across a short Markov chain of length T,
exactly following Eqs. (1)-(10) of the paper.

Contents:
  1. Noise schedule construction                         -> Eqs. (9)-(10)
  2. Forward process q(x_t | x_0, y0)                     -> Eq. (2)
  3. Posterior q(x_{t-1} | x_t, x_0, y0)                    -> Eq. (6)
  4. Reverse process p_theta(x_{t-1} | x_t, y0)             -> Eq. (7)
  5. Denoising network f_theta(x_t, y0, t) predicting x0   -> Sec. 2.1 / 4.1
  6. ResShiftMSTPP wrapper: runs MST++ to get y0, then runs the residual-shifting
     diffusion process on top of it (training-time forward + inference-time sample)

No loss functions are included here (per request) -- the training `forward()` below
returns everything needed (x0_pred, x0_gt, loss_weight for Eq. 8) so the objective
of Eq. (8) can be implemented in a separate file, e.g. `resshift_losses.py`. Note
the paper reports that simply using an unweighted MSE (dropping wt) performs best
in practice, matching the finding of Ho et al. (DDPM).

Usage: replace the MST++ import placeholder below with your actual model file, then
instantiate `ResShiftMSTPP(mst_plus_plus=your_model, ...)`.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# >>> Replace this with the import of your existing MST++ model file <<<
from MST_Plus_Plus import MST_Plus_Plus
# ---------------------------------------------------------------------------


# =============================================================================
# 1. Noise schedule  (Sec. 2.2, Eqs. 9-10)
# =============================================================================
def make_eta_schedule(T: int, kappa: float, p: float = 0.3,
                       min_noise_level: float = 0.04) -> torch.Tensor:
    """
    Builds the monotonically increasing shifting sequence {eta_t}_{t=1}^{T},
    exactly as described in Sec. 2.2:

        eta_1 -> min((min_noise_level / kappa)^2, 0.001)   (so kappa*sqrt(eta_1) small)
        eta_T -> 0.999                                      (so eta_T -> 1)
        eta_t, 2<=t<=T-1, follows the non-uniform geometric schedule of Eq. (9)-(10):

            sqrt(eta_t) = sqrt(eta_1) * b0^{beta_t}
            beta_t = ((t-1)/(T-1))^p * (T-1)
            b0 = exp( 1/(2(T-1)) * log(eta_T / eta_1) )

    Returns a tensor of shape (T,) with etas[t-1] == eta_t (1-indexed in the paper).
    """
    eta_1 = min((min_noise_level / kappa) ** 2, 0.001)
    eta_T = 0.999

    if T == 1:
        return torch.tensor([eta_T], dtype=torch.float64).float()

    sqrt_eta_1 = math.sqrt(eta_1)
    b0 = math.exp((1.0 / (2 * (T - 1))) * math.log(eta_T / eta_1))

    sqrt_etas = [sqrt_eta_1]
    for t in range(2, T):  # t = 2, ..., T-1
        beta_t = ((t - 1) / (T - 1)) ** p * (T - 1)
        sqrt_etas.append(sqrt_eta_1 * (b0 ** beta_t))
    sqrt_etas.append(math.sqrt(eta_T))  # t = T

    etas = torch.tensor(sqrt_etas, dtype=torch.float64) ** 2
    return etas.float()


class ResShiftSchedule(nn.Module):
    """
    Pre-computes and stores (as registered buffers, so they move with .to(device)
    and are (de)serialized with state_dict) the schedule quantities eta_t, alpha_t,
    and the posterior mean/variance coefficients of Eq. (6)-(8).
    """

    def __init__(self, T: int = 15, kappa: float = 2.0, p: float = 0.3,
                 min_noise_level: float = 0.04):
        super().__init__()
        self.T = T
        self.kappa = kappa
        self.p = p

        etas = make_eta_schedule(T, kappa, p, min_noise_level)           # eta_t,  t=1..T
        etas_prev = torch.cat([torch.zeros(1), etas[:-1]])               # eta_{t-1}; eta_0 := 0
        alphas = etas - etas_prev                                        # alpha_t = eta_t - eta_{t-1}
        alphas[0] = etas[0]                                              # alpha_1 = eta_1 (Eq. 1)

        self.register_buffer("etas", etas)                               # (T,)
        self.register_buffer("etas_prev", etas_prev)                     # (T,)
        self.register_buffer("alphas", alphas)                           # (T,)

        # q(x_{t-1}|x_t,x_0,y0) = N(mu, kappa^2 * eta_{t-1}/eta_t * alpha_t)   (Eq. 6, 17)
        self.register_buffer(
            "posterior_variance",
            (kappa ** 2) * etas_prev * alphas / etas.clamp(min=1e-12),
        )
        self.register_buffer("posterior_mean_coef_xt", etas_prev / etas.clamp(min=1e-12))
        self.register_buffer("posterior_mean_coef_x0", alphas / etas.clamp(min=1e-12))

        # wt = alpha_t / (2 * kappa^2 * eta_t * eta_{t-1})   (Eq. 8; optional, paper omits it)
        wt = alphas / (2 * (kappa ** 2) * etas * etas_prev.clamp(min=1e-12))
        wt[0] = alphas[0] / (2 * (kappa ** 2) * etas[0] * etas[0])  # avoid /0 at t=1 (eta_0=0)
        self.register_buffer("loss_weight", wt)

    @staticmethod
    def _extract(arr: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Gathers per-timestep coefficients (t is 0-indexed) and reshapes to broadcast."""
        out = arr.to(t.device).gather(0, t)
        return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))

    # ---------------------------------------------------------------------
    # Forward process: q(x_t | x_0, y0) = N(x0 + eta_t*e0, kappa^2*eta_t*I)   (Eq. 2)
    # ---------------------------------------------------------------------
    def q_sample(self, x0: torch.Tensor, y0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """t is 0-indexed here (t=0 corresponds to the paper's t=1)."""
        if noise is None:
            noise = torch.randn_like(x0)
        e0 = y0 - x0
        eta_t = self._extract(self.etas, t, x0.shape)
        mean = x0 + eta_t * e0
        std = self.kappa * eta_t.sqrt()
        return mean + std * noise

    # ---------------------------------------------------------------------
    # Posterior q(x_{t-1} | x_t, x_0, y0)   (Eq. 6)
    # ---------------------------------------------------------------------
    def q_posterior_mean_variance(self, x0: torch.Tensor, xt: torch.Tensor,
                                   t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coef_xt = self._extract(self.posterior_mean_coef_xt, t, xt.shape)
        coef_x0 = self._extract(self.posterior_mean_coef_x0, t, xt.shape)
        mean = coef_xt * xt + coef_x0 * x0
        var = self._extract(self.posterior_variance, t, xt.shape)
        return mean, var

    # ---------------------------------------------------------------------
    # Reverse process p_theta(x_{t-1} | x_t, y0)   (Eq. 7)
    # mu_theta = (eta_{t-1}/eta_t) x_t + (alpha_t/eta_t) f_theta(x_t,y0,t)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def p_mean_variance(self, model: nn.Module, xt: torch.Tensor, y0: torch.Tensor,
                         t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x0_pred = model(xt, y0, t)
        model_mean, model_var = self.q_posterior_mean_variance(x0_pred, xt, t)
        return model_mean, model_var, x0_pred

    @torch.no_grad()
    def p_sample(self, model: nn.Module, xt: torch.Tensor, y0: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        mean, var, _ = self.p_mean_variance(model, xt, y0, t)
        noise = torch.randn_like(xt)
        # posterior_variance is already exactly 0 at t=0 (paper's t=1, since eta_0=0),
        # so the final step is naturally deterministic without any extra masking.
        return mean + var.sqrt() * noise

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, y0: torch.Tensor,
                       return_intermediates: bool = False):
        """
        Full reverse sampling: x_T ~ N(y0, kappa^2 I)  (Eq. 4 approx. prior)
        iteratively refined down to x_0, i.e. the ground-truth prediction refined
        starting from the MST++ rough guess y0.
        """
        device = y0.device
        b = y0.shape[0]
        xt = y0 + self.kappa * torch.randn_like(y0)

        intermediates = [xt] if return_intermediates else None
        for i in reversed(range(self.T)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            xt = self.p_sample(model, xt, y0, t)
            if return_intermediates:
                intermediates.append(xt)

        return (xt, intermediates) if return_intermediates else xt


# =============================================================================
# 2. Denoising network f_theta(x_t, y0, t) -> predicted x_0
#    A UNet in the spirit of the DDPM backbone used by the paper (Sec 4.1), with
#    conditioning on y0 via channel-wise concatenation at every forward pass, just
#    like ResShift conditions on the LR image. Swap in your own backbone if you
#    prefer -- the diffusion math above is architecture-agnostic.
# =============================================================================
def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_ch, out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Self-attention block (stand-in for the Swin-Transformer block the paper
    substitutes in for arbitrary-resolution robustness)."""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).permute(1, 0, 3, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(c), dim=-1)
        out = (attn @ v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class ResShiftUNet(nn.Module):
    """
    f_theta(x_t, y0, t): predicts x_0 directly (the Eq. 7-8 parameterization).
    x_t and y0 are concatenated channel-wise, so the network is conditioned on
    the MST++ rough prediction at every denoising step.
    """

    def __init__(self, in_channels: int = 31, base_ch: int = 64,
                 ch_mult: Tuple[int, ...] = (1, 2, 4), num_res_blocks: int = 2,
                 attn_resolutions_idx: Tuple[int, ...] = (2,)):
        super().__init__()
        self.in_channels = in_channels
        temb_ch = base_ch * 4
        self.temb_dim = base_ch

        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, temb_ch), nn.SiLU(), nn.Linear(temb_ch, temb_ch)
        )

        # x_t and y0 concatenated -> 2 * in_channels input
        self.in_conv = nn.Conv2d(in_channels * 2, base_ch, 3, padding=1)

        # ---- Down path ----
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_ch
        chs = [ch]
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, temb_ch))
                ch = out_ch
                if i in attn_resolutions_idx:
                    blocks.append(AttnBlock(ch))
                chs.append(ch)
            self.down_blocks.append(blocks)
            if i != len(ch_mult) - 1:
                self.downsamples.append(Downsample(ch))
                chs.append(ch)
            else:
                self.downsamples.append(None)

        # ---- Middle ----
        self.mid_block1 = ResBlock(ch, ch, temb_ch)
        self.mid_attn = AttnBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, temb_ch)

        # ---- Up path ----
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = chs.pop()
                blocks.append(ResBlock(ch + skip_ch, out_ch, temb_ch))
                ch = out_ch
                if i in attn_resolutions_idx:
                    blocks.append(AttnBlock(ch))
            self.up_blocks.append(blocks)
            if i != 0:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(None)

        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, xt: torch.Tensor, y0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.temb_dim))
        h = self.in_conv(torch.cat([xt, y0], dim=1))

        hs = [h]
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for layer in blocks:
                h = layer(h, temb) if isinstance(layer, ResBlock) else layer(h)
                hs.append(h)
            if down is not None:
                h = down(h)
                hs.append(h)

        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        for blocks, up in zip(self.up_blocks, self.upsamples):
            for layer in blocks:
                if isinstance(layer, ResBlock):
                    h = torch.cat([h, hs.pop()], dim=1)
                    h = layer(h, temb)
                else:
                    h = layer(h)
            if up is not None:
                h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))  # predicted x_0


# =============================================================================
# 3. Full wrapper: MST++ (rough predictor) + ResShift residual-shifting diffusion
# =============================================================================
class ResShiftMSTPP(nn.Module):
    """
    End-to-end module:

        rgb --MST++--> y0  (rough prediction, plays the role of the LR image)
        (x0_gt, y0) --ResShift residual-shifting diffusion--> refined prediction

    Training: call `forward(rgb, x0_gt)`; it returns a dict with x0_pred / x0_gt /
    loss_weight so you can plug it into the (separately defined) Eq. (8) objective.

    Inference: call `sample(rgb)` to get the final T-step refined prediction.
    """

    def __init__(self, mst_plus_plus: nn.Module, out_channels: int = 31,
                 T: int = 15, kappa: float = 2.0, p: float = 0.3,
                 min_noise_level: float = 0.04, unet_base_ch: int = 64,
                 freeze_mstpp: bool = True):
        super().__init__()

        # --- rough predictor (already trained; provided by the user) ---
        self.mst_plus_plus = mst_plus_plus
        self.freeze_mstpp = freeze_mstpp
        if freeze_mstpp:
            for param in self.mst_plus_plus.parameters():
                param.requires_grad_(False)
            self.mst_plus_plus.eval()

        # --- diffusion schedule & denoiser ---
        self.schedule = ResShiftSchedule(T=T, kappa=kappa, p=p,
                                          min_noise_level=min_noise_level)
        self.denoiser = ResShiftUNet(in_channels=out_channels, base_ch=unet_base_ch)
        self.T = T

    @torch.no_grad()
    def get_rough_prediction(self, rgb: torch.Tensor) -> torch.Tensor:
        """Runs the (frozen) MST++ backbone to obtain y0 (analogue of the LR image)."""
        was_training = self.mst_plus_plus.training
        if self.freeze_mstpp:
            self.mst_plus_plus.eval()
        y0 = self.mst_plus_plus(rgb)
        if self.freeze_mstpp and was_training:
            self.mst_plus_plus.train()
        return y0

    # ---------------------------------------------------------------------
    # Training-time forward: draws a random t, builds x_t via q_sample, and runs
    # the denoiser. Returns everything needed for the (externally-defined) Eq. (8)
    # loss -- no loss computation happens in this file.
    # ---------------------------------------------------------------------
    def forward(self, rgb: torch.Tensor, x0_gt: torch.Tensor,
                t: Optional[torch.Tensor] = None):
        y0 = self.get_rough_prediction(rgb) if self.freeze_mstpp else self.mst_plus_plus(rgb)

        b = x0_gt.shape[0]
        if t is None:
            t = torch.randint(0, self.T, (b,), device=x0_gt.device, dtype=torch.long)

        xt = self.schedule.q_sample(x0_gt, y0, t)
        x0_pred = self.denoiser(xt, y0, t)

        return {
            "x0_pred": x0_pred,        # f_theta(x_t, y0, t), Eq. (7)-(8)
            "x0_gt": x0_gt,
            "y0": y0,
            "xt": xt,
            "t": t,
            "loss_weight": self.schedule.loss_weight.to(x0_gt.device).gather(0, t),
        }

    # ---------------------------------------------------------------------
    # Inference-time sampling: MST++ rough guess -> T-step residual-shifting
    # reverse diffusion -> refined prediction.
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, rgb: torch.Tensor, return_intermediates: bool = False):
        y0 = self.get_rough_prediction(rgb)
        return self.schedule.p_sample_loop(self.denoiser, y0,
                                            return_intermediates=return_intermediates)


