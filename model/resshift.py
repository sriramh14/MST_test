from tqdm import tqdm

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: import MST_Plus_Plus from your model file, e.g.:
from .MST_Plus_Plus import MST_Plus_Plus


def get_resshift_schedule(T=15, p=0.3, kappa=2.0):
    eta_T = 0.999 
    eta_1 = min((0.04 / kappa)**2, 0.001) 
    b0 = torch.exp(torch.tensor((1.0 / (2 * (T - 1))) * torch.log(torch.tensor(eta_T / eta_1))))
    t = torch.arange(1, T + 1, dtype=torch.float32)
    beta_t = ((t - 1) / (T - 1))**p * (T - 1)
    sqrt_eta_t = torch.sqrt(torch.tensor(eta_1)) * (b0 ** beta_t)
    eta_t = sqrt_eta_t ** 2
    alpha_t = torch.zeros_like(eta_t)
    alpha_t[0] = eta_t[0]
    alpha_t[1:] = eta_t[1:] - eta_t[:-1]
    return eta_t, alpha_t

class ResShiftDiffusion(nn.Module):
    def __init__(self, T=15, p=0.3, kappa=2.0):
        super().__init__()
        self.T = T
        self.kappa = kappa
        eta, alpha = self._get_resshift_schedule(T, p, kappa)
        self.register_buffer('eta', eta)
        self.register_buffer('alpha', alpha)
        
    def _get_resshift_schedule(self, T, p, kappa):
        eta_T = 0.999 
        eta_1 = min((0.04 / kappa)**2, 0.001) 
        b0 = torch.exp(torch.tensor((1.0 / (2 * (T - 1))) * torch.log(torch.tensor(eta_T / eta_1))))
        t = torch.arange(1, T + 1, dtype=torch.float32)
        beta_t = ((t - 1) / (T - 1))**p * (T - 1)
        sqrt_eta_t = torch.sqrt(torch.tensor(eta_1)) * (b0 ** beta_t)
        eta_t = sqrt_eta_t ** 2
        alpha_t = torch.zeros_like(eta_t)
        alpha_t[0] = eta_t[0]
        alpha_t[1:] = eta_t[1:] - eta_t[:-1]
        return eta_t, alpha_t

    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0, y0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        eta_t = self.extract(self.eta, t, x0.shape)
        residual = y0 - x0
        mean = x0 + eta_t * residual
        std = self.kappa * torch.sqrt(eta_t)
        x_t = mean + std * noise
        return x_t

    def p_losses(self, denoise_model, x0, y0, t):
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, y0, t, noise=noise)
        pred_x0 = denoise_model(x_t, y0, t)
        loss = nn.functional.mse_loss(pred_x0, x0)
        return loss

    @torch.no_grad()
    def q_posterior_mean_variance(self, x_start, x_t, y0, t):
        eta_t = self.extract(self.eta, t, x_t.shape)
        eta_t_minus_1 = self.extract(self.eta, t - 1, x_t.shape)
        alpha_t = self.extract(self.alpha, t, x_t.shape)
        coef1 = eta_t_minus_1 / eta_t
        coef2 = alpha_t / eta_t
        posterior_mean = coef1 * x_t + coef2 * x_start
        posterior_variance = (self.kappa ** 2) * (eta_t_minus_1 * alpha_t) / eta_t
        return posterior_mean, posterior_variance

    @torch.no_grad()
    def p_sample(self, denoise_model, x_t, y0, t):
        pred_x_start = denoise_model(x_t, y0, t)
        if (t == 0).all():
            return pred_x_start
        model_mean, model_variance = self.q_posterior_mean_variance(pred_x_start, x_t, y0, t)
        noise = torch.randn_like(x_t)
        x_t_minus_1 = model_mean + torch.sqrt(model_variance) * noise
        return x_t_minus_1

    @torch.no_grad()
    def p_sample_loop(self, denoise_model, y0):
        device = y0.device
        b = y0.shape[0]
        x_t = y0 + self.kappa * torch.sqrt(self.eta[-1]) * torch.randn_like(y0)
        for i in tqdm(reversed(range(0, self.T)), desc='ResShift Sampling', total=self.T):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x_t = self.p_sample(denoise_model, x_t, y0, t)
        return x_t


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        freqs = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -freqs)
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.embed = SinusoidalTimeEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        return self.mlp(self.embed(t))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(t_emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class SpectralSelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv.unbind(1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.softmax((q.transpose(-2, -1) @ k) / math.sqrt(q.shape[-2]), dim=-1)
        out = v @ attn.transpose(-2, -1)
        out = out.reshape(b, c, h, w)
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
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class ResShiftDenoiser(nn.Module):
    def __init__(self, channels=31, base_dim=64, dim_mults=(1, 2, 4), time_dim=256, num_res_blocks=2):
        super().__init__()
        self.time_mlp = TimeMLP(time_dim, time_dim)
        in_ch = channels * 2
        self.conv_in = nn.Conv2d(in_ch, base_dim, 3, padding=1)
        dims = [base_dim] + [base_dim * m for m in dim_mults]
        self.num_levels = len(dim_mults)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(self.num_levels):
            dim_in, dim_out = dims[i], dims[i + 1]
            blocks = nn.ModuleList([
                ResBlock(dim_in if j == 0 else dim_out, dim_out, time_dim)
                for j in range(num_res_blocks)
            ])
            self.down_blocks.append(blocks)
            is_last = i == self.num_levels - 1
            self.downsamples.append(nn.Identity() if is_last else Downsample(dim_out))

        mid_dim = dims[-1]
        self.mid_block1 = ResBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = SpectralSelfAttention(mid_dim)
        self.mid_block2 = ResBlock(mid_dim, mid_dim, time_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in reversed(range(self.num_levels)):
            dim_in, dim_out = dims[i], dims[i + 1]
            blocks = nn.ModuleList([
                ResBlock(dim_out * 2 if j == 0 else dim_in, dim_in, time_dim)
                for j in range(num_res_blocks)
            ])
            self.up_blocks.append(blocks)
            is_last = i == self.num_levels - 1
            self.upsamples.append(nn.Identity() if is_last else Upsample(dim_out))

        self.norm_out = nn.GroupNorm(min(8, base_dim), base_dim)
        self.conv_out = nn.Conv2d(base_dim, channels, 3, padding=1)

    def forward(self, x_t, y0, t):
        t_emb = self.time_mlp(t)
        h = self.conv_in(torch.cat([x_t, y0], dim=1))
        skips = []
        for i in range(self.num_levels):
            for block in self.down_blocks[i]:
                h = block(h, t_emb)
            skips.append(h)
            h = self.downsamples[i](h)
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        for idx, i in enumerate(reversed(range(self.num_levels))):
            h = self.upsamples[idx](h)
            h = torch.cat([h, skips[i]], dim=1)
            for block in self.up_blocks[idx]:
                h = block(h, t_emb)
        return self.conv_out(F.silu(self.norm_out(h)))


class ResShiftSSR(nn.Module):
    def __init__(self, mst_ckpt_path=None, channels=31, T=15, p=0.3, kappa=2.0,
                 base_dim=64, dim_mults=(1, 2, 4), num_res_blocks=2, freeze_coarse=True):
        super().__init__()

        if "MST_Plus_Plus" not in globals():
            raise ImportError("MST_Plus_Plus is not imported.")
        self.coarse_net = MST_Plus_Plus(in_channels=3, out_channels=channels)

        if mst_ckpt_path is not None:
            ckpt = torch.load(mst_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.coarse_net.load_state_dict(state_dict)

        self.freeze_coarse = freeze_coarse
        if freeze_coarse:
            for param in self.coarse_net.parameters():
                param.requires_grad = False
            self.coarse_net.eval()

        if "ResShiftDiffusion" not in globals():
            raise ImportError("ResShiftDiffusion is not imported.")
        self.diffusion = ResShiftDiffusion(T=T, p=p, kappa=kappa)

        self.denoiser = ResShiftDenoiser(
            channels=channels, base_dim=base_dim, dim_mults=dim_mults, num_res_blocks=num_res_blocks
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_coarse:
            self.coarse_net.eval()
        return self

    def forward(self, rgb, hsi_gt):
        with torch.set_grad_enabled(not self.freeze_coarse):
            y0 = self.coarse_net(rgb)
        b = hsi_gt.shape[0]
        t = torch.randint(0, self.diffusion.T, (b,), device=hsi_gt.device, dtype=torch.long)
        loss = self.diffusion.p_losses(self.denoiser, hsi_gt, y0, t)
        return loss

    @torch.no_grad()
    def sample(self, rgb):
        y0 = self.coarse_net(rgb)
        fine_hsi = self.diffusion.p_sample_loop(self.denoiser, y0)
        return fine_hsi, y0
