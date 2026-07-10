"""
Residual Diffusion Bridge Model (RDBM) for Hyperspectral Image Reconstruction.

This file consolidates the original RDBM implementation (networks.py + rdbm.py,
Wang et al., "Residual Diffusion Bridge Model for Image Restoration", CVPR 2026)
into a single module and adapts it to the hyperspectral reconstruction setting:

    - The conditioning image "mu" of the original RDBM (previously the raw
      degraded/low-quality RGB image) is replaced by the prediction of a
      frozen, pretrained MST++ network (RGB -> 31-channel HSI).
    - The diffusion U-Net therefore no longer bridges (clean image <-> raw
      degraded image), but bridges (ground-truth HSI <-> MST++ prediction),
      i.e. it is trained to model the *residual* gt_hsi - mst_pred, exactly
      as the original RDBM already internally models the residual between
      x0 and mu (see Unet.forward: `return x + mu`, and the paper's
      pi = x0 - mu modulation term).
    - The paper's residual-modulated forward process is used explicitly:
          x_t = mu + (x_0 - mu) * Theta_t
                   + (x_0 - mu) * Sigma_t * epsilon.
      Thus, regions or spectral values already reconstructed correctly by
      MST++ receive little or no perturbation.
    - This implementation uses the stable x_0-prediction parameterization:
      the U-Net predicts the clean HSI as mu plus a learned residual. The
      reverse recursion is the paper's residual form with pi_hat=x0_hat-mu.

Usage:
    model = RDBMHSI(mst_ckpt="path/to/mstplusplus.pth")
    loss = model(rgb, gt_hsi)                 # training
    hsi_pred = model.reconstruct(rgb)         # inference
"""

import math
from collections import namedtuple
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum, nn
from tqdm.auto import tqdm

# -----------------------------------------------------------------------
# Import the pretrained MST++ model. Adjust this import to match wherever
# the MST++ definition actually lives in your codebase.
# -----------------------------------------------------------------------
from .MST_Plus_Plus import MST_Plus_Plus  # noqa: E402  (import path to be adjusted later)


ModelResPrediction = namedtuple('ModelResPrediction', ['pred_noise', 'pred_x_start'])


# =========================================================================
# ------------------------------- Utils ----------------------------------
# =========================================================================

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


# =========================================================================
# --------------------------- U-Net building blocks ----------------------
# (kept structurally identical to the original networks.py)
# =========================================================================

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """https://arxiv.org/abs/1903.10520"""

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding,
                         self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8"""

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    """
    Same U-Net architecture as the original RDBM (networks.py), with the
    default number of channels changed from 3 (RGB) to 31 (hyperspectral).

    `condition=True` concatenates the conditioning image `mu` (here: the
    frozen MST++ prediction) with the noisy state `x_t` along the channel
    dimension, exactly as in the original implementation.
    """

    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=31,                 # <-- hyperspectral channels (was 3 for RGB)
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        condition=True,               # True: residual bridge conditioned on mu
    ):
        super().__init__()

        self.channels = channels
        self.depth = len(dim_mults)
        input_channels = channels + channels * (1 if condition else 0)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x_t, mu, time):
        """
        x_t : noisy HSI state at diffusion time `time`     [B, 31, H, W]
        mu  : conditioning image, i.e. the frozen MST++    [B, 31, H, W]
              prediction (analogue of the LQ image in the
              original RDBM formulation)
        """
        x = torch.cat((x_t, mu), dim=1)
        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = x[..., :H, :W].contiguous()
        # x is the learned clean residual pi_hat. Adding the fixed endpoint
        # mu yields an x0 prediction: x0_hat = mu + pi_hat.
        return x + mu


# =========================================================================
# ------------------------------ RDBM core --------------------------------
# (kept identical to the original rdbm.py RDBM class)
# =========================================================================

class RDBM(nn.Module):
    """Residual-modulated diffusion bridge with x0 prediction.

    The paper's forward process is

        x_t = mu + pi * Theta_t + pi * Sigma_t * epsilon,
        pi  = x_0 - mu.

    The denoiser predicts x_0 directly. Equivalently, because the U-Net adds
    ``mu`` to its residual output, it predicts ``pi_hat = x0_hat - mu``.

    Only ``pred_x_start`` is exposed. The previous ``pred_noise`` option was
    inconsistent with the U-Net output and could divide by zero at the
    terminal bridge endpoint, where Sigma_T = 0.
    """

    def __init__(
        self,
        model,
        *,
        image_size=256,
        objective='pred_x_start',
        sampling_type='pred_x_start',
        timesteps=100,
        sampling_timesteps=10,
        condition=True,
        lamb=10.0 / 255.0,
        loss_type='l1',
    ):
        super().__init__()

        if objective != 'pred_x_start':
            raise ValueError(
                "This implementation supports only objective='pred_x_start'. "
                "The U-Net outputs x0 = mu + predicted_residual."
            )
        if sampling_type != 'pred_x_start':
            raise ValueError(
                "This implementation supports only sampling_type='pred_x_start'."
            )
        if timesteps < 2:
            raise ValueError('timesteps must be at least 2.')
        if sampling_timesteps is None:
            sampling_timesteps = timesteps
        if not 1 <= sampling_timesteps <= timesteps:
            raise ValueError(
                'sampling_timesteps must be between 1 and timesteps, inclusive.'
            )
        if lamb <= 0:
            raise ValueError('lamb must be positive.')
        if loss_type not in {'l1', 'l2'}:
            raise ValueError("loss_type must be either 'l1' or 'l2'.")
        if not condition:
            raise ValueError('RDBMHSI requires condition=True.')

        self.model = model
        self.channels = self.model.channels
        self.image_size = image_size
        self.condition = condition
        self.objective = objective
        self.sampling_type = sampling_type
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = int(sampling_timesteps)
        self.lamb = float(lamb)
        self.loss_type = loss_type

        thetas = betas_for_alpha_bar(self.num_timesteps)
        thetas_cumsum_0_to_t = thetas.cumsum(dim=0)
        thetas_cumsum_0_to_T = thetas_cumsum_0_to_t[-1]
        thetas_cumsum_t_to_T = thetas_cumsum_0_to_T - thetas_cumsum_0_to_t

        sinh_thetas_cumsum_0_to_t = torch.sinh(thetas_cumsum_0_to_t)
        sinh_thetas_cumsum_0_to_T = torch.sinh(thetas_cumsum_0_to_T)
        sinh_thetas_cumsum_t_to_T = torch.sinh(thetas_cumsum_t_to_T)

        Theta = sinh_thetas_cumsum_t_to_T / sinh_thetas_cumsum_0_to_T
        Sigma2 = (
            2
            * self.lamb
            * sinh_thetas_cumsum_0_to_t
            * sinh_thetas_cumsum_t_to_T
            / sinh_thetas_cumsum_0_to_T
        )
        # Numerical guard only; the analytical quantity is non-negative.
        Sigma = torch.sqrt(Sigma2.clamp_min(0.0))

        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps

        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer('thetas', thetas)
        register_buffer('thetas_cumsum_0_to_t', thetas_cumsum_0_to_t)
        register_buffer('thetas_cumsum_0_to_T', thetas_cumsum_0_to_T)
        register_buffer('thetas_cumsum_t_to_T', thetas_cumsum_t_to_T)
        register_buffer('sinh_thetas_cumsum_0_to_t', sinh_thetas_cumsum_0_to_t)
        register_buffer('sinh_thetas_cumsum_0_to_T', sinh_thetas_cumsum_0_to_T)
        register_buffer('sinh_thetas_cumsum_t_to_T', sinh_thetas_cumsum_t_to_T)
        register_buffer('Theta', Theta)
        register_buffer('Sigma2', Sigma2)
        register_buffer('Sigma', Sigma)

    def model_predictions(self, x_t, mu, t, clip_denoised=False):
        """Predict x0 without deriving a noise estimate at Sigma_T=0."""
        x_start = self.model(x_t, mu, t)

        if clip_denoised:
            x_start = x_start.clamp(-1.0, 1.0)

        return ModelResPrediction(pred_noise=None, pred_x_start=x_start)

    @torch.no_grad()
    def ddim_sample(self, mu, shape, last=True):
        """Deterministic reverse recursion from the endpoint ``mu`` to x0."""
        if tuple(mu.shape) != tuple(shape):
            raise ValueError(
                f'mu shape {tuple(mu.shape)} does not match requested shape {tuple(shape)}.'
            )

        batch = shape[0]
        device = mu.device
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps

        times = torch.linspace(
            -1,
            total_timesteps - 1,
            steps=sampling_timesteps + 1,
            device=device,
        )
        times = list(reversed(times.long().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # A diffusion bridge has the deterministic terminal endpoint x_T=mu.
        img = mu
        trajectory = [] if not last else None

        for time, time_next in tqdm(
            time_pairs,
            desc='sampling loop time step',
            disable=True,
        ):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long
            )
            x_start = self.model_predictions(
                img, mu, time_cond
            ).pred_x_start

            if time_next < 0:
                img = x_start
                if trajectory is not None:
                    trajectory.append(img)
                continue

            theta_now = self.Theta[time]
            theta_next = self.Theta[time_next]
            sigma_now = self.Sigma[time]
            sigma_next = self.Sigma[time_next]

            if time == total_timesteps - 1:
                # At T, Sigma_T=Theta_T=0 and img=mu. Use the analytical
                # endpoint limit instead of forming a 0/0 ratio.
                img = mu + theta_next * (x_start - mu)
            else:
                sigma_ratio = sigma_next / sigma_now
                img = (
                    mu
                    + sigma_ratio * (img - mu)
                    + (theta_next - theta_now * sigma_ratio)
                    * (x_start - mu)
                )

            if trajectory is not None:
                trajectory.append(img)

        if last:
            return [mu, img]
        return [mu, *trajectory]

    def sample(self, x_input=None, gt=None, batch_size=16, last=True):
        del gt, batch_size  # retained in the signature for backward compatibility

        if x_input is None:
            raise ValueError('x_input (the MST++ prediction mu) is required.')
        if x_input.ndim != 4:
            raise ValueError(
                f'x_input must have shape [B,C,H,W], got {tuple(x_input.shape)}.'
            )
        if x_input.shape[1] != self.channels:
            raise ValueError(
                f'Expected {self.channels} HSI channels, got {x_input.shape[1]}.'
            )

        return self.ddim_sample(x_input, tuple(x_input.shape), last=last)

    def q_sample(self, x_start, mu, t, noise=None):
        """Sample the paper's residual-modulated forward bridge q(x_t|x0,mu)."""
        if x_start.shape != mu.shape:
            raise ValueError(
                f'x_start shape {tuple(x_start.shape)} and mu shape '
                f'{tuple(mu.shape)} must match.'
            )

        noise = default(noise, lambda: torch.randn_like(x_start))
        if noise.shape != x_start.shape:
            raise ValueError(
                f'noise shape {tuple(noise.shape)} must match x_start '
                f'shape {tuple(x_start.shape)}.'
            )

        residual = x_start - mu
        theta_t = extract(self.Theta, t, x_start.shape)
        sigma_t = extract(self.Sigma, t, x_start.shape)

        # Crucial RDBM term: residual also modulates the noise amplitude.
        return mu + residual * theta_t + residual * sigma_t * noise

    def _loss(self, prediction, target):
        if self.loss_type == 'l1':
            return F.l1_loss(prediction, target)
        return F.mse_loss(prediction, target)

    def p_losses(self, imgs, t, noise=None):
        # imgs = [x_start (GT HSI), mu (frozen MST++ prediction)]
        if not isinstance(imgs, (list, tuple)) or len(imgs) != 2:
            raise ValueError('imgs must be [gt_hsi, mu].')

        x_start, mu = imgs
        if x_start.shape != mu.shape:
            raise ValueError(
                f'GT HSI shape {tuple(x_start.shape)} does not match MST++ '
                f'prediction shape {tuple(mu.shape)}.'
            )

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start, mu, t, noise=noise)

        # x0-prediction parameterization. Since Unet.forward returns
        # mu + predicted_residual, this is equivalent to supervising the
        # predicted clean residual against x_start - mu.
        x_start_pred = self.model(x_t, mu, t)
        return self._loss(x_start_pred, x_start)

    def forward(self, img, *args, **kwargs):
        if not isinstance(img, (list, tuple)) or len(img) != 2:
            raise ValueError('img must be [gt_hsi, mu].')

        x_start = img[0]
        if x_start.ndim != 4:
            raise ValueError(
                f'gt_hsi must have shape [B,C,H,W], got {tuple(x_start.shape)}.'
            )

        batch = x_start.shape[0]
        t = torch.randint(
            0,
            self.num_timesteps,
            (batch,),
            device=x_start.device,
            dtype=torch.long,
        )
        return self.p_losses(img, t, *args, **kwargs)


# =========================================================================
# ---------------- RDBM + frozen MST++ for HSI reconstruction ------------
# =========================================================================

class RDBMHSI(nn.Module):
    """
    Wraps:
      1) a frozen, pretrained MST++ network that maps RGB -> 31-channel HSI
         and provides the RDBM conditioning image mu.
      2) the RDBM diffusion bridge / U-Net, which is trained to model the
         residual between the ground-truth HSI (x0) and the MST++
         prediction (mu), i.e. the bridge endpoint pair (x0 = gt_hsi,
         mu = mst_pred).

    Training:
        loss = model(rgb, gt_hsi)

    Inference:
        hsi_pred = model.reconstruct(rgb)
        # internally: mu = MST++(rgb) (frozen)
        #             residual is sampled via the RDBM reverse process
        #             final HSI = mu + predicted residual (handled inside
        #             the U-Net / ddim_sample, consistent with the
        #             original RDBM formulation)
    """

    def __init__(
        self,
        mst_ckpt=None,
        hsi_channels=31,
        unet_dim=64,
        dim_mults=(1, 2, 2, 4),
        timesteps=100,
        sampling_timesteps=10,
        image_size=256,
        objective='pred_x_start',
        sampling_type='pred_x_start',
        lamb=10.0 / 255.0,
        loss_type='l1',
        mst_kwargs=None,
        mst_strict=True,
        check_finite=False,
    ):
        super().__init__()

        # ---- 1) Frozen pretrained MST++ conditioning network ----
        mst_kwargs = default(mst_kwargs, {})
        self.mst = MST_Plus_Plus(**mst_kwargs)
        if mst_ckpt is not None:
            checkpoint = torch.load(mst_ckpt, map_location='cpu')
            state_dict = self._extract_mst_state_dict(checkpoint)
            self.mst.load_state_dict(state_dict, strict=mst_strict)

        self.check_finite = bool(check_finite)
        self.hsi_channels = int(hsi_channels)

        self.mst.eval()
        for p in self.mst.parameters():
            p.requires_grad_(False)

        # ---- 2) RDBM diffusion bridge over the residual ----
        self.unet = Unet(
            dim=unet_dim,
            dim_mults=dim_mults,
            channels=hsi_channels,
            condition=True,
        )
        self.rdbm = RDBM(
            model=self.unet,
            image_size=image_size,
            objective=objective,
            sampling_type=sampling_type,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            condition=True,
            lamb=lamb,
            loss_type=loss_type,
        )

    @staticmethod
    def _extract_mst_state_dict(checkpoint):
        """Extract a state dict from common MST++ checkpoint layouts."""
        if not isinstance(checkpoint, dict):
            raise TypeError(
                'MST++ checkpoint must be a state-dict-like dictionary.'
            )

        state_dict = checkpoint
        for key in ('state_dict', 'model_state_dict', 'model', 'mst', 'mst_state_dict'):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                state_dict = value
                break

        cleaned = {}
        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            while key.startswith('module.'):
                key = key[len('module.'):]
            if key.startswith('mst.'):
                key = key[len('mst.'):]
            cleaned[key] = value

        if not cleaned:
            raise ValueError('No tensor parameters were found in the MST++ checkpoint.')
        return cleaned

    def train(self, mode=True):
        """Keep the frozen MST++ conditioner in evaluation mode."""
        super().train(mode)
        self.mst.eval()
        return self

    def _frozen_mst_predict(self, rgb):
        if rgb.ndim != 4:
            raise ValueError(
                f'rgb must have shape [B,3,H,W], got {tuple(rgb.shape)}.'
            )

        self.mst.eval()
        with torch.no_grad():
            mu = self.mst(rgb)

        if not torch.is_tensor(mu):
            raise TypeError('MST++ must return a tensor.')
        if mu.ndim != 4:
            raise ValueError(
                f'MST++ output must have shape [B,C,H,W], got {tuple(mu.shape)}.'
            )
        if mu.shape[1] != self.hsi_channels:
            raise ValueError(
                f'MST++ returned {mu.shape[1]} channels; expected '
                f'{self.hsi_channels}.'
            )
        if self.check_finite and not torch.isfinite(mu).all():
            raise FloatingPointError('MST++ output contains NaN or Inf values.')
        return mu

    def forward(self, rgb, gt_hsi):
        """
        Training step. Returns the RDBM residual-bridge L1 loss between the
        predicted and ground-truth HSI, with mu fixed to the frozen MST++
        output.
        """
        if gt_hsi.ndim != 4:
            raise ValueError(
                f'gt_hsi must have shape [B,C,H,W], got {tuple(gt_hsi.shape)}.'
            )
        if gt_hsi.shape[1] != self.hsi_channels:
            raise ValueError(
                f'gt_hsi has {gt_hsi.shape[1]} channels; expected '
                f'{self.hsi_channels}.'
            )
        if self.check_finite and not torch.isfinite(gt_hsi).all():
            raise FloatingPointError('Ground-truth HSI contains NaN or Inf values.')

        mu = self._frozen_mst_predict(rgb)
        if mu.shape != gt_hsi.shape:
            raise ValueError(
                f'MST++ output shape {tuple(mu.shape)} does not match GT HSI '
                f'shape {tuple(gt_hsi.shape)}.'
            )

        return self.rdbm([gt_hsi, mu])

    @torch.no_grad()
    def reconstruct(self, rgb, last=True):
        """
        Inference: obtain the MST++ prediction (mu), then run the RDBM
        reverse (DDIM-style) process to refine it into the final
        hyperspectral reconstruction: hsi_pred = mu + predicted_residual.
        """
        mu = self._frozen_mst_predict(rgb)
        samples = self.rdbm.sample(x_input=mu, last=last)

        # Preserve the convenient tensor return for normal inference, while
        # returning the complete trajectory when explicitly requested.
        if last:
            return samples[-1]
        return samples
