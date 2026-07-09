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
    - All diffusion equations, the (Theta, Sigma) scheduler derived from
      Doob's h-transform / OU bridge, the forward q_sample process, the
      deterministic reverse (DDIM-style) sampling recursion, and the U-Net
      architecture (ResNet blocks, linear/full attention, sinusoidal time
      embeddings, conditioning-by-concatenation) are kept as close to the
      original implementation as possible. Only the channel count (3 -> 31)
      and the introduction of the frozen MST++ conditioning network are new.

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
        # The network internally predicts the residual (x_start - mu) and
        # adds mu back -> this is exactly the "residual-only" learning
        # objective requested: the U-Net never has to reproduce mu itself.
        return x + mu


# =========================================================================
# ------------------------------ RDBM core --------------------------------
# (kept identical to the original rdbm.py RDBM class)
# =========================================================================

class RDBM(nn.Module):
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
    ):
        super().__init__()

        assert objective in ['pred_noise', 'pred_x_start']
        assert sampling_type in ['pred_noise', 'pred_x_start']

        self.model = model
        self.channels = self.model.channels
        self.image_size = image_size
        self.condition = condition
        self.objective = objective
        self.sampling_type = sampling_type
        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps

        lamb = 1e-4
        thetas = betas_for_alpha_bar(timesteps)
        thetas_cumsum_0_to_t = thetas.cumsum(dim=0)
        thetas_cumsum_0_to_T = thetas_cumsum_0_to_t[-1]
        thetas_cumsum_t_to_T = thetas_cumsum_0_to_T - thetas_cumsum_0_to_t

        sinh_thetas_cumsum_0_to_t = torch.sinh(thetas_cumsum_0_to_t)
        sinh_thetas_cumsum_0_to_T = torch.sinh(thetas_cumsum_0_to_T)
        sinh_thetas_cumsum_t_to_T = torch.sinh(thetas_cumsum_t_to_T)

        Theta = sinh_thetas_cumsum_t_to_T / (sinh_thetas_cumsum_0_to_T)
        Sigma2 = 2 * lamb * (sinh_thetas_cumsum_0_to_t) * (sinh_thetas_cumsum_t_to_T) / (sinh_thetas_cumsum_0_to_T)
        Sigma = torch.sqrt(Sigma2)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps

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

    def predict_x_start_from_noise(self, x_t, t, mu, noise):
        return (
            ((x_t - mu - (extract(self.Sigma, t, x_t.shape) * noise)) / (extract(self.Theta, t, x_t.shape))) + mu
        )

    def predict_noise_from_x_start(self, x_t, t, mu, x_start):
        return (
            (x_t - mu - extract(self.Theta, t, x_t.shape) * (x_start - mu)) / extract(self.Sigma, t, x_t.shape)
        )

    def model_predictions(self, x_t, mu, t, clip_denoised=False):
        # NOTE: clip_denoised defaults to False here because HSI reflectance
        # values are not necessarily bounded to [-1, 1] like RGB pixels.
        # Set clip_denoised=True if your HSI data is normalized accordingly.
        model_output = self.model(x_t, mu, t)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_denoised else identity

        if self.objective == "pred_noise":
            noise = model_output
            x_start = self.predict_x_start_from_noise(x_t, t, mu, noise)
            x_start = maybe_clip(x_start)
        elif self.objective == "pred_x_start":
            x_start = model_output
            x_start = maybe_clip(x_start)
            noise = self.predict_noise_from_x_start(x_t, t, mu, x_start)
        else:
            exit('please specify the prediction mode')

        return ModelResPrediction(noise, x_start)

    @torch.no_grad()
    def ddim_sample(self, x_input, gt, shape, last=True):
        mu = x_input[0]
        batch, device, total_timesteps, sampling_timesteps, objective = shape[
            0], self.thetas.device, self.num_timesteps, self.sampling_timesteps, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        if self.condition:
            img = mu
        else:
            img = torch.randn(shape, device=device)

        x_start = None

        if not last:
            img_list = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step', disable=True):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            preds = self.model_predictions(img, mu, time_cond)

            noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            Theta_now = self.Theta[time]
            Theta_next = self.Theta[time_next]
            Sigma_now = self.Sigma[time]
            Sigma_next = self.Sigma[time_next]

            if self.sampling_type == "pred_noise":
                if time == (self.num_timesteps - 1):
                    img = mu
                else:
                    img = mu + (Theta_next / Theta_now) * (img - mu) - (
                        ((Theta_next / Theta_now) * Sigma_now) - Sigma_next) * noise
            elif self.sampling_type == "pred_x_start":
                if time == (self.num_timesteps - 1):
                    img = mu + Theta_next * (x_start - mu)
                else:
                    img = mu + (Sigma_next / Sigma_now) * (img - mu) + (
                        Theta_next - (Theta_now * Sigma_next / Sigma_now)) * (x_start - mu)
            else:
                exit('Illegal objective')

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [mu] + img_list
            else:
                img_list = [mu, img]
            return img_list
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return img_list

    def sample(self, x_input=None, gt=None, batch_size=16, last=True):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.ddim_sample
        if self.condition:
            x_input = x_input.unsqueeze(0)
            batch_size, channels, h, w = x_input[0].shape
            size = (batch_size, channels, h, w)
        else:
            size = (batch_size, channels, image_size, image_size)

        samples = sample_fn(x_input, gt, size, last=last)
        return samples

    def q_sample(self, x_start, mu, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            mu + (x_start - mu) * extract(self.Theta, t, x_start.shape) + extract(self.Sigma, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self, loss_type='l1'):
        if loss_type == 'l1':
            return F.l1_loss
        elif loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def p_losses(self, imgs, t, noise=None):
        # imgs = [x_start (gt HSI), mu (frozen MST++ prediction)]
        if isinstance(imgs, list):
            x_start = imgs[0]
            mu = imgs[1]
        else:
            raise ValueError('imgs must be a list [gt_hsi, mu]')

        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start, mu, t, noise=noise)
        model_out = self.model(x, mu, t)
        target = x_start

        loss = F.l1_loss(model_out, target)
        return loss

    def forward(self, img, *args, **kwargs):
        if isinstance(img, list):
            b, c, h, w, device = *img[0].shape, img[0].device
        else:
            b, c, h, w, device = *img.shape, img.device

        t = torch.randint(0, int(self.num_timesteps), (b,), device=device).long()
        t = torch.clamp(t, min=0, max=self.num_timesteps - 1)

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
        mst_kwargs=None,
    ):
        super().__init__()

        # ---- 1) Frozen pretrained MST++ conditioning network ----
        mst_kwargs = default(mst_kwargs, {})
        self.mst = MST_Plus_Plus(**mst_kwargs)
        if mst_ckpt is not None:
            state_dict = torch.load(mst_ckpt, map_location='cpu')
            state_dict = state_dict.get('state_dict', state_dict)
            self.mst.load_state_dict(state_dict, strict=True)

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
        )

    def _frozen_mst_predict(self, rgb):
        self.mst.eval()
        with torch.no_grad():
            mu = self.mst(rgb)
        return mu

    def forward(self, rgb, gt_hsi):
        """
        Training step. Returns the RDBM residual-bridge L1 loss between the
        predicted and ground-truth HSI, with mu fixed to the frozen MST++
        output.
        """
        mu = self._frozen_mst_predict(rgb)
        loss = self.rdbm([gt_hsi, mu])
        return loss

    @torch.no_grad()
    def reconstruct(self, rgb, last=True):
        """
        Inference: obtain the MST++ prediction (mu), then run the RDBM
        reverse (DDIM-style) process to refine it into the final
        hyperspectral reconstruction: hsi_pred = mu + predicted_residual.
        """
        mu = self._frozen_mst_predict(rgb)
        samples = self.rdbm.sample(x_input=mu, last=last)
        # samples = [mu, x0_pred] (or full trajectory if last=False);
        # the final reconstructed HSI is the last element.
        hsi_pred = samples[-1]
        return hsi_pred


