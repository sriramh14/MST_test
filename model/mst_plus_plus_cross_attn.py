import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    else:
        raise ValueError(f"invalid mode {mode}")

    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


def shift_back(inputs, step=2):
    # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


# -----------------------------------------------------------------------------
# REMOVED ORIGINAL ATTENTION BLOCK
# Only the original MS_MSA block is removed. The rest of MST++ is unchanged.
# It is left commented below for direct comparison.
# -----------------------------------------------------------------------------
# class MS_MSA(nn.Module):
#     def __init__(
#             self,
#             dim,
#             dim_head,
#             heads,
#     ):
#         super().__init__()
#         self.num_heads = heads
#         self.dim_head = dim_head
#         self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
#         self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
#         self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
#         self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
#         self.proj = nn.Linear(dim_head * heads, dim, bias=True)
#         self.pos_emb = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
#             GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
#         )
#         self.dim = dim
#
#     def forward(self, x_in):
#         """
#         x_in: [b,h,w,c]
#         return out: [b,h,w,c]
#         """
#         b, h, w, c = x_in.shape
#         x = x_in.reshape(b, h * w, c)
#         q_inp = self.to_q(x)
#         k_inp = self.to_k(x)
#         v_inp = self.to_v(x)
#         q, k, v = map(
#             lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
#             (q_inp, k_inp, v_inp),
#         )
#         q = q.transpose(-2, -1)
#         k = k.transpose(-2, -1)
#         v = v.transpose(-2, -1)
#         q = F.normalize(q, dim=-1, p=2)
#         k = F.normalize(k, dim=-1, p=2)
#         attn = (k @ q.transpose(-2, -1))
#         attn = attn * self.rescale
#         attn = attn.softmax(dim=-1)
#         x = attn @ v
#         x = x.permute(0, 3, 1, 2)
#         x = x.reshape(b, h * w, self.num_heads * self.dim_head)
#         out_c = self.proj(x).view(b, h, w, c)
#         out_p = self.pos_emb(
#             v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)
#         ).permute(0, 2, 3, 1)
#         return out_c + out_p


# -----------------------------------------------------------------------------
# NEW MODULES FOR VARIANCE-GUIDED SPECTRAL-SPATIAL ATTENTION
# -----------------------------------------------------------------------------
class ConvFeatureHead(nn.Module):
    """Lightweight full-resolution feature/value prediction head."""

    def __init__(self, in_channels, out_channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(16, min(64, in_channels))

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=False),
            GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                1,
                1,
                bias=False,
                groups=hidden_channels,
            ),
            GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class LocalSpatialVariance(nn.Module):
    """
    Predicts a single-channel spatial-variance gate from the current MST++
    feature map.

    Input:  [B, C, H, W]
    Output: [B, 1, H, W] in [0, 1]
    """

    def __init__(self, dim, kernel_size=5):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("variance kernel_size must be odd")

        variance_channels = max(8, min(32, dim // 2))
        self.kernel_size = kernel_size
        self.proj = nn.Conv2d(dim, variance_channels, 1, 1, 0, bias=False)
        self.to_gate = nn.Sequential(
            nn.Conv2d(
                variance_channels,
                variance_channels,
                3,
                1,
                1,
                bias=False,
                groups=variance_channels,
            ),
            GELU(),
            nn.Conv2d(variance_channels, 1, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        f = self.proj(x)
        pad = self.kernel_size // 2

        # Reflection padding avoids treating zeros outside the image as texture.
        pad_mode = 'reflect' if f.shape[-2] > pad and f.shape[-1] > pad else 'replicate'
        f_pad = F.pad(f, [pad, pad, pad, pad], mode=pad_mode)

        mean = F.avg_pool2d(f_pad, self.kernel_size, stride=1)
        mean_sq = F.avg_pool2d(f_pad * f_pad, self.kernel_size, stride=1)
        variance = (mean_sq - mean * mean).clamp_min(0.0)

        # log1p limits very large values while preserving the variance ordering.
        variance = torch.log1p(variance)
        return self.to_gate(variance)


class MS_MSA(nn.Module):
    """
    Variance-Guided Spectral-Spatial Attention (VGSSA).

    This class intentionally keeps the name ``MS_MSA`` so every attention
    block in the original MST++ is replaced automatically without modifying
    MSAB, MST, or MST_Plus_Plus.

    Input/output layout:
        x_in: [B, H, W, C]
        out:  [B, H, W, C]

    Main operations:
        1. Each feature channel creates one query from its pooled spatial map.
        2. Downsampled spatial features create keys.
        3. QK^T produces a coarse channel-by-space attention map.
        4. The map is bilinearly upsampled.
        5. Local spatial variance gates a learned high-resolution correction.
        6. The refined attention modulates a learned full-resolution value map.

    Because this block replaces attention at all MST++ depths, sigmoid is used
    rather than channel-wise softmax. Deeper 62/124-channel features are latent
    channels, and multiple channels should be allowed to activate together.
    """

    def __init__(
        self,
        dim,
        dim_head,
        heads,
        spatial_reduction=4,
        query_pool_size=4,
        variance_kernel=5,
    ):
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be at least 1")
        if spatial_reduction < 1:
            raise ValueError("spatial_reduction must be at least 1")

        self.dim = dim
        self.num_heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.spatial_reduction = spatial_reduction
        self.query_pool_size = query_pool_size

        # Spectral/latent-channel queries.
        # [B, C, query_pool_size^2] -> [B, C, heads*dim_head]
        self.to_q = nn.Linear(
            query_pool_size * query_pool_size,
            self.inner_dim,
            bias=False,
        )

        # Spatial keys are generated from the complete feature tensor.
        self.to_k = nn.Conv2d(dim, self.inner_dim, 1, 1, 0, bias=False)

        # Per-head similarity scale and learned head aggregation.
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.head_logits = nn.Parameter(torch.zeros(heads))

        # Full-resolution learned intensity/base and value/correction branches.
        self.intensity_head = ConvFeatureHead(dim, dim)
        self.value_head = ConvFeatureHead(dim, dim)

        # High-resolution local detail features used by the attention refiner.
        detail_channels = max(8, min(32, dim // 4))
        self.detail_head = ConvFeatureHead(dim, detail_channels)

        # Spatial variance does not determine intensity directly. It determines
        # where the smoothly upsampled attention map may be corrected strongly.
        self.variance_gate = LocalSpatialVariance(dim, kernel_size=variance_kernel)

        refine_hidden = max(16, min(64, dim // 2))
        refine_in_channels = dim + dim + 1 + detail_channels
        self.attention_refiner = nn.Sequential(
            nn.Conv2d(refine_in_channels, refine_hidden, 1, 1, 0, bias=False),
            GELU(),
            nn.Conv2d(
                refine_hidden,
                refine_hidden,
                3,
                1,
                1,
                bias=False,
                groups=refine_hidden,
            ),
            GELU(),
            nn.Conv2d(refine_hidden, dim, 1, 1, 0, bias=True),
        )

        # Output projection after feature-amplitude modulation.
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)

        # Preserve the useful local positional branch from the original block.
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

        # MSAB already applies ``x = attn(x) + x``. Therefore this block returns
        # only a residual update. A small nonzero initial scale stabilizes the
        # complete replacement while still allowing gradients through the block.
        self.output_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_in):
        """
        x_in: [B, H, W, C]
        return: [B, H, W, C]
        """
        b, h, w, c = x_in.shape
        if c != self.dim:
            raise ValueError(f"Expected {self.dim} channels, but received {c}")

        x = x_in.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        # ------------------------------------------------------------------
        # 1. CHANNEL/SPECTRAL QUERIES
        # ------------------------------------------------------------------
        q_source = F.adaptive_avg_pool2d(
            x,
            output_size=(self.query_pool_size, self.query_pool_size),
        )                                                   # [B, C, P, P]
        q_source = q_source.flatten(2)                      # [B, C, P^2]
        q = self.to_q(q_source)                             # [B, C, inner_dim]
        q = rearrange(
            q,
            'b c (heads d) -> b heads c d',
            heads=self.num_heads,
        )                                                   # [B, heads, C, d]

        # ------------------------------------------------------------------
        # 2. COARSE SPATIAL KEYS
        # ------------------------------------------------------------------
        coarse_h = max(1, math.ceil(h / self.spatial_reduction))
        coarse_w = max(1, math.ceil(w / self.spatial_reduction))

        k_map = self.to_k(x)                                # [B, inner_dim, H, W]
        k_map = F.adaptive_avg_pool2d(
            k_map,
            output_size=(coarse_h, coarse_w),
        )                                                   # [B, inner_dim, Hc, Wc]
        k = k_map.flatten(2).transpose(1, 2)                # [B, N, inner_dim]
        k = rearrange(
            k,
            'b n (heads d) -> b heads n d',
            heads=self.num_heads,
        )                                                   # [B, heads, N, d]

        # Cosine similarity is more stable across the different MST++ depths.
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        # Each channel query attends to every coarse spatial position.
        logits = torch.einsum('bhcd,bhnd->bhcn', q, k)      # [B, heads, C, N]
        logits = logits * self.rescale.unsqueeze(0)

        # Sigmoid permits multiple spectral/latent channels to be active at the
        # same spatial position. This is safer when replacing every block.
        attention_per_head = torch.sigmoid(logits)

        # Learned weighted average of the attention heads.
        head_weights = torch.softmax(self.head_logits, dim=0)
        head_weights = head_weights.view(1, self.num_heads, 1, 1)
        attention_low = (attention_per_head * head_weights).sum(dim=1)
        attention_low = attention_low.view(b, c, coarse_h, coarse_w)

        # ------------------------------------------------------------------
        # 3. SMOOTH UPSAMPLING + VARIANCE-GUIDED REFINEMENT
        # ------------------------------------------------------------------
        attention_base = F.interpolate(
            attention_low,
            size=(h, w),
            mode='bilinear',
            align_corners=False,
        )                                                   # [B, C, H, W]

        intensity = self.intensity_head(x)                  # [B, C, H, W]
        value = self.value_head(x)                          # [B, C, H, W]
        detail = self.detail_head(x)                        # [B, Cd, H, W]
        variance = self.variance_gate(x)                    # [B, 1, H, W]

        refine_input = torch.cat(
            [attention_base, intensity, variance, detail],
            dim=1,
        )
        delta_attention = self.attention_refiner(refine_input)

        # Refine in logit space so the final attention remains in [0, 1].
        eps = 1e-4
        base_logits = torch.logit(attention_base.clamp(eps, 1.0 - eps))
        attention_refined = torch.sigmoid(
            base_logits + variance * delta_attention
        )                                                   # [B, C, H, W]

        # ------------------------------------------------------------------
        # 4. LEARNED INTENSITY/VALUE MODULATION
        # ------------------------------------------------------------------
        # Variance controls the refinement of the attention map, not intensity.
        # The attention then decides where the learned value correction is used.
        modulated = intensity + attention_refined * value   # [B, C, H, W]

        out_content = self.proj(modulated)
        out_position = self.pos_emb(x)
        out = self.output_scale * (out_content + out_position)

        return out.permute(0, 2, 3, 1).contiguous()


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(
                dim * mult,
                dim * mult,
                3,
                1,
                1,
                bias=False,
                groups=dim * mult,
            ),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class MSAB(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim)),
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class MST(nn.Module):
    def __init__(self, in_dim=31, out_dim=31, dim=31, stage=2, num_blocks=[2, 4, 4]):
        super(MST, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage,
                    num_blocks=num_blocks[i],
                    dim_head=dim,
                    heads=dim_stage // dim,
                ),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = MSAB(
            dim=dim_stage,
            dim_head=dim,
            heads=dim_stage // dim,
            num_blocks=num_blocks[-1],
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(
                    dim_stage,
                    dim_stage // 2,
                    stride=2,
                    kernel_size=2,
                    padding=0,
                    output_padding=0,
                ),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2,
                    num_blocks=num_blocks[stage - 1 - i],
                    dim_head=dim,
                    heads=(dim_stage // 2) // dim,
                ),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        fea = self.embedding(x)

        fea_encoder = []
        for (MSAB_layer, FeaDownSample) in self.encoder_layers:
            fea = MSAB_layer(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        fea = self.bottleneck(fea)

        for i, (FeaUpSample, Fusion, LeWinBlock) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            fea = LeWinBlock(fea)

        out = self.mapping(fea) + x
        return out


class MST_Plus_Plus(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, n_feat=31, stage=3):
        super(MST_Plus_Plus, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(
            in_channels,
            n_feat,
            kernel_size=3,
            padding=(3 - 1) // 2,
            bias=False,
        )
        modules_body = [
            MST(dim=31, stage=2, num_blocks=[1, 1, 1])
            for _ in range(stage)
        ]
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(
            n_feat,
            out_channels,
            kernel_size=3,
            padding=(3 - 1) // 2,
            bias=False,
        )

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x = self.conv_in(x)
        h = self.body(x)
        h = self.conv_out(h)
        h += x
        return h[:, :, :h_inp, :w_inp]


