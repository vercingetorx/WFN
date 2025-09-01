import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wave_attn import SpatialWaveNetwork

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def spectral_conv(*args, **kwargs):
    # return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))
    return nn.Conv2d(*args, **kwargs)

def spectral_linear(*args, **kwargs):
    # return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))
    return nn.Linear(*args, **kwargs)

# -----------------------------------------------------------------------------
# Anti-aliased downsample (BlurPool)
# -----------------------------------------------------------------------------

class BlurPool2d(nn.Module):
    """
    Depthwise low-pass then stride-2 (or 1) downsample.
    Kernel is separable [1,2,1] x [1,2,1].
    """
    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        if stride == 1:
            self.op = nn.Identity()
        else:
            k = torch.tensor([1., 2., 1.], dtype=torch.float32)
            kernel = (k[:, None] @ k[None, :]) / k.sum()**2  # 3x3 normalized
            kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
            self.register_buffer("kernel", kernel)  # [C,1,3,3]
            self.groups = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x
        x = F.conv2d(x, self.kernel, stride=1, padding=1, groups=self.groups)
        return x[:, :, ::self.stride, ::self.stride]

# -----------------------------------------------------------------------------
# Frequency (Haar) head
# -----------------------------------------------------------------------------

def _haar_filters(in_c: int, device=None):
    LL = torch.tensor([[ 0.5,  0.5],
                       [ 0.5,  0.5]], dtype=torch.float32, device=device)
    LH = torch.tensor([[-0.5, -0.5],
                       [ 0.5,  0.5]], dtype=torch.float32, device=device)
    HL = torch.tensor([[-0.5,  0.5],
                       [-0.5,  0.5]], dtype=torch.float32, device=device)
    HH = torch.tensor([[ 0.5, -0.5],
                       [-0.5,  0.5]], dtype=torch.float32, device=device)
    filt = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)  # [4,1,2,2]
    return filt.repeat(in_c, 1, 1, 1)  # [4*C,1,2,2]

class HaarHFHead(nn.Module):
    """
    Simple PatchGAN on concatenated high-frequency subbands (LH, HL, HH).
    Input: RGB image
    """
    def __init__(self, in_channels: int = 3, base_c: int = 32):
        super().__init__()
        self.register_buffer("filters", _haar_filters(in_channels))
        self.groups = in_channels
        hf_c = in_channels * 3  # LH,HL,HH
        self.head = nn.Sequential(
            spectral_conv(hf_c, base_c, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv(base_c, base_c * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv(base_c * 2, base_c * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv(base_c * 4, 1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # wavelet @ stride=2
        w = F.conv2d(x, self.filters, stride=2, groups=self.groups)  # [B,4C,H/2,W/2]
        C = x.shape[1]
        hf = torch.cat([w[:, C:2*C], w[:, 2*C:3*C], w[:, 3*C:4*C]], dim=1)  # LH,HL,HH
        patch_map = self.head(hf)  # [B,1,h,w]
        return patch_map

# -----------------------------------------------------------------------------
# Helper blocks
# -----------------------------------------------------------------------------

class DWConvMix(nn.Module):
    """Cheap spatial mixing: DW 3x3 + 1x1, SN + LeakyReLU."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            spectral_conv(channels, channels, 3, 1, 1, groups=channels),
            spectral_conv(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class SpectralNormResidualBlock(nn.Module):
    """Residual block with spectral norm and LeakyReLU."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = spectral_conv(in_channels, out_channels, 3, stride, 1)
        self.conv2 = spectral_conv(out_channels, out_channels, 3, 1, 1)
        self.shortcut = (
            spectral_conv(in_channels, out_channels, 1, stride)
            if stride != 1 or in_channels != out_channels else nn.Identity()
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        h = h + self.shortcut(x)
        return self.act(h)

class CondProj(nn.Module):
    """Projection-based conditional term (Miyato et al.)."""
    def __init__(self, deg_dim: int, feat_dim: int):
        super().__init__()
        self.embed = spectral_linear(deg_dim, feat_dim)

    def forward(self, feat: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        proj = (self.embed(d) * feat).sum(dim=1, keepdim=True)
        return proj

# -----------------------------------------------------------------------------
# Scale branch
# -----------------------------------------------------------------------------

class ScaleBranch(nn.Module):
    def __init__(self, in_c: int, base_c: int, num_heads: int, downsample: int, use_wave_start: int = 2):
        """
        Two-stage stem: blurpool (if needed) + conv. Body ups channels 3 times.
        `use_wave_start`: depth index where we switch to SpatialWaveNetwork.
        """
        super().__init__()
        self.blur = BlurPool2d(in_c, stride=downsample)
        self.head = nn.Sequential(
            spectral_conv(in_c, base_c, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        blocks = []
        channels = base_c
        for depth in range(3):  # C -> 2C -> 4C -> 8C
            next_c = channels * 2
            mix_block = (
                SpatialWaveNetwork(channels, max(1, num_heads // (2 ** depth)))
                if depth >= use_wave_start else
                DWConvMix(channels)
            )
            blocks.extend([
                SpectralNormResidualBlock(channels, channels),
                mix_block,
                SpectralNormResidualBlock(channels, next_c, stride=2)
            ])
            channels = next_c
        blocks.append(SpatialWaveNetwork(channels, num_heads=max(1, num_heads // 4)))
        self.body = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.out_dim = channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.blur(x)
        h = self.head(h)
        h = self.body(h)
        patch_feat = h                          # (B,C,H',W')
        vec = self.flatten(self.pool(h))        # (B,C)
        return vec, patch_feat

# -----------------------------------------------------------------------------
# Full discriminator
# -----------------------------------------------------------------------------

class WaveFusionDiscriminator(nn.Module):
    """
    Two-scale, patch-aware, conditional discriminator with an optional
    frequency (Haar HF) patch head.
    """
    def __init__(self,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 num_heads: int = 8,
                 deg_dim: int = 128,
                 lambda_patch: float = 0.5,
                 lambda_freq: float = 0.0,   # good to start low and ramp up
                 freq_base_c: int = 32):
        super().__init__()
        self.lambda_patch = lambda_patch
        self.lambda_freq  = lambda_freq

        # two scales: 1× and ½×
        self.scale1 = ScaleBranch(in_channels, base_channels, num_heads, downsample=1)
        self.scale2 = ScaleBranch(in_channels, base_channels, num_heads, downsample=2)

        feat_dim = self.scale1.out_dim  # highest channel dim
        self.mix_proj = spectral_linear(feat_dim, feat_dim)
        self.cond_proj = CondProj(deg_dim, feat_dim)
        self.fc_final = spectral_linear(feat_dim, 1)

        # Patch head on scale1 feature map
        self.patch_head = spectral_conv(feat_dim, 1, 3, 1, 1)

        # Optional frequency head
        self.freq_head = HaarHFHead(in_channels, base_c=freq_base_c)

        # Normalize + detach d_vec in D
        self.d_norm = nn.LayerNorm(deg_dim)

    def forward(self, x: torch.Tensor, d_vec: torch.Tensor, return_maps: bool = False):
        # Prepare conditioning (detach so D doesn't backprop into deg-estimator)
        d = self.d_norm(d_vec)

        # global + patch features from two scales
        s1_vec, s1_map = self.scale1(x)
        s2_vec, _      = self.scale2(x)

        # simple proj-add fusion + 2-way norm-based soft weighting
        s2_vec = s2_vec + self.mix_proj(s1_vec)
        w = torch.softmax(torch.stack([s1_vec.norm(dim=1), s2_vec.norm(dim=1)], dim=1), dim=1)
        fused = w[:, 0:1] * s1_vec + w[:, 1:2] * s2_vec

        # global score with conditional projection
        global_score = self.fc_final(fused) + self.cond_proj(fused, d)  # [B,1]

        # patch score (scale-1 map)
        patch_map = self.patch_head(s1_map)               # [B,1,h,w]
        patch_score = patch_map.mean([2, 3], keepdim=True)

        # frequency head
        freq_map = self.freq_head(x)                  # [B,1,hf_w,hf_h]
        freq_score = freq_map.mean([2, 3], keepdim=True)

        total = global_score + self.lambda_patch * patch_score + self.lambda_freq * freq_score

        if return_maps:
            return {
                "logit": total,          # [B,1]
                "global": global_score,  # [B,1]
                "patch_map": patch_map,  # [B,1,h,w]
                "freq_map": freq_map     # [B,1,*,*] or None
            }
        return total
