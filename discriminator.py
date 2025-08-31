import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wave_attn import SpatialWaveNetwork

# -----------------------------------------------------------------------------
# Helper blocks
# -----------------------------------------------------------------------------

class DWConvMix(nn.Module):
    """Cheap spatial mixing replacement for early Wave blocks."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)),
            nn.utils.spectral_norm(nn.Conv2d(channels, channels, 1)),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class SpectralNormResidualBlock(nn.Module):
    """Residual block with spectral norm and GELU."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        self.shortcut = (
            nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 1, stride))
            if stride != 1 or in_channels != out_channels else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        h = h + self.shortcut(x)
        return self.act(h)


class CondProj(nn.Module):
    """Projection‑based conditional discriminator term (Miyato et al.)."""
    def __init__(self, deg_dim: int, feat_dim: int):
        super().__init__()
        self.embed = nn.utils.spectral_norm(nn.Linear(deg_dim, feat_dim))

    def forward(self, feat: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # <f, W·d>
        proj = (self.embed(d) * feat).sum(dim=1, keepdim=True)
        return proj


# -----------------------------------------------------------------------------
# Scale branch that returns both global vector and feature map for PatchGAN
# -----------------------------------------------------------------------------

class ScaleBranch(nn.Module):
    def __init__(self, in_c: int, base_c: int, num_heads: int, downsample: int, use_wave_start: int = 2):
        """Create discriminator scale.

        Args:
            use_wave_start: which residual depth begins to use SpatialWaveNetwork
        """
        super().__init__()
        stride = downsample
        self.head = nn.Sequential(
            nn.AvgPool2d(stride) if stride > 1 else nn.Identity(),
            nn.utils.spectral_norm(nn.Conv2d(in_c, base_c, 3, 1, 1)),
            nn.GELU(),
        )
        blocks = []
        channels = base_c
        for depth in range(3):  # produce C,2C,4C,8C
            next_c = channels * 2
            # choose mixing block
            mix_block = SpatialWaveNetwork(channels, num_heads=num_heads // (2 ** depth)) if depth >= use_wave_start else DWConvMix(channels)
            blocks.extend([
                SpectralNormResidualBlock(channels, channels),
                mix_block,
                SpectralNormResidualBlock(channels, next_c, stride=2)
            ])
            channels = next_c
        # one final wave block at highest res channels
        blocks.append(SpatialWaveNetwork(channels, num_heads=max(1, num_heads // 4)))
        self.body = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.out_dim = channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.head(x)
        h = self.body(h)
        patch_feat = h                          # (B,C,H',W') for PatchGAN
        vec = self.flatten(self.pool(h))        # (B,C)
        return vec, patch_feat


# -----------------------------------------------------------------------------
# Full discriminator
# -----------------------------------------------------------------------------

class WaveFusionDiscriminator(nn.Module):
    """Multi‑scale, patch‑aware, conditional discriminator."""
    def __init__(self,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 num_heads: int = 8,
                 deg_dim: int = 128,
                 lambda_patch: float = 0.5):
        super().__init__()
        self.lambda_patch = lambda_patch

        # two scales: 1× and ½×
        self.scale1 = ScaleBranch(in_channels, base_channels, num_heads, downsample=1)
        self.scale2 = ScaleBranch(in_channels, base_channels, num_heads, downsample=2)

        feat_dim = self.scale1.out_dim  # dim*8
        self.mix_proj = nn.utils.spectral_norm(nn.Linear(feat_dim, feat_dim))
        self.cond_proj = CondProj(deg_dim, feat_dim)
        self.fc_final = nn.utils.spectral_norm(nn.Linear(feat_dim, 1))

        # Patch head on scale1 feature map
        self.patch_head = nn.utils.spectral_norm(nn.Conv2d(feat_dim, 1, 3, 1, 1))

    def forward(self, x: torch.Tensor, d_vec: torch.Tensor) -> torch.Tensor:
        # global + patch features
        s1_vec, s1_map = self.scale1(x)
        s2_vec, _      = self.scale2(x)

        # simple proj‑add fusion
        s2_vec = s2_vec + self.mix_proj(s1_vec)

        # attention‑like weighting (2‑way softmax)
        w = torch.softmax(torch.stack([s1_vec.norm(dim=1), s2_vec.norm(dim=1)], dim=1), dim=1)
        fused = w[:, 0:1] * s1_vec + w[:, 1:2] * s2_vec

        # global score with conditional projection
        global_score = self.fc_final(fused) + self.cond_proj(fused, d_vec)

        # patch score
        patch_score = self.patch_head(s1_map).mean([2, 3], keepdim=True)

        return global_score + self.lambda_patch * patch_score


# -----------------------------------------------------------------------------
# Smoke‑test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    d = WaveFusionDiscriminator().cuda()
    img = torch.randn(2, 3, 128, 128).cuda()
    deg = torch.randn(2, 128).cuda()
    out = d(img, deg)
    print(out.shape)  # -> (B,1,1,1)
