import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from wave_attn import SpatialWaveNetwork


###########################################
# HAAR Wavelet Transforms
###########################################

def create_haar_filters(in_channels, device=None):
    # Define HAAR wavelet filters for a 2x2 block.
    LL = torch.tensor([[ 0.5,  0.5],
                       [ 0.5,  0.5]], dtype=torch.float32, device=device)
    LH = torch.tensor([[-0.5, -0.5],
                       [ 0.5,  0.5]], dtype=torch.float32, device=device)
    HL = torch.tensor([[-0.5,  0.5],
                       [-0.5,  0.5]], dtype=torch.float32, device=device)
    HH = torch.tensor([[ 0.5, -0.5],
                       [-0.5,  0.5]], dtype=torch.float32, device=device)
    filters = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1) # shape: [4, 1, 2, 2]
    # Repeat filters for each input channel.
    # For in_channels = 3, result shape is [12, 1, 2, 2] with ordering:
    # [LL_R, LH_R, HL_R, HH_R, LL_G, LH_G, HL_G, HH_G, LL_B, LH_B, HL_B, HH_B]
    filters = filters.repeat(in_channels, 1, 1, 1)
    return filters


class HaarWaveletTransform(nn.Module):
    def __init__(self, in_channels=3, device="cpu"):
        super(HaarWaveletTransform, self).__init__()
        filters = create_haar_filters(in_channels, device)
        self.register_buffer("filters", filters)
        self.groups = in_channels

    def forward(self, x):
        # x: [B, C, H, W]
        # Output: [B, 4 * in_channels, H/2, W/2]
        return F.conv2d(x, self.filters, stride=2, groups=self.groups)


class InverseHaarWaveletTransform(nn.Module):
    def __init__(self, in_channels=3, device="cpu"):
        super(InverseHaarWaveletTransform, self).__init__()
        filters = create_haar_filters(in_channels, device)
        self.register_buffer("filters", filters)
        self.groups = in_channels

    def forward(self, x):
        # x: [B, 4 * in_channels, H, W]
        B, ch4, H, W = x.shape
        C = ch4 // 4
        # Output: [B, in_channels, H*2, W*2]
        return F.conv_transpose2d(x, self.filters, stride=2, groups=C)

###########################################
# Pixel Upsampling Module (PUM)
###########################################

class PUM(nn.Module):
    """
    Multi-stage upsampling module.
    Each stage applies:
      - Convolution for channel expansion,
      - PixelShuffle for spatial upscaling,
      - GELU activation,
      - SpatialWaveNetwork attention for refinement,
      - Implicit Detail Enhancement to mitigate blocking.
    """
    def __init__(self, in_channels, out_channels, upscale_factor, num_heads=4, mlp_ratio=4.0):
        super(PUM, self).__init__()
        layers = []
        num_upsamples = int(math.log2(upscale_factor))
        for _ in range(num_upsamples):
            layers.extend([
                nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
                SpatialWaveNetwork(in_channels, num_heads=num_heads, mlp_ratio=mlp_ratio),
                ImplicitDetailEnhancer(in_channels, mlp_ratio=mlp_ratio)
            ])
        self.layers = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.layers(x)
        return self.final_conv(x)

###########################################
# Multi-Scale Feature Extraction Module (MSFEM)
###########################################

class MSFEM(nn.Module):
    """
    Applies multiple convolutions with receptive fields 3x, 5x, 7x and a dilated 3x,
    then fuses them.
    """
    def __init__(self, in_channels, out_channels):
        super(MSFEM, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.activation = nn.GELU()
        self.fusion = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        feat3 = self.activation(self.conv3(x))
        feat5 = self.activation(self.conv5(x))
        feat7 = self.activation(self.conv7(x))
        feat_dilated = self.activation(self.dilated_conv(x))
        concatenated = torch.cat([feat3, feat5, feat7, feat_dilated], dim=1)
        return self.fusion(concatenated)

###########################################
# Implicit Detail Enhancer
###########################################

class ImplicitDetailEnhancer(nn.Module):
    """
    Patch-based detail enhancement with partial NeuralSlice refinement.
    """
    def __init__(self, dim, patch_size=8, stride=4, mlp_ratio=4.0):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.dim = dim
        hidden_dim = int(dim * mlp_ratio)

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.ns = NeuralSlice(dim, kernel_size=5, mlp_ratio=mlp_ratio)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x, d=None):
        B, C, H, W = x.shape
        p = self.patch_size
        s = self.stride

        # Replicate padding so patches fit evenly
        pad_h = (p - (H % p)) % p
        pad_w = (p - (W % p)) % p
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        Hp, Wp = x_padded.shape[2], x_padded.shape[3]

        # Extract patches [B, C, Hp, Wp] -> [B, C, #patches, p, p]
        patches = x_padded.unfold(2, p, s).unfold(3, p, s)  # B, C, H_num, W_num, p, p
        patches = patches.contiguous().view(B, C, -1, p, p)
        # Reorder to [B, #patches, C, p, p]
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        orig_shape = patches.shape  # [B, #patches, C, p, p]

        # Flatten to pass through attention + NeuralSlice
        patches = patches.view(-1, C, p, p)  # [B * #patches, C, p, p]
        attn_weights = self.spatial_attn(patches)
        patches = patches * attn_weights
        refined_patches = self.ns(patches)

        # Reshape for folding
        refined_patches = refined_patches.view(*orig_shape)  # [B, #patches, C, p, p]
        refined_patches = refined_patches.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, #patches, p, p]
        refined_patches = refined_patches.view(B, C * p * p, -1)

        # Fold back
        output = F.fold(
            refined_patches,
            output_size=(Hp, Wp),
            kernel_size=(p, p),
            stride=s
        )
        # Create divisor map
        ones = torch.ones_like(refined_patches)
        divisor = F.fold(
            ones,
            output_size=(Hp, Wp),
            kernel_size=(p, p),
            stride=s
        )
        output = output / (divisor + 1e-8)
        output = output[:, :, :H, :W]

        # Residual
        return self.alpha * output + x


class DegradationEstimator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, embed_dim)
        )
    def forward(self, x):
        return self.model(x)


class NeuralSlice(nn.Module):
    """Multi‑kernel convolutional MLP replacement."""
    def __init__(self, dim: int, mlp_ratio: float = 4.0, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (5, 7)
        hidden = int(dim * mlp_ratio)
        small_k = kernel_size - 2
        self.conv_large = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2)
        self.conv_small = nn.Conv2d(dim, dim, small_k,     padding=small_k // 2)
        self.fuse       = nn.Conv2d(dim, dim, 3, padding=1)
        self.expand  = nn.Conv2d(dim, hidden, 1)
        self.mix     = nn.Conv2d(hidden, hidden, 3, padding=1, groups=max(1, hidden // 4))
        self.project = nn.Conv2d(hidden, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        g     = torch.sigmoid(self.fuse(x))
        fused = g * self.conv_large(x) + (1 - g) * self.conv_small(x)
        y     = self.act(fused)
        y     = self.act(self.expand(y))
        y     = self.act(self.mix(y))
        return self.project(y)


class FiLMLayer(nn.Module):
    """Per‑channel (or spatial) FiLM modulation."""
    def __init__(self, channels: int, embed_dim: int, spatial: bool = False):
        super().__init__()
        self.spatial = spatial
        if spatial:
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, channels * 8), nn.GELU(),
                nn.Linear(channels * 8, channels * 32)
            )
        else:
            self.proj = nn.Linear(embed_dim, channels * 2)

    def forward(self, feat: torch.Tensor, d: torch.Tensor):
        B, C, H, W = feat.shape
        gb = self.proj(d)
        if self.spatial:
            gb = gb.view(B, 2 * C, 4, 4)
            gb = F.interpolate(gb, size=(H, W), mode="bilinear", align_corners=False)
        else:
            gb = gb.view(B, 2 * C, 1, 1)
        gamma, beta = torch.chunk(gb, 2, 1)
        return gamma * feat + beta


class DecoupledBlock(nn.Module):
    def __init__(self, c: int, num_heads: int, mlp_ratio: float, embed_dim: int, spatial_film: bool = False):
        super().__init__()
        self.norm  = nn.GroupNorm(max(1, num_heads), c)
        self.attn  = SpatialWaveNetwork(c, num_heads, mlp_ratio)
        self.film  = FiLMLayer(c, embed_dim, spatial_film)
        self.mlp   = NeuralSlice(c, mlp_ratio)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, d):
        h = self.attn(self.norm(x))
        h = self.film(h, d)
        h = self.mlp(h)
        return x + self.scale * h

###########################################
# Processors using Decoupled Blocks
###########################################

class LowFrequencyProcessor(nn.Module):
    """
    Processes the LL (low-frequency) subband.
    """
    def __init__(self, channels, num_blocks=4, num_heads=4, mlp_ratio=4.0, embed_dim=128):
        super(LowFrequencyProcessor, self).__init__()
        self.msfem = MSFEM(channels, channels)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(DecoupledBlock(channels, num_heads=num_heads, mlp_ratio=mlp_ratio, embed_dim=embed_dim, spatial_film=True))

    def forward(self, x, d):
        x = self.msfem(x)
        for block in self.blocks:
            x = block(x, d)
        return x


class HighFrequencyProcessorIndependent(nn.Module):
    """
    Processes each high-frequency subband independently for an initial set of blocks.
    """
    def __init__(self, channels, num_blocks=4, num_heads=4, mlp_ratio=4.0, embed_dim=128):
        super(HighFrequencyProcessorIndependent, self).__init__()
        self.msfem = MSFEM(channels, channels)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(DecoupledBlock(channels, num_heads=num_heads, mlp_ratio=mlp_ratio, embed_dim=embed_dim, spatial_film=True))
            if (i + 1) % 2 == 0:
                self.blocks.append(ImplicitDetailEnhancer(channels, mlp_ratio=mlp_ratio))

    def forward(self, x, d):
        x = self.msfem(x)
        for block in self.blocks:
            x = block(x, d)
        return x


class HighFrequencyProcessorFused(nn.Module):
    """
    Processes fused high-frequency features after intermediate fusion.
    """
    def __init__(self, channels, num_blocks=4, num_heads=4, mlp_ratio=4.0, embed_dim=128):
        super(HighFrequencyProcessorFused, self).__init__()
        self.msfem = MSFEM(channels, channels)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(DecoupledBlock(channels, num_heads=num_heads, mlp_ratio=mlp_ratio, embed_dim=embed_dim, spatial_film=True))
            if (i + 1) % 2 == 0:
                self.blocks.append(ImplicitDetailEnhancer(channels, mlp_ratio=mlp_ratio))

    def forward(self, x, d):
        x = self.msfem(x)
        for block in self.blocks:
            x = block(x, d)
        return x

###########################################
# Fusion Modules
###########################################

class IntraHighFrequencyFusion(nn.Module):
    """
    Fuses the three high-frequency subbands (LH, HL, HH) with learnable weights.
    """
    def __init__(self, channels, num_heads=4, mlp_ratio=4.0, embed_dim=128):
        super(IntraHighFrequencyFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(3))
        self.conv = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.act = nn.GELU()
        self.attn = DecoupledBlock(channels, num_heads=num_heads, mlp_ratio=mlp_ratio, embed_dim=embed_dim, spatial_film=True)
    
    def forward(self, hf_list, d):
        # hf_list: list of three tensors, each [B, C, H, W]
        stacked = torch.cat(hf_list, dim=1)  # [B, 3C, H, W]
        B, threeC, H, W = stacked.shape
        C = threeC // 3
        weighted = []
        for i in range(3):
            sub = stacked[:, i * C:(i + 1) * C, :, :]
            weighted.append(self.weights[i] * sub)
        fused = torch.cat(weighted, dim=1)
        fused = self.act(self.conv(fused))
        fused = self.attn(fused, d)
        return fused


class GlobalFusion(nn.Module):
    """
    Fuses low-frequency (LL) features with the fused high-frequency features.
    """
    def __init__(self, channels, num_heads=4, mlp_ratio=4.0, embed_dim=128):
        super(GlobalFusion, self).__init__()
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.act = nn.GELU()
        self.attn = DecoupledBlock(channels, num_heads=num_heads, mlp_ratio=mlp_ratio, embed_dim=embed_dim, spatial_film=True)
    
    def forward(self, ll_feat, hf_feat, d):
        fused = torch.cat([ll_feat, hf_feat], dim=1)
        fused = self.act(self.conv(fused))
        fused = self.attn(fused, d)
        return fused

###########################################
# Main Network: WaveFusionNet
###########################################

class WaveFusionNet(nn.Module):
    """
    WaveFusionNet:
      - Decomposes the input via HAAR wavelet transform.
      - Reassembles subbands for a 3-channel image.
      - Processes the LL subband independently.
      - Processes each high-frequency subband (LH, HL, HH) independently for an initial stage,
        then fuses them via an intermediate fusion layer, and further processes the fused features.
      - Globally fuses the LL and high-frequency features, applies global refinement,
        and then uses a robust multi-stage upsampling module (PUM).
      - Finally, reconstructs the output via the inverse HAAR transform with a residual connection.
    """
    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        embed_dim=128,
        num_ll_blocks=4,
        num_hf_indep_blocks=2,
        num_hf_fused_blocks=2,
        num_global_blocks=3,
        num_heads=4,
        mlp_ratio=2.0,
        upscale_factor=1,
        device="cpu"
    ):
        super(WaveFusionNet, self).__init__()
        self.upscale_factor = upscale_factor

        self.d_est = DegradationEstimator(embed_dim)

        # Wavelet transforms
        self.haar = HaarWaveletTransform(in_channels, device=device)
        self.ihaar = InverseHaarWaveletTransform(in_channels, device=device)

        # Projection for each subband: since the HAAR transform yields 12 channels (for 3-channel input)
        # in the order [LL_R, LH_R, HL_R, HH_R, LL_G, LH_G, HL_G, HH_G, LL_B, LH_B, HL_B, HH_B],
        # we reassemble by frequency:
        self.subband_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Low-frequency processor for the LL subband
        self.ll_processor = LowFrequencyProcessor(base_channels, num_blocks=num_ll_blocks,
                                                  num_heads=num_heads, mlp_ratio=mlp_ratio, embed_dim=embed_dim)
        # High-frequency processing:
        # Process each high-frequency subband (LH, HL, HH) independently first.
        self.hf_indep = HighFrequencyProcessorIndependent(base_channels, num_blocks=num_hf_indep_blocks,
                                                          num_heads=num_heads, mlp_ratio=mlp_ratio, embed_dim=embed_dim)
        # Intermediate fusion for high-frequency features
        self.intra_hf_fusion = IntraHighFrequencyFusion(base_channels, num_heads=num_heads, mlp_ratio=mlp_ratio, embed_dim=embed_dim)
        # Further processing of fused high-frequency features
        self.hf_fused = HighFrequencyProcessorFused(base_channels, num_blocks=num_hf_fused_blocks,
                                                    num_heads=num_heads, mlp_ratio=mlp_ratio, embed_dim=embed_dim)

        # Global fusion: fuse LL features with high-frequency features
        self.global_fusion = GlobalFusion(base_channels, num_heads=num_heads, mlp_ratio=mlp_ratio, embed_dim=embed_dim)
        # Global refinement (stack of DecoupledBlock blocks)
        self.global_refine = nn.ModuleList()
        for _ in range(num_global_blocks):
            self.global_refine.append(DecoupledBlock(base_channels, num_heads=num_heads, mlp_ratio=mlp_ratio, embed_dim=embed_dim, spatial_film=True))

        # Upsampling
        if upscale_factor > 1:
            self.upsample = PUM(base_channels, base_channels, upscale_factor=upscale_factor,
                                num_heads=num_heads, mlp_ratio=mlp_ratio)
            self.global_residual = nn.Upsample(scale_factor=upscale_factor, mode="bicubic", align_corners=False)
        else:
            self.upsample = nn.Sequential(
                ImplicitDetailEnhancer(base_channels, mlp_ratio=mlp_ratio),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
            )
            self.global_residual = nn.Identity()

        # Map features to wavelet coefficient space (expects 4 * in_channels channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels * 4, kernel_size=3, padding=1)
        self.act = nn.Tanh()
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        residual = x
        d = self.d_est(x)
        # HAAR transform: for 3-channel input, output shape: [B, 12, H/2, W/2]
        wave_coeffs = self.haar(x)

        # Split into 12 channels:
        (ll_r, lh_r, hl_r, hh_r,
         ll_g, lh_g, hl_g, hh_g,
         ll_b, lh_b, hl_b, hh_b) = wave_coeffs.chunk(12, dim=1)
        # Reassemble by frequency:
        ll_band = torch.stack([ll_r, ll_g, ll_b], dim=2).flatten(1,2)
        lh_band = torch.stack([lh_r, lh_g, lh_b], dim=2).flatten(1,2)
        hl_band = torch.stack([hl_r, hl_g, hl_b], dim=2).flatten(1,2)
        hh_band = torch.stack([hh_r, hh_g, hh_b], dim=2).flatten(1,2)

        # Process LL subband:
        ll_proj = self.subband_proj(ll_band) # [B, base_channels, H/2, W/2]
        ll_feat = self.ll_processor(ll_proj, d)
        # Process high-frequency subbands independently:
        lh_proj = self.subband_proj(lh_band)
        hl_proj = self.subband_proj(hl_band)
        hh_proj = self.subband_proj(hh_band)
        lh_feat_ind = self.hf_indep(lh_proj, d)
        hl_feat_ind = self.hf_indep(hl_proj, d)
        hh_feat_ind = self.hf_indep(hh_proj, d)

        # Intermediate fusion of high-frequency features:
        hf_fused_initial = self.intra_hf_fusion([lh_feat_ind, hl_feat_ind, hh_feat_ind], d)
        # Further process fused high-frequency features:
        hf_feat = self.hf_fused(hf_fused_initial, d)
        # Global fusion of LL and high-frequency features:
        fused = self.global_fusion(ll_feat, hf_feat, d)

        # Global refinement:
        refined = fused
        for refine in self.global_refine:
            refined = refine(refined, d)
        # Upsample:
        upsampled = self.upsample(refined)

        # Map to wavelet coefficient space:
        wave_out = self.out_conv(upsampled)
        # Inverse HAAR transform:
        out = self.ihaar(wave_out)
        # Global residual
        out = self.alpha * out + self.global_residual(residual)
        return self.act(out), d

###########################################
# Testing
###########################################

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WaveFusionNet(
        in_channels=3,
        base_channels=128,
        num_ll_blocks=8,
        num_hf_indep_blocks=4,
        num_hf_fused_blocks=4,
        num_global_blocks=6,
        num_heads=8,
        mlp_ratio=4.0,
        upscale_factor=8,
        device=device
    )
    model.to(device)
    dummy_input = torch.randn(1, 3, 64, 64, device=device)
    output = model(dummy_input)
    print("Output shape:", output.shape)
