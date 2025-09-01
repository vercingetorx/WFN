import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveEmission(nn.Module):
    def __init__(self, in_channels, wave_channels, expansion=4):
        super().__init__()
        hidden_dim = int(in_channels * expansion)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, wave_channels, kernel_size=1)
        )

        # Learnable edge detection hybrid
        self.edge_conv = nn.Conv2d(1, wave_channels, kernel_size=3, padding=1)
        nn.init.dirac_(self.edge_conv.weight[:,:1])  # Preserve Sobel pattern
        self.edge_conv.weight.requires_grad_(True)  # Allow fine-tuning
        
        # Coordinate embedding with scale control
        self.coord_scale = nn.Parameter(torch.tensor(0.1))
        self.coord_conv = nn.Conv2d(2, wave_channels, kernel_size=1)

    def get_coord_map(self, x):
        b, _, h, w = x.shape
        x_coord = torch.linspace(-1, 1, w, device=x.device).view(1,1,1,w)
        y_coord = torch.linspace(-1, 1, h, device=x.device).view(1,1,h,1)
        return torch.cat([
            x_coord.expand(b,-1,h,w),
            y_coord.expand(b,-1,h,w)
        ], dim=1) * self.coord_scale

    def forward(self, x):
        # Edge-modulated wave generation
        edges = self.edge_conv(x.mean(dim=1, keepdim=True))
        waves = self.mlp(x) * torch.sigmoid(edges)
        
        # Add coordinate information
        return waves + self.coord_conv(self.get_coord_map(x))


class WaveDiffusion(nn.Module):
    def __init__(self, channels, num_heads, num_steps=4):
        super().__init__()
        self.num_steps = num_steps
        self.num_heads = num_heads
        
        # Physics-informed initialization
        self.laplacian = nn.Conv2d(channels, channels, kernel_size=3, 
                                 padding=1, groups=channels, bias=False)
        nn.init.dirac_(self.laplacian.weight)
        self.laplacian.weight.data[:,:,1,1] = -4  # Center pixel
        
        # Adaptive interaction network with group convolutions
        hidden_dim = max(4, channels//4)
        # Ensure hidden_dim divisible by num_heads
        if hidden_dim % num_heads != 0:
            hidden_dim += num_heads - (hidden_dim % num_heads)
        self.interaction = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, groups=num_heads),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, groups=num_heads)
        )
        
        # Normalization scheme
        self.norm = nn.GroupNorm(min(4, channels), channels)
        
        # Learnable coefficients with constraint
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.register_constraint()

    def register_constraint(self):
        for p in [self.alpha, self.beta]:
            p.register_hook(lambda grad: torch.clamp(grad, -0.1, 0.1))

    def forward(self, w):
        identity = w
        for _ in range(self.num_steps):
            w = self.norm(w)
            diffusion = self.laplacian(w)
            interaction = self.interaction(w)
            w = w + torch.tanh(self.alpha)*diffusion + torch.sigmoid(self.beta)*(w * interaction)
        return w + identity


class InterferenceAggregation(nn.Module):
    def __init__(self, channels, num_heads, kernel_size=5):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.gamma = nn.Parameter(torch.tensor(1.0))
        
        # Distance kernel with per-head patterns
        self.register_buffer("dist_kernel", self.create_distance_kernel(kernel_size, channels, num_heads))
        self.adapt_conv = nn.Conv2d(channels, num_heads, kernel_size=kernel_size, padding=kernel_size//2)

        # Sharpening parameter per head
        self.sharpening = nn.Parameter(torch.ones(num_heads))

    def create_distance_kernel(self, size, channels, num_heads):
        r = size//2
        kernel = torch.zeros(num_heads, 1, size, size)
        for i in range(size):
            for j in range(size):
                dist = math.sqrt((i - r)**2 + (j - r)**2)
                kernel[:, 0, i, j] = dist
        # Repeat kernel for each channel in head
        head_dim = channels // num_heads
        return kernel.repeat_interleave(head_dim, dim=0)

    def forward(self, w):
        # Channel-wise decay factors
        decay = torch.exp(-self.gamma * self.dist_kernel)
        
        # Depthwise convolution for interference pattern
        aggregated = F.conv2d(w.abs(), decay, 
                            padding=self.dist_kernel.shape[-1]//2,
                            groups=self.channels)
        
        # Per-head sharpening and sigmoid
        return torch.sigmoid(self.adapt_conv(aggregated) * self.sharpening.view(1, -1, 1, 1))


class SpatialWaveNetwork(nn.Module):
    def __init__(self, channels, wave_ratio=4, mlp_ratio=4.0, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        wave_channels = max(8, channels // wave_ratio)
        # Ensure divisible by num_heads
        wave_channels = (wave_channels // num_heads) * num_heads
        wave_channels = max(wave_channels, num_heads)  # At least 1 per head
        
        self.wave_emission = WaveEmission(channels, wave_channels, mlp_ratio)
        self.wave_diffusion = WaveDiffusion(wave_channels, num_heads)
        self.aggregation = InterferenceAggregation(wave_channels, num_heads)

    def forward(self, x):
        # Wave processing
        waves = self.wave_emission(x)
        waves = self.wave_diffusion(waves)
        attn = self.aggregation(waves)
        
        # Split channels into heads and modulate
        B, C, H, W = x.shape
        x_reshaped = x.view(B, self.num_heads, -1, H, W)
        attn = torch.sigmoid(attn).unsqueeze(2)  # Add channel dim
        modulated = x_reshaped * (1 + attn)
        
        # Combine heads and project
        return modulated.view(B, C, H, W)
