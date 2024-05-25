import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.increase_channels = None
        if in_channels != out_channels:
            self.increase_channels = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

        self.block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(32, out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

    def forward(self, x):
        if self.increase_channels is not None:
            return self.increase_channels(x) + self.block(x)
        else:
            return x + self.block(x)

class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        x = self.conv(x)
        return x

class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        
        self.theta = nn.Conv2d(in_channels, in_channels, 1, 1)
        self.phi = nn.Conv2d(in_channels, in_channels, 1, 1)
        self.g = nn.Conv2d(in_channels, in_channels, 1, 1)
        
        self.out = nn.Conv2d(in_channels, in_channels, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        phi = self.phi(x)
        theta = self.theta(x)
        g = self.g(x)
        
        phi = rearrange(phi, "b c h w -> b c (h w)")
        theta = rearrange(theta, "b c h w -> b (h w) c")
        g = rearrange(g, "b c h w -> b c (h w)")

        scaling_factor = 1 / C
        Y = F.softmax(phi @ theta * scaling_factor, dim=2) @ g
        Y = rearrange(Y, "b c (h w) -> b c h w", h=H, w=W)

        A = self.out(Y)
        x = x + A

        return x

