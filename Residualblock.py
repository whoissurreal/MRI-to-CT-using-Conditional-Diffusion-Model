import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Skip connection with 1x1 conv if channels change
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        
    def forward(self, x):
        identity = self.skip(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x)
        
        return x + identity
