import torch
import torch.nn as nn
import torch.nn.functional as F
from CBAM import CBAM
from Utils import SinusoidalPositionEmbeddings
from Residualblock import ResidualBlock

def nonlinearity(x):
    # swish activation
    return x * torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                      in_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                      in_channels,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_dim=256, base_channels=96):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # Initial conv with residual connection using ResidualBlock
        self.init_block = ResidualBlock(in_channels * 2, base_channels)
        
        # Encoder path with residual blocks
        self.down1 = nn.ModuleList([
            ResidualBlock(base_channels, base_channels),
            CBAM(base_channels),
            ResidualBlock(base_channels, base_channels)
        ])
        
        self.down2 = nn.ModuleList([
            ResidualBlock(base_channels, base_channels * 2),
            CBAM(base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2)
        ])
        
        self.down3 = nn.ModuleList([
            ResidualBlock(base_channels * 2, base_channels * 4),
            CBAM(base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 4)
        ])

        # Bottleneck with residual blocks
        self.bottleneck = nn.ModuleList([
            ResidualBlock(base_channels * 4, base_channels * 8),
            CBAM(base_channels * 8),
            ResidualBlock(base_channels * 8, base_channels * 4)
        ])

        # Downsampling and upsampling
        self.down_ops = nn.ModuleList([
            Downsample(base_channels, True),
            Downsample(base_channels * 2, True),
            Downsample(base_channels * 4, True)
        ])
        
        self.up_ops = nn.ModuleList([
            Upsample(base_channels * 4, True),
            Upsample(base_channels * 2, True),
            Upsample(base_channels, True)
        ])

        # Decoder path with residual blocks
        self.up1 = nn.ModuleList([
            ResidualBlock(base_channels * 8, base_channels * 4),
            CBAM(base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 2)
        ])
        
        self.up2 = nn.ModuleList([
            ResidualBlock(base_channels * 4, base_channels * 2),
            CBAM(base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels)
        ])
        
        self.up3 = nn.ModuleList([
            ResidualBlock(base_channels * 2, base_channels),
            CBAM(base_channels),
            ResidualBlock(base_channels, base_channels)
        ])

        # Output layers with residual connection
        self.final_residual = ResidualBlock(base_channels, base_channels)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Low-level skip connection
        self.low_level_proj = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

    def forward(self, x, t, condition):
        # Combine input and condition
        x_input = torch.cat([x, condition], dim=1)
        low_level_features = self.low_level_proj(x_input)
        
        # Time embedding
        temb = self.time_mlp(t)
        
        # Initial residual block
        x = self.init_block(x_input)
        
        # Encoder path with residual connections
        # Down 1
        x1 = x
        for layer in self.down1:
            x1 = layer(x1)
        x1_skip = x1
        x1 = self.down_ops[0](x1)
        
        # Down 2
        x2 = x1
        for layer in self.down2:
            x2 = layer(x2)
        x2_skip = x2
        x2 = self.down_ops[1](x2)
        
        # Down 3
        x3 = x2
        for layer in self.down3:
            x3 = layer(x3)
        x3_skip = x3
        x3 = self.down_ops[2](x3)
        
        # Bottleneck
        x = x3
        for layer in self.bottleneck:
            x = layer(x)
        
        # Decoder path with skip and residual connections
        # Up 1
        x = self.up_ops[0](x)
        x = torch.cat([x, x3_skip], dim=1)
        for layer in self.up1:
            x = layer(x)
        
        # Up 2
        x = self.up_ops[1](x)
        x = torch.cat([x, x2_skip], dim=1)
        for layer in self.up2:
            x = layer(x)
        
        # Up 3
        x = self.up_ops[2](x)
        x = torch.cat([x, x1_skip], dim=1)
        for layer in self.up3:
            x = layer(x)
        
        # Final layers with residual connection
        x = self.final_residual(x)
        x = self.final_conv(x)
        
        # Add low-level features
        x = x + low_level_features
        
        return x