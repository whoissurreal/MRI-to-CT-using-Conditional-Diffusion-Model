import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ExponentialMovingAverage:
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}
        
    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    self.decay * self.shadow[name].data +
                    (1 - self.decay) * param.data
                )

class RandomNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
        
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class RandomBrightness(nn.Module):
    def __init__(self, brightness_factor=0.2):
        super().__init__()
        self.brightness_factor = brightness_factor
        
    def forward(self, x):
        if self.training:
            factor = 1.0 + torch.rand(1).item() * self.brightness_factor * 2 - self.brightness_factor
            return torch.clamp(x * factor, 0, 1)
        return x

class RandomContrast(nn.Module):
    def __init__(self, contrast_factor=0.2):
        super().__init__()
        self.contrast_factor = contrast_factor
        
    def forward(self, x):
        if self.training:
            factor = 1.0 + torch.rand(1).item() * self.contrast_factor * 2 - self.contrast_factor
            mean = torch.mean(x)
            return torch.clamp((x - mean) * factor + mean, 0, 1)
        return x

class ImageAugmentation(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.augmentations = nn.ModuleList([
            RandomNoise(0.1),
            RandomBrightness(0.2),
            RandomContrast(0.2)
        ])
    
    def forward(self, x):
        if self.training and torch.rand(1).item() < self.p:
            # Randomly apply augmentations
            for aug in self.augmentations:
                if torch.rand(1).item() < self.p:
                    x = aug(x)
        return x

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'Total Parameters': f"{total_params:,}",
        
    }

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
