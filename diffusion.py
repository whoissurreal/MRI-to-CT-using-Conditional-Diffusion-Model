import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import ExponentialMovingAverage, RandomBrightness, RandomContrast, RandomNoise
from unet import DiffusionUNet
from SwinEncoder import DINOv2Encoder
from noisescheduler import NoiseScheduleVP

class DiffusionModel(nn.Module):
    def __init__(self,
                 input_channels=1,
                 output_channels=1,
                 noise_steps=1000,
                 schedule='cosine',
                 beta_start=1e-4,
                 beta_end=0.02,
                 use_ema=True):
        super().__init__()
        
        self.use_ema = use_ema
        if use_ema:
            self.ema = ExponentialMovingAverage(0.995)
        
        self.lr_warmup_steps = 1000
        self.lr_decay_steps = 50000
        
        self.encoder = DINOv2Encoder()
        self.encoder.model.gradient_checkpointing_enable()
        
        self.noise_scheduler = NoiseScheduleVP(
            schedule=schedule,
            continuous_beta_0=beta_start,
            continuous_beta_1=beta_end
        )
        
        sample_input = torch.randn(1, 1, 224, 224)
        encoder_feature_dim = self.encoder(sample_input).size(1)
        
        self.unet = DiffusionUNet(
            in_channels=input_channels + encoder_feature_dim,
            out_channels=output_channels,
            time_dim=256,
            base_channels=128
        )
        
        self.augmentation = nn.ModuleList([
            RandomBrightness(0.2),
            RandomContrast(0.2)
        ])
        
    def sample_timesteps(self, batch_size):
        if self.noise_scheduler.schedule == 'discrete':
            timesteps = torch.randint(0, self.noise_scheduler.total_N, (batch_size,))
            return timesteps / self.noise_scheduler.total_N
        else:
            return torch.rand(batch_size) * self.noise_scheduler.T
    
    def noise_images(self, x_0, t):
        alpha_t = self.noise_scheduler.marginal_alpha(t)
        sigma_t = self.noise_scheduler.marginal_std(t)
        
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1)
        
        epsilon = torch.randn_like(x_0)
        
        x_t = alpha_t * x_0 + sigma_t * epsilon
        
        return x_t, epsilon

    def forward(self, x, t, condition=None):
        if self.training:
            for aug in self.augmentation:
                x = aug(x)
                if condition is not None:
                    condition = aug(condition)
        
        condition_features = self.encoder(condition)
        
        return self.unet(x, t, condition_features)
    
    @torch.no_grad()
    def sample(self, condition, num_steps=100, guidance_scale=7.5):
        self.eval()
        device = condition.device
        batch_size = condition.shape[0]
        
        condition_features = self.encoder(condition)
        null_features = torch.zeros_like(condition_features)
        
        x = torch.randn_like(condition)
        
        timesteps = torch.linspace(self.noise_scheduler.T, 0, num_steps, device=device)
        
        for i in range(num_steps - 1):
            t = timesteps[i].repeat(batch_size)
            next_t = timesteps[i + 1].repeat(batch_size)
            
            alpha = self.noise_scheduler.marginal_alpha(t)
            alpha_next = self.noise_scheduler.marginal_alpha(next_t)
            
            sigma = self.noise_scheduler.marginal_std(t)
            sigma_next = self.noise_scheduler.marginal_std(next_t)
            
            alpha = alpha.view(-1, 1, 1, 1)
            alpha_next = alpha_next.view(-1, 1, 1, 1)
            sigma = sigma.view(-1, 1, 1, 1)
            sigma_next = sigma_next.view(-1, 1, 1, 1)
            

            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([condition_features, null_features])
            
            noise_pred = self.unet(x_in, t_in, c_in)
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)

            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            

            x_0_pred = (x - sigma * noise_pred) / alpha
            
            if i < num_steps - 2:

                noise = torch.randn_like(x)
                x = alpha_next * x_0_pred + sigma_next * noise
            else:

                x = alpha_next * x_0_pred
        
        return x