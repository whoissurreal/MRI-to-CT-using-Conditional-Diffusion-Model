import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

class DINOv2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        self.model = AutoModel.from_pretrained("facebook/dinov2-large")
        self.model = self.model.float()
        for param in self.model.parameters():
            param.data = param.data.float()
            
        self.hidden_dim = self.model.config.hidden_size
        
    def forward(self, x):
        x = x.float()
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = x.repeat(1, 3, 1, 1)
        x_np = x.permute(0, 2, 3, 1).detach().cpu().numpy()
        inputs = self.processor(images=x_np, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(x.device).float()
        outputs = self.model(pixel_values=pixel_values)
        features = outputs.last_hidden_state
        batch_size = features.shape[0]
        input_size = 224
        patch_size = 14
        spatial_size = input_size // patch_size
        patch_tokens = features[:, 1:]
        features = patch_tokens.reshape(batch_size, spatial_size, spatial_size, self.hidden_dim)
        features = features.permute(0, 3, 1, 2)
        features = F.interpolate(features, size=(input_size, input_size), mode='bilinear', align_corners=False)
        
        return features