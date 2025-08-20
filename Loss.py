import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=0.5, perceptual_weight=0.5, perceptual_layers=[2, 7, 12, 21, 30]):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
        # L1 component
        self.l1 = nn.L1Loss()
        
        # Perceptual Loss Setup
        vgg = vgg16(pretrained=True).features
        self.perceptual_layers = perceptual_layers
        self.perceptual_model = nn.Sequential()
        for i, layer in enumerate(list(vgg.children())):
            self.perceptual_model.add_module(str(i), layer)
            if i in self.perceptual_layers:
                break
        
        # Freeze VGG layers
        for param in self.perceptual_model.parameters():
            param.requires_grad = False

    def perceptual_loss(self, pred, target):
        def extract_features(x):
            features = []
            for i, layer in enumerate(self.perceptual_model):
                x = layer(x)
                if i in self.perceptual_layers:
                    features.append(x)
            return features

        # Repeat single channel to match VGG input requirements
        pred_features = extract_features(pred.repeat(1, 3, 1, 1))
        target_features = extract_features(target.repeat(1, 3, 1, 1))
        
        # Compute perceptual loss as weighted sum of L1 distances between features
        loss = 0
        weights = [1.0, 0.75, 0.5, 0.25, 0.125]  # Higher weight for early layers
        for p_feat, t_feat, weight in zip(pred_features, target_features, weights):
            loss += weight * F.l1_loss(p_feat, t_feat)
        
        return loss

    def forward(self, pred, target):
        # L1 Loss
        l1_loss = self.l1(pred, target)
        
        # Perceptual Loss
        perceptual_loss = self.perceptual_loss(pred, target)
        
        # Combine losses with weights
        total_loss = (self.l1_weight * l1_loss + 
                     self.perceptual_weight * perceptual_loss)
        
        # Store individual losses for logging
        self.current_losses = {
            'l1': l1_loss.item(),
            'perceptual': perceptual_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss
