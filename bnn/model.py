import torch
import torch.nn as nn
import timm
from torchvision.models import (
    efficientnet_b3, EfficientNet_B3_Weights,
    vit_b_16, ViT_B_16_Weights,
)
from bnn_layer import BayesianLinear

class _TimmBackbone(nn.Module):
    """Wraps a timm model: raw conv features -> global avg pool -> flatten.
    Matches the training-time forward path, which bypasses timm's own head.norm."""
    def __init__(self, name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(name, pretrained=pretrained, num_classes=0)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feats = self.model.forward_features(x)
        feats = self.pool(feats)
        feats = torch.flatten(feats, 1)
        return feats

class BNNMultiInputModel(nn.Module):
    def __init__(self, dropout=0.5, name="efficientnet_b3", prior_sigma=1.0):
        super().__init__()
        
        # Load pre-trained backbones
        if name == "efficientnet_b3":
            backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            num_ftrs = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        elif name == "vit_b_16":
            backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            num_ftrs = backbone.heads.head.in_features  # 768
            backbone.heads = nn.Identity()
        elif name == "convnextv2_tiny":
            backbone = _TimmBackbone("convnextv2_tiny", pretrained=True)
            num_ftrs = backbone.model.num_features
        else:
            raise ValueError(f"Unknown backbone: {name!r}")

        self.base_model = backbone

        # Dense layer for sex input
        self.sex_fc = BayesianLinear(1, 32, prior_sigma=prior_sigma)

        # Build the new head: feature + 32 (for processed sex_input)
        self.fc1 = BayesianLinear(num_ftrs + 32, 256, prior_sigma=prior_sigma)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.out = BayesianLinear(256, 1, prior_sigma=prior_sigma)

    def forward(self, img, sex, sample=True):
        # Extract features (B, num_ftrs)
        feat = self.base_model(img)

        # Process sex through dense layer
        sex = self.relu(self.sex_fc(sex, sample=sample))

        # Concatenate features with processed sex
        x = torch.cat((feat, sex), dim=1)
        
        # FC layers
        x = self.relu(self.fc1(x, sample=sample))
        x = self.dropout(x)
        out = self.out(x, sample=sample)
        
        return out
        
    def kl_divergence(self):
        kl = 0.0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl += module.kl_divergence()
        return kl

def build_multi_input_model(dropout=0.5, learning_rate=1e-4, name="efficientnet_b3", prior_sigma=1.0):
    """
    Returns the PyTorch multi-input model with Bayesian layers.
    Note: learning_rate is handled in the optimizer in PyTorch,
    but we keep the signature compatible.
    """
    model = BNNMultiInputModel(dropout=dropout, name=name, prior_sigma=prior_sigma)
    return model
