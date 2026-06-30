import torch
import torch.nn as nn
import timm
from torchvision.models import (
    efficientnet_b3, EfficientNet_B3_Weights,
    vit_b_16, ViT_B_16_Weights,
)


class MultiInputModel(nn.Module):
    def __init__(self, dropout=0.5,name="efficientnet_b3"):
        super().__init__()
        
        # Load pre-trained EfficientNetB3
        if name == "efficientnet_b3":
            backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            num_ftrs = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        elif name == "vit_b_16":
            backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            num_ftrs = backbone.heads.head.in_features  # 768
            backbone.heads = nn.Identity()
        elif name == "convnextv2_tiny":
            inner = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0)
            backbone = nn.Module()
            backbone.model = inner          # checkpoint keys are "base_model.model.*"
            backbone.forward = inner.forward
            num_ftrs = inner.num_features
        else:
            raise ValueError(f"Unknown backbone: {name!r}")

        self.base_model = backbone

        # Dense layer for sex input
        self.sex_fc = nn.Linear(1, 32)

        # Build the new head: feature + 32 (for processed sex_input)
        self.fc1 = nn.Linear(num_ftrs + 32, 256)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, 1)

    def forward(self, img, sex):
        # Extract features (B, num_ftrs)
        feat = self.base_model(img)

        # Process sex through dense layer
        sex = self.relu(self.sex_fc(sex))

        # Concatenate features with processed sex
        x = torch.cat((feat, sex), dim=1)
        
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.out(x)
        
        return out

def build_multi_input_model(dropout=0.5, learning_rate=1e-4, name="efficientnet_b3"):
    """
    Returns the PyTorch multi-input model.
    Note: learning_rate is handled in the optimizer in PyTorch,
    but we keep the signature compatible.
    """
    model = MultiInputModel(dropout=dropout,name=name)
    return model
