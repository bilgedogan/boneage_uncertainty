"""Bayesian Neural Network model for bone age regression.

The backbone (EfficientNet-B3 / ViT-B/16 / ConvNeXtV2-Tiny) stays a deterministic
fine-tuned feature extractor; only the regression head (fc1 + out) is turned into
Bayesian variational layers using Intel Labs' ``bayesian-torch`` library
(``dnn_to_bnn`` with mean-field Reparameterization layers).

This mirrors the point-prediction ``MultiInputModel`` in the project root
(../model.py) so results stay comparable across UQ methods.
"""

import torch
import torch.nn as nn
import timm
from torchvision.models import (
    efficientnet_b3, EfficientNet_B3_Weights,
    vit_b_16, ViT_B_16_Weights,
)
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss


def make_prior_parameters(prior_sigma=1.0):
    """Return the ``const_bnn_prior_parameters`` dict expected by ``dnn_to_bnn``.

    Mean-field Gaussian prior N(prior_mu, prior_sigma^2), Reparameterization
    variational posterior. MOPED is disabled so the posterior is initialised
    from scratch rather than from pretrained weights.
    """
    return {
        "prior_mu": 0.0,
        "prior_sigma": prior_sigma,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": False,
        "moped_delta": 0.5,
    }


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


class _BayesianHead(nn.Module):
    """The regression head whose linear layers become Bayesian after conversion.

    Kept as its own module so ``dnn_to_bnn`` only converts fc1 and out, leaving
    the backbone and sex branch deterministic.
    """
    def __init__(self, in_features, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.out(x)


class BNNMultiInputModel(nn.Module):
    """Image + sex -> bone age, with a Bayesian regression head."""

    def __init__(self, dropout=0.5, name="efficientnet_b3", prior_sigma=1.0):
        super().__init__()

        # Load pre-trained backbone (deterministic feature extractor)
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

        # Deterministic dense layer for the sex input
        self.sex_fc = nn.Linear(1, 32)
        self.relu = nn.ReLU()

        # Bayesian head over [backbone features | processed sex]
        self.head = _BayesianHead(num_ftrs + 32, dropout=dropout)
        dnn_to_bnn(self.head, make_prior_parameters(prior_sigma))

    def forward(self, img, sex):
        feat = self.base_model(img)
        sex = self.relu(self.sex_fc(sex))
        x = torch.cat((feat, sex), dim=1)
        return self.head(x)

    def kl_loss(self):
        """Total KL divergence of the Bayesian head (backbone contributes 0)."""
        return get_kl_loss(self.head)


def build_bnn_model(dropout=0.5, name="efficientnet_b3", prior_sigma=1.0):
    """Factory for the Bayesian multi-input model."""
    return BNNMultiInputModel(dropout=dropout, name=name, prior_sigma=prior_sigma)
