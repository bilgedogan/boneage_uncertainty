import torch.nn as nn

from model import build_multi_input_model


def build_cqr_model(dropout=0.5, learning_rate=1e-4, name="efficientnet_b3"):
    """Reuse the point model and replace only the scalar head with two quantiles."""
    model = build_multi_input_model(
        dropout=dropout,
        learning_rate=learning_rate,
        name=name,
    )
    model.out = nn.Linear(model.out.in_features, 2)
    return model
