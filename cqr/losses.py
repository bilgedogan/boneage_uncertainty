import torch.nn as nn
import torch.nn.functional as F

from cqr.torchcp_regression import QuantileLoss


class NonCrossingQuantileLoss(nn.Module):
    """TorchCP quantile loss plus ReLU crossing penalty."""

    def __init__(
        self,
        alpha=0.10,
        lambda_cross=10.0,
        margin=0.0,
        crossing_power=1,
    ):
        super().__init__()
        quantiles = [alpha / 2.0, 1.0 - alpha / 2.0]
        self.quantile_loss = QuantileLoss(quantiles)
        self.lambda_cross = lambda_cross
        self.margin = margin
        self.crossing_power = crossing_power

    def forward(self, preds, target):
        if preds.ndim != 2 or preds.size(1) != 2:
            raise ValueError("CQR predictions must have shape [batch, 2].")

        preds = preds.float()
        q_low = preds[:, 0]
        q_high = preds[:, 1]

        crossing = F.relu(q_low - q_high + self.margin)
        if self.crossing_power == 1:
            crossing_penalty = crossing.mean()
        elif self.crossing_power == 2:
            crossing_penalty = crossing.pow(2).mean()
        else:
            raise ValueError("crossing_power should be 1 or 2.")

        return self.quantile_loss(preds, target.float().view(-1)) + self.lambda_cross * crossing_penalty


def build_quantile_loss(
    alpha=0.10,
    lambda_cross=10.0,
    margin=0.0,
    crossing_power=1,
):
    return NonCrossingQuantileLoss(
        alpha=alpha,
        lambda_cross=lambda_cross,
        margin=margin,
        crossing_power=crossing_power,
    )
