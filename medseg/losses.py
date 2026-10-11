"""Loss functions for binary segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_DIMS = (1, 2, 3)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits.float())
        inter = (p * targets).sum(_DIMS)
        denom = p.sum(_DIMS) + targets.sum(_DIMS)
        return (1 - (2 * inter + self.smooth) / (denom + self.smooth)).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.w, self.dice = bce_weight, DiceLoss(smooth)

    def forward(self, logits, targets):
        return self.w * F.binary_cross_entropy_with_logits(logits.float(), targets) + (1 - self.w) * self.dice(logits, targets)


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1.0):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits.float())
        tp = (p * targets).sum(_DIMS)
        fp = (p * (1 - targets)).sum(_DIMS)
        fn = ((1 - p) * targets).sum(_DIMS)
        ti = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return ((1 - ti) ** self.gamma).mean()


LOSS_CHOICES = ["bce", "dice", "bce_dice", "focal_tversky"]


def build_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss()
    if name == "bce_dice":
        return BCEDiceLoss()
    if name == "focal_tversky":
        return FocalTverskyLoss()
    raise ValueError(f"Unknown loss '{name}'. Choices: {LOSS_CHOICES}")
