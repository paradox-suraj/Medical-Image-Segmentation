"""Segmentation evaluation metrics."""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

EPS = 1e-7


@torch.no_grad()
def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, torch.Tensor]:
    """Compute per-sample segmentation metrics for a batch."""
    preds = (torch.sigmoid(logits.float()) > threshold).float()
    dims = (1, 2, 3)
    tp = (preds * targets).sum(dims)
    fp = (preds * (1 - targets)).sum(dims)
    fn = ((1 - preds) * targets).sum(dims)
    tn = ((1 - preds) * (1 - targets)).sum(dims)
    return {
        "dice": (2 * tp + EPS) / (2 * tp + fp + fn + EPS),
        "iou": (tp + EPS) / (tp + fp + fn + EPS),
        "precision": (tp + EPS) / (tp + fp + EPS),
        "recall": (tp + EPS) / (tp + fn + EPS),
        "specificity": (tn + EPS) / (tn + fp + EPS),
        "gt_pixels": targets.sum(dims),
        "pred_pixels": preds.sum(dims),
    }


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile Hausdorff distance in pixels between the two boundaries (NaN if either mask is empty)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    to_gt = distance_transform_edt(~gt)
    to_pred = distance_transform_edt(~pred)
    pred_surface = pred & ~binary_erosion(pred)
    gt_surface = gt & ~binary_erosion(gt)
    d = np.concatenate([to_gt[pred_surface], to_pred[gt_surface]])
    return float(np.percentile(d, 95))
