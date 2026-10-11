"""Small helpers: seeding, device selection, JSON I/O, visualisation and plotting."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def denormalize(img: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=img.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img.device).view(3, 1, 1)
    x = (img.float() * std + mean).clamp(0, 1)
    return (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def to_display(img_u8: np.ndarray, dataset: str = "folder") -> np.ndarray:
    if dataset == "lgg":
        g = img_u8[..., 1]
        return np.stack([g, g, g], axis=-1)
    return img_u8


def overlay_masks(image: np.ndarray, gt: np.ndarray | None = None,
                  pred: np.ndarray | None = None, alpha: float = 0.35) -> np.ndarray:
    """Draw ground truth (green) and prediction (red) on an RGB uint8 image."""
    out = image.astype(np.float32).copy()
    layers = []
    if gt is not None:
        layers.append((gt, (0, 255, 0)))
    if pred is not None:
        layers.append((pred, (255, 0, 0)))
    for mask, color in layers:
        m = mask.astype(bool)
        if m.any():
            out[m] = (1 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    out = np.ascontiguousarray(out.clip(0, 255).astype(np.uint8))
    for mask, color in layers:
        if mask.any():
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cnts, -1, color, 1)
    return out


def plot_history(csv_path, out_path) -> None:
    """Training curves (loss, Dice/IoU, learning rate) from history.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(df["epoch"], df["train_loss"], label="train")
    ax[0].plot(df["epoch"], df["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(df["epoch"], df["val_dice"], label="val Dice (all slices)")
    ax[1].plot(df["epoch"], df["val_dice_pos"], label="val Dice (tumour slices)")
    ax[1].plot(df["epoch"], df["val_iou"], label="val IoU (all slices)")
    ax[1].set_title("Validation metrics"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(df["epoch"], df["lr"])
    ax[2].set_title("Learning rate"); ax[2].set_xlabel("epoch"); ax[2].set_yscale("log"); ax[2].grid(alpha=.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
