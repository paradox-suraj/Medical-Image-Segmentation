"""Loading a trained run and predicting with it (shared by evaluate.py, predict.py and app.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .augment import get_eval_transforms
from .data import read_image
from .models import build_model
from .utils import load_json


def load_run(run_dir, device, ckpt: str = "best.pt"):
    run = Path(run_dir)
    cfg = load_json(run / "config.json")
    model = build_model(cfg["model"], pretrained=False, base_ch=cfg.get("base_ch", 32))
    state = torch.load(run / ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def predict_probs(model, x: torch.Tensor, tta: bool = False) -> torch.Tensor:
    """Sigmoid probabilities (B,1,H,W). tta=True averages the prediction with its horizontal flip."""
    p = torch.sigmoid(model(x).float())
    if tta:
        p = (p + torch.sigmoid(model(torch.flip(x, dims=[3])).float()).flip(dims=[3])) / 2
    return p


@torch.no_grad()
def predict_image(model, cfg, device, image_rgb: np.ndarray, threshold: float = 0.5, tta: bool = False):
    """image_rgb: HxWx3 uint8. Returns (probability map HxW float32, binary mask HxW uint8) at the model's input size."""
    if cfg.get("greyscale", False):
        if image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
            is_lgg_multi = np.std(image_rgb[..., 0].astype(float) - image_rgb[..., 1].astype(float)) > 8
            if is_lgg_multi:
                g = image_rgb[..., 1]
            else:
                import cv2
                g = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            image_rgb = np.stack([g, g, g], axis=-1)
        elif image_rgb.ndim == 2:
            image_rgb = np.stack([image_rgb, image_rgb, image_rgb], axis=-1)
    tf = get_eval_transforms(cfg["img_size"])
    x, _ = tf(image_rgb, np.zeros(image_rgb.shape[:2], dtype=np.uint8))
    probs = predict_probs(model, x[None].to(device), tta)[0, 0].cpu().numpy()
    return probs, (probs > threshold).astype(np.uint8)


def predict_path(model, cfg, device, path, threshold: float = 0.5, tta: bool = False):
    return predict_image(model, cfg, device, read_image(path), threshold, tta)
