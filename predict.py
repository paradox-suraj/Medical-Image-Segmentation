#!/usr/bin/env python
"""Run segmentation inference on a single image slice."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from medseg.data import read_image, read_mask
from medseg.inference import load_run, predict_image
from medseg.utils import get_device, overlay_masks, to_display


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--mask", default=None, help="optional ground-truth mask for comparison")
    p.add_argument("--out", default="outputs/prediction", help="output prefix (no extension)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    device = get_device(a.device)
    model, cfg = load_run(a.run, device)
    image = read_image(a.image)
    probs, pred = predict_image(model, cfg, device, image, a.threshold, a.tta)

    size = cfg["img_size"]
    base = to_display(cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA), cfg["dataset"])
    gt = None
    if a.mask:
        gt = cv2.resize(read_mask(a.mask), (size, size), interpolation=cv2.INTER_NEAREST)
    overlay = overlay_masks(base, gt=gt, pred=pred)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out) + "_mask.png", pred * 255)
    cv2.imwrite(str(out) + "_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    area = int(pred.sum())
    print(f"Predicted region: {area} px ({100 * area / pred.size:.2f}% of the slice)")
    if gt is not None:
        inter = (gt.astype(bool) & pred.astype(bool)).sum()
        dice = (2 * inter + 1e-7) / (gt.sum() + pred.sum() + 1e-7)
        print(f"Dice vs ground truth: {dice:.4f}")
    print(f"Saved {out}_mask.png and {out}_overlay.png")


if __name__ == "__main__":
    main()
