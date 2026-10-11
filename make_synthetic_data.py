#!/usr/bin/env python
"""Generate synthetic MRI slices and segmentation masks for testing."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_slice(rng, size, lesion: bool):
    yy, xx = np.mgrid[:size, :size]
    cy, cx = size / 2 + rng.normal(0, 3), size / 2 + rng.normal(0, 3)
    brain = (((yy - cy) / (0.42 * size)) ** 2 + ((xx - cx) / (0.36 * size)) ** 2) < 1
    img = np.zeros((size, size, 3), np.float32)
    for c in range(3):
        img[..., c] = brain * (0.35 + 0.05 * c) + rng.normal(0, 0.03, (size, size))
    mask = np.zeros((size, size), np.uint8)
    if lesion:
        ly, lx = cy + rng.normal(0, 0.12 * size), cx + rng.normal(0, 0.12 * size)
        ry, rx = rng.uniform(0.04, 0.11) * size, rng.uniform(0.04, 0.11) * size
        blob = (((yy - ly) / ry) ** 2 + ((xx - lx) / rx) ** 2) < 1
        mask = (blob & brain).astype(np.uint8)
        img[..., 1] += mask * 0.35
        img[..., 0] += mask * 0.10
    img = cv2.GaussianBlur(np.clip(img, 0, 1), (3, 3), 0)
    return (img * 255).astype(np.uint8), mask * 255


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--layout", choices=["folder", "lgg"], default="folder")
    p.add_argument("--patients", type=int, default=14)
    p.add_argument("--slices", type=int, default=10)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    for pi in range(a.patients):
        pid = f"PAT_{pi:03d}"
        for k in range(a.slices):
            lesion = 3 <= k <= a.slices - 4
            img, msk = make_slice(rng, a.size, lesion)
            if a.layout == "folder":
                (out / "images").mkdir(parents=True, exist_ok=True)
                (out / "masks").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / "images" / f"{pid}__{k:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out / "masks" / f"{pid}__{k:03d}.png"), msk)
            else:
                d = out / "kaggle_3m" / pid
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{pid}_{k + 1}.tif"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(d / f"{pid}_{k + 1}_mask.tif"), msk)
    print(f"Synthetic dataset written to {out} ({a.patients} patients x {a.slices} slices, {a.size}x{a.size}).")


if __name__ == "__main__":
    main()
