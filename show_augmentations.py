#!/usr/bin/env python
"""Visualize image and mask data augmentation samples."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from medseg.augment import get_train_transforms
from medseg.data import load_samples, read_image, read_mask
from medseg.utils import denormalize, overlay_masks, seed_everything, to_display


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["lgg", "folder"], default="lgg")
    p.add_argument("--data-root", required=True)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--strength", choices=["light", "medium", "strong"], default="medium")
    p.add_argument("--n", type=int, default=7, help="number of augmented versions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="figures/augmentations.png")
    a = p.parse_args()

    seed_everything(a.seed)
    samples = load_samples(a.dataset, a.data_root)
    random.shuffle(samples)
    sample = next((s for s in samples if read_mask(s.mask).sum() > 150), samples[0])
    image, mask = read_image(sample.image), read_mask(sample.mask)
    tf = get_train_transforms(a.img_size, a.strength)

    cols = 4
    rows = int(np.ceil((a.n + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    first = to_display(image, a.dataset)
    axes[0, 0].imshow(overlay_masks(first, gt=mask)); axes[0, 0].set_title("original + mask", fontsize=9)
    for k in range(a.n):
        x, y = tf(image, mask)
        img = to_display(denormalize(x), a.dataset)
        ax = axes[(k + 1) // cols, (k + 1) % cols]
        ax.imshow(overlay_masks(img, gt=y[0].numpy().astype(np.uint8)))
        ax.set_title(f"augmented #{k + 1}", fontsize=9)
    fig.suptitle("Augmented Image and Mask Samples")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
