#!/usr/bin/env python
"""Evaluate a trained segmentation model on held-out test or validation splits."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from medseg.augment import get_eval_transforms
from medseg.data import SegDataset, load_samples, select_by_patients
from medseg.inference import load_run, predict_probs
from medseg.metrics import hd95, segmentation_metrics
from medseg.utils import denormalize, get_device, load_json, overlay_masks, save_json, to_display


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="run folder created by train.py")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--data-root", default=None, help="override the data root stored in config.json")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tta", action="store_true", help="enable test-time augmentation")
    p.add_argument("--no-hd95", action="store_true", help="skip Hausdorff distance computation")
    p.add_argument("--n-vis", type=int, default=3, help="number of cases per visualization category")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def case_grid(model, cfg, device, ds, picks, thr, tta, path):
    fig, axes = plt.subplots(len(picks), 4, figsize=(12, 3 * len(picks)), squeeze=False)
    for r, (i, caption) in enumerate(picks):
        x, y = ds[i]
        p = predict_probs(model, x[None].to(device), tta)[0, 0].cpu().numpy()
        img = to_display(denormalize(x), cfg["dataset"])
        gt, pr = y[0].numpy().astype(np.uint8), (p > thr).astype(np.uint8)
        tiles = [img, overlay_masks(img, gt=gt), overlay_masks(img, pred=pr), overlay_masks(img, gt, pr)]
        titles = [caption, "ground truth (green)", "prediction (red)", "overlap"]
        for c in range(4):
            axes[r, c].imshow(tiles[c])
            axes[r, c].set_title(titles[c], fontsize=9)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    device = get_device(args.device)
    run = Path(args.run)
    model, cfg = load_run(run, device)
    split = load_json(run / "split.json")
    samples = select_by_patients(load_samples(cfg["dataset"], args.data_root or cfg["data_root"]), split[args.split])
    ds = SegDataset(samples, get_eval_transforms(cfg["img_size"]), greyscale=cfg.get("greyscale", False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print(f"Evaluating '{cfg['model']}' on {len(ds)} {args.split} slices from {len(split[args.split])} unseen patients ...")

    rows, k = [], 0
    for x, y in tqdm(loader):
        probs = predict_probs(model, x.to(device), args.tta).cpu()
        m = segmentation_metrics(torch.logit(probs, eps=1e-6), y, args.threshold)
        preds_np, gt_np = (probs > args.threshold).numpy()[:, 0], y.numpy()[:, 0]
        for b in range(x.size(0)):
            s = samples[k]
            rows.append({"image": s.image.name, "patient": s.patient,
                         **{key: m[key][b].item() for key in
                            ("dice", "iou", "precision", "recall", "specificity", "gt_pixels", "pred_pixels")},
                         "hd95_px": np.nan if args.no_hd95 else hd95(preds_np[b], gt_np[b])})
            k += 1
    df = pd.DataFrame(rows)

    out = run / f"eval_{args.split}"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "per_image_metrics.csv", index=False)

    pos, neg = df["gt_pixels"] > 0, df["gt_pixels"] == 0
    summary = {
        "run": str(run), "model": cfg["model"], "split": args.split, "tta": args.tta, "threshold": args.threshold,
        "params_million": round(cfg["n_params"] / 1e6, 2),
        "n_slices": int(len(df)), "n_tumour_slices": int(pos.sum()), "n_patients": int(df["patient"].nunique()),
        "dice_all": df["dice"].mean(), "iou_all": df["iou"].mean(),
        "dice_tumour": df.loc[pos, "dice"].mean(), "iou_tumour": df.loc[pos, "iou"].mean(),
        "precision_tumour": df.loc[pos, "precision"].mean(), "recall_tumour": df.loc[pos, "recall"].mean(),
        "hd95_median_px": df.loc[pos, "hd95_px"].median(), "hd95_mean_px": df.loc[pos, "hd95_px"].mean(),
        "slice_sensitivity": float((df.loc[pos, "pred_pixels"] > 0).mean()) if pos.any() else float("nan"),
        "slice_specificity": float((df.loc[neg, "pred_pixels"] == 0).mean()) if neg.any() else float("nan"),
    }
    summary = {k_: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k_, v in summary.items()}
    save_json(summary, out / "summary.json")

    picks = []
    ranked = df[pos].sort_values("dice", ascending=False)
    n = max(1, args.n_vis)
    if len(ranked):
        mid = len(ranked) // 2
        picks += [(i, f"BEST dice={df.at[i, 'dice']:.2f}") for i in ranked.index[:n]]
        picks += [(i, f"MEDIAN dice={df.at[i, 'dice']:.2f}") for i in ranked.index[max(0, mid - n // 2): max(0, mid - n // 2) + n]]
        picks += [(i, f"WORST dice={df.at[i, 'dice']:.2f}") for i in ranked.index[-n:]]
    fa = df[neg & (df["pred_pixels"] > 0)].sort_values("pred_pixels", ascending=False)
    picks += [(i, f"FALSE ALARM ({int(df.at[i, 'pred_pixels'])} px)") for i in fa.index[:2]]
    if picks:
        case_grid(model, cfg, device, ds, picks, args.threshold, args.tta, out / "qualitative_cases.png")

    if pos.any():
        plt.figure(figsize=(6, 4))
        plt.hist(df.loc[pos, "dice"], bins=20, color="#3b7dd8", edgecolor="white")
        plt.axvline(df.loc[pos, "dice"].mean(), color="k", ls="--", label=f"mean {df.loc[pos, 'dice'].mean():.3f}")
        plt.xlabel("Dice per tumour slice"); plt.ylabel("number of slices"); plt.legend(); plt.tight_layout()
        plt.savefig(out / "dice_histogram.png", dpi=150)
        plt.close()

    print("\n" + "=" * 62)
    for key, v in summary.items():
        print(f"{key:>20}: {v}")
    print("=" * 62)
    print(f"Saved to {out}/  (summary.json, per_image_metrics.csv, qualitative_cases.png, dice_histogram.png)")


if __name__ == "__main__":
    main()
