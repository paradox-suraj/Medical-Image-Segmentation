#!/usr/bin/env python
"""Convert 3D NIfTI volumes into 2D PNG slices for CT and MRI workflows."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def window_ct(sl: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return ((np.clip(sl, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def normalise_mri(vol: np.ndarray, lo_p: float = 1, hi_p: float = 99) -> np.ndarray:
    nz = vol[vol > 0]
    if nz.size == 0:
        return np.zeros_like(vol, dtype=np.uint8)
    lo, hi = np.percentile(nz, [lo_p, hi_p])
    return (np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True, help="folder with image volumes (*.nii or *.nii.gz)")
    p.add_argument("--labels", required=True, help="folder with label volumes of the same names")
    p.add_argument("--out", required=True)
    p.add_argument("--modality", choices=["ct", "mri"], required=True)
    p.add_argument("--label-ids", type=int, nargs="*", default=None, help="labels that count as foreground (default: all > 0)")
    p.add_argument("--channels", type=int, nargs="*", default=[0], help="MRI channels to use (up to 3)")
    p.add_argument("--ct-window", type=float, nargs=2, default=[-100, 400], help="HU window (liver: -100 400)")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--min-pixels", type=int, default=10, help="slice counts as 'positive' above this many mask pixels")
    p.add_argument("--keep-empty", type=float, default=0.15, help="fraction of empty slices to keep")
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    try:
        import nibabel as nib
    except ImportError:
        raise SystemExit("pip install nibabel")

    rng = random.Random(a.seed)
    out_i, out_m = Path(a.out, "images"), Path(a.out, "masks")
    out_i.mkdir(parents=True, exist_ok=True)
    out_m.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in Path(a.images).iterdir()
                   if f.name.endswith((".nii", ".nii.gz")) and not f.name.startswith("."))
    if a.max_cases:
        files = files[: a.max_cases]
    kept = pos_total = 0
    for f in files:
        case = f.name.replace(".nii.gz", "").replace(".nii", "")
        lab_path = Path(a.labels, f.name)
        if not lab_path.exists():
            print(f"[skip] no label for {case}")
            continue
        vol = np.asarray(nib.load(str(f)).dataobj, dtype=np.float32)
        lab = np.asarray(nib.load(str(lab_path)).dataobj)
        if vol.ndim == 3:
            vol = vol[..., None]
        chans = [vol[..., c] for c in a.channels[:3]]
        chans = [window_ct(c, *a.ct_window) if a.modality == "ct" else normalise_mri(c) for c in chans]
        while len(chans) < 3:
            chans.append(chans[0])
        fg = np.isin(lab, a.label_ids) if a.label_ids else lab > 0
        for k in range(vol.shape[2]):
            m = fg[:, :, k]
            positive = m.sum() >= a.min_pixels
            if not positive and rng.random() > a.keep_empty:
                continue
            img = np.rot90(np.stack([c[:, :, k] for c in chans], axis=-1))
            msk = np.rot90(m.astype(np.uint8)) * 255
            img = cv2.resize(np.ascontiguousarray(img), (a.size, a.size), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(np.ascontiguousarray(msk), (a.size, a.size), interpolation=cv2.INTER_NEAREST)
            name = f"{case}__{k:03d}.png"
            cv2.imwrite(str(out_i / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_m / name), msk)
            kept += 1
            pos_total += int(positive)
        print(f"{case}: done")
    print(f"\nSaved {kept} slices ({pos_total} with target) to {a.out}")


if __name__ == "__main__":
    main()
