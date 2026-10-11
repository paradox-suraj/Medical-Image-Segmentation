#!/usr/bin/env python
"""Download LGG MRI Segmentation dataset from Kaggle."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

HANDLE = "mateuszbuda/lgg-mri-segmentation"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None, help="optional folder to copy the dataset into")
    a = p.parse_args()
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("kagglehub required: pip install kagglehub")

    path = Path(kagglehub.dataset_download(HANDLE))
    print(f"Downloaded to: {path}")
    root = next(iter(sorted(path.rglob("kaggle_3m"))), path)
    if a.out:
        dst = Path(a.out)
        if dst.exists():
            print(f"{dst} already exists - leaving it untouched.")
        else:
            shutil.copytree(root, dst)
            print(f"Copied to: {dst}")
            root = dst
    print(f"\nUse:   --dataset lgg --data-root {os.fspath(root)}")


if __name__ == "__main__":
    main()
