#!/usr/bin/env python
"""Compare metrics and generate summary tables and charts across evaluated runs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from medseg.utils import load_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    rows = []
    for r in a.runs:
        run = Path(r)
        sm = run / f"eval_{a.split}" / "summary.json"
        if not sm.exists():
            print(f"[skip] {run}: run evaluate.py first (missing {sm})")
            continue
        s, cfg = load_json(sm), load_json(run / "config.json")
        rows.append({"run": run.name, "model": s["model"], "aug": cfg.get("aug"), "loss": cfg.get("loss"),
                     "params (M)": s["params_million"], "Dice (all)": s["dice_all"], "IoU (all)": s["iou_all"],
                     "Dice (tumour)": s["dice_tumour"], "IoU (tumour)": s["iou_tumour"],
                     "Recall": s["recall_tumour"], "Precision": s["precision_tumour"],
                     "HD95 med (px)": s["hd95_median_px"],
                     "slice sens.": s["slice_sensitivity"], "slice spec.": s["slice_specificity"]})
    if not rows:
        raise SystemExit("Nothing to compare.")
    df = pd.DataFrame(rows).sort_values("Dice (tumour)", ascending=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "comparison.csv", index=False)
    try:
        (out / "comparison.md").write_text(df.to_markdown(index=False, floatfmt=".3f"))
    except ImportError:  # to_markdown needs 'tabulate'
        (out / "comparison.md").write_text(df.to_string(index=False))
    print(df.to_string(index=False))

    ax = df.set_index("run")[["Dice (tumour)", "IoU (tumour)", "Dice (all)"]].plot.bar(figsize=(8, 4.5), rot=20, ylim=(0, 1))
    ax.set_ylabel("score"); ax.set_title(f"Model comparison on the {a.split} set"); ax.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig(out / "comparison.png", dpi=150)
    print(f"\nSaved {out}/comparison.csv, comparison.md, comparison.png")


if __name__ == "__main__":
    main()
