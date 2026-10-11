#!/usr/bin/env python
"""Train a segmentation model (U-Net, FCN, or SMP)."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from tqdm import tqdm

from medseg.data import build_loaders, load_samples, select_by_patients, split_patients
from medseg.losses import LOSS_CHOICES, build_loss
from medseg.metrics import segmentation_metrics
from medseg.models import MODEL_CHOICES, build_model
from medseg.utils import count_parameters, get_device, plot_history, save_json, seed_everything


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["lgg", "folder"], default="lgg")
    p.add_argument("--data-root", required=True)
    p.add_argument("--model", default="unet", help=f"{MODEL_CHOICES} or smp-<arch>-<encoder>")
    p.add_argument("--base-ch", type=int, default=32, help="U-Net base channels")
    p.add_argument("--no-pretrained", action="store_true", help="do not load ImageNet weights")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss", choices=LOSS_CHOICES, default="bce_dice")
    p.add_argument("--aug", choices=["none", "light", "medium", "strong"], default="medium")
    p.add_argument("--balance", action="store_true", help="oversample slices containing lesions")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=12, help="early-stopping patience (epochs)")
    p.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    p.add_argument("--greyscale", action="store_true", help="train on single-channel greyscale/FLAIR slices")
    p.add_argument("--max-samples", type=int, default=0, help="limit number of training slices")
    p.add_argument("--out", default=None, help="run folder (default: runs/<model>)")
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False, desc=""):
    train = optimizer is not None
    model.train(train)
    loss_sum, n = 0.0, 0
    dice, iou, pos = [], [], []
    bar = tqdm(loader, desc=desc, leave=False)
    for x, y in bar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        else:
            m = segmentation_metrics(logits.detach(), y)
            dice.append(m["dice"].cpu())
            iou.append(m["iou"].cpu())
            pos.append((y.sum((1, 2, 3)) > 0).cpu())
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
        bar.set_postfix(loss=f"{loss_sum / n:.4f}")
    out = {"loss": loss_sum / max(n, 1)}
    if not train:
        dice, iou, pos = torch.cat(dice), torch.cat(iou), torch.cat(pos)
        out.update(dice=dice.mean().item(), iou=iou.mean().item(),
                   dice_pos=dice[pos].mean().item() if pos.any() else float("nan"))
    return out


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    out = Path(args.out or f"runs/{args.model}")
    out.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.dataset, args.data_root)
    split = split_patients(samples, args.val_frac, args.test_frac, args.seed)
    by_split = {k: select_by_patients(samples, v) for k, v in split.items()}
    if args.max_samples:
        by_split["train"] = by_split["train"][: args.max_samples]
    print(f"device={device} amp={amp} | patients train/val/test = "
          f"{len(split['train'])}/{len(split['val'])}/{len(split['test'])} | slices = "
          f"{len(by_split['train'])}/{len(by_split['val'])}/{len(by_split['test'])}")
    train_loader, val_loader, _ = build_loaders(
        by_split, args.img_size, args.batch_size, args.aug, args.workers, args.balance,
        pin_memory=device.type == "cuda", greyscale=args.greyscale)

    model = build_model(args.model, pretrained=not args.no_pretrained, base_ch=args.base_ch).to(device)
    n_params = count_parameters(model)
    print(f"model={args.model} | trainable parameters: {n_params / 1e6:.2f} M")
    criterion = build_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    save_json({**vars(args), "n_params": n_params,
               "n_slices": {k: len(v) for k, v in by_split.items()}}, out / "config.json")
    save_json(split, out / "split.json")

    fields = ["epoch", "lr", "train_loss", "val_loss", "val_dice", "val_iou", "val_dice_pos", "seconds"]
    best, bad_epochs = -1.0, 0
    with open(out / "history.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr = run_epoch(model, train_loader, criterion, device, optimizer, scaler, amp, f"train {epoch}")
            va = run_epoch(model, val_loader, criterion, device, amp=amp, desc=f"val   {epoch}")
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step(va["dice"])
            row = {"epoch": epoch, "lr": lr, "train_loss": tr["loss"], "val_loss": va["loss"],
                   "val_dice": va["dice"], "val_iou": va["iou"], "val_dice_pos": va["dice_pos"],
                   "seconds": round(time.time() - t0, 1)}
            writer.writerow(row)
            fh.flush()
            print(f"epoch {epoch:3d} | lr {lr:.1e} | train {tr['loss']:.4f} | val {va['loss']:.4f} | "
                  f"Dice {va['dice']:.4f} | IoU {va['iou']:.4f} | Dice(tumour slices) {va['dice_pos']:.4f} | {row['seconds']}s")

            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": va["dice"]}, out / "last.pt")
            if va["dice"] > best + 1e-4:
                best, bad_epochs = va["dice"], 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": va["dice"]}, out / "best.pt")
                print(f"           -> new best model saved (val Dice {best:.4f})")
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"Early stopping: no improvement for {args.patience} epochs.")
                    break

    plot_history(out / "history.csv", out / "curves.png")
    print(f"\nDone. Best val Dice = {best:.4f}. Files in {out}/")
    print(f"Next:  python evaluate.py --run {out}")


if __name__ == "__main__":
    main()
