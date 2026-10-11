"""Dataset discovery, patient-wise splitting and PyTorch Dataset / DataLoader helpers.

Two on-disk layouts are supported
  lgg     Kaggle "LGG MRI Segmentation": <root>/<PATIENT>/<PATIENT>_<k>.tif and <PATIENT>_<k>_mask.tif
  folder  any dataset exported as <root>/images/<name>.png and <root>/masks/<name>.png
          (use "<patient>__<slice>.png" names so slices of one patient stay together;
           a file without "__" is treated as its own patient)
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .augment import get_eval_transforms, get_train_transforms

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm", ".nii", ".gz"}


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    patient: str


def read_dicom(path) -> tuple[np.ndarray, dict]:
    """Read a DICOM file (.dcm) with Rescale and Windowing applied."""
    import pydicom
    dcm = pydicom.dcmread(str(path))
    arr = dcm.pixel_array.astype(np.float32)
    slope = getattr(dcm, "RescaleSlope", 1)
    intercept = getattr(dcm, "RescaleIntercept", 0)
    arr = arr * float(slope) + float(intercept)

    wc = getattr(dcm, "WindowCenter", None)
    ww = getattr(dcm, "WindowWidth", None)
    if wc is not None and ww is not None:
        if isinstance(wc, (list, pydicom.multival.MultiValue)):
            wc = wc[0]
        if isinstance(ww, (list, pydicom.multival.MultiValue)):
            ww = ww[0]
        wc, ww = float(wc), float(ww)
        low, high = wc - ww / 2, wc + ww / 2
        arr = np.clip(arr, low, high)
        arr = ((arr - low) / max(high - low, 1e-5) * 255.0).astype(np.uint8)
    else:
        mn, mx = arr.min(), arr.max()
        arr = ((arr - mn) / max(mx - mn, 1e-5) * 255.0).astype(np.uint8)

    rgb = np.stack([arr, arr, arr], axis=-1)
    spacing = [float(x) for x in getattr(dcm, "PixelSpacing", [1.0, 1.0])]
    meta = {
        "PatientID": str(getattr(dcm, "PatientID", "ANONYMOUS")),
        "Modality": str(getattr(dcm, "Modality", "MR")),
        "StudyDate": str(getattr(dcm, "StudyDate", "")),
        "PixelSpacing": spacing,
        "SliceThickness": float(getattr(dcm, "SliceThickness", 1.0)),
    }
    return rgb, meta


def read_nifti(path, slice_idx=None) -> np.ndarray:
    """Read a 2D slice from a NIfTI volume (.nii or .nii.gz)."""
    import nibabel as nib
    nii = nib.load(str(path))
    data = nii.get_fdata()
    if data.ndim == 3:
        idx = data.shape[2] // 2 if slice_idx is None else slice_idx
        sl = data[:, :, idx]
    elif data.ndim == 2:
        sl = data
    else:
        sl = data[:, :, data.shape[2] // 2, 0]
    mn, mx = sl.min(), sl.max()
    norm = ((sl - mn) / max(mx - mn, 1e-5) * 255.0).astype(np.uint8)
    return np.stack([norm, norm, norm], axis=-1)


def read_image(path, greyscale: bool = False) -> np.ndarray:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".dcm":
        rgb, _ = read_dicom(p)
        return rgb
    if ext in (".nii", ".gz") or str(p).endswith(".nii.gz"):
        return read_nifti(p)

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if greyscale:
        is_lgg_multi = str(path).endswith(".tif") and np.std(rgb[..., 0].astype(float) - rgb[..., 1].astype(float)) > 8
        if is_lgg_multi:
            g = rgb[..., 1]
        else:
            g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return np.stack([g, g, g], axis=-1)
    return rgb


def read_mask(path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    thr = 0 if m.max() <= 1 else 127
    return (m > thr).astype(np.uint8)


def find_lgg_samples(root) -> list[Sample]:
    root = Path(root)
    samples, seen = [], set()
    for mp in sorted(root.rglob("*_mask.tif")):
        ip = mp.with_name(mp.name.replace("_mask.tif", ".tif"))
        if not ip.exists() or ip.name in seen:
            continue
        seen.add(ip.name)
        samples.append(Sample(ip, mp, mp.parent.name))
    if not samples:
        raise FileNotFoundError(f"No '*_mask.tif' files found under {root}. Check --data-root.")
    return samples


def find_folder_samples(root, images_dir: str = "images", masks_dir: str = "masks") -> list[Sample]:
    root = Path(root)
    img_dir, msk_dir = root / images_dir, root / masks_dir
    if not img_dir.is_dir() or not msk_dir.is_dir():
        raise FileNotFoundError(f"Expected folders '{img_dir}' and '{msk_dir}'.")
    masks = {p.stem: p for p in msk_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    samples = []
    for ip in sorted(img_dir.iterdir()):
        if ip.suffix.lower() not in IMG_EXTS:
            continue
        mp = masks.get(ip.stem) or masks.get(ip.stem + "_mask") or masks.get(ip.stem + "_segmentation")
        if mp is None:
            continue
        patient = ip.stem.split("__")[0]
        samples.append(Sample(ip, mp, patient))
    if not samples:
        raise FileNotFoundError(f"No image/mask pairs found in {root}.")
    return samples


def load_samples(dataset: str, root) -> list[Sample]:
    if dataset == "lgg":
        return find_lgg_samples(root)
    if dataset == "folder":
        return find_folder_samples(root)
    raise ValueError(f"Unknown dataset '{dataset}' (use 'lgg' or 'folder').")


def split_patients(samples: list[Sample], val_frac: float = 0.15, test_frac: float = 0.15,
                   seed: int = 42) -> dict[str, list[str]]:
    patients = sorted({s.patient for s in samples})
    if len(patients) < 3:
        raise ValueError("Need at least 3 patients/groups to create train/val/test splits.")
    random.Random(seed).shuffle(patients)
    n_test = max(1, round(len(patients) * test_frac))
    n_val = max(1, round(len(patients) * val_frac))
    return {"test": patients[:n_test],
            "val": patients[n_test:n_test + n_val],
            "train": patients[n_test + n_val:]}


def select_by_patients(samples: list[Sample], patients: list[str]) -> list[Sample]:
    keep = set(patients)
    return [s for s in samples if s.patient in keep]


class SegDataset(Dataset):
    def __init__(self, samples: list[Sample], transform=None, greyscale: bool = False):
        self.samples = samples
        self.transform = transform
        self.greyscale = greyscale

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        image, mask = read_image(s.image, greyscale=self.greyscale), read_mask(s.mask)
        if self.transform is not None:
            return self.transform(image, mask)
        return (torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0,
                torch.from_numpy(mask).float()[None])


def positive_flags(samples: list[Sample]) -> np.ndarray:
    return np.array([bool(read_mask(s.mask).any()) for s in samples])


def make_balanced_sampler(samples: list[Sample], pos_weight: float = 3.0) -> WeightedRandomSampler:
    w = np.where(positive_flags(samples), pos_weight, 1.0)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(samples), replacement=True)


def build_loaders(split_samples: dict[str, list[Sample]], img_size: int, batch_size: int,
                  aug_strength: str = "medium", workers: int = 2, balance: bool = False,
                  pin_memory: bool = False, greyscale: bool = False):
    train_ds = SegDataset(split_samples["train"], get_train_transforms(img_size, aug_strength), greyscale=greyscale)
    val_ds = SegDataset(split_samples["val"], get_eval_transforms(img_size), greyscale=greyscale)
    test_ds = SegDataset(split_samples["test"], get_eval_transforms(img_size), greyscale=greyscale)
    sampler = make_balanced_sampler(split_samples["train"]) if balance else None
    common = dict(num_workers=workers, pin_memory=pin_memory, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
                              drop_last=len(train_ds) >= batch_size, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader
