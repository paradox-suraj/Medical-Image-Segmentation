"""Image and mask data augmentation pipelines."""
from __future__ import annotations

import random

import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from .utils import IMAGENET_MEAN, IMAGENET_STD

STRENGTH = {"light": 0.5, "medium": 1.0, "strong": 1.5}


class SegTransform:
    def __init__(self, img_size: int = 256, train: bool = True, strength: str = "medium"):
        self.img_size = img_size
        resize = v2.Resize((img_size, img_size), antialias=True)
        self.photometric = None
        self.noise_p, self.noise_std = 0.0, (0.01, 0.05)

        if train and strength != "none":
            s = STRENGTH[strength]
            self.geometric = v2.Compose([
                resize,
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply([v2.RandomAffine(degrees=15 * s,
                                                translate=(0.05 * s, 0.05 * s),
                                                scale=(1 - 0.10 * s, 1 + 0.10 * s))], p=0.7),
                v2.RandomApply([v2.ElasticTransform(alpha=30.0 * s, sigma=5.0)], p=0.3),
            ])
            self.photometric = v2.Compose([
                v2.RandomApply([v2.ColorJitter(brightness=0.2 * s, contrast=0.2 * s)], p=0.5),
                v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.2))], p=0.2),
            ])
            self.noise_p = 0.3
        else:
            self.geometric = resize

        self.normalize = v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        img = tv_tensors.Image(torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))))
        msk = tv_tensors.Mask(torch.from_numpy(np.ascontiguousarray(mask))[None])
        img, msk = self.geometric(img, msk)
        if self.photometric is not None:
            img = self.photometric(img)
        img = v2.functional.to_dtype(img, torch.float32, scale=True)
        if self.noise_p and random.random() < self.noise_p:
            img = (img + torch.randn_like(img) * random.uniform(*self.noise_std)).clamp(0, 1)
        img = self.normalize(img)
        img = img.as_subclass(torch.Tensor)
        msk = (msk.as_subclass(torch.Tensor) > 0).float()
        return img, msk


def get_train_transforms(img_size: int = 256, strength: str = "medium") -> SegTransform:
    return SegTransform(img_size, train=True, strength=strength)


def get_eval_transforms(img_size: int = 256) -> SegTransform:
    return SegTransform(img_size, train=False)
