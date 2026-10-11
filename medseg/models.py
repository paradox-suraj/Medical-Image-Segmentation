"""Segmentation network architectures."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_BN_Weights, vgg16_bn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_ch: int = 32,
                 depth: int = 4, dropout: float = 0.2):
        super().__init__()
        ch = [base_ch * 2 ** i for i in range(depth + 1)]
        self.inc = DoubleConv(in_channels, ch[0])
        self.downs = nn.ModuleList(
            [Down(ch[i], ch[i + 1], dropout if i == depth - 1 else 0.0) for i in range(depth)])
        self.ups = nn.ModuleList([Up(ch[i + 1], ch[i]) for i in reversed(range(depth))])
        self.outc = nn.Conv2d(ch[0], num_classes, kernel_size=1)
        self.apply(_init_weights)

    def forward(self, x):
        feats = [self.inc(x)]
        for down in self.downs:
            feats.append(down(feats[-1]))
        x = feats.pop()
        for up in self.ups:
            x = up(x, feats.pop())
        return self.outc(x)


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ================================================================================ FCN
def _bilinear_kernel(channels: int, k: int) -> torch.Tensor:
    factor = (k + 1) // 2
    center = factor - 1 if k % 2 == 1 else factor - 0.5
    og = torch.arange(k, dtype=torch.float32)
    f1 = 1 - torch.abs(og - center) / factor
    w = torch.zeros(channels, channels, k, k)
    for c in range(channels):
        w[c, c] = f1[:, None] * f1[None, :]
    return w


def _deconv(ch: int, k: int, s: int, p: int) -> nn.ConvTranspose2d:
    layer = nn.ConvTranspose2d(ch, ch, kernel_size=k, stride=s, padding=p, bias=False)
    with torch.no_grad():
        layer.weight.copy_(_bilinear_kernel(ch, k))
    return layer


def _fit(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return x if x.shape[2:] == ref.shape[2:] else F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)


class FCN(nn.Module):
    def __init__(self, num_classes: int = 1, variant: str = "8s", pretrained: bool = True,
                 fc_channels: int = 1024, fc_kernel: int = 3, dropout: float = 0.3):
        super().__init__()
        if variant not in ("32s", "16s", "8s"):
            raise ValueError("variant must be '32s', '16s' or '8s'")
        self.variant = variant
        vgg = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None)

        stages, cur = [], []
        for layer in vgg.features:
            cur.append(layer)
            if isinstance(layer, nn.MaxPool2d):
                stages.append(nn.Sequential(*cur))
                cur = []
        self.to_pool3 = nn.Sequential(*stages[:3])
        self.to_pool4 = stages[3]
        self.to_pool5 = stages[4]

        self.head = nn.Sequential(
            nn.Conv2d(512, fc_channels, fc_kernel, padding=fc_kernel // 2), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(fc_channels, fc_channels, 1), nn.ReLU(inplace=True), nn.Dropout2d(dropout))
        self.score_fr = nn.Conv2d(fc_channels, num_classes, 1)

        if variant in ("16s", "8s"):
            self.score_pool4 = nn.Conv2d(512, num_classes, 1)
            self.up2_a = _deconv(num_classes, 4, 2, 1)
        if variant == "8s":
            self.score_pool3 = nn.Conv2d(256, num_classes, 1)
            self.up2_b = _deconv(num_classes, 4, 2, 1)
        k, s, p = {"32s": (64, 32, 16), "16s": (32, 16, 8), "8s": (16, 8, 4)}[variant]
        self.up_final = _deconv(num_classes, k, s, p)

        for m in [self.head, self.score_fr]:
            m.apply(_init_weights)
        for name in ("score_pool4", "score_pool3"):
            if hasattr(self, name):
                nn.init.zeros_(getattr(self, name).weight)
                nn.init.zeros_(getattr(self, name).bias)

    def forward(self, x):
        h, w = x.shape[2:]
        p3 = self.to_pool3(x)
        p4 = self.to_pool4(p3)
        p5 = self.to_pool5(p4)
        s = self.score_fr(self.head(p5))
        if self.variant != "32s":
            s4 = self.score_pool4(p4)
            s = _fit(self.up2_a(s), s4) + s4
            if self.variant == "8s":
                s3 = self.score_pool3(p3)
                s = _fit(self.up2_b(s), s3) + s3
        out = self.up_final(s)
        return out if out.shape[2:] == (h, w) else F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)


def _build_smp(name: str, in_channels: int, num_classes: int, pretrained: bool) -> nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as e:  # pragma: no cover
        raise ImportError("Model names starting with 'smp-' need:  pip install segmentation-models-pytorch") from e
    _, arch, encoder = name.split("-", 2)
    archs = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus, "fpn": smp.FPN,
             "deeplabv3plus": smp.DeepLabV3Plus, "manet": smp.MAnet}
    if arch not in archs:
        raise ValueError(f"smp architecture must be one of {sorted(archs)}")
    return archs[arch](encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                       in_channels=in_channels, classes=num_classes)


MODEL_CHOICES = ["unet", "fcn32s", "fcn16s", "fcn8s", "smp-unet-resnet34"]


def build_model(name: str, in_channels: int = 3, num_classes: int = 1, pretrained: bool = True,
                base_ch: int = 32) -> nn.Module:
    name = name.lower()
    if name == "unet":
        return UNet(in_channels, num_classes, base_ch=base_ch)
    if name in ("fcn32s", "fcn16s", "fcn8s"):
        return FCN(num_classes, variant=name[3:], pretrained=pretrained)
    if name.startswith("smp-"):
        return _build_smp(name, in_channels, num_classes, pretrained)
    raise ValueError(f"Unknown model '{name}'. Choices: {MODEL_CHOICES} (or any 'smp-<arch>-<encoder>')")
