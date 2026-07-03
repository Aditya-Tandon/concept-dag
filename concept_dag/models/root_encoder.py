"""
RootEncoder — pluggable frozen feature extractor.

All encoders expose the same interface:
    encoder(x: Tensor[B, 3, H, W]) -> Tensor[B, feature_dim]

They are always frozen (no optimizer step touches their params).
The encoder is a singleton shared across all DAG nodes; it lives
outside the DAGNode and is called only during feature caching.

Supported backends
------------------
  "smallcnn"       — the original task-trained SmallCNN (legacy; keep for comparison)
  "dinov2_vits14"  — DINOv2 ViT-S/14 (384-dim, 32×32→224×224 via resize)  [PRIMARY]
  "clip_vitb16"    — CLIP ViT-B/16 (512-dim) — optional, needs open_clip
  "resnet50"       — ImageNet-pretrained ResNet-50 (2048-dim) — sanity check

Usage
-----
    enc = build_encoder("dinov2_vits14", device="cuda")
    # enc is frozen; call cache_features() from feature_cache.py, not enc() directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class RootEncoder(nn.Module):
    """Abstract frozen feature extractor."""

    feature_dim: int           # must be set by subclass
    input_size:  int = 224     # expected H=W after resize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# DINOv2 ViT-S/14 (primary)
# ---------------------------------------------------------------------------

class DINOv2Encoder(RootEncoder):
    """
    DINOv2 ViT-S/14 — CLS token output, 384-dim.
    Loaded via torch.hub; weights cached in ~/.cache/torch/hub by default.

    Input:  (B, 3, H, W) — will be resized to 224×224 internally.
    Output: (B, 384)
    """

    feature_dim = 384
    input_size  = 224

    def __init__(self, device: str = "cpu"):
        super().__init__()
        try:
            import torchvision.transforms.functional as TF  # noqa: F401
        except ImportError:
            raise ImportError("torchvision required for DINOv2Encoder.")

        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            pretrained=True, verbose=False,
        )
        self.model = model
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resize if needed (CIFAR inputs are 32×32)
        if x.shape[-1] != 224:
            import torch.nn.functional as F
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            return self.model(x)   # (B, 384) — CLS token


# ---------------------------------------------------------------------------
# CLIP ViT-B/16 (optional)
# ---------------------------------------------------------------------------

class CLIPEncoder(RootEncoder):
    """
    CLIP ViT-B/16 image encoder, 512-dim.
    Requires: pip install open_clip_torch
    """

    feature_dim = 512
    input_size  = 224

    def __init__(self, device: str = "cpu"):
        super().__init__()
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch required for CLIPEncoder. "
                "Install: pip install open_clip_torch --break-system-packages"
            )
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        self.model = model.visual
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 224:
            import torch.nn.functional as F
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            return self.model(x)   # (B, 512)


# ---------------------------------------------------------------------------
# ResNet-50 (sanity-check baseline)
# ---------------------------------------------------------------------------

class ResNet50Encoder(RootEncoder):
    """
    ImageNet-pretrained ResNet-50 with the final FC stripped, 2048-dim.
    """

    feature_dim = 2048
    input_size  = 224

    def __init__(self, device: str = "cpu"):
        super().__init__()
        try:
            import torchvision.models as tvm
        except ImportError:
            raise ImportError("torchvision required for ResNet50Encoder.")
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        # Strip the classification head
        self.model = nn.Sequential(*list(base.children())[:-1])  # up to AvgPool
        self.flatten = nn.Flatten()
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 224:
            import torch.nn.functional as F
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            return self.flatten(self.model(x))   # (B, 2048)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ENCODERS = {
    "dinov2_vits14": DINOv2Encoder,
    "clip_vitb16":   CLIPEncoder,
    "resnet50":      ResNet50Encoder,
}


def build_encoder(name: str, device: str = "cpu") -> RootEncoder:
    """
    Instantiate and return a frozen RootEncoder by name.

    Args:
        name:   One of "dinov2_vits14", "clip_vitb16", "resnet50".
        device: Device string ("cpu", "cuda", "mps").

    Returns:
        Frozen RootEncoder on the specified device.
    """
    if name not in _ENCODERS:
        raise ValueError(
            f"Unknown encoder '{name}'. Available: {list(_ENCODERS)}\n"
            f"For SmallCNN (legacy), set backbone='smallcnn' in config — "
            f"no encoder is built in that case."
        )
    enc = _ENCODERS[name](device=device)
    enc = enc.to(device)
    enc.freeze()
    return enc
