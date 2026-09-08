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
    DINOv2 ViT-S/14 — 384-dim. CLS token by default; optionally the patch-token grid.

    Loaded via torch.hub; weights cached in ~/.cache/torch/hub by default.

    Input:  (B, 3, H, W) — will be resized to 224×224 internally.
    Output: (B, 384)              when return_tokens=False   [DEFAULT, unchanged]
            (B, 1 + T, 384)       when return_tokens=True

    With return_tokens=True index 0 of the sequence is the normalised CLS token
    (``x_norm_clstoken``) and the remaining T entries are the normalised patch
    tokens (``x_norm_patchtokens``, 256 = 16×16 at 224/14) reshaped to their 16×16
    spatial grid and average-pooled by ``token_pool``:

        token_pool = 1 → 16×16 =  256 tokens   (T = 256, sequence 257)
        token_pool = 2 →  8×8  =   64 tokens   (T =  64, sequence  65)
        token_pool = 4 →  4×4  =   16 tokens   (T =  16, sequence  17)

    ``feature_dim`` is 384 in both modes; ``n_tokens`` is the returned sequence
    length (1 in CLS mode, 1 + T in token mode).
    """

    feature_dim = 384
    input_size  = 224

    def __init__(self, device: str = "cpu", return_tokens: bool = False,
                 token_pool: int = 1):
        super().__init__()
        try:
            import torchvision.transforms.functional as TF  # noqa: F401
        except ImportError:
            raise ImportError("torchvision required for DINOv2Encoder.")

        if int(token_pool) < 1:
            raise ValueError(f"token_pool must be >= 1, got {token_pool}")
        self.return_tokens = bool(return_tokens)
        self.token_pool    = int(token_pool)
        self._grid         = self.input_size // 14          # 16 for 224/14
        if self.return_tokens and self._grid % self.token_pool != 0:
            raise ValueError(
                f"token_pool={self.token_pool} does not divide the {self._grid}×{self._grid} "
                f"DINOv2 patch grid."
            )

        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            pretrained=True, verbose=False,
        )
        self.model = model
        self.freeze()

    @property
    def n_tokens(self) -> int:
        """Length of the token sequence `forward` returns (1 in CLS-only mode)."""
        if not self.return_tokens:
            return 1
        side = self._grid // self.token_pool
        return 1 + side * side

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resize if needed (CIFAR inputs are 32×32)
        if x.shape[-1] != 224:
            import torch.nn.functional as F
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        if self.return_tokens:
            with torch.no_grad():
                return self._forward_tokens(x)   # (B, 1 + T, 384)
        with torch.no_grad():
            return self.model(x)   # (B, 384) — CLS token

    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """CLS + pooled patch tokens via the dinov2 hub `forward_features` dict.

        Fails loudly rather than guessing if the hub API ever changes shape: the
        cache this feeds is written once and read by every arm of the ablation, so
        a silently wrong token layout would be invisible downstream.
        """
        import torch.nn.functional as F

        feats = self.model.forward_features(x)
        if not isinstance(feats, dict):
            raise RuntimeError(
                f"dinov2 forward_features returned {type(feats).__name__}, expected a dict "
                f"with 'x_norm_clstoken' / 'x_norm_patchtokens'."
            )
        missing = [k for k in ("x_norm_clstoken", "x_norm_patchtokens") if k not in feats]
        if missing:
            raise RuntimeError(
                f"dinov2 forward_features dict is missing {missing}; got keys {sorted(feats)}."
            )

        cls     = feats["x_norm_clstoken"]      # (B, 384)
        patches = feats["x_norm_patchtokens"]   # (B, 256, 384)
        if cls.ndim != 2 or patches.ndim != 3:
            raise RuntimeError(
                f"unexpected dinov2 token shapes: cls {tuple(cls.shape)}, "
                f"patches {tuple(patches.shape)}"
            )
        B, N, D = patches.shape
        side = int(round(N ** 0.5))
        if side * side != N:
            raise RuntimeError(f"{N} patch tokens do not form a square grid.")
        if side % self.token_pool != 0:
            raise RuntimeError(
                f"token_pool={self.token_pool} does not divide the {side}×{side} patch grid."
            )

        grid = patches.transpose(1, 2).reshape(B, D, side, side)   # (B, 384, 16, 16)
        if self.token_pool > 1:
            grid = F.avg_pool2d(grid, self.token_pool)             # (B, 384, s, s)
        tok = grid.flatten(2).transpose(1, 2)                      # (B, T, 384)
        return torch.cat([cls.unsqueeze(1), tok], dim=1)           # (B, 1 + T, 384)


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
# ResNet-18 (light laptop backbone)
# ---------------------------------------------------------------------------

class ResNet18Encoder(RootEncoder):
    """
    ImageNet-pretrained ResNet-18 with the final FC stripped, 512-dim.

    The light backbone for on-laptop (M1/MPS) validation runs: ~11M frozen params,
    weights auto-download once via torchvision. Not the headline encoder — use
    DINOv2/CLIP on HPC — but enough to give the heterogeneous stream a real shared
    representation so the grow/reuse/merge paths fire.
    """

    feature_dim = 512
    input_size  = 224

    def __init__(self, device: str = "cpu"):
        super().__init__()
        try:
            import torchvision.models as tvm
        except ImportError:
            raise ImportError("torchvision required for ResNet18Encoder.")
        base = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(base.children())[:-1])  # up to AvgPool
        self.flatten = nn.Flatten()
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 224:
            import torch.nn.functional as F
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            return self.flatten(self.model(x))   # (B, 512)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ENCODERS = {
    "dinov2_vits14": DINOv2Encoder,
    "clip_vitb16":   CLIPEncoder,
    "resnet50":      ResNet50Encoder,
    "resnet18":      ResNet18Encoder,
}


def build_encoder(name: str, device: str = "cpu", **kwargs) -> RootEncoder:
    """
    Instantiate and return a frozen RootEncoder by name.

    Args:
        name:   One of "dinov2_vits14", "clip_vitb16", "resnet50".
        device: Device string ("cpu", "cuda", "mps").
        kwargs: Encoder-specific options forwarded to the constructor
                (DINOv2Encoder: `return_tokens`, `token_pool`). Passing none
                leaves every existing call site's behaviour untouched.

    Returns:
        Frozen RootEncoder on the specified device.
    """
    if name not in _ENCODERS:
        raise ValueError(
            f"Unknown encoder '{name}'. Available: {list(_ENCODERS)}\n"
            f"For SmallCNN (legacy), set backbone='smallcnn' in config — "
            f"no encoder is built in that case."
        )
    enc = _ENCODERS[name](device=device, **kwargs)
    enc = enc.to(device)
    enc.freeze()
    return enc
