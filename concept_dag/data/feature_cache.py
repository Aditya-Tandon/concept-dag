"""
Feature caching for frozen SSL encoders.

Runs the encoder once over every task split (train / val / test),
saves (features, labels) tensors to disk, and returns new task dicts
whose DataLoaders yield feature tensors instead of raw images.

This decouples encoder cost from training: once the cache exists,
experiments run without touching the encoder at all.

Usage
-----
    from concept_dag.models.root_encoder import build_encoder
    from concept_dag.data.feature_cache import cache_features
    from concept_dag.data.loaders import make_split_cifar100

    raw_tasks = make_split_cifar100(data_root="./data", n_tasks=20)
    encoder   = build_encoder("dinov2_vits14", device="cuda")
    tasks     = cache_features(encoder, raw_tasks, cache_dir="./data/dino_features",
                               device="cuda")
    # tasks is a drop-in replacement for raw_tasks; loaders yield (B, 384) tensors.

Cache layout
------------
    cache_dir/
        task_{t}_train_features.pt   Tensor[N_train, D]
        task_{t}_train_labels.pt     Tensor[N_train]
        task_{t}_val_features.pt
        task_{t}_val_labels.pt
        task_{t}_test_features.pt
        task_{t}_test_labels.pt
        meta.json                    {encoder_name, feature_dim, n_tasks}
"""

from __future__ import annotations

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# In-memory feature dataset
# ---------------------------------------------------------------------------

class FeatureTensorDataset(Dataset):
    """Simple (feature, label) dataset backed by in-memory or mmap tensors."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        assert len(features) == len(labels)
        self.features = features
        self.labels   = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Core cache function
# ---------------------------------------------------------------------------

def cache_features(
    encoder,
    raw_tasks:   List[Dict],
    cache_dir:   str,
    device:      str  = "cpu",
    batch_size:  int  = 256,
    force_redo:  bool = False,
) -> List[Dict]:
    """
    Extract and cache encoder features for all tasks.

    Args:
        encoder:    A frozen RootEncoder instance.
        raw_tasks:  List of task dicts with DataLoaders yielding (image, label).
        cache_dir:  Directory to write / read .pt files.
        device:     Device to run the encoder on.
        batch_size: Batch size for feature extraction (can be larger than training bs).
        force_redo: If True, ignore existing cache files and recompute.

    Returns:
        List of task dicts with the same schema as raw_tasks, but DataLoaders
        now yield (feature_tensor, label) pairs — same interface, just faster.
    """
    import pathlib

    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "meta.json")

    encoder_name = getattr(encoder, "__class__", type(encoder)).__name__
    feature_dim  = encoder.feature_dim

    # Write / verify meta
    if not force_redo and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("encoder_name") != encoder_name or meta.get("n_tasks") != len(raw_tasks):
            print(f"[feature_cache] Cache meta mismatch — recomputing.")
            force_redo = True
    else:
        meta = {"encoder_name": encoder_name, "feature_dim": feature_dim,
                "n_tasks": len(raw_tasks)}
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    encoder = encoder.to(device).eval()
    new_tasks = []

    for task in raw_tasks:
        t      = task["task_id"]
        splits = {"train": task["train"], "val": task["val"], "test": task["test"]}
        loaders = {}

        for split_name, loader in splits.items():
            feat_path  = os.path.join(cache_dir, f"task_{t}_{split_name}_features.pt")
            label_path = os.path.join(cache_dir, f"task_{t}_{split_name}_labels.pt")

            if not force_redo and os.path.exists(feat_path) and os.path.exists(label_path):
                features = torch.load(feat_path, map_location="cpu", weights_only=True)
                labels   = torch.load(label_path, map_location="cpu", weights_only=True)
            else:
                print(f"  [cache] task {t} / {split_name} ...", end=" ", flush=True)
                all_feats, all_labels = [], []
                # Use a fresh loader at the requested batch_size (larger = faster)
                fast_loader = _make_fast_loader(loader.dataset, batch_size)
                with torch.no_grad():
                    for x, y in fast_loader:
                        feats = encoder(x.to(device)).cpu()
                        all_feats.append(feats)
                        all_labels.append(y.cpu())
                features = torch.cat(all_feats, dim=0)   # (N, D)
                labels   = torch.cat(all_labels, dim=0)  # (N,)
                torch.save(features, feat_path)
                torch.save(labels,   label_path)
                print(f"{len(features)} samples, shape {tuple(features.shape)}")

            ds = FeatureTensorDataset(features, labels)
            # Preserve original loader's shuffle setting
            shuffle = (split_name == "train")
            loaders[split_name] = DataLoader(
                ds, batch_size=loader.batch_size or 128,
                shuffle=shuffle, num_workers=0, pin_memory=False,
            )

        new_task = dict(task)
        new_task["train"] = loaders["train"]
        new_task["val"]   = loaders["val"]
        new_task["test"]  = loaders["test"]
        new_task["feature_dim"] = feature_dim
        new_tasks.append(new_task)

    return new_tasks


def _make_fast_loader(dataset, batch_size: int) -> DataLoader:
    """Temporary loader at higher batch_size for feature extraction."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)


# ---------------------------------------------------------------------------
# Normalisation helper (DINO doesn't renormalise; CIFAR loaders use CIFAR stats)
# ---------------------------------------------------------------------------

def get_imagenet_transform():
    """
    Returns a torchvision transform that converts CIFAR-normalised tensors
    to ImageNet-normalised tensors suitable for DINOv2 / CLIP / ResNet-50.

    Not needed if using feature caching (encoder handles the normalisation
    internally via the raw images). Provided here for completeness.
    """
    try:
        import torchvision.transforms as T
    except ImportError:
        raise ImportError("torchvision required.")
    return T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
