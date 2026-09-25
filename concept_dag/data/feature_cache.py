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
        meta.json                    {encoder_name, feature_dim, n_tasks, seed,
                                      tokens, n_tokens}

Token caches (`tokens=True`)
----------------------------
With an encoder configured to return patch tokens (see `DINOv2Encoder(return_tokens=True,
token_pool=P)`), the cached tensors are Tensor[N, 1 + T, D] — index 0 of the sequence is the
CLS token, so a CLS-only view is just `features[:, 0]`. They are stored as **float16 on disk**
(halving a 65-token ViT-S cache to ~2 kB/image) and returned as float32 in memory; the values
handed to callers are the round-tripped ones, so an in-memory run and a cache-hit run see
bit-identical inputs. The directory gets a `_tok{T}` suffix so a token cache can never collide
with the CLS-only cache of the same encoder. `FeatureTensorDataset` returns the (1 + T, D)
tensor as `x`, so every DataLoader-consuming code path is unchanged.
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
    """Simple (feature, label) dataset backed by in-memory or mmap tensors.

    A float16 backing tensor is cast to float32 on access. Token caches are held
    half-precision in RAM (a 65-token ViT-S stream of 300k images is 7.5 GB half vs
    15 GB single) and the cast is exact — the values already round-tripped through
    float16 on disk — so every consumer sees the same float32 numbers it would have
    seen from a float32 buffer. CLS-only caches are float32 and pass through untouched.
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        assert len(features) == len(labels)
        self.features = features
        self.labels   = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        if x.dtype == torch.float16:
            x = x.float()
        return x, self.labels[idx]


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
    seed:        Optional[int] = None,
    tokens:      bool = False,
    token_pool:  int  = 1,
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
        tokens:     If True the encoder is expected to return (B, 1 + T, D) token
                    sequences; the cache is written as float16, read back as float32,
                    and `cache_dir` gains a `_tok{T}` suffix. The encoder must expose
                    `n_tokens`. Default False = the CLS-only cache, unchanged.
        token_pool: Recorded in meta for provenance only — the pooling itself is the
                    encoder's job (`DINOv2Encoder(token_pool=...)`).

    Returns:
        List of task dicts with the same schema as raw_tasks, but DataLoaders
        now yield (feature_tensor, label) pairs — same interface, just faster.
    """
    if tokens:
        n_tokens = getattr(encoder, "n_tokens", None)
        if not isinstance(n_tokens, int) or n_tokens < 2:
            raise ValueError(
                "cache_features(tokens=True) needs an encoder exposing n_tokens >= 2 "
                f"(the 1 + T sequence length); got {n_tokens!r}."
            )
        cache_dir = f"{cache_dir}_tok{n_tokens - 1}"
    else:
        n_tokens = None

    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "meta.json")

    encoder_name = getattr(encoder, "__class__", type(encoder)).__name__
    feature_dim  = encoder.feature_dim

    # Verify / (re)write meta. `seed` is included because it determines the task
    # class splits: reusing a cache built under a different seed would silently
    # serve stale features for the wrong classes.
    current_meta = {"encoder_name": encoder_name, "feature_dim": feature_dim,
                    "n_tasks": len(raw_tasks), "seed": seed}
    if tokens:
        current_meta.update({"tokens": True, "n_tokens": n_tokens,
                             "token_pool": int(token_pool)})
    if not force_redo and os.path.exists(meta_path):
        with open(meta_path) as f:
            existing = json.load(f)
        # `.get(..., default)` on the token keys keeps pre-token caches valid: an old
        # meta with no "tokens" key reads as a CLS-only cache, which is what it is.
        if (existing.get("encoder_name") != encoder_name
                or existing.get("n_tasks") != len(raw_tasks)
                or existing.get("seed") != seed
                or bool(existing.get("tokens", False)) != bool(tokens)
                or existing.get("n_tokens", None) != n_tokens):
            print(f"[feature_cache] Cache meta mismatch "
                  f"(encoder/n_tasks/seed/tokens) — recomputing.")
            force_redo = True
    # Always (re)write meta so a mismatch-triggered recompute updates the on-disk
    # key. Otherwise the stale meta would keep mismatching and every subsequent
    # run would recompute from scratch, defeating the cache.
    with open(meta_path, "w") as f:
        json.dump(current_meta, f, indent=2)

    encoder = encoder.to(device).eval()
    new_tasks = []

    for task in raw_tasks:
        t      = task["task_id"]
        splits = {"train": task["train"], "val": task["val"], "test": task["test"]}
        loaders = {}

        for split_name, loader in splits.items():
            feat_path  = os.path.join(cache_dir, f"task_{t}_{split_name}_features.pt")
            label_path = os.path.join(cache_dir, f"task_{t}_{split_name}_labels.pt")

            features, labels = _load_or_encode(
                encoder, loader.dataset, feat_path, label_path,
                device=device, batch_size=batch_size, force_redo=force_redo,
                tokens=tokens, label=f"task {t} / {split_name}",
                # Token features stay float16 in RAM; FeatureTensorDataset casts per item.
                as_float=not tokens,
            )

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
        if tokens:
            new_task["n_tokens"] = n_tokens
        new_tasks.append(new_task)

    return new_tasks


def cache_split_features(
    encoder,
    dataset,
    cache_dir:  str,
    key:        str,
    device:     str  = "cpu",
    batch_size: int  = 256,
    force_redo: bool = False,
    tokens:     bool = False,
):
    """Cache ONE split's features and return them as (features, labels) tensors.

    The single-split entry point behind `cache_features`, exposed for single-task
    studies (e.g. the module-family ablation) that encode a whole dataset split once
    and then index subsets out of it, rather than materialising a stream of task dicts.

    Files are `<cache_dir>/<key>_features.pt` and `<cache_dir>/<key>_labels.pt`.
    Returns float32 features (round-tripped through float16 on disk when `tokens=True`).
    The caller owns `cache_dir` (including any `_tok{T}` suffix) and its meta.
    """
    os.makedirs(cache_dir, exist_ok=True)
    return _load_or_encode(
        encoder, dataset,
        os.path.join(cache_dir, f"{key}_features.pt"),
        os.path.join(cache_dir, f"{key}_labels.pt"),
        device=device, batch_size=batch_size, force_redo=force_redo,
        tokens=tokens, label=key,
    )


def _load_or_encode(encoder, dataset, feat_path: str, label_path: str, *,
                    device: str, batch_size: int, force_redo: bool,
                    tokens: bool, label: str, as_float: bool = True):
    """Read a cached (features, labels) pair, or run the encoder over `dataset` and write it.

    Token caches are stored float16; the freshly-computed tensor is put through the same
    half() cast before being returned, so a cache-miss run and a cache-hit run hand the
    caller bit-identical values. ``as_float`` (default True) casts them back to float32 on
    return; ``as_float=False`` leaves them float16 for the caller to cast lazily — used by
    `cache_features`, whose `FeatureTensorDataset` casts per item, halving resident RAM
    without changing a single value.
    """
    if not force_redo and os.path.exists(feat_path) and os.path.exists(label_path):
        features = torch.load(feat_path, map_location="cpu", weights_only=True)
        labels   = torch.load(label_path, map_location="cpu", weights_only=True)
    else:
        print(f"  [cache] {label} ...", end=" ", flush=True)
        all_feats, all_labels = [], []
        # Use a fresh loader at the requested batch_size (larger = faster)
        fast_loader = _make_fast_loader(dataset, batch_size)
        with torch.no_grad():
            for x, y in fast_loader:
                feats = encoder(x.to(device)).cpu()
                all_feats.append(feats.half() if tokens else feats)
                all_labels.append(y.cpu())
        features = torch.cat(all_feats, dim=0)   # (N, D) or (N, 1 + T, D) float16
        labels   = torch.cat(all_labels, dim=0)  # (N,)
        torch.save(features, feat_path)
        torch.save(labels,   label_path)
        print(f"{len(features)} samples, shape {tuple(features.shape)}")

    if tokens:
        if features.ndim != 3:
            raise ValueError(
                f"token cache {feat_path} has shape {tuple(features.shape)}; expected (N, 1 + T, D)."
            )
        if as_float:
            features = features.float()
    return features, labels


def _make_fast_loader(dataset, batch_size: int) -> DataLoader:
    """Temporary loader at higher batch_size for feature extraction."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)
