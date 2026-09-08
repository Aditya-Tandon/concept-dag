#!/usr/bin/env python
"""
module_family_ablation.py — H6: does SVHN need patch tokens, attention, or a trained backbone?

Implements the single-task ablation pre-registered in
``Central Library/Hypotheses/concept-dag/kan-gated-growth/module-family-ablation.md``.
Six root families are trained on ONE task at a time (no DAG, no gate, no routing —
nothing in this script touches the DAG code path):

    A   mlp_cls            DINOv2 CLS (384)            → 2-layer MLP      (the current root)
    B   mlp_cls_meanpool   CLS ⊕ mean patch tokens     → the same MLP     (H6a: information)
    C0  proj_uniform_pool  tokens, uniform pooling     → MLP              (control for C/D)
    C   attn_pool          tokens, learned-query pool  → MLP              (H6b: attention)
    D   self_attn          tokens, 1 self-attn layer   → pool → MLP       (H6b: attention)
    E   cnn                raw 32×32×3 pixels          → SmallCNN + MLP   (H6c: substrate)

(`mlp_meanpool`, patch tokens only, is the superseded first-pass arm B — still runnable.)

Usage
-----
    # full data, pool 2, 3 seeds
    python scripts/module_family_ablation.py \\
        --dataset svhn --n_train full --seeds 42 43 44 --epochs 40 \\
        --backbone dinov2_vits14 --token_pool 2 \\
        --data_root /tmp/data --cache_dir /tmp/dino_tok --out results/h6 --device cuda

    # @400, the token_pool sweep, 5 seeds
    python scripts/module_family_ablation.py \\
        --dataset svhn --n_train 400 --seeds 42 43 44 45 46 --epochs 40 \\
        --backbone dinov2_vits14 --token_pool 1 2 4 \\
        --data_root /tmp/data --cache_dir /tmp/dino_tok --out results/h6 --device cuda

    # CPU smoke run, no encoder download, no torch.hub (what the tests use):
    python scripts/module_family_ablation.py --dataset synthetic --backbone stub \\
        --n_train 120 --epochs 1 --n_val 40 --n_test 60 --token_pool 4 --out /tmp/h6

Design decisions (recorded because the spec left them open)
-----------------------------------------------------------
* **One token cache per (dataset, token_pool), shared by every token arm.** The backbone
  runs ONCE over the whole train split and the whole test split of a dataset at each
  requested `token_pool`; every family, regime and seed then indexes subsets out of that one
  cache. Arm A's CLS view is index 0 of the same tensor (`--token_pool` is a treatment for
  arms C0/C/D and cannot change arm A).
* **`--token_pool` takes a list.** Results are written per pool value to
  `<out>/<dataset>_<n_train>_pool<p>/`. Image-input families (arm E) do not read tokens, so
  they are trained once — under the first pool value, with `token_pool: null` recorded —
  rather than re-run identically for every pool.
* **One loader per family.** Arms A–D read the cached tokens (a lazy index-into-the-cache
  dataset, so no per-family copy of a multi-GB tensor); arm E reads the raw images through
  the loaders' standard stream transform (RGB, 32×32, ImageNet-normalised — MNIST-family
  sets are 3-channel replicated by `.convert("RGB")`, as everywhere else in the repo).
  All families see the SAME image indices for a given (regime, seed).
* **Validation split.** 2000 images drawn from the TRAIN pool *in addition to* `n_train`
  (the CTrL convention: `val_frac` is on top of, not carved out of, the training images).
  `--n_train full` = every remaining training image after that 2000.
* **Test set.** `full` → the entire test split; `400` → a 5000-image subset, mirroring
  `make_ctrl_stream(n_test=5000)`. Both drawn with the run's seed.
* **Pairing.** Train / validation / test indices are drawn ONCE per (regime, seed) and
  reused by every family and every pool value, so all arms are compared on identical images;
  the SHA-1 of those index arrays is recorded in each JSON as `split_hash`.
* **Training recipe** is one shared loop for every family, mirroring
  `exp3_growing_dag.train_node` exactly: AdamW lr 1e-3,
  weight_decay 1e-4, cosine schedule over `--epochs`, grad-norm clip 1.0, safe
  cross-entropy, plus `orth_weight * orth_loss()` for the families that expose one. A root
  `ConceptModule` has NO aggregator, so that orth term is identically 0 for every family
  here — it is kept only so the recipe is literally the DAG's.
* **Memory.** A 65-token ViT-S cache is ~2 kB/image on disk (float16) and ~4× that in
  memory as float32: SVHN full ≈ 7 GB train + 2.6 GB test. Use `--token_pool 4`
  (17 tokens) if the pod is tight.

Determinism: each (family, seed) run seeds torch + numpy from `seed`; the data subsample
is drawn inside a `torch.random.fork_rng()` from `numpy.random.default_rng(seed)`, so the
splits are identical across families and unaffected by model initialisation order.

Output
------
    <out>/<dataset>_<n_train>_pool<p>/family_<family>_seed_<seed>.json   one run
    <out>/<dataset>_<n_train>_pool<p>/summary.json                       this regime, aggregated
    <out>/summary.json                                                   everything under <out>

Each run JSON records test_acc, val_acc, val_bits AND test_bits, params, wall_s, the
per-epoch validation curve, the split sizes and `split_hash`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from concept_dag.data.feature_cache import cache_split_features            # noqa: E402
from concept_dag.data.loaders import _build_stream_transform, _load_raw_dataset  # noqa: E402
from concept_dag.models.baselines import LinearHead                        # noqa: E402
from concept_dag.modules.token_roots import (                              # noqa: E402
    ARM_OF, FAMILIES, FAMILY_INPUT, PREREGISTERED, build_root, count_params,
)
from concept_dag.utils.metrics import accuracy, safe_cross_entropy         # noqa: E402

_LOG2 = math.log(2.0)
N_CLASSES = 10


# ---------------------------------------------------------------------------
# Stub encoder — CPU/test substitute for DINOv2 (no torch.hub, no download)
# ---------------------------------------------------------------------------

class StubTokenEncoder(nn.Module):
    """Deterministic fake ViT: random fixed projection of image patches → (B, 1 + T, 384).

    Exists so `--backbone stub` exercises the whole pipeline (cache → per-family loaders →
    train → JSON) on a laptop with no hub access. It is class-informative (it is a linear
    map of the pixels) but carries no pretrained knowledge; never use it for a result.
    """

    feature_dim = 384

    def __init__(self, device: str = "cpu", token_pool: int = 1, grid: int = 8,
                 seed: int = 0):
        super().__init__()
        if grid % token_pool != 0:
            raise ValueError(f"token_pool={token_pool} does not divide the {grid}×{grid} grid.")
        self.token_pool = int(token_pool)
        self.grid = int(grid)
        g = torch.Generator().manual_seed(seed)
        # Fixed random projection from a patch's pixels to 384 dims.
        self.register_buffer("proj", torch.randn(3, self.feature_dim, generator=g) * 0.5)
        self.eval()

    @property
    def n_tokens(self) -> int:
        side = self.grid // self.token_pool
        return 1 + side * side

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        side = self.grid // self.token_pool
        pooled = F.adaptive_avg_pool2d(x, side)                 # (B, 3, side, side)
        tok = pooled.flatten(2).transpose(1, 2) @ self.proj     # (B, side*side, 384)
        cls = (x.mean(dim=(2, 3)) @ self.proj).unsqueeze(1)     # (B, 1, 384)
        return torch.cat([cls, torch.tanh(tok)], dim=1)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class _SyntheticImages(Dataset):
    """Class-separable 32×32×3 images, for `--dataset synthetic` (tests only).

    `centers_seed` is shared by the train and test splits — they must be draws from the
    SAME class-conditional distribution, or test accuracy is chance by construction —
    while `sample_seed` differs so the splits hold different images.
    """

    def __init__(self, n: int, sample_seed: int, centers_seed: int = 1234,
                 n_classes: int = N_CLASSES):
        gc = torch.Generator().manual_seed(centers_seed)
        centers = torch.randn(n_classes, 3, 4, 4, generator=gc) * 2.0
        g = torch.Generator().manual_seed(sample_seed)
        self.y = torch.randint(0, n_classes, (n,), generator=g)
        base = torch.nn.functional.interpolate(
            centers[self.y], size=(32, 32), mode="nearest")
        self.x = base + 0.5 * torch.randn(n, 3, 32, 32, generator=g)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class _CachedTokens(Dataset):
    """Lazy view: sample i is `features[idx[i]]` (optionally only the CLS slot).

    Indexing lazily instead of materialising `features[idx]` keeps a multi-GB token
    cache single-copy across the five families.
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor,
                 idx: np.ndarray, cls_only: bool = False):
        self.features = features
        self.labels = labels
        self.idx = np.asarray(idx)
        self.cls_only = cls_only

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = self.features[j]
        return (x[0] if self.cls_only else x), self.labels[j]


def load_raw_splits(dataset: str, data_root: str, download: bool,
                    synthetic_n: int, seed: int = 0):
    """Return (train_dataset, test_dataset) of transformed 3×32×32 images."""
    if dataset == "synthetic":
        return (_SyntheticImages(synthetic_n, sample_seed=seed),
                _SyntheticImages(max(synthetic_n // 2, 20), sample_seed=seed + 10_000))
    transform = _build_stream_transform(32)
    train = _load_raw_dataset(dataset, data_root, True, transform, download)
    test = _load_raw_dataset(dataset, data_root, False, transform, download)
    return train, test


def build_splits(n_train_pool: int, n_test_pool: int, regime: str, seed: int,
                 n_val: int, n_test_small: int) -> Dict[str, np.ndarray]:
    """Index sets shared by every family at this (regime, seed).

    The 2000-image validation split comes out of the train pool IN ADDITION to n_train.
    `full` keeps every remaining train image and the whole test split; a finite n_train
    takes a `n_test_small` test subset (CTrL's n_test=5000).
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_train_pool)
    if n_val >= n_train_pool:
        raise ValueError(f"n_val={n_val} exceeds the {n_train_pool}-image train pool.")
    val_idx = order[:n_val]
    rest = order[n_val:]
    if regime == "full":
        train_idx = rest
    else:
        n_train = int(regime)
        if n_train > len(rest):
            raise ValueError(
                f"n_train={n_train} + n_val={n_val} exceeds the {n_train_pool}-image train pool.")
        train_idx = rest[:n_train]

    if regime == "full":
        test_idx = np.arange(n_test_pool)
    else:
        test_order = rng.permutation(n_test_pool)
        test_idx = test_order[: min(n_test_small, n_test_pool)]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def split_hash(splits: Dict[str, np.ndarray]) -> str:
    """Fingerprint of the three index arrays — the paired-design check.

    Every family and every token_pool at a given (regime, seed) must report the same hash;
    if two arms ever disagree, they were not compared on the same images.
    """
    h = hashlib.sha1()
    for name in ("train", "val", "test"):
        h.update(name.encode())
        h.update(np.ascontiguousarray(splits[name], dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Token cache — one per (dataset, backbone, token_pool), shared by every token arm
# ---------------------------------------------------------------------------

def build_encoder_for(backbone: str, device: str, token_pool: int):
    if backbone == "stub":
        return StubTokenEncoder(device=device, token_pool=token_pool).to(device)
    from concept_dag.models.root_encoder import build_encoder
    return build_encoder(backbone, device=device, return_tokens=True, token_pool=token_pool)


def get_token_cache(dataset: str, backbone: str, token_pool: int, cache_dir: str,
                    train_ds, test_ds, device: str, batch_size: int,
                    force_redo: bool = False) -> Dict:
    """Encode the whole train + test splits once; return float32 token tensors."""
    encoder = build_encoder_for(backbone, device, token_pool)
    n_tokens = int(encoder.n_tokens)
    cdir = os.path.join(cache_dir, f"{dataset}_{backbone}_p{token_pool}_tok{n_tokens - 1}")
    os.makedirs(cdir, exist_ok=True)

    meta = {"dataset": dataset, "backbone": backbone, "token_pool": token_pool,
            "n_tokens": n_tokens, "feature_dim": int(encoder.feature_dim),
            "n_train_pool": len(train_ds), "n_test_pool": len(test_ds), "tokens": True}
    meta_path = os.path.join(cdir, "meta.json")
    if os.path.exists(meta_path) and not force_redo:
        old = json.load(open(meta_path))
        if any(old.get(k) != v for k, v in meta.items()):
            print(f"[h6] token cache meta mismatch in {cdir} — recomputing.")
            force_redo = True
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    out = {"n_tokens": n_tokens, "feature_dim": int(encoder.feature_dim), "dir": cdir}
    for split, ds in (("train", train_ds), ("test", test_ds)):
        feats, labels = cache_split_features(
            encoder, ds, cdir, f"{dataset}_{split}", device=device,
            batch_size=batch_size, force_redo=force_redo, tokens=True,
        )
        if feats.shape[1] != n_tokens:
            raise RuntimeError(
                f"cached {split} tokens have sequence length {feats.shape[1]}, expected {n_tokens}.")
        out[f"{split}_features"], out[f"{split}_labels"] = feats, labels
    del encoder
    return out


# ---------------------------------------------------------------------------
# Per-family loaders
# ---------------------------------------------------------------------------

def make_loaders(family: str, splits: Dict[str, np.ndarray], cache: Optional[Dict],
                 train_ds, test_ds, batch_size: int) -> Dict[str, DataLoader]:
    kind = FAMILY_INPUT[family]
    loaders = {}
    for split in ("train", "val", "test"):
        idx = splits[split]
        pool = "test" if split == "test" else "train"
        if kind == "image":
            ds = Subset(test_ds if pool == "test" else train_ds, [int(i) for i in idx])
        else:
            if cache is None:
                raise RuntimeError(f"family '{family}' needs a token cache.")
            ds = _CachedTokens(cache[f"{pool}_features"], cache[f"{pool}_labels"],
                               idx, cls_only=(kind == "cls"))
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    num_workers=0, pin_memory=False)
    return loaders


# ---------------------------------------------------------------------------
# Train / eval — `train_node`'s recipe, single task, no DAG
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_split(root: nn.Module, head: nn.Module, loader, device: str) -> Dict[str, float]:
    """Accuracy and mean held-out code length in bits (the gate's currency)."""
    root.eval()
    head.eval()
    correct, total, bits_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = head(root(x))
        correct += (logits.argmax(-1) == y).sum().item()
        bits_sum += float(safe_cross_entropy(logits, y, reduction="sum").item()) / _LOG2
        total += y.size(0)
    return {"acc": correct / max(total, 1), "bits": bits_sum / max(total, 1), "n": total}


def train_family(root: nn.Module, head: nn.Module, loaders, epochs: int, lr: float,
                 device: str, orth_weight: float, log_every: int, name: str) -> Dict:
    root.to(device)
    head.to(device)
    params = list(root.trainable_parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    has_orth = hasattr(root, "orth_loss")

    history = {"loss": [], "accuracy": [], "val_acc": [], "val_bits": []}
    for epoch in range(1, epochs + 1):
        root.train()
        head.train()
        ep_loss, ep_acc, n = 0.0, 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            logits = head(root(x))
            loss = safe_cross_entropy(logits, y)
            if has_orth:
                loss = loss + orth_weight * root.orth_loss().to(loss.device)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += loss.item()
            ep_acc += accuracy(logits, y)
            n += 1
        sched.step()
        val = eval_split(root, head, loaders["val"], device)
        history["loss"].append(ep_loss / max(n, 1))
        history["accuracy"].append(ep_acc / max(n, 1))
        history["val_acc"].append(val["acc"])
        history["val_bits"].append(val["bits"])
        if log_every and (epoch % log_every == 0 or epoch == 1 or epoch == epochs):
            print(f"    [{name} | ep {epoch:3d}/{epochs}] loss={history['loss'][-1]:.4f} "
                  f"acc={history['accuracy'][-1]:.3f} val_acc={val['acc']:.3f} "
                  f"val_bits={val['bits']:.4f}")
    root.eval()
    head.eval()
    return history


def run_one(family: str, seed: int, args, cache: Optional[Dict], train_ds, test_ds,
            splits: Dict[str, np.ndarray], token_pool: Optional[int]) -> Dict:
    """Train one (family, seed) on the pre-drawn `splits` shared by every arm."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    feature_dim = cache["feature_dim"] if cache is not None else 3
    root = build_root(family, feature_dim=feature_dim, concept_dim=args.concept_dim)
    head = LinearHead(root.out_dim, N_CLASSES)
    loaders = make_loaders(family, splits, cache, train_ds, test_ds, args.batch_size)

    name = f"{args.dataset}_{args.n_train}/{family}/seed{seed}"
    if token_pool is not None:
        name += f"/pool{token_pool}"
    print(f"\n[h6] {name}: params={count_params(root) + count_params(head)} "
          f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    t0 = time.time()
    history = train_family(root, head, loaders, args.epochs, args.lr, args.device,
                           args.orth_weight, args.log_every, name)
    val = eval_split(root, head, loaders["val"], args.device)
    test = eval_split(root, head, loaders["test"], args.device)
    wall = time.time() - t0

    return {
        "dataset": args.dataset, "n_train": args.n_train, "family": family, "seed": seed,
        "test_acc": test["acc"], "val_acc": val["acc"], "val_bits": val["bits"],
        "test_bits": test["bits"],
        "params": count_params(root), "params_with_head": count_params(root) + count_params(head),
        "wall_s": wall, "epochs": args.epochs, "lr": args.lr,
        "backbone": args.backbone, "token_pool": token_pool,
        "n_tokens": (cache["n_tokens"] if cache is not None else None),
        "input_kind": FAMILY_INPUT[family], "arm": ARM_OF.get(family),
        "split_hash": split_hash(splits),
        "n_train_images": len(splits["train"]), "n_val_images": len(splits["val"]),
        "n_test_images": len(splits["test"]),
        "device": args.device, "concept_dim": args.concept_dim,
        "val_curve": [{"epoch": i + 1, "val_acc": a, "val_bits": b,
                       "train_loss": l, "train_acc": ta}
                      for i, (a, b, l, ta) in enumerate(zip(
                          history["val_acc"], history["val_bits"],
                          history["loss"], history["accuracy"]))],
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean_se_sd(values: List[float]):
    n = len(values)
    if n == 0:
        return None, None, None
    mean = float(np.mean(values))
    if n == 1:
        return mean, 0.0, 0.0
    sd = float(np.std(values, ddof=1))
    return mean, sd / math.sqrt(n), sd


def summarise(runs: List[Dict]) -> Dict:
    """Aggregate per-run records into {dataset_regime: {family: stats}}."""
    groups: Dict[str, Dict[str, List[Dict]]] = {}
    for r in runs:
        pool = r.get("token_pool")
        key = f"{r['dataset']}_{r['n_train']}" + ("" if pool is None else f"_pool{pool}")
        groups.setdefault(key, {}).setdefault(r["family"], []).append(r)
    out = {}
    for key, fams in sorted(groups.items()):
        out[key] = {}
        for fam, rs in sorted(fams.items()):
            accs = [r["test_acc"] for r in rs]
            bits = [r["val_bits"] for r in rs]
            tbits = [r.get("test_bits") for r in rs]
            vaccs = [r["val_acc"] for r in rs]
            acc_mean, acc_se, acc_sd = _mean_se_sd(accs)
            bits_mean, bits_se, bits_sd = _mean_se_sd(bits)
            out[key][fam] = {
                "n_seeds": len(rs), "seeds": sorted(r["seed"] for r in rs),
                "test_acc_mean": acc_mean, "test_acc_se": acc_se, "test_acc_sd": acc_sd,
                "val_bits_mean": bits_mean, "val_bits_se": bits_se, "val_bits_sd": bits_sd,
                "val_acc_mean": _mean_se_sd(vaccs)[0],
                "test_bits_mean": _mean_se_sd([b for b in tbits if b is not None])[0],
                "params": rs[0]["params"],
                "wall_s_mean": float(np.mean([r["wall_s"] for r in rs])),
                "test_accs": accs, "val_bits_all": bits,
            }
    return out


def collect_runs(out_dir: str) -> List[Dict]:
    """Read every per-run JSON under `out_dir` (across regimes and past invocations)."""
    runs = []
    for root, _dirs, files in os.walk(out_dir):
        for fn in sorted(files):
            if fn.startswith("family_") and fn.endswith(".json"):
                with open(os.path.join(root, fn)) as f:
                    runs.append(json.load(f))
    return runs


def print_table(summary: Dict) -> None:
    for key, fams in summary.items():
        print(f"\n=== {key} ===")
        print(f"{'family':<20}{'n':>3}  {'test acc':>16}  {'val bits':>10}  "
              f"{'test bits':>10}  {'params':>10}")
        for fam, st in fams.items():
            acc = f"{st['test_acc_mean']:.4f} ± {st['test_acc_se']:.4f}"
            tb = st.get("test_bits_mean")
            print(f"{fam:<20}{st['n_seeds']:>3}  {acc:>16}  {st['val_bits_mean']:>10.4f}  "
                  f"{(float('nan') if tb is None else tb):>10.4f}  {st['params']:>10,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="svhn",
                    choices=["svhn", "fashion", "kmnist", "mnist", "cifar10", "synthetic"],
                    help="kmnist is the pre-registered control; "
                         "synthetic = in-memory images for CPU smoke tests")
    ap.add_argument("--n_train", default="full",
                    help="'full' (all train images minus the val split) or an integer (e.g. 400)")
    ap.add_argument("--families", nargs="+", default=list(PREREGISTERED),
                    choices=list(FAMILIES),
                    help="default = the six pre-registered arms A, B, C0, C, D, E")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                    help="the note asks for 3 seeds at full data, 5 at @400")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--backbone", default="dinov2_vits14",
                    help="dinov2_vits14 (pod) | stub (CPU tests) | any build_encoder name")
    ap.add_argument("--token_pool", nargs="+", type=int, default=[2],
                    help="pool the 16x16 DINO patch grid by these factors "
                         "(1 -> 256 tokens, 2 -> 64, 4 -> 16); one cache and one results "
                         "directory per value")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--cache_dir", default="./data/h6_token_cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--encode_batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--concept_dim", type=int, default=128)
    ap.add_argument("--orth_weight", type=float, default=0.01)
    ap.add_argument("--n_val", type=int, default=2000,
                    help="validation images drawn from the train pool IN ADDITION to n_train")
    ap.add_argument("--n_test", type=int, default=5000,
                    help="test subset size when n_train is finite; 'full' uses the whole test set")
    ap.add_argument("--synthetic_n", type=int, default=600,
                    help="train-pool size for --dataset synthetic")
    ap.add_argument("--download", action="store_true",
                    help="allow torchvision to download the dataset")
    ap.add_argument("--force_redo", action="store_true", help="recompute the token cache")
    ap.add_argument("--log_every", type=int, default=5)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n_train != "full":
        try:
            int(args.n_train)
        except ValueError:
            raise SystemExit(f"--n_train must be 'full' or an integer, got {args.n_train!r}")

    train_ds, test_ds = load_raw_splits(args.dataset, args.data_root, args.download,
                                        args.synthetic_n)
    n_train_pool, n_test_pool = len(train_ds), len(test_ds)
    print(f"[h6] {args.dataset}: train pool {n_train_pool}, test pool {n_test_pool}")

    # Paired design: one index draw per seed, reused by every family and every pool value.
    splits_by_seed = {}
    for seed in args.seeds:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            splits_by_seed[seed] = build_splits(n_train_pool, n_test_pool, args.n_train,
                                                seed, args.n_val, args.n_test)
        print(f"[h6] seed {seed}: train={len(splits_by_seed[seed]['train'])} "
              f"val={len(splits_by_seed[seed]['val'])} test={len(splits_by_seed[seed]['test'])} "
              f"split_hash={split_hash(splits_by_seed[seed])}")

    token_families = [f for f in args.families if FAMILY_INPUT[f] != "image"]
    image_families = [f for f in args.families if FAMILY_INPUT[f] == "image"]
    written_dirs = []

    for pool_i, pool in enumerate(args.token_pool):
        regime_dir = os.path.join(args.out, f"{args.dataset}_{args.n_train}_pool{pool}")
        os.makedirs(regime_dir, exist_ok=True)
        written_dirs.append(regime_dir)

        cache = None
        if token_families:
            cache = get_token_cache(args.dataset, args.backbone, pool, args.cache_dir,
                                    train_ds, test_ds, args.device,
                                    args.encode_batch_size, force_redo=args.force_redo)
            print(f"[h6] token cache: {cache['dir']}  "
                  f"({cache['n_tokens']} tokens x {cache['feature_dim']}d)")

        # Image arms read no tokens: train them once, under the first pool value only.
        families = list(token_families) + (image_families if pool_i == 0 else [])
        for family in families:
            is_image = FAMILY_INPUT[family] == "image"
            for seed in args.seeds:
                record = run_one(family, seed, args, cache, train_ds, test_ds,
                                 splits_by_seed[seed], None if is_image else pool)
                path = os.path.join(regime_dir, f"family_{family}_seed_{seed}.json")
                with open(path, "w") as f:
                    json.dump(record, f, indent=2)
                print(f"[h6] {family} seed {seed}: test_acc={record['test_acc']:.4f} "
                      f"val_bits={record['val_bits']:.4f} test_bits={record['test_bits']:.4f} "
                      f"wall={record['wall_s']:.1f}s -> {path}")

        with open(os.path.join(regime_dir, "summary.json"), "w") as f:
            json.dump(summarise(collect_runs(regime_dir)), f, indent=2)
        del cache

    full_summary = summarise(collect_runs(args.out))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(full_summary, f, indent=2)
    print_table(full_summary)
    print(f"\nWrote {len(written_dirs)} regime summaries and {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
