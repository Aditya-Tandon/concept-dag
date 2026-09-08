#!/usr/bin/env python
"""
module_family_probe_noise.py — H6 gate M5: how noisy is each root family *inside the gate*?

The ablation (`module_family_ablation.py`) asks which family is more accurate. This script
asks the question the gate actually cares about: if the Kan gate's raw-root grow probe were
built from family F instead of the CLS-MLP, would its held-out code length be a noisier
statistic? A family that wins 0.02 accuracy but doubles the probe's split-to-split SD makes
every grow/reuse decision worse, not better.

Protocol (mirrors `concept_dag.training.kan_gate._held_out_codelength`, which is what the
probe calls):

  * take the SAME 400 gate samples the ablation trains on at `--n_train 400` (same seed,
    same `build_splits`, so the probe and the accuracy run are paired);
  * draw `--n_splits` random 280/120 partitions of those 400;
  * train root + linear head on the 280 for `--gate_epochs` epochs — AdamW `--lr`,
    weight_decay 1e-4, grad-norm clip 1.0, loss = mean held-out-currency bits (NOT the
    ablation's cosine-scheduled cross-entropy: this is the probe's recipe, not the root
    training recipe);
  * evaluate the 120 held-out images every epoch and keep the MINIMUM (the probe's
    min-over-epochs early stopping — the source of its selection optimism), recording the
    test-split bits of that best-held-out model;
  * report per family: mean/SD over splits of the min held-out bits, and the mean
    *optimism* = final-epoch held-out bits − min held-out bits.

M5 (in `eval_module_family.py`): a family whose split SD or optimism exceeds arm A's by
more than 1.5x is GATE-NOISIER.

Usage
-----
    python scripts/module_family_probe_noise.py --dataset svhn --n_train 400 \\
        --families mlp_cls mlp_cls_meanpool proj_uniform_pool attn_pool self_attn cnn \\
        --seeds 42 43 44 45 46 --n_splits 20 --gate_epochs 40 \\
        --backbone dinov2_vits14 --token_pool 2 --data_root /tmp/data \\
        --cache_dir /tmp/dino_tok --out results/h6_probe --device cuda

    # CPU smoke run (what the tests use)
    python scripts/module_family_probe_noise.py --dataset synthetic --backbone stub \\
        --n_train 60 --n_splits 2 --gate_epochs 1 --token_pool 4 --out /tmp/probe

Output
------
    <out>/<dataset>_<n_train>_pool<p>/probe_<family>_seed_<seed>.json
    <out>/probe_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import module_family_ablation as ablation                                  # noqa: E402
from concept_dag.models.baselines import LinearHead                        # noqa: E402
from concept_dag.modules.token_roots import (                              # noqa: E402
    ARM_OF, FAMILIES, FAMILY_INPUT, PREREGISTERED, build_root, count_params,
)
from concept_dag.training.kan_gate import classification_task              # noqa: E402

N_CLASSES = ablation.N_CLASSES


# ---------------------------------------------------------------------------
# One probe fit — the gate's `_held_out_codelength` recipe
# ---------------------------------------------------------------------------

@torch.no_grad()
def _bits(root, head, loader, spec, device: str) -> float:
    root.eval()
    head.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += float(spec.nll_bits(head(root(x)), y).sum().item())
        n += y.size(0)
    return total / max(n, 1)


def probe_split(family: str, seed: int, split_seed: int, loaders, epochs: int, lr: float,
                device: str, concept_dim: int, feature_dim: int) -> Dict:
    """Fit one 280/120 probe; return min / final held-out bits and the best model's test bits."""
    spec = classification_task(N_CLASSES)
    torch.manual_seed(seed * 1000 + split_seed)
    root = build_root(family, feature_dim=feature_dim, concept_dim=concept_dim).to(device)
    head = LinearHead(root.out_dim, N_CLASSES).to(device)
    params = list(root.trainable_parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    best_bits, best_epoch, best_test_bits = float("inf"), -1, None
    final_bits = float("nan")
    for epoch in range(1, epochs + 1):
        root.train()
        head.train()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            loss = spec.nll_bits(head(root(x)), y).mean()   # bits/sample, as the gate does
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        held = _bits(root, head, loaders["val"], spec, device)
        final_bits = held
        if held < best_bits:
            best_bits, best_epoch = held, epoch
            best_test_bits = _bits(root, head, loaders["test"], spec, device)
    return {"split_seed": split_seed, "min_held_out_bits": best_bits,
            "final_held_out_bits": final_bits, "best_epoch": best_epoch,
            "optimism": final_bits - best_bits, "test_bits_at_best": best_test_bits,
            "params": count_params(root)}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(v):
    return float(np.mean(v)) if len(v) else None


def _sd(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def summarise_family(splits: List[Dict]) -> Dict:
    mins = [s["min_held_out_bits"] for s in splits]
    opts = [s["optimism"] for s in splits]
    tests = [s["test_bits_at_best"] for s in splits if s["test_bits_at_best"] is not None]
    return {"n_splits": len(splits),
            "mean_min_held_out_bits": _mean(mins), "sd_min_held_out_bits": _sd(mins),
            "mean_optimism": _mean(opts), "sd_optimism": _sd(opts),
            "mean_test_bits_at_best": _mean(tests), "sd_test_bits_at_best": _sd(tests)}


def collect_probe_runs(out_dir: str) -> List[Dict]:
    runs = []
    for root, _dirs, files in os.walk(out_dir):
        for fn in sorted(files):
            if fn.startswith("probe_") and fn.endswith(".json") and fn != "probe_summary.json":
                with open(os.path.join(root, fn)) as f:
                    runs.append(json.load(f))
    return runs


def probe_summary(runs: List[Dict]) -> Dict:
    """{"<dataset>_<n_train>_pool<p>": {family: pooled stats over seeds x splits}}"""
    groups: Dict[str, Dict[str, List[Dict]]] = {}
    for r in runs:
        pool = r.get("token_pool")
        key = f"{r['dataset']}_{r['n_train']}" + ("" if pool is None else f"_pool{pool}")
        groups.setdefault(key, {}).setdefault(r["family"], []).append(r)
    out = {}
    for key, fams in sorted(groups.items()):
        out[key] = {}
        for fam, rs in sorted(fams.items()):
            all_splits = [s for r in rs for s in r["splits"]]
            agg = summarise_family(all_splits)
            agg.update({"n_seeds": len(rs), "seeds": sorted(r["seed"] for r in rs),
                        "arm": rs[0].get("arm"), "params": rs[0].get("params")})
            out[key][fam] = agg
    return out


def print_table(summary: Dict) -> None:
    for key in sorted(summary):
        print(f"\n=== {key} (gate probe noise) ===")
        print(f"{'family':<20}{'splits':>7}  {'min bits':>10}  {'SD':>8}  {'optimism':>9}  "
              f"{'test bits':>10}")
        for fam in sorted(summary[key]):
            a = summary[key][fam]
            print(f"{fam:<20}{a['n_splits']:>7}  {a['mean_min_held_out_bits']:>10.4f}  "
                  f"{a['sd_min_held_out_bits']:>8.4f}  {a['mean_optimism']:>9.4f}  "
                  f"{(a['mean_test_bits_at_best'] or float('nan')):>10.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="svhn",
                    choices=["svhn", "fashion", "kmnist", "mnist", "cifar10", "synthetic"])
    ap.add_argument("--n_train", default="400",
                    help="the gate regime: the probe uses these images as the gate samples")
    ap.add_argument("--families", nargs="+", default=list(PREREGISTERED),
                    choices=list(FAMILIES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--n_splits", type=int, default=20)
    ap.add_argument("--holdout", type=int, default=120,
                    help="held-out images per split (the gate's 280/120 of 400)")
    ap.add_argument("--gate_epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--token_pool", type=int, default=2)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--cache_dir", default="./data/h6_token_cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--encode_batch_size", type=int, default=256)
    ap.add_argument("--concept_dim", type=int, default=128)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=5000)
    ap.add_argument("--synthetic_n", type=int, default=600)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--force_redo", action="store_true")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    train_ds, test_ds = ablation.load_raw_splits(args.dataset, args.data_root,
                                                 args.download, args.synthetic_n)
    regime_dir = os.path.join(args.out, f"{args.dataset}_{args.n_train}_pool{args.token_pool}")
    os.makedirs(regime_dir, exist_ok=True)

    token_families = [f for f in args.families if FAMILY_INPUT[f] != "image"]
    cache = None
    if token_families:
        cache = ablation.get_token_cache(args.dataset, args.backbone, args.token_pool,
                                         args.cache_dir, train_ds, test_ds, args.device,
                                         args.encode_batch_size, force_redo=args.force_redo)
    feature_dim = cache["feature_dim"] if cache is not None else 3

    for seed in args.seeds:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            splits = ablation.build_splits(len(train_ds), len(test_ds), args.n_train,
                                           seed, args.n_val, args.n_test)
        gate_idx = np.asarray(splits["train"])
        if len(gate_idx) <= args.holdout:
            raise SystemExit(
                f"--holdout {args.holdout} needs fewer images than the {len(gate_idx)} "
                f"gate samples at --n_train {args.n_train}.")

        for family in args.families:
            is_image = FAMILY_INPUT[family] == "image"
            t0 = time.time()
            records = []
            for split_i in range(args.n_splits):
                rng = np.random.default_rng(seed * 100_000 + split_i)
                perm = rng.permutation(len(gate_idx))
                held = gate_idx[perm[: args.holdout]]
                fit = gate_idx[perm[args.holdout:]]
                probe_splits = {"train": fit, "val": held, "test": splits["test"]}
                loaders = ablation.make_loaders(family, probe_splits, cache,
                                                train_ds, test_ds, args.batch_size)
                records.append(probe_split(family, seed, split_i, loaders,
                                           args.gate_epochs, args.lr, args.device,
                                           args.concept_dim, feature_dim))
            agg = summarise_family(records)
            record = {"dataset": args.dataset, "n_train": args.n_train, "family": family,
                      "arm": ARM_OF.get(family), "seed": seed,
                      "token_pool": None if is_image else args.token_pool,
                      "backbone": args.backbone, "gate_epochs": args.gate_epochs,
                      "holdout": args.holdout, "n_gate_samples": int(len(gate_idx)),
                      "params": records[0]["params"], "wall_s": time.time() - t0,
                      "splits": records, **agg}
            path = os.path.join(regime_dir, f"probe_{family}_seed_{seed}.json")
            with open(path, "w") as f:
                json.dump(record, f, indent=2)
            print(f"[h6-probe] {family} seed {seed}: min_bits="
                  f"{agg['mean_min_held_out_bits']:.4f} sd={agg['sd_min_held_out_bits']:.4f} "
                  f"optimism={agg['mean_optimism']:.4f} -> {path}")

    summary = probe_summary(collect_probe_runs(args.out))
    with open(os.path.join(args.out, "probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print_table(summary)
    print(f"\nWrote {args.out}/probe_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
