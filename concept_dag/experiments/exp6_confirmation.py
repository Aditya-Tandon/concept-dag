"""
Experiment 6 — Confirmation run at larger embedding size.

Purpose:
    After Exp 5a/5b identify the optimal (n_parents*, subspace_k*) pair at
    concept_dim=128, this single run verifies whether scaling concept_dim to
    256 — with subspace_k scaled proportionally to preserve the k/concept_dim
    ratio — yields the expected capacity gain.

Design:
    - concept_dim      : 256  (2× the baseline)
    - cnn_out_dim      : 256  (already at the target; SmallCNN emits 256)
    - subspace_k       : k* × 2   (scaled proportionally)
    - n_parents        : n_parents* (from Exp 5a)
    - soft_pca_k       : same as subspace_k
    - All other hyperparameters identical to Exp 3a

    After growing the DAG, the perturbation test from Exp 3b is run so the
    degree-vs-drift correlation can be compared at two scales.

Usage:
    # Auto-load best pair from Exp 5 result JSONs
    python run_experiment.py --exp 6 --device cuda --data_root ./data \
        --exp5a results/exp5/exp5a_results.json \
        --exp5b results/exp5/exp5b_results.json

    # Or specify the pair explicitly
    python run_experiment.py --exp 6 --device cuda --data_root ./data \
        --best_n_parents 2 --best_subspace_k 8

    # Override concept_dim if you want something other than 256
    python run_experiment.py --exp 6 --device cuda --data_root ./data \
        --best_n_parents 2 --best_subspace_k 8 --confirm_concept_dim 384
"""

import os
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..data.loaders import make_split_cifar100
from .exp3_growing_dag import (
    Exp3Config, run_exp3a, run_exp3b,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _best_from_exp5a(path: str) -> Tuple[int, float]:
    """Return (best_n_parents, best_AA) from an exp5a_results.json file."""
    with open(path) as f:
        data = json.load(f)
    sweep = data["sweep"]
    best  = max(sweep, key=lambda s: s["average_accuracy"])
    return int(best["n_parents"]), float(best["average_accuracy"])


def _best_from_exp5b(path: str) -> Tuple[int, float]:
    """Return (best_subspace_k, best_AA) from an exp5b_results.json file."""
    with open(path) as f:
        data = json.load(f)
    sweep = data["sweep"]
    best  = max(sweep, key=lambda s: s["average_accuracy"])
    return int(best["subspace_k"]), float(best["average_accuracy"])


def _load_baseline_aa(path: Optional[str]) -> Optional[float]:
    """Try to read the baseline (concept_dim=128) AA for direct comparison."""
    if path is None or not os.path.exists(path):
        return None
    with open(path) as f:
        return float(json.load(f).get("average_accuracy", float("nan")))


# ──────────────────────────────────────────────────────────────────────────────
# Exp 6 runner
# ──────────────────────────────────────────────────────────────────────────────

def run_exp6(
    data_root:          str,
    device:             str,
    epochs:             int,
    batch_size:         int,
    seed:               int,
    out_dir:            str,
    best_n_parents:     Optional[int]  = None,
    best_subspace_k:    Optional[int]  = None,
    exp5a_path:         Optional[str]  = None,
    exp5b_path:         Optional[str]  = None,
    exp3a_baseline_path:Optional[str]  = None,
    confirm_concept_dim:int  = 256,
    baseline_concept_dim:int = 128,
    n_tasks:            int  = 20,
    run_perturbation:   bool = True,
) -> Dict:
    """
    Single-config confirmation run at larger concept_dim with k scaled
    proportionally.

    Returns a dict with:
        baseline_aa      : AA from Exp 3a at concept_dim=128  (if available)
        confirm_aa       : AA from this run at concept_dim=confirm_concept_dim
        delta_aa         : confirm_aa − baseline_aa
        config_used      : the exact hyperparameters applied
        per_task         : per-task accuracies
        perturbation     : exp3b-style drift/degree results (if run)
    """
    # ── 1.  Resolve best pair
    if best_n_parents is None:
        if exp5a_path is None:
            raise ValueError("Provide --best_n_parents or --exp5a result path")
        best_n_parents, aa_5a = _best_from_exp5a(exp5a_path)
        print(f"[5a] Loaded best n_parents={best_n_parents} (AA={aa_5a:.4f})")

    if best_subspace_k is None:
        if exp5b_path is None:
            raise ValueError("Provide --best_subspace_k or --exp5b result path")
        best_subspace_k, aa_5b = _best_from_exp5b(exp5b_path)
        print(f"[5b] Loaded best subspace_k={best_subspace_k} (AA={aa_5b:.4f})")

    # ── 2.  Scale k proportionally to keep k / concept_dim constant
    scale         = confirm_concept_dim / baseline_concept_dim
    scaled_k_raw  = best_subspace_k * scale
    # Clamp to concept_dim and snap to the nearest even power-of-two-ish value
    scaled_k      = min(int(round(scaled_k_raw)), confirm_concept_dim)
    if scaled_k < 2:
        scaled_k = 2

    print("\n" + "=" * 70)
    print("Experiment 6: Confirmation Run at Larger Embedding Size")
    print("=" * 70)
    print(f"  Baseline     : concept_dim={baseline_concept_dim}, "
          f"k={best_subspace_k}, n_parents={best_n_parents}")
    print(f"  Confirmation : concept_dim={confirm_concept_dim}, "
          f"k={scaled_k}  (scaled {best_subspace_k}×{scale:.1f}), "
          f"n_parents={best_n_parents}")
    print(f"  k/concept_dim ratio preserved: "
          f"{best_subspace_k/baseline_concept_dim:.4f} → "
          f"{scaled_k/confirm_concept_dim:.4f}")

    # ── 3.  Build the Exp3Config with overrides
    cfg = Exp3Config(
        data_root    = data_root,
        results_dir  = out_dir,
        cnn_out_dim  = 256,                  # SmallCNN already emits 256
        concept_dim  = confirm_concept_dim,
        n_mlp_layers = 2,
        soft_pca_k   = scaled_k,
        n_tasks      = n_tasks,
        n_parents    = best_n_parents,
        subspace_k   = scaled_k,
        root_epochs  = epochs,
        child_epochs = epochs,
        batch_size   = batch_size,
        seed         = seed,
        device       = device,
    )

    os.makedirs(cfg.results_dir, exist_ok=True)

    # ── 4.  Run the full DAG (reuses Exp 3a's runner)
    results_3a, nodes, heads, parent_map, tasks = run_exp3a(cfg)

    # ── 5.  Perturbation test (reuses Exp 3b's runner)
    results_3b = None
    if run_perturbation:
        results_3b = run_exp3b(cfg, nodes, heads, parent_map, tasks)

    # ── 6.  Compare to baseline AA
    baseline_aa = _load_baseline_aa(exp3a_baseline_path)
    confirm_aa  = results_3a["average_accuracy"]
    delta       = (confirm_aa - baseline_aa) if baseline_aa is not None else None

    print("\n" + "=" * 70)
    print("Experiment 6 — Summary")
    print("=" * 70)
    if baseline_aa is not None:
        print(f"  Baseline   AA (concept_dim={baseline_concept_dim}): {baseline_aa:.4f}")
    print(f"  Confirm    AA (concept_dim={confirm_concept_dim}): {confirm_aa:.4f}")
    if delta is not None:
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"  Δ AA       : {delta:+.4f}  {arrow}")
    if results_3b is not None:
        print(f"  Perturbation ρ(degree, drift): "
              f"{results_3b.get('corr_out_degree_drift', float('nan')):+.4f}")
        print(f"  (Natural baseline at concept_dim=128 was −0.586)")

    # ── 7.  Save everything
    output = {
        "config_used": {
            "concept_dim":        confirm_concept_dim,
            "baseline_concept_dim": baseline_concept_dim,
            "n_parents":          best_n_parents,
            "subspace_k":         scaled_k,
            "baseline_subspace_k": best_subspace_k,
            "k_over_dim_ratio":   scaled_k / confirm_concept_dim,
            "n_tasks":            n_tasks,
            "epochs":             epochs,
            "seed":                seed,
        },
        "baseline_aa":     baseline_aa,
        "confirm_aa":      confirm_aa,
        "delta_aa":        delta,
        "per_task":        results_3a["test_accs"],
        "out_degree":      results_3a["out_degree"],
        "orth_scores":     results_3a["orth_scores"],
        "parent_map":      results_3a["parent_map"],
        "perturbation":    results_3b,
    }
    out_path = os.path.join(cfg.results_dir, "exp6_confirmation_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output
