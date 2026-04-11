"""
Experiment 5 — Hyperparameter Sensitivity Studies.

Two orthogonal sweeps, each holding everything else at the Exp 3a defaults:

  5a  n_parents sweep   : vary n_parents ∈ {1, 2, 3, 4, 5}
      Hypothesis: accuracy peaks at n_parents=2 and plateaus / degrades beyond,
      because routing becomes noisier with many candidates and SoftPCA must
      compress more vectors into the same concept_dim space.

  5b  subspace_k sweep  : vary subspace_k (= soft_pca_k) ∈ {2, 4, 8, 16, 32}
      Hypothesis: accuracy rises up to k≈8–16 (the natural intrinsic dimensionality
      of 5-class feature spaces in CIFAR-100) and then plateaus, while orthogonality
      scores degrade as k approaches concept_dim (over-packing).

Each sweep grows the full 20-task DAG (or fewer tasks if --n_tasks is overridden)
for every parameter value and records:
  - per-task test accuracy
  - average accuracy (AA)
  - concept subspace orthogonality scores (orth_score per node)
  - parent assignment map (to check whether routing changes with n_parents)

Run:
    python run_experiment.py --exp 5a --device cuda --data_root ./data
    python run_experiment.py --exp 5b --device cuda --data_root ./data
    python run_experiment.py --exp 5  --device cuda --data_root ./data   # both sweeps
    python run_experiment.py --exp 5a --n_tasks 10 --epochs 15           # fast mode
"""

import gc
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..data.loaders import make_split_cifar100
from .exp3_growing_dag import (
    Exp3Config, DAGNode, LinearHead,
    train_node, eval_node, route_for_task, _flush,
)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Exp5Config:
    """
    All base hyperparameters match Exp 3a defaults.
    The sweep lists define which values to iterate over for each sub-experiment.
    """
    data_root:      str   = "./data"
    results_dir:    str   = "results/exp5"
    # Model (fixed across sweep)
    cnn_out_dim:    int   = 256
    concept_dim:    int   = 128
    n_mlp_layers:   int   = 2
    # DAG growth defaults
    n_tasks:        int   = 20
    routing_batches: int  = 20
    # Training (fixed across sweep)
    root_epochs:    int   = 25
    child_epochs:   int   = 25
    lr:             float = 1e-3
    batch_size:     int   = 128
    orth_weight:    float = 0.01
    # Sweep definitions
    n_parents_sweep:  List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    subspace_k_sweep: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    # Misc
    seed:           int   = 42
    device:         str   = "cpu"
    log_every:      int   = 5


# ──────────────────────────────────────────────────────────────────────────────
# Single-configuration DAG run (factored out for reuse across sweep values)
# ──────────────────────────────────────────────────────────────────────────────

def _run_one_config(
    cfg:        Exp5Config,
    n_parents:  int,
    subspace_k: int,
    tasks:      List[Dict],
    label:      str,
) -> Dict:
    """
    Grow the Concept DAG with a specific (n_parents, subspace_k) pair.
    Returns a dict of metrics for this configuration.
    `tasks` is the pre-loaded list from make_split_cifar100.
    """
    assert subspace_k <= cfg.concept_dim, (
        f"subspace_k={subspace_k} must be ≤ concept_dim={cfg.concept_dim}. "
        "Larger values would make the SoftPCA projection and principal-angle "
        "routing degenerate (more directions than the embedding space supports)."
    )
    torch.manual_seed(cfg.seed)
    device      = cfg.device
    soft_pca_k  = subspace_k   # they control the same quantity

    nodes:      List[DAGNode]    = []
    heads:      List[LinearHead] = []
    parent_map: Dict[int, List[int]] = {}
    test_accs:  List[float]      = []
    orth_scores_per_task: Dict[int, float] = {}

    for t, task in enumerate(tasks):
        n_par = min(n_parents, t)   # task 0 is always root

        # ── routing
        if n_par == 0:
            selected_idx = []
        else:
            selected_idx, _ = route_for_task(
                existing_nodes   = nodes,
                new_task_loader  = task["train"],
                n_parents        = n_par,
                subspace_k       = subspace_k,
                device           = device,
                routing_batches  = cfg.routing_batches,
            )
        parent_map[t] = selected_idx
        selected_parents = [nodes[i] for i in selected_idx]

        # ── build node
        node = DAGNode(
            task_id       = t,
            concept_dim   = cfg.concept_dim,
            cnn_out_dim   = cfg.cnn_out_dim,
            n_mlp_layers  = cfg.n_mlp_layers,
            parent_models = selected_parents if selected_parents else None,
            soft_pca_k    = soft_pca_k,
        )
        head = LinearHead(cfg.concept_dim, task["n_classes"])

        # ── train
        n_epochs = cfg.root_epochs if n_par == 0 else cfg.child_epochs
        train_node(
            node, head, task["train"],
            n_epochs, cfg.lr, device, cfg.log_every,
            name=f"{label}/t{t}", orth_weight=cfg.orth_weight,
        )
        test_acc = eval_node(node, head, task["test"], device)
        test_accs.append(test_acc)

        # ── subspace + orthogonality
        node.compute_concept_subspace(
            task["train"], device,
            top_k=subspace_k, max_batches=cfg.routing_batches,
        )
        cs = node.concept_module.get_concept_subspace()
        if cs is not None:
            gram = cs.t() @ cs
            off  = (gram - torch.eye(gram.shape[0])).abs().mean().item()
            orth_scores_per_task[t] = float(off)

        node.freeze()
        nodes.append(node)
        heads.append(head)
        _flush(device)

    # ── compute out-degree
    out_degree = {t: 0 for t in range(len(tasks))}
    for t, parents in parent_map.items():
        for p in parents:
            out_degree[p] += 1

    aa = float(np.mean(test_accs))
    avg_orth = float(np.mean(list(orth_scores_per_task.values()))) if orth_scores_per_task else float("nan")

    print(f"  [{label}] AA={aa:.4f}  avg_orth={avg_orth:.2e}  "
          f"out_degree_max={max(out_degree.values())}")

    result = {
        "n_parents":         n_parents,
        "subspace_k":        subspace_k,
        "test_accs":         test_accs,
        "average_accuracy":  aa,
        "orth_scores":       orth_scores_per_task,
        "avg_orth":          avg_orth,
        "parent_map":        {str(k): v for k, v in parent_map.items()},
        "out_degree":        {str(k): v for k, v in out_degree.items()},
    }

    # Explicitly release per-config DAG state before returning so the outer
    # sweep loop's _flush() can actually reclaim GPU memory. Nothing downstream
    # holds references to these objects — the sweep only needs the `result`
    # dict above.
    del nodes, heads, parent_map, test_accs, orth_scores_per_task
    _flush(device)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Exp 5a — n_parents sweep
# ──────────────────────────────────────────────────────────────────────────────

def run_exp5a(cfg: Exp5Config) -> Dict:
    """
    Sweep n_parents ∈ cfg.n_parents_sweep, holding subspace_k fixed at
    the Exp 3a default (8).

    This isolates the effect of the number of parents on:
      - Average accuracy (task performance)
      - Routing quality (do we select more relevant parents?)
      - DAG topology (does higher n_parents create deeper or denser graphs?)
    """
    print("\n" + "=" * 70)
    print("Experiment 5a: n_parents sweep")
    print(f"  subspace_k fixed = 8 | n_parents ∈ {cfg.n_parents_sweep}")
    print("=" * 70)

    os.makedirs(cfg.results_dir, exist_ok=True)
    fixed_k = 8   # Exp 3a default

    print(f"\n[Step 0] Loading Split-CIFAR-100 ({cfg.n_tasks} tasks)...")
    tasks = make_split_cifar100(
        data_root  = cfg.data_root,
        n_tasks    = cfg.n_tasks,
        batch_size = cfg.batch_size,
        seed       = cfg.seed,
    )

    sweep_results = []
    for n_par in cfg.n_parents_sweep:
        print(f"\n{'─'*60}")
        print(f"  n_parents = {n_par}  (subspace_k = {fixed_k})")
        print(f"{'─'*60}")
        res = _run_one_config(
            cfg        = cfg,
            n_parents  = n_par,
            subspace_k = fixed_k,
            tasks      = tasks,
            label      = f"5a/npar={n_par}",
        )
        sweep_results.append(res)
        _flush(cfg.device)

    # ── Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Exp 5a: n_parents sweep")
    print(f"  {'n_parents':>10}  {'AA':>8}  {'avg_orth':>12}  {'max_degree':>12}")
    print("-" * 50)
    for res in sweep_results:
        od = {int(k): v for k, v in res["out_degree"].items()}
        print(f"  {res['n_parents']:>10}  {res['average_accuracy']:>8.4f}  "
              f"{res['avg_orth']:>12.2e}  {max(od.values()):>12}")

    output = {
        "experiment":   "5a_n_parents_sweep",
        "fixed_subspace_k": fixed_k,
        "n_tasks":      cfg.n_tasks,
        "sweep":        sweep_results,
    }
    path = os.path.join(cfg.results_dir, "exp5a_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {path}")
    return output


# ──────────────────────────────────────────────────────────────────────────────
# Exp 5b — subspace_k sweep
# ──────────────────────────────────────────────────────────────────────────────

def run_exp5b(cfg: Exp5Config) -> Dict:
    """
    Sweep subspace_k ∈ cfg.subspace_k_sweep, holding n_parents fixed at 2.

    subspace_k controls two things simultaneously:
      (a) Routing: how many right singular vectors define each node's concept
          subspace for principal-angle matching.
      (b) SoftPCA projection: how many near-orthogonal directions are extracted
          from parent outputs (soft_pca_k = subspace_k).

    As k grows:
      - Routing gets richer but potentially noisier (more directions = more
        variance captured, but also more noise dimensions).
      - SoftPCA can express finer-grained crystallized subspaces, but the
        orthogonality regularisation becomes harder to satisfy (more columns
        to orthogonalise in concept_dim space).
    """
    print("\n" + "=" * 70)
    print("Experiment 5b: subspace_k sweep")
    print(f"  n_parents fixed = 2 | subspace_k ∈ {cfg.subspace_k_sweep}")
    print("=" * 70)

    os.makedirs(cfg.results_dir, exist_ok=True)
    fixed_npar = 2   # Exp 3a default

    # Clamp k values that exceed concept_dim (would be degenerate)
    valid_ks = [k for k in cfg.subspace_k_sweep if k <= cfg.concept_dim]
    skipped  = [k for k in cfg.subspace_k_sweep if k > cfg.concept_dim]
    if skipped:
        print(f"  [info] subspace_k values {skipped} > concept_dim={cfg.concept_dim}, skipping.")

    print(f"\n[Step 0] Loading Split-CIFAR-100 ({cfg.n_tasks} tasks)...")
    tasks = make_split_cifar100(
        data_root  = cfg.data_root,
        n_tasks    = cfg.n_tasks,
        batch_size = cfg.batch_size,
        seed       = cfg.seed,
    )

    sweep_results = []
    for k in valid_ks:
        print(f"\n{'─'*60}")
        print(f"  subspace_k = {k}  (n_parents = {fixed_npar})")
        print(f"{'─'*60}")
        res = _run_one_config(
            cfg        = cfg,
            n_parents  = fixed_npar,
            subspace_k = k,
            tasks      = tasks,
            label      = f"5b/k={k}",
        )
        sweep_results.append(res)
        _flush(cfg.device)

    # ── Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Exp 5b: subspace_k sweep")
    print(f"  {'subspace_k':>12}  {'AA':>8}  {'avg_orth':>12}")
    print("-" * 36)
    for res in sweep_results:
        print(f"  {res['subspace_k']:>12}  {res['average_accuracy']:>8.4f}  {res['avg_orth']:>12.2e}")

    output = {
        "experiment":    "5b_subspace_k_sweep",
        "fixed_n_parents": fixed_npar,
        "n_tasks":       cfg.n_tasks,
        "sweep":         sweep_results,
    }
    path = os.path.join(cfg.results_dir, "exp5b_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {path}")
    return output
