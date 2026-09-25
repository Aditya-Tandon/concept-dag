"""A fixed, tiny, CPU-only CLS-mode gated run — the reference the byte-identity test replays.

The adoption of the attention-pooling root ([[attn-root-adoption]]) must leave the published
`mlp_cls` path bit-for-bit unchanged: every re-baseline arm, and the null-validity check that
licenses the comparison, rests on it. A synthetic four-task feature stream with `--raw_grow_probe`
and the three-way ladder exercises the code paths the change actually touched — `DAGNode` root
construction, `root_module_factory`, the gate cache, consolidation — while running in seconds.

This module is deliberately dependency-light and uses only API that exists BOTH before and after
the change, so the same file can be dropped into a pre-change checkout to mint the fixture
`tests/fixtures/mlp_cls_baseline.json` (see the test's docstring for the exact command).
"""

from __future__ import annotations

import os
import tempfile

import torch
from torch.utils.data import TensorDataset, DataLoader


#: JSON keys that legitimately differ after the change (new fields) or between runs (wall time).
NEW_OR_NONDETERMINISTIC_KEYS = {
    "root_family", "n_tokens", "feature_dim", "gate_cache_max",
    "param_curve_total", "params_total_pre_consolidation", "params_per_root",
}
DECISION_KEYS_TO_IGNORE = {"gate_seconds"}


def make_cls_tasks(n_tasks: int = 4, n_classes: int = 4, feature_dim: int = 24,
                   n_per_class: int = 48, batch_size: int = 16, seed: int = 0):
    """CLS-mode (2-D) feature tasks, the schema `feature_cache.cache_features` returns."""
    g = torch.Generator().manual_seed(seed)
    tasks = []
    for t in range(n_tasks):
        feats, labels = [], []
        for c in range(n_classes):
            center = torch.randn(feature_dim, generator=g) * 3.0
            x = center + torch.randn(n_per_class, feature_dim, generator=g)
            feats.append(x)
            labels.append(torch.full((n_per_class,), c, dtype=torch.long))
        X = torch.cat(feats)
        Y = torch.cat(labels)
        perm = torch.randperm(len(X), generator=g)
        X, Y = X[perm], Y[perm]
        n = len(X)
        n_tr, n_val = int(0.6 * n), int(0.2 * n)

        def _loader(lo, hi, shuffle):
            return DataLoader(TensorDataset(X[lo:hi], Y[lo:hi]),
                              batch_size=batch_size, shuffle=shuffle)

        tasks.append({
            "task_id": t, "n_classes": n_classes, "name": f"cls{t}",
            "train": _loader(0, n_tr, True),
            "val":   _loader(n_tr, n_tr + n_val, False),
            "test":  _loader(n_tr + n_val, n, False),
            "feature_dim": feature_dim,
        })
    return tasks


def run_reference(results_dir: str) -> dict:
    """Run the reference stream and return the results dict. Seeded end to end."""
    from concept_dag.experiments.kan_exp import KanExpConfig, run_exp3a_kan

    tasks = make_cls_tasks()
    cfg = KanExpConfig(
        results_dir=results_dir, device="cpu", backbone="dinov2_vits14",
        feature_dim=24, concept_dim=16, seed=42,
        root_epochs=3, child_epochs=3, gate_epochs=3,
        n_tasks=len(tasks), n_parents=2, routing_batches=4, gate_cache_max=256,
        batch_size=16, raw_grow_probe=True, enable_search=True, search_skip=True,
        oracle_rungs=True, consolidate_every=0, log_every=100,
    )
    return run_exp3a_kan(cfg, tasks=tasks)


def comparable(results: dict) -> dict:
    """Strip fields that the change legitimately adds, or that are wall-clock."""
    out = {k: v for k, v in results.items() if k not in NEW_OR_NONDETERMINISTIC_KEYS}
    out["decisions"] = [
        {k: v for k, v in d.items() if k not in DECISION_KEYS_TO_IGNORE}
        for d in results["decisions"]
    ]
    return out


if __name__ == "__main__":  # mint the fixture from whichever checkout is on sys.path
    import json
    import sys

    with tempfile.TemporaryDirectory() as d:
        res = run_reference(os.path.join(d, "r"))
    target = sys.argv[1] if len(sys.argv) > 1 else "mlp_cls_baseline.json"
    with open(target, "w") as f:
        json.dump(comparable(res), f, indent=2, sort_keys=True)
    print("wrote", target)
