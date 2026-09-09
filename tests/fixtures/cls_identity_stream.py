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


def make_duplicate_cls_tasks(**kw):
    """`make_cls_tasks` with the LAST task redrawn from the FIRST task's class centres.

    The plain reference stream never gets a merge past the redundancy detector, so
    `distill_merge` — the one consolidation path that trains weights — goes unexercised by the
    byte-identity fixture. A deliberate duplicate makes the detector fire, so a change to the
    merge path can no longer slip through the identity check (PR #7 review, should-fix 3).
    """
    tasks = make_cls_tasks(**kw)
    dup = dict(tasks[0])
    dup["task_id"] = len(tasks)
    dup["name"] = f"cls{len(tasks)}dup0"
    tasks.append(dup)
    return tasks


def run_reference_consolidating(results_dir: str) -> dict:
    """The reference stream plus a duplicate task and consolidation every 2 tasks.

    Same knobs as `run_reference` except `consolidate_every=2` and the duplicated final task, so
    the run exercises `consolidate_nodes` mid-stream, `_functional_similarity`, `distill_merge`
    and the backward-safety accept/veto.
    """
    from concept_dag.experiments.kan_exp import KanExpConfig, run_exp3a_kan

    tasks = make_duplicate_cls_tasks()
    cfg = KanExpConfig(
        results_dir=results_dir, device="cpu", backbone="dinov2_vits14",
        feature_dim=24, concept_dim=16, seed=42,
        # More epochs than the plain reference: the redundancy detector compares the two roots'
        # FUNCTIONS, and at 3 epochs their outputs are still dominated by initialisation, so the
        # canonical correlation never reaches the 0.9 trigger and `distill_merge` is never called.
        root_epochs=12, child_epochs=12, gate_epochs=6,
        n_tasks=len(tasks), n_parents=2, routing_batches=4, gate_cache_max=256,
        # subspace_k=3 of a 16-d concept, not the default 8. The functional detector averages the
        # top-k canonical correlations, and only ~3 directions of a 4-class concept carry class
        # information; averaging 8 dilutes them with noise directions and the 0.9 trigger is never
        # reached (measured: 0.99 at k=3 for the true duplicate, 0.75 at k=8). k/concept_dim here
        # is 3/16, close to the production 8/128.
        subspace_k=3,
        batch_size=16, raw_grow_probe=True, enable_search=True, search_skip=True,
        oracle_rungs=True, consolidate_every=2, distill=True, distill_epochs=4,
        # Force-grow the duplicate, exactly as `--inject_dup` does on 5-Datasets: the gate would
        # (correctly) reuse it, and then there would be no redundant pair for the detector to
        # find and no `distill_merge` call to pin down.
        force_grow_ids=(len(tasks) - 1,),
        log_every=100,
    )
    return run_exp3a_kan(cfg, tasks=tasks)


if __name__ == "__main__":  # mint the fixture from whichever checkout is on sys.path
    import json
    import sys

    which = sys.argv[2] if len(sys.argv) > 2 else "plain"
    runner = run_reference if which == "plain" else run_reference_consolidating
    with tempfile.TemporaryDirectory() as d:
        res = runner(os.path.join(d, "r"))
    target = sys.argv[1] if len(sys.argv) > 1 else "mlp_cls_baseline.json"
    with open(target, "w") as f:
        json.dump(comparable(res), f, indent=2, sort_keys=True)
    print("wrote", target)
