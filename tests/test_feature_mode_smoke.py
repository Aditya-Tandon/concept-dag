"""
Smoke test for the feature-mode (SSL-backbone) DAG path.

ref: cross-attention-backbone-effect

The DINO backbone swap routes cached frozen-encoder features into the DAG via
`use_cnn=False` roots (DAGNode consumes (B, feature_dim) vectors directly instead
of running a SmallCNN). That path is exercised end-to-end here WITHOUT building any
encoder: we synthesise class-separable feature tensors and feed them in as
pre-cached tasks, exactly as `feature_cache.cache_features` would.

This validates the new code path (feature-mode roots, child aggregation over
parent outputs, principal-angle routing, subspace computation, freeze/eval) on a
tiny 3-task DAG that runs in seconds on CPU — no GPU, no torch.hub, no timm/open_clip.

Run directly:   python tests/test_feature_mode_smoke.py
Or via pytest:  pytest tests/test_feature_mode_smoke.py
"""

from __future__ import annotations

import tempfile

import torch
from torch.utils.data import TensorDataset, DataLoader


def _make_synthetic_feature_tasks(
    n_tasks: int = 3,
    n_classes: int = 5,
    feature_dim: int = 16,
    n_per_class: int = 40,
    batch_size: int = 16,
    seed: int = 0,
):
    """Build pre-cached feature tasks with class-dependent mean shifts so the
    concept modules have a learnable signal. Matches the task-dict schema returned
    by `feature_cache.cache_features` / `loaders.make_split_cifar100`."""
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
            return DataLoader(
                TensorDataset(X[lo:hi], Y[lo:hi]),
                batch_size=batch_size, shuffle=shuffle,
            )

        tasks.append({
            "task_id": t,
            "train": _loader(0, n_tr, True),
            "val": _loader(n_tr, n_tr + n_val, False),
            "test": _loader(n_tr + n_val, n, False),
            "n_classes": n_classes,
            "class_ids": list(range(t * n_classes, (t + 1) * n_classes)),
            "name": f"synthetic-task-{t}",
            "feature_dim": feature_dim,
        })
    return tasks


def test_feature_mode_smoke():
    from concept_dag.experiments.exp3_growing_dag import Exp3Config, run_exp3a

    feature_dim = 16
    tasks = _make_synthetic_feature_tasks(feature_dim=feature_dim)

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Exp3Config(
            # any non-"smallcnn" backbone forces use_cnn=False; no encoder is built
            # because we pass `tasks=` directly (encoder construction lives in
            # run_experiment.py, not run_exp3a).
            backbone="dinov2_vits14",
            feature_dim=feature_dim,
            concept_dim=32,
            soft_pca_k=4,
            subspace_k=4,
            n_tasks=3,
            n_parents=2,
            routing_batches=2,
            root_epochs=2,
            child_epochs=2,
            device="cpu",
            seed=0,
            results_dir=tmp,
        )
        results, nodes, heads, parent_map, _ = run_exp3a(cfg, tasks=tasks)

    aa = results["average_accuracy"]
    assert 0.0 <= aa <= 1.0, f"AA out of range: {aa}"
    assert len(nodes) == 3, f"expected 3 nodes, got {len(nodes)}"
    # feature-mode roots must NOT carry a CNN backbone
    assert nodes[0].is_root and nodes[0].cnn is None, "feature-mode root should have cnn=None"
    # at least one child should have been routed to a parent
    assert any(parent_map[t] for t in parent_map), "no child nodes acquired parents"

    print(f"[smoke] feature-mode DAG ran end-to-end: AA={aa:.4f}, nodes={len(nodes)}, "
          f"parent_map={parent_map}")
    return aa


if __name__ == "__main__":
    test_feature_mode_smoke()
    print("PASS")
