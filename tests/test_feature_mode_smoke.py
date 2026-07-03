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


def test_cross_attention_no_silent_flip():
    """Regression test for the H2 confound (see [[cross-attention-backbone-effect]]).

    With the default query_from_input=False, a cross-attention child's output must
    NOT depend on its input `x` at all — in particular it must be identical whether
    feature_dim == concept_dim or feature_dim != concept_dim. Previously the query
    source silently flipped to the raw input exactly when those dims coincided
    (e.g. DINOv2-384 with concept_dim=384), contaminating the ablation.
    """
    from concept_dag.modules.concept_module import ConceptModule

    concept_dim = 16
    cm = ConceptModule(
        module_id="child", in_dim=concept_dim, hidden_dim=concept_dim,
        out_dim=concept_dim, n_layers=2, n_parents=2,
        aggregation="cross_attention", agg_kwargs={"n_heads": 4},  # query_from_input defaults False
    )
    cm.eval()  # disable dropout for determinism

    torch.manual_seed(0)
    parents = [torch.randn(4, concept_dim), torch.randn(4, concept_dim)]
    x_eq = torch.randn(4, concept_dim)        # feature_dim == concept_dim (the old flip trigger)
    x_neq = torch.randn(4, concept_dim + 8)   # feature_dim != concept_dim

    with torch.no_grad():
        o_eq = cm(x_eq, parent_outputs=parents)
        o_neq = cm(x_neq, parent_outputs=parents)
        o_none = cm(None, parent_outputs=parents)

    assert o_eq.shape == (4, concept_dim)
    assert torch.allclose(o_eq, o_neq, atol=1e-6), "output changed with feature_dim — silent flip present"
    assert torch.allclose(o_eq, o_none, atol=1e-6), "output depends on input when query_from_input=False"
    print("[cross_attn] no-silent-flip: output is input-independent when query_from_input=False ✓")


def test_cross_attention_input_query_mode():
    """The opt-in input-query mode must work at ANY feature_dim (dim-safe projection),
    actually use the input (output depends on it), and raise on a missing 2-D input."""
    from concept_dag.modules.concept_module import ConceptModule

    concept_dim, feature_dim = 16, 24  # deliberately unequal
    cm = ConceptModule(
        module_id="child", in_dim=concept_dim, hidden_dim=concept_dim,
        out_dim=concept_dim, n_layers=2, n_parents=2,
        aggregation="cross_attention",
        agg_kwargs={"n_heads": 4, "query_from_input": True, "query_input_dim": feature_dim},
    )
    cm.eval()

    torch.manual_seed(0)
    parents = [torch.randn(4, concept_dim), torch.randn(4, concept_dim)]
    x1 = torch.randn(4, feature_dim)
    x2 = torch.randn(4, feature_dim)

    with torch.no_grad():
        o1 = cm(x1, parent_outputs=parents)
        o2 = cm(x2, parent_outputs=parents)

    assert o1.shape == (4, concept_dim), "projection feature_dim->concept_dim failed"
    assert not torch.allclose(o1, o2), "input-query mode ignored the input"

    # Missing 2-D input (e.g. smallcnn mode where x is a 4-D image) must fail loudly.
    raised = False
    try:
        with torch.no_grad():
            cm(None, parent_outputs=parents)
    except ValueError:
        raised = True
    assert raised, "input-query mode must raise when no 2-D query input is available"
    print("[cross_attn] input-query mode: dim-safe, input-driven, raises on misuse ✓")


def test_feature_cache_seed_meta_rewrite():
    """A seed change must invalidate the cache AND update the on-disk meta, so a
    second run at the new seed does not re-detect a mismatch and recompute forever."""
    import json
    import os
    import torch.nn as nn
    from concept_dag.data.feature_cache import cache_features

    class DummyEncoder(nn.Module):
        feature_dim = 8

        def forward(self, x):
            return x.flatten(1)[:, : self.feature_dim]  # (B, 8)

    def _raw_tasks():
        tasks = []
        for t in range(2):
            imgs = torch.randn(12, 3, 4, 4)
            lbls = torch.randint(0, 5, (12,))
            def _dl(shuffle):
                return DataLoader(TensorDataset(imgs, lbls), batch_size=6, shuffle=shuffle)
            tasks.append({
                "task_id": t, "train": _dl(True), "val": _dl(False), "test": _dl(False),
                "n_classes": 5, "class_ids": list(range(t * 5, t * 5 + 5)), "name": f"t{t}",
            })
        return tasks

    enc = DummyEncoder()
    with tempfile.TemporaryDirectory() as tmp:
        meta_path = os.path.join(tmp, "meta.json")

        cache_features(enc, _raw_tasks(), cache_dir=tmp, device="cpu", seed=0)
        assert json.load(open(meta_path))["seed"] == 0

        # Re-run at the same seed: meta stays 0, cache is reused (no crash).
        cache_features(enc, _raw_tasks(), cache_dir=tmp, device="cpu", seed=0)
        assert json.load(open(meta_path))["seed"] == 0

        # Seed change: meta MUST update to 1 (the bug left it stale at 0).
        cache_features(enc, _raw_tasks(), cache_dir=tmp, device="cpu", seed=1)
        assert json.load(open(meta_path))["seed"] == 1, "meta not rewritten after seed change"

    print("[feature_cache] seed change invalidates cache and rewrites meta ✓")


if __name__ == "__main__":
    test_feature_mode_smoke()
    test_cross_attention_no_silent_flip()
    test_cross_attention_input_query_mode()
    test_feature_cache_seed_meta_rewrite()
    print("PASS")
