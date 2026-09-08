"""
Tests for `KanExpConfig.gate_cache_max` — the gate's probe-cache sample budget, decoupled from
`routing_batches` (which still caps `route_for_task` / `compute_concept_subspace`).

Before this change, `_cache_parent_stack(..., max_batches=cfg.routing_batches)` capped the gate's
sample cache at routing_batches * batch_size (2,560 by default) — far below the 50-70k a
5-Datasets root deploys on. See Central Library concept-dag-open-directions.md §0 (Hygiene).
"""

import torch

from concept_dag.experiments.kan_exp import KanExpConfig, run_exp3a_kan


def _make_two_task_stream(n_per_class=500, feature_dim=16, batch_size=64, train_frac=0.8, seed=0):
    """Two tasks (root + one gated task) with a known, exact train-set size, so the gate's cache
    N is fully determined by gate_cache_max (capped) or the loader (uncapped)."""
    g = torch.Generator().manual_seed(seed)
    tasks = []
    for t in range(2):
        base = t % 2
        mu = torch.zeros(feature_dim)
        mu[base * 2] = 2.0
        mu[base * 2 + 1] = -2.0
        xs, ys = [], []
        for c in range(2):
            centre = mu if c == 0 else -mu
            xs.append(centre + torch.randn(n_per_class, feature_dim, generator=g))
            ys.append(torch.full((n_per_class,), c, dtype=torch.long))
        X = torch.cat(xs); Y = torch.cat(ys)
        perm = torch.randperm(X.shape[0], generator=g)
        X, Y = X[perm], Y[perm]
        n_tr = int(train_frac * X.shape[0])
        tr = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X[:n_tr], Y[:n_tr]),
                                         batch_size=batch_size, shuffle=True)
        te = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X[n_tr:], Y[n_tr:]),
                                         batch_size=batch_size)
        tasks.append({"train": tr, "test": te, "n_classes": 2, "class_ids": [2 * t, 2 * t + 1],
                     "n_train": n_tr})
    return tasks


def _base_cfg(tmp_path, *, gate_cache_max, routing_batches, batch_size, feature_dim):
    return KanExpConfig(
        backbone="synthetic", feature_dim=feature_dim, concept_dim=feature_dim, cnn_out_dim=feature_dim,
        n_tasks=2, n_parents=1, subspace_k=4, soft_pca_k=4, routing_batches=routing_batches,
        root_epochs=4, child_epochs=4, gate_epochs=10, gate_lr=3e-3, lr=3e-3,
        eps_rel=0.10, similarity_threshold=7.0, results_dir=str(tmp_path), device="cpu",
        log_every=1000, batch_size=batch_size, gate_cache_max=gate_cache_max,
    )


def test_gate_cache_max_caps_gate_cache_independent_of_routing(tmp_path):
    """gate_cache_max=256, batch_size=64, a 1000-example synthetic task (800 train) -> the gate's
    live decision sees exactly 256 samples (gate_cache_n), while routing_batches (2 -> 128 samples)
    is a different, smaller cap that governs route_for_task / compute_concept_subspace instead."""
    tasks = _make_two_task_stream(n_per_class=500, feature_dim=16, batch_size=64)
    assert tasks[1]["n_train"] == 800  # 1000 examples/task, 80% train split
    cfg = _base_cfg(tmp_path, gate_cache_max=256, routing_batches=2, batch_size=64, feature_dim=16)
    res = run_exp3a_kan(cfg, tasks)

    d1 = next(d for d in res["decisions"] if d["task"] == 1)
    assert d1["gate_cache_n"] == 256
    # Not the routing cap (routing_batches * batch_size = 128) — the two caps are independent.
    assert d1["gate_cache_n"] != cfg.routing_batches * cfg.batch_size
    # Not the full task either — gate_cache_max actually capped it below the 800 available.
    assert d1["gate_cache_n"] < tasks[1]["n_train"]


def test_gate_cache_max_zero_uses_full_task(tmp_path):
    """gate_cache_max=0 means unlimited: the gate's live decision sees the WHOLE task's train set."""
    tasks = _make_two_task_stream(n_per_class=500, feature_dim=16, batch_size=64)
    assert tasks[1]["n_train"] == 800
    cfg = _base_cfg(tmp_path, gate_cache_max=0, routing_batches=2, batch_size=64, feature_dim=16)
    res = run_exp3a_kan(cfg, tasks)

    d1 = next(d for d in res["decisions"] if d["task"] == 1)
    assert d1["gate_cache_n"] == tasks[1]["n_train"] == 800


if __name__ == "__main__":
    import tempfile, pathlib
    test_gate_cache_max_caps_gate_cache_independent_of_routing(pathlib.Path(tempfile.mkdtemp()))
    test_gate_cache_max_zero_uses_full_task(pathlib.Path(tempfile.mkdtemp()))
    print("\ngate_cache_max tests passed.")
