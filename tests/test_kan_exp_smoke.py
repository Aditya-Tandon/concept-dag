"""
End-to-end smoke test for the Kan-gated growth experiment (DAGNode path).

Runs `run_exp3a_kan` on tiny synthetic feature-mode tasks (no CIFAR, no encoder, CPU). Verifies the
full pipeline wires together: routing → Kan gate → grow|reuse → heterogeneous predictors → CL metrics
→ consolidation. It does NOT assert a particular grow/reuse split (that is data-dependent and is what
the real 20-task run measures) — only that the machinery runs and returns a coherent result.
"""

import torch

from concept_dag.experiments.kan_exp import KanExpConfig, run_exp3a_kan


def _make_synth_tasks(n_tasks=4, feature_dim=32, n_per_class=120, seed=0):
    g = torch.Generator().manual_seed(seed)
    tasks = []
    for t in range(n_tasks):
        # Two Gaussian blobs; tasks 2..3 reuse task 0/1's directions (encourages some reuse).
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
        n_tr = int(0.7 * X.shape[0])
        tr = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X[:n_tr], Y[:n_tr]),
                                         batch_size=64, shuffle=True)
        te = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X[n_tr:], Y[n_tr:]),
                                         batch_size=64)
        tasks.append({"train": tr, "test": te, "n_classes": 2, "class_ids": [2 * t, 2 * t + 1]})
    return tasks


def test_run_exp3a_kan_end_to_end(tmp_path):
    tasks = _make_synth_tasks()
    cfg = KanExpConfig(
        backbone="synthetic", feature_dim=32, concept_dim=32, cnn_out_dim=32,
        n_tasks=4, n_parents=2, subspace_k=8, soft_pca_k=8, routing_batches=10,
        root_epochs=6, child_epochs=6, gate_epochs=25, gate_lr=3e-3, lr=3e-3,
        eps_rel=0.10, similarity_threshold=7.0, results_dir=str(tmp_path), device="cpu",
        log_every=100,
    )
    res = run_exp3a_kan(cfg, tasks)

    assert 0.0 <= res["average_accuracy"] <= 1.0
    assert len(res["param_curve"]) == 4
    assert res["n_grow"] >= 1                          # the root always grows
    assert res["n_grow"] + res["n_reuse"] == 4
    assert 0.0 <= res["reuse_rate"] <= 1.0
    assert "consolidation" in res and "params_after" in res["consolidation"]
    # param curve is non-decreasing during growth (reduction happens only in the consolidation pass)
    assert all(b >= a for a, b in zip(res["param_curve"], res["param_curve"][1:]))
    # decisions recorded for every task
    assert len(res["decisions"]) == 4


if __name__ == "__main__":
    import tempfile, pathlib
    test_run_exp3a_kan_end_to_end(pathlib.Path(tempfile.mkdtemp()))
    print("\nKan-exp end-to-end smoke test passed.")
