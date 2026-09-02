"""
Smoke tests for the oracle-rungs, gate-tensor-dump, and update-rung instrumentation in
`concept_dag.experiments.kan_exp.run_exp3a_kan` (gate-estimator / denominator / update-arm branch).

All tasks are tiny, in-memory, synthetic feature-mode ("no CNN, no encoder") streams so the whole
suite runs on CPU in seconds. Two task-generation helpers are used:

  * `_make_synth_tasks` — the two-Gaussian-blob generator from the existing kan_exp smoke test,
    reused here for the oracle-rungs and gate-dump tests (any mix of grow/reuse/search is fine).
  * `_binary_stream` — a single linear-threshold direction shared across two tasks, used for the
    update-rung tests: task 0 is a deliberately UNDER-TRAINED root (root_epochs=1) and task 1 draws
    from the SAME direction with more data, so a warm-started `update_probe` refinement of task 0's
    concept (which starts from its already-partially-fit weights) can out-compete a cold-started
    fresh `grow` probe under the same small epoch budget — the mechanism the update arm exists to
    exploit ([[reuse-with-update-arm-stress-test]]).
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from concept_dag.experiments.kan_exp import KanExpConfig, run_exp3a_kan


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------


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
        tr = DataLoader(TensorDataset(X[:n_tr], Y[:n_tr]), batch_size=64, shuffle=True)
        te = DataLoader(TensorDataset(X[n_tr:], Y[n_tr:]), batch_size=64)
        tasks.append({"train": tr, "test": te, "n_classes": 2, "class_ids": [2 * t, 2 * t + 1]})
    return tasks


def _binary_stream(feature_dim, w, n_train, n_test, seed, batch_size=64, noise=0.05):
    """A single linear-threshold direction `w` (unit-norm), split train/test, no `val` key (the
    gate/update code must fall back to `test` for backward safety when `val` is absent)."""
    g = torch.Generator().manual_seed(seed)
    n = n_train + n_test
    X = torch.randn(n, feature_dim, generator=g)
    y = ((X @ w) > 0).long()
    flip = torch.rand(n, generator=g) < noise
    y = torch.where(flip, 1 - y, y)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]
    return {
        "train": DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True),
        "test": DataLoader(TensorDataset(Xte, yte), batch_size=batch_size),
        "n_classes": 2,
    }


def _update_stream(feature_dim=24, seed=0):
    """Task 0: an under-trained root over a random linear direction `w` (root_epochs=1 in the
    caller's config). Task 1: the SAME direction, more data — a distribution an update-refined
    root should serve at least as well as a freshly grown one, without minting a new concept."""
    torch.manual_seed(seed)
    w = torch.randn(feature_dim)
    w = w / w.norm()
    task0 = _binary_stream(feature_dim, w, n_train=80, n_test=100, seed=seed + 1)
    task1 = _binary_stream(feature_dim, w, n_train=400, n_test=150, seed=seed + 2)
    return [task0, task1]


def _update_cfg(tmp_path, feature_dim=24, eps_rel=0.9, enable_update=True):
    return KanExpConfig(
        backbone="synthetic", feature_dim=feature_dim, concept_dim=16, cnn_out_dim=16,
        n_mlp_layers=2, n_tasks=2, n_parents=1, subspace_k=8, soft_pca_k=8, routing_batches=20,
        root_epochs=1, child_epochs=15, gate_epochs=10, gate_lr=3e-3, lr=3e-3,
        eps_rel=eps_rel, eps_search=0.05, similarity_threshold=7.0, results_dir=str(tmp_path),
        device="cpu", log_every=1000,
        raw_grow_probe=True, enable_search=True, search_skip=True,
        enable_update=enable_update, update_lr=1e-3, eps_update=0.1, update_tolerance=0.02,
    )


# ---------------------------------------------------------------------------
# 1. Oracle rungs
# ---------------------------------------------------------------------------


def test_oracle_rungs_writes_all_three_and_matches_chosen(tmp_path):
    tasks = _make_synth_tasks(n_tasks=4, feature_dim=24, n_per_class=90, seed=3)
    cfg = KanExpConfig(
        backbone="synthetic", feature_dim=24, concept_dim=16, cnn_out_dim=16, n_mlp_layers=2,
        n_tasks=4, n_parents=2, subspace_k=8, soft_pca_k=8, routing_batches=10,
        root_epochs=6, child_epochs=6, gate_epochs=15, gate_lr=3e-3, lr=3e-3,
        eps_rel=0.1, eps_search=0.05, enable_search=True, search_skip=True,
        oracle_rungs=True, similarity_threshold=7.0, results_dir=str(tmp_path), device="cpu",
        log_every=1000,
    )
    res = run_exp3a_kan(cfg, tasks)

    gated = [d for d in res["decisions"] if d["task"] >= 1]
    assert gated, "expected at least one gated (non-root) task"
    for d in gated:
        assert "oracle_accs" in d
        oa = d["oracle_accs"]
        assert set(oa.keys()) == {"reuse", "search", "grow"}
        chosen = d["decision"]
        assert chosen in ("reuse", "search", "grow")
        t = d["task"]
        # The chosen rung's oracle entry is the SAME NUMBER as the real predictor's test accuracy
        # (reused, not retrained) — exact equality, not merely close.
        assert oa[chosen] == res["test_accs"][t]


# ---------------------------------------------------------------------------
# 2. gate_dump.pt
# ---------------------------------------------------------------------------


def test_dump_gate_tensors_writes_gate_dump(tmp_path):
    tasks = _make_synth_tasks(n_tasks=3, feature_dim=20, n_per_class=80, seed=4)
    cfg = KanExpConfig(
        backbone="synthetic", feature_dim=20, concept_dim=16, cnn_out_dim=16, n_mlp_layers=2,
        n_tasks=3, n_parents=2, subspace_k=8, soft_pca_k=8, routing_batches=10,
        root_epochs=5, child_epochs=5, gate_epochs=10, gate_lr=3e-3, lr=3e-3,
        eps_rel=0.05, raw_grow_probe=True, dump_gate_tensors=True,
        similarity_threshold=7.0, results_dir=str(tmp_path), device="cpu", log_every=1000,
    )
    res = run_exp3a_kan(cfg, tasks)

    dump_path = tmp_path / "gate_dump.pt"
    assert dump_path.exists()
    dump = torch.load(dump_path, weights_only=False)

    for key in ("feature_mode", "concept_dim", "feature_dim", "n_parents", "seed", "config",
               "tasks", "nodes", "predictors", "decisions"):
        assert key in dump
    assert dump["feature_mode"] is True
    assert dump["feature_dim"] == cfg.feature_dim
    assert len(dump["tasks"]) == 3

    for tdump in dump["tasks"]:
        assert tdump["train_raw"].shape[1] == cfg.feature_dim
        assert tdump["train_raw"].shape[0] == tdump["train_y"].shape[0]
        assert tdump["val_raw"].shape[1] == cfg.feature_dim
        assert tdump["test_raw"].shape[1] == cfg.feature_dim

    # raw_grow_probe mints every grow decision as a parallel ROOT (no parent_models), so exactly
    # one node per grow decision — no reduction (consolidation defaults keep the threshold high
    # enough that these deliberately-distinct synthetic tasks never merge).
    assert len(dump["nodes"]) == res["n_grow"]
    for ndump in dump["nodes"]:
        assert set(("index", "task_id", "is_root", "parent_indices", "state_dict")) <= set(ndump.keys())
        assert isinstance(ndump["state_dict"], dict)

    assert len(dump["predictors"]) == len(tasks)
    for pdump in dump["predictors"]:
        assert set(("task", "kind", "head_state", "node_index", "parent_indices",
                    "composer_kind", "composer_state", "search_meta")) <= set(pdump.keys())
        assert pdump["kind"] in ("grow", "reuse", "search", "update")


# ---------------------------------------------------------------------------
# 3. Update rung
# ---------------------------------------------------------------------------


def test_update_rung_fires_or_is_masked_by_grow(tmp_path):
    """Task 0 is under-trained (root_epochs=1); task 1 shares its direction with more data. With
    enable_update on, the frozen ladder's decision on task 1 is either replaced by "update" (the
    warm-started refinement out-competes reuse/search enough, and backward safety holds), or — if
    the frozen ladder itself chose "grow" — the probe is logged only and the decision is left
    untouched (the masking guard: update can never override a grow). Either branch is a valid
    outcome of this stochastic synthetic setup; we assert whichever one actually happened."""
    tasks = _update_stream(feature_dim=24, seed=0)
    cfg = _update_cfg(tmp_path, enable_update=True)
    res = run_exp3a_kan(cfg, tasks)

    d1 = res["decisions"][1]
    assert "update" in d1
    upd = d1["update"]
    assert upd["probed"] is True
    assert "backward_deltas_val" in upd and isinstance(upd["backward_deltas_val"], dict)
    assert "backward_deltas_test" in upd and isinstance(upd["backward_deltas_test"], dict)
    assert upd["parent"] == 0            # single root parent, index 0 into `nodes`
    assert upd["parent_task_id"] == 0

    if d1["decision"] == "update":
        print("update-rung test: branch = FIRED (frozen ladder was reuse/search, update selected)")
        assert upd["selected"] is True
        assert upd["logged_only"] is False
        assert upd["rel_update"] > cfg.eps_update
        assert upd["backward_safe"] is True
        assert res["n_update"] == 1
    else:
        print(f"update-rung test: branch = MASKED (frozen ladder chose {d1['decision']!r})")
        assert d1["decision"] == "grow"
        assert upd["selected"] is False
        assert upd["logged_only"] is True
        assert res["n_update"] == 0

    # n_update is consistent with the decisions list either way.
    assert res["n_update"] == sum(1 for d in res["decisions"] if d["decision"] == "update")


def test_update_rung_off_reproduces_frozen_ladder(tmp_path):
    """`enable_update=False` never runs the update block at all, so its decision on task 1 IS, by
    construction, the frozen ladder's decision. With `enable_update=True` on the identical
    tasks/config, task 1's decision must either match that frozen decision exactly (update
    probed-but-not-selected, or masked by grow) or be "update" (the frozen decision was overridden
    — only possible when the OFF run's frozen decision was "reuse" or "search")."""
    tasks = _update_stream(feature_dim=24, seed=0)

    cfg_off = _update_cfg(tmp_path / "off", enable_update=False)
    res_off = run_exp3a_kan(cfg_off, tasks)
    frozen_decision = res_off["decisions"][1]["decision"]
    assert "update" not in res_off["decisions"][1]
    assert res_off["n_update"] == 0

    cfg_on = _update_cfg(tmp_path / "on", enable_update=True)
    res_on = run_exp3a_kan(cfg_on, tasks)
    on_decision = res_on["decisions"][1]["decision"]

    if on_decision == "update":
        assert frozen_decision in ("reuse", "search")
    else:
        assert on_decision == frozen_decision


# ---------------------------------------------------------------------------
# 4. Masking guard — update can never override a grow decision
# ---------------------------------------------------------------------------


def test_update_never_overrides_grow(tmp_path):
    """A low eps_rel (eps_grow) biases the ladder toward "grow" on the update-stream's task 1
    (task 0's under-trained root is a weak enough obstruction that a fresh concept clears the
    grow threshold easily). update must then be probed-but-masked: selected False, logged_only
    True, and the recorded decision must stay "grow"."""
    tasks = _update_stream(feature_dim=24, seed=0)
    cfg = _update_cfg(tmp_path, eps_rel=0.15, enable_update=True)
    res = run_exp3a_kan(cfg, tasks)

    d1 = res["decisions"][1]
    assert d1["decision"] == "grow"          # frozen ladder chose grow on this eps_rel
    assert "update" in d1
    assert d1["update"]["probed"] is True
    assert d1["update"]["selected"] is False
    assert d1["update"]["logged_only"] is True
    assert res["n_update"] == 0


# ---------------------------------------------------------------------------
# 5. Regression — the update probe must not perturb the main RNG stream
# ---------------------------------------------------------------------------


def _update_probe_rng_stream(feature_dim=24, seed=0):
    """Three tasks sharing one under-trained root direction `w`. Task 0 is the forced root; tasks
    1 and 2 are both routed through it, so the update probe (root_parent_idxs non-empty) runs on
    each of them whenever enable_update=True."""
    torch.manual_seed(seed)
    w = torch.randn(feature_dim); w = w / w.norm()
    task0 = _binary_stream(feature_dim, w, n_train=80, n_test=100, seed=seed + 1)
    task1 = _binary_stream(feature_dim, w, n_train=120, n_test=100, seed=seed + 2)
    task2 = _binary_stream(feature_dim, w, n_train=150, n_test=100, seed=seed + 3)
    return [task0, task1, task2]


def test_update_probe_does_not_perturb_main_rng_stream(tmp_path):
    """Regression test for the update-probe RNG leak ([[reuse-with-update-arm-stress-test]]): with
    enable_update=True, `update_probe`'s training loop (randperm, deepcopy init) used to run on the
    GLOBAL torch RNG stream, so every task AFTER the probed one got a different init/batch order
    than the enable_update=False control — a borderline gate decision on a later task could then
    flip for an RNG reason, not because of the update rung. Fixed by running the probe +
    backward-safety measurement inside `torch.random.fork_rng`, reseeded deterministically from
    `cfg.seed * 1000 + t + 500` (mirrors `_oracle_rungs`'s fork_rng usage).

    `eps_update` is pinned absurdly high so `rel_update > eps_update` can never hold — the update
    rung is therefore NEVER selected on either task 1 or task 2, whatever the frozen ladder's own
    decision turns out to be. That keeps the DAG's actual state (nodes, predictors, params)
    IDENTICAL between the enable_update=False and enable_update=True runs, isolating the one thing
    that's allowed to differ: whether the probe's own RNG consumption leaks into the global stream
    and perturbs later tasks. With the fix, every gated task's decision and L_reuse_bits/L_grow_bits
    must be bit-for-bit identical between the two runs. Before the fix, this test fails on task 2
    (the task after the first probed one)."""
    tasks = _update_probe_rng_stream(feature_dim=24, seed=1)

    cfg_off = _update_cfg(tmp_path / "off", enable_update=False)
    cfg_off.n_tasks = 3
    res_off = run_exp3a_kan(cfg_off, tasks)

    cfg_on = _update_cfg(tmp_path / "on", enable_update=True)
    cfg_on.n_tasks = 3
    cfg_on.eps_update = 10.0   # rel_update in [-1, 1] roughly -> can never clear this -> never selected
    res_on = run_exp3a_kan(cfg_on, tasks)

    probed_any = False
    for t in range(1, 3):
        d_off = res_off["decisions"][t]
        d_on = res_on["decisions"][t]
        assert "update" not in d_off

        if "update" in d_on:
            # Masking guaranteed by eps_update, not incidental to this test.
            probed_any = True
            assert d_on["update"]["probed"] is True
            assert d_on["update"]["selected"] is False

        # The only thing left that could diverge is RNG leakage — assert it doesn't.
        assert d_on["decision"] == d_off["decision"]
        assert d_on["L_reuse_bits"] == d_off["L_reuse_bits"]
        assert d_on["L_grow_bits"] == d_off["L_grow_bits"]

    assert probed_any, "expected the update probe to run on at least one of tasks 1, 2"
    assert res_on["test_accs"] == res_off["test_accs"]


if __name__ == "__main__":
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        test_oracle_rungs_writes_all_three_and_matches_chosen(pathlib.Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_dump_gate_tensors_writes_gate_dump(pathlib.Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_update_rung_fires_or_is_masked_by_grow(pathlib.Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_update_rung_off_reproduces_frozen_ladder(pathlib.Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_update_never_overrides_grow(pathlib.Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_update_probe_does_not_perturb_main_rng_stream(pathlib.Path(td))
    print("\nUpdate/oracle/dump smoke tests passed.")
