"""
Tests for the attention-pooling DAG root adoption (token-mode `root_family="attn_pool"`).

Context: `concept_dag/modules/token_roots.py` already held the five root families from the H6
module-family ablation (see `tests/test_module_family.py`); this change WIRES arm C
(`AttnPoolRoot`) into the live DAG/gate path behind `Exp3Config.root_family` /
`KanExpConfig.root_family`, so `DAGNode` can read DINOv2 patch tokens through learned-query
attention pooling instead of only the CLS vector. Everything here runs on CPU in seconds: no
encoder, no network, no torch.hub — token streams are synthesised exactly as
`feature_cache.cache_features(tokens=True)` would hand them to the DAG (a task dict with
`train/val/test` DataLoaders yielding `(B, 1 + T, feature_dim)` tensors).

Covers:
  1. CLI/config defaults keep the published CLS-vector root unless opted in.
  2. A token-mode DAG smoke run end-to-end through `run_exp3a_kan` (structure, not which rung wins).
  3. The raw-root grow probe is capacity- and shape-matched to the root family it would deploy,
     in BOTH token mode (attn_pool) and the published CLS mode (mlp_cls) — the load-bearing test.
  4. The update rung explicitly refuses token roots (`pack_update_input`, `run_exp3a_kan`).
  5. The token feature cache stays float16 in RAM and round-trips to identical float32 values.
  6. `--dump_gate_tensors` is force-disabled in token mode (loudly, not silently).
  7. ... and is unaffected in CLS mode, so test 6 is about token mode, not a broken flag.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from concept_dag.experiments.exp3_growing_dag import DAGNode
from concept_dag.experiments.kan_exp import KanExpConfig, run_exp3a_kan
from concept_dag.modules.concept_module import ConceptModule
from concept_dag.modules.token_roots import AttnPoolRoot
from concept_dag.training.kan_gate import pack_update_input


# ---------------------------------------------------------------------------
# Synthetic task builders (no encoder, no network — matches feature_cache's schema)
# ---------------------------------------------------------------------------


def _make_synthetic_token_tasks(
    n_tasks: int = 4,
    n_classes: int = 4,
    feature_dim: int = 24,
    n_tokens: int = 7,
    n_per_class: int = 48,
    batch_size: int = 16,
    seed: int = 0,
):
    """Token-mode task dicts with the `feature_cache.cache_features(tokens=True)` schema:
    `task_id, n_classes, name, train, val, test, feature_dim, n_tokens`. Class signal is put on
    BOTH the CLS slot (index 0) and one patch token (index 1) so attention pooling has to read
    more than the CLS position to find all of it."""
    T = n_tokens - 1
    g = torch.Generator().manual_seed(seed)
    tasks = []
    for t in range(n_tasks):
        toks, labels = [], []
        for c in range(n_classes):
            cls_center = torch.randn(feature_dim, generator=g) * 3.0
            patch_center = torch.randn(feature_dim, generator=g) * 3.0
            x = torch.randn(n_per_class, n_tokens, feature_dim, generator=g) * 0.5
            x[:, 0, :] += cls_center      # CLS carries class signal
            x[:, 1, :] += patch_center    # one patch token carries (different) class signal
            toks.append(x)
            labels.append(torch.full((n_per_class,), c, dtype=torch.long))
        X = torch.cat(toks)
        Y = torch.cat(labels)
        perm = torch.randperm(len(X), generator=g)
        X, Y = X[perm], Y[perm]

        n = len(X)
        n_tr, n_val = int(0.6 * n), int(0.2 * n)

        def _loader(lo, hi, shuffle):
            return DataLoader(TensorDataset(X[lo:hi], Y[lo:hi]), batch_size=batch_size,
                              shuffle=shuffle)

        tasks.append({
            "task_id": t,
            "train": _loader(0, n_tr, True),
            "val": _loader(n_tr, n_tr + n_val, False),
            "test": _loader(n_tr + n_val, n, False),
            "n_classes": n_classes,
            "name": f"synthetic-token-task-{t}",
            "feature_dim": feature_dim,
            "n_tokens": n_tokens,
        })
    return tasks


def _make_synthetic_cls_tasks(
    n_tasks: int = 2,
    n_classes: int = 4,
    feature_dim: int = 20,
    n_per_class: int = 40,
    batch_size: int = 16,
    seed: int = 0,
):
    """CLS-mode (2-D feature) task dicts, matching `tests/test_feature_mode_smoke.py`'s
    generator, used to confirm the dump-disable in test 6 is a token-mode-only effect."""
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
            return DataLoader(TensorDataset(X[lo:hi], Y[lo:hi]), batch_size=batch_size,
                              shuffle=shuffle)

        tasks.append({
            "task_id": t,
            "train": _loader(0, n_tr, True),
            "val": _loader(n_tr, n_tr + n_val, False),
            "test": _loader(n_tr + n_val, n, False),
            "n_classes": n_classes,
            "name": f"synthetic-cls-task-{t}",
            "feature_dim": feature_dim,
        })
    return tasks


# ===========================================================================
# 1. CLI / config defaults keep the published root
# ===========================================================================


def test_cli_defaults_keep_the_published_root():
    from run_experiment import build_parser

    parser = build_parser()
    args = parser.parse_args([])
    assert args.root_family == "mlp_cls"
    assert args.token_pool == 2

    args2 = parser.parse_args(["--root_family", "attn_pool"])
    assert args2.root_family == "attn_pool"

    cfg = KanExpConfig()
    assert cfg.root_family == "mlp_cls"
    assert cfg.n_tokens is None


# ===========================================================================
# 2. Token-mode DAG smoke run through run_exp3a_kan
# ===========================================================================


def test_token_mode_dag_smoke(tmp_path):
    feature_dim = 24
    n_tokens = 7
    tasks = _make_synthetic_token_tasks(n_tasks=4, n_classes=4, feature_dim=feature_dim,
                                        n_tokens=n_tokens, n_per_class=48, batch_size=16, seed=0)

    cfg = KanExpConfig(
        backbone="dinov2_vits14", feature_dim=feature_dim, n_tokens=n_tokens,
        root_family="attn_pool", concept_dim=16, root_epochs=3, child_epochs=3,
        gate_epochs=3, n_parents=2, routing_batches=4, gate_cache_max=256, batch_size=16,
        raw_grow_probe=True, enable_search=True, search_skip=True, oracle_rungs=True,
        log_every=100, device="cpu", results_dir=str(tmp_path),
    )
    results = run_exp3a_kan(cfg, tasks)

    assert len(results["decisions"]) == len(tasks)     # one decision per task
    assert results["root_family"] == "attn_pool"
    assert results["n_tokens"] == n_tokens

    gated = [d for d in results["decisions"] if d["task"] > 0]
    assert gated, "expected at least one gated (non-root) task"
    for d in gated:
        assert d["grow_probe_input"] == "raw-root"
        assert isinstance(d["gate_cache_n"], int) and d["gate_cache_n"] > 0
        assert isinstance(d["gate_seconds"], float) and d["gate_seconds"] >= 0
        if "oracle_accs" in d:
            for v in d["oracle_accs"].values():
                float(v)   # parses as a float

    curve, curve_total = results["param_curve"], results["param_curve_total"]
    assert len(curve) == len(curve_total) == len(tasks)
    assert all(t >= c for c, t in zip(curve, curve_total))
    assert curve_total[0] > curve[0]   # the root's AttentionPool is counted only in the total


# ===========================================================================
# 3. Grow probe family matches the deployed root (capacity match) — load-bearing
# ===========================================================================


def test_grow_probe_family_matches_the_deployed_root():
    # --- token mode: DAGNode(root_family="attn_pool") vs AttnPoolRoot ---
    feature_dim, concept_dim = 384, 128
    node = DAGNode(task_id=0, concept_dim=concept_dim, parent_models=None, use_cnn=False,
                  feature_dim=feature_dim, root_family="attn_pool")
    probe = AttnPoolRoot(feature_dim=feature_dim, concept_dim=concept_dim)

    node_total = sum(p.numel() for p in node.parameters())
    probe_total = sum(p.numel() for p in probe.parameters())
    assert node_total == probe_total

    node_shapes = sorted(tuple(p.shape) for p in node.parameters())
    probe_shapes = sorted(tuple(p.shape) for p in probe.parameters())
    assert node_shapes == probe_shapes

    x = torch.randn(5, 65, feature_dim)
    assert node(x).shape == (5, concept_dim)
    assert probe(x).shape == (5, concept_dim)

    # --- CLS mode: DAGNode(root_family="mlp_cls") vs the published ConceptModule probe ---
    cls_node = DAGNode(task_id=0, concept_dim=concept_dim, parent_models=None, use_cnn=False,
                       feature_dim=feature_dim, root_family="mlp_cls")
    assert cls_node.root_pool is None
    assert cls_node.token_mode is False

    cls_probe = ConceptModule(module_id="__root_probe__", in_dim=feature_dim,
                              hidden_dim=concept_dim, out_dim=concept_dim, n_layers=2,
                              n_parents=0)
    cls_node_total = sum(p.numel() for p in cls_node.parameters())
    cls_probe_total = sum(p.numel() for p in cls_probe.parameters())
    assert cls_node_total == cls_probe_total


# ===========================================================================
# 4. The update rung refuses token roots
# ===========================================================================


def test_update_rung_refuses_token_roots(tmp_path):
    parent_stack = torch.randn(8, 2, 16)
    raw_stack = torch.randn(8, 7, 24)   # 3-D: a token set, not a 2-D feature matrix
    with pytest.raises(NotImplementedError) as excinfo:
        pack_update_input(parent_stack, raw_stack)
    msg = str(excinfo.value)
    assert "attn_pool" in msg or "mlp_cls" in msg

    cfg = KanExpConfig(
        backbone="dinov2_vits14", feature_dim=24, n_tokens=7, root_family="attn_pool",
        concept_dim=16, enable_update=True, results_dir=str(tmp_path), device="cpu",
    )
    with pytest.raises(NotImplementedError):
        run_exp3a_kan(cfg, tasks=[])   # must raise before touching the (empty) task list


# ===========================================================================
# 5. Token feature cache stays half in RAM, yields identical float32 values
# ===========================================================================


def test_token_feature_cache_stays_half_in_memory_but_yields_identical_floats(tmp_path):
    from concept_dag.data.feature_cache import cache_features

    T, D = 3, 5
    n_tok = T + 1

    class _StubTokenEncoder(nn.Module):
        """Deterministic pure function of `x` (not of call order), so a cache-miss encoding
        and an independent re-encoding for comparison always agree."""
        feature_dim = D
        n_tokens = n_tok

        def forward(self, x):
            B = x.shape[0]
            base = x.reshape(B, -1).mean(dim=1).view(B, 1, 1)
            offsets = torch.arange(n_tok * D, dtype=torch.float32).view(1, n_tok, D)
            return base.expand(B, n_tok, D) + offsets

    enc = _StubTokenEncoder()
    images = torch.randn(9, 3, 4, 4)
    labels = torch.randint(0, 3, (9,))

    def _dl(shuffle):
        # batch_size > len(images) so every split comes back as ONE batch, in dataset order.
        return DataLoader(TensorDataset(images, labels), batch_size=32, shuffle=shuffle)

    task = {"task_id": 0, "train": _dl(True), "val": _dl(False), "test": _dl(False),
           "n_classes": 3, "name": "t0"}

    cache_dir = str(tmp_path / "cache")
    tasks = cache_features(enc, [task], cache_dir=cache_dir, device="cpu",
                           tokens=True, token_pool=2)

    assert os.path.isdir(f"{cache_dir}_tok{T}")

    out = tasks[0]
    assert out["train"].dataset.features.dtype is torch.float16
    assert out["n_tokens"] == n_tok
    assert out["feature_dim"] == D

    xb, yb = next(iter(out["test"]))
    assert xb.dtype == torch.float32
    assert xb.shape == (9, n_tok, D)
    assert torch.equal(yb, labels)

    expected = enc(images).half().float()
    assert torch.equal(xb, expected)


# ===========================================================================
# 6 / 7. --dump_gate_tensors: disabled (loudly) in token mode, unaffected in CLS mode
# ===========================================================================


def test_dump_gate_tensors_is_disabled_in_token_mode(tmp_path, capsys):
    feature_dim, n_tokens = 24, 7
    tasks = _make_synthetic_token_tasks(n_tasks=2, n_classes=4, feature_dim=feature_dim,
                                        n_tokens=n_tokens, n_per_class=48, batch_size=16, seed=1)
    cfg = KanExpConfig(
        backbone="dinov2_vits14", feature_dim=feature_dim, n_tokens=n_tokens,
        root_family="attn_pool", concept_dim=16, root_epochs=2, child_epochs=2, gate_epochs=2,
        n_parents=1, routing_batches=2, gate_cache_max=128, batch_size=16,
        dump_gate_tensors=True, log_every=1000, device="cpu", results_dir=str(tmp_path),
    )
    run_exp3a_kan(cfg, tasks)

    assert not os.path.exists(os.path.join(str(tmp_path), "gate_dump.pt"))
    out = capsys.readouterr().out
    assert "dump_gate_tensors" in out


def test_cls_mode_dump_still_written(tmp_path):
    tasks = _make_synthetic_cls_tasks(n_tasks=2, n_classes=4, feature_dim=20, n_per_class=40,
                                      batch_size=16, seed=1)
    cfg = KanExpConfig(
        backbone="dinov2_vits14", feature_dim=20, root_family="mlp_cls", concept_dim=16,
        root_epochs=2, child_epochs=2, gate_epochs=2, n_parents=1, routing_batches=2,
        gate_cache_max=128, batch_size=16, dump_gate_tensors=True, log_every=1000,
        device="cpu", results_dir=str(tmp_path),
    )
    run_exp3a_kan(cfg, tasks)

    assert os.path.exists(os.path.join(str(tmp_path), "gate_dump.pt"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
