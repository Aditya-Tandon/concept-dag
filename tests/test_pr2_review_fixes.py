"""Fixes for the PR #2 review (Aditya-Tandon/concept-dag#2, review comment of 2026-09-25).

  1. A merge must re-point search/update predictors too, else the dropped node is still run at
     inference while `params_saved` claims it was freed (`--merge_fixes`; the published path now
     records the leak on the op as `drop_still_read_by`).
  4. A merge must be accepted on VAL accuracy, not selected on test (`--merge_fixes`).
  5. A parameter count that includes each task's head/composer and the merge adapters
     (`--full_param_count`, reporting only).
  9. `--exp 3a-kan` must pass the search flags through instead of silently dropping them.

The default path is unchanged by all four: the whole-dict identity fixtures
(`test_mlp_cls_byte_identity.py`, `test_dump_flag_identity.py`) pass untouched, and the flagged
keys are written only when their flag is set.
"""

from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn

import concept_dag.experiments.kan_exp as ke
from concept_dag.experiments.kan_exp import (TaskPredictor, _full_param_count, _merge_nodes,
                                             _restore_topology, _snapshot_topology,
                                             make_accuracy_accept_fn, run_exp3a_kan)
from tests.test_provisional_gate import _base_cfg, _make_feature_tasks


# ---------------------------------------------------------------------------
# Minimal stand-ins: _merge_nodes touches only parent_models / parent_adapters / n_parents
# ---------------------------------------------------------------------------


class _Node(nn.Module):
    def __init__(self, name, parents=(), width=4):
        super().__init__()
        self.name = name
        self.parent_models = list(parents)
        self.parent_adapters = None
        self.concept_module = types.SimpleNamespace(n_parents=len(self.parent_models),
                                                    state_dict=lambda: {})
        self.lin = nn.Linear(width, width)


def _world():
    keep, drop, other = _Node("keep"), _Node("drop"), _Node("other")
    nodes = [keep, drop, other]
    preds = [
        TaskPredictor("grow", nn.Linear(4, 2), node=keep),
        TaskPredictor("grow", nn.Linear(4, 2), node=drop),
        TaskPredictor("reuse", nn.Linear(4, 2), parents=[keep, drop], composer=nn.Linear(8, 2)),
        TaskPredictor("search", nn.Linear(4, 2), parents=[drop, other], composer=nn.Linear(8, 2)),
        TaskPredictor("update", nn.Linear(4, 2), parents=[drop], composer=nn.Linear(4, 2)),
    ]
    return nodes, preds, keep, drop, other


# ---------------------------------------------------------------------------
# Finding 1 — re-point every parent-reading predictor
# ---------------------------------------------------------------------------


def test_published_merge_leaves_search_and_update_reading_the_dropped_node_and_says_so():
    nodes, preds, keep, drop, other = _world()
    still = _merge_nodes(nodes, preds, keep=keep, drop=drop)
    assert drop not in nodes
    assert still == [3, 4], "the published merge must REPORT the leak, not hide it"
    assert any(q is drop for q in preds[3].parents)            # search still runs `drop`
    assert all(q is keep for q in preds[2].parents)            # reuse was re-pointed


def test_repoint_all_readers_frees_the_dropped_node():
    nodes, preds, keep, drop, other = _world()
    W_drop = nn.Linear(4, 4)
    still = _merge_nodes(nodes, preds, keep=keep, drop=drop, W_drop=W_drop,
                         repoint_all_readers=True)
    assert still == []
    assert [q.name for q in preds[3].parents] == ["keep", "other"]
    assert [q.name for q in preds[4].parents] == ["keep"]
    assert preds[3].parent_adapters[0] is W_drop and preds[3].parent_adapters[1] is None
    assert preds[1].node is keep                                # drop's own task reads keep


def test_rollback_restores_search_and_update_edges():
    nodes, preds, keep, drop, other = _world()
    snap = _snapshot_topology(nodes, preds, keep=keep)
    snap["keep_id"] = None                                      # stand-ins carry no weights
    _merge_nodes(nodes, preds, keep=keep, drop=drop, repoint_all_readers=True)
    _restore_topology(nodes, preds, snap)
    assert drop in nodes
    assert [q.name for q in preds[3].parents] == ["drop", "other"]
    assert [q.name for q in preds[4].parents] == ["drop"]
    assert preds[1].node is drop


# ---------------------------------------------------------------------------
# Finding 4 — the accept decision is measured on the requested split
# ---------------------------------------------------------------------------


class _SpyPred:
    kind = "reuse"

    def __init__(self, log):
        self.parents, self.log = [], log

    def accuracy(self, loader, device):
        self.log.append(loader)
        return 1.0


def _spy_accept(split):
    log = []
    tasks = [{"val": "VAL", "test": "TEST"}]
    fn = (make_accuracy_accept_fn([], [_SpyPred(log)], tasks, "cpu", split=split) if split
          else make_accuracy_accept_fn([], [_SpyPred(log)], tasks, "cpu"))
    assert fn(None, None, {0}) is True
    return log


def test_accept_fn_decides_on_test_by_default_and_on_val_when_asked():
    assert set(_spy_accept(None)) == {"TEST"}
    assert set(_spy_accept("val")) == {"VAL"}


# ---------------------------------------------------------------------------
# Finding 5 — the full count sees composers and adapters, once each
# ---------------------------------------------------------------------------


def _n(m):
    return sum(p.numel() for p in m.parameters())


def test_full_param_count_includes_composers_adapters_and_dedups():
    nodes, preds, keep, drop, other = _world()
    base = sum(_n(x) for x in nodes)
    extra = sum(_n(p.head) + (_n(p.composer) if p.composer is not None else 0) for p in preds)
    assert _full_param_count(nodes, preds) == base + extra
    # a composer that holds the task head is counted once, not twice
    preds[2].head = preds[2].composer
    assert _full_param_count(nodes, preds) == base + extra - _n(nn.Linear(4, 2))


def test_full_param_count_still_counts_a_node_the_published_merge_leaked():
    nodes, preds, keep, drop, other = _world()
    before = _full_param_count(nodes, preds)
    _merge_nodes(nodes, preds, keep=keep, drop=drop)             # published: search still reads drop
    assert _full_param_count(nodes, preds) == before, "a leaked node is not a saving"
    nodes, preds, keep, drop, other = _world()
    _merge_nodes(nodes, preds, keep=keep, drop=drop, repoint_all_readers=True)
    assert _full_param_count(nodes, preds) == before - _n(drop)


# ---------------------------------------------------------------------------
# End to end: flagged run self-identifies and reports; default run is unchanged
# ---------------------------------------------------------------------------


def test_flagged_run_self_identifies_and_reports_full_params(tmp_path):
    tasks = _make_feature_tasks(n_tasks=4, seed=7)
    res = run_exp3a_kan(_base_cfg(tmp_path / "on", consolidate_every=1, force_grow_ids=(1, 2, 3),
                                  merge_fixes=True, full_param_count=True), tasks)
    assert res["merge_fixes"] is True
    assert len(res["param_curve_full"]) == len(tasks)
    assert all(f >= t for f, t in zip(res["param_curve_full"], res["param_curve_total"]))
    assert res["params_full_final"] > 0
    ops = [op for p in res["consolidation_passes"] for op in p["ops"]]
    assert sum(op.get("gate_split") == "val" for op in ops) >= 1, "no distilled merge exercised"
    for op in ops:
        assert "drop_still_read_by" not in op
        if op["op"] in ("merge", "merge_rejected") and "backward_deltas" in op:
            assert op["gate_split"] == "val" and "backward_deltas_test" in op


def test_default_run_writes_none_of_the_new_keys(tmp_path):
    tasks = _make_feature_tasks(n_tasks=4, seed=7)
    res = run_exp3a_kan(_base_cfg(tmp_path / "off", consolidate_every=1,
                                  force_grow_ids=(1, 2, 3)), tasks)
    assert not {"merge_fixes", "param_curve_full", "params_full_final"} & set(res)
    ops = [op for p in res["consolidation_passes"] for op in p["ops"]]
    assert not any("gate_split" in op or "backward_deltas_test" in op for op in ops)


# ---------------------------------------------------------------------------
# Finding 9 — 3a-kan passes the search flags through
# ---------------------------------------------------------------------------


def test_3a_kan_cli_passes_search_and_review_flags(monkeypatch):
    import concept_dag.data.loaders as loaders
    import run_experiment

    seen = {}
    monkeypatch.setattr(loaders, "make_split_cifar100", lambda **kw: [])
    monkeypatch.setattr(ke, "run_exp3a_kan", lambda cfg, tasks: seen.setdefault("cfg", cfg))
    monkeypatch.setattr(sys, "argv", [
        "run_experiment.py", "--exp", "3a-kan", "--device", "cpu", "--n_tasks", "2",
        "--enable_search", "--eps_search", "0.07", "--search_budget", "3", "--search_skip",
        "--search_device_rng_fix", "--merge_fixes", "--full_param_count"])
    run_experiment.main()
    cfg = seen["cfg"]
    assert cfg.enable_search is True and cfg.eps_search == 0.07 and cfg.search_budget == 3
    assert cfg.search_skip is True and cfg.search_device_rng_fix is True
    assert cfg.merge_fixes is True and cfg.full_param_count is True
