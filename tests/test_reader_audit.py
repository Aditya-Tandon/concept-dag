"""
Tests for the reader-audit gate ([[post-mint-audit-gap]] P5): every post-mint change to a node —
low-rank truncation, an accepted/rejected merge, an update-rung commit — is recorded in
`results["reader_audit"]`, and a truncation that hurts a reader is rolled back.

Background: `consolidate_nodes`' step (1) called `low_rank_factorize_final_layer` on every node at
every consolidation pass with no audit and no rollback at all — unlike accepted merges, which
already re-evaluate affected readers and roll back on a forgetting-gate failure. It fired zero
times across the whole H8 archive, so the fix must not cost a run in which nothing truncates any
extra test-set evaluations or RNG draws (A1): `consolidate_nodes` now probes on a `copy.deepcopy`
of the module to learn whether truncation would apply, and only pays for reader accuracies (before
truncating for real) when it would.

Also covers Amendment D: `test_accs`/`average_accuracy` price the DAG BEFORE its last
consolidation pass (each `test_accs[t]` is appended ahead of that iteration's `consolidate_every`
pass); `test_accs_final`/`average_accuracy_final` re-evaluate every task once more, after the run's
last consolidation, so a merge's or truncation's cost actually reaches them.

Covers:
  1. Nothing changes (`tests/fixtures/cls_identity_stream.make_cls_tasks`'s plain reference
     stream): `reader_audit == []` and `test_accs_final` == `test_accs`.
  2. A truncation forced to apply and found harmless: recorded `kind="truncate"`,
     `verdict="kept"`, correct readers, small drift/rel_error.
  3. A truncation forced to apply and found to hurt a reader (`reader_tolerance=0.0`): recorded
     `verdict="rolled_back"`, `node.concept_module.mlp` restored to the ORIGINAL module object,
     and the reader's accuracy is back to its pre-truncation value.
  4. No extra cost when nothing would truncate: `TaskPredictor.accuracy` is called exactly the
     same number of times whether `truncate_energy` is disabled (`None`) or enabled but inert.
  5. An accepted merge contributes a `reader_audit` entry and its cost reaches
     `test_accs_final`/`average_accuracy_final` (differing from `test_accs`/`average_accuracy`);
     every task whose recorded accuracy actually changed is named as a reader somewhere in
     `reader_audit`.
  6. A selected update-rung commit contributes a `reader_audit` entry
     (`kind="update_commit"`), matching its own `backward_deltas_val`/`_test`.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from concept_dag.experiments.exp3_growing_dag import DAGNode, train_node
from concept_dag.experiments.kan_exp import (
    KanExpConfig, TaskPredictor, consolidate_nodes, make_accuracy_accept_fn, run_exp3a_kan,
)
from concept_dag.models.baselines import LinearHead
from concept_dag.training.kan_gate import ReuseComposer

from tests.fixtures.cls_identity_stream import make_cls_tasks, make_duplicate_cls_tasks, run_reference
from tests.test_update_oracle_dump import _update_cfg, _update_stream


# ===========================================================================
# 1. Nothing changes
# ===========================================================================


def test_reader_audit_empty_and_final_accs_agree_when_nothing_consolidates(tmp_path):
    res = run_reference(str(tmp_path))
    assert res["reader_audit"] == []
    assert res["test_accs_final"] == res["test_accs"]
    assert res["average_accuracy_final"] == res["average_accuracy"]


# ===========================================================================
# Shared scaffolding for the direct `consolidate_nodes` truncation tests (2-4)
# ===========================================================================


def _make_low_rank_root(signal_at: int, kept: bool, concept_dim: int = 4):
    """A root DAGNode (`n_mlp_layers=1`, so its whole MLP is one Linear(hidden->out_dim)) whose
    final-layer weight is an EXACT rank-1 matrix built from a hand-picked SVD: `S = [100, 1, 1,
    0.01]` and `V`'s columns a coordinate permutation. At `energy=0.99` this keeps only the r=1
    top singular direction (`v0`) and saves parameters (72->... see the docstring math in the P5
    spec's own desk check). `signal_at` is the INPUT coordinate the two classes are separated
    along; `kept=True` places it on `v0` (survives rank-1 truncation), `kept=False` places it on
    `v3` (the direction rank-1 truncation drops). Returns `(node, head, loader)` with the head
    reading exactly the OUTPUT coordinate the signal lands on, so accuracy is 100% before any
    truncation regardless of which case this is.
    """
    assert concept_dim == 4, "the S/V construction below is hand-picked for concept_dim=4"
    node = DAGNode(task_id=0, concept_dim=concept_dim, cnn_out_dim=concept_dim, n_mlp_layers=1,
                   parent_models=None, soft_pca_k=4, use_cnn=False, feature_dim=concept_dim,
                   root_family="mlp_cls")
    U = torch.eye(concept_dim)
    V = torch.eye(concept_dim)[:, [1, 2, 3, 0]]   # v0=e1, v1=e2, v2=e3, v3=e0
    S = torch.tensor([100., 1., 1., 0.01])
    W = (U * S) @ V.T
    final = node.concept_module.mlp[-1]
    with torch.no_grad():
        final.weight.copy_(W)
        final.bias.zero_()
    node.freeze()

    # v0 pairs with input coord 1 (e1) and output coord 0 (u0=e0); v3 pairs with input coord 0
    # (e0) and output coord 3 (u3=e3).
    in_coord = 1 if kept else 0
    out_coord = 0 if kept else 3
    assert signal_at == in_coord or signal_at is None
    head = LinearHead(concept_dim, 2)
    with torch.no_grad():
        head.fc.weight.zero_()
        head.fc.weight[0, out_coord] = 1.0
        head.fc.weight[1, out_coord] = -1.0
        head.fc.bias.zero_()

    n = 20
    x_pos = torch.zeros(concept_dim); x_pos[in_coord] = 1.0
    x_neg = torch.zeros(concept_dim); x_neg[in_coord] = -1.0
    X = torch.cat([x_pos.unsqueeze(0).repeat(n, 1), x_neg.unsqueeze(0).repeat(n, 1)])
    Y = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    loader = DataLoader(TensorDataset(X, Y), batch_size=8)
    return node, head, loader


def _consolidate_single_node(node, head, loader, reader_tolerance, max_rel_error=0.05):
    tasks = [{"train": loader, "val": loader, "test": loader, "n_classes": 2}]
    predictor = TaskPredictor("grow", head, node=node)
    nodes = [node]
    predictors = [predictor]
    accept = make_accuracy_accept_fn(nodes, predictors, tasks, "cpu", tolerance=0.01)
    summ = consolidate_nodes(
        nodes, predictors, tasks, "cpu", accept_fn=accept, similarity_threshold=7.0,
        reader_tolerance=reader_tolerance, subspace_k=4, truncate_energy=0.99,
        truncate_max_rel_error=max_rel_error, distill=True, distill_epochs=1,
        merge_tolerance=0.01, functional_threshold=None)
    return summ, predictor, tasks


# ===========================================================================
# 2. A harmless forced truncation is applied and KEPT
# ===========================================================================


def test_truncation_kept_when_it_does_not_hurt_the_reader():
    node, head, loader = _make_low_rank_root(signal_at=1, kept=True)
    predictor = TaskPredictor("grow", head, node=node)
    assert predictor.accuracy(loader, "cpu") == 1.0

    summ, predictor, _ = _consolidate_single_node(node, head, loader, reader_tolerance=0.05)

    truncate_ops = [op for op in summ["ops"] if op["op"] == "truncate"]
    assert len(truncate_ops) == 1
    assert summ["params_saved"] > 0

    entries = [e for e in summ["reader_audit"] if e["kind"] == "truncate"]
    assert len(entries) == 1
    e = entries[0]
    assert e["node"] == node.task_id
    assert e["readers"] == [0]                 # the sole task whose predictor routes through it
    assert e["verdict"] == "kept"
    assert e["tolerance"] == 0.05
    assert e["rel_error"] is not None and e["rel_error"] < 0.05
    assert e["drift"] >= 0.0
    assert e["reader_deltas_test"] == {"0": 0.0}

    # The change actually happened and is functionally harmless.
    assert predictor.accuracy(loader, "cpu") == 1.0


# ===========================================================================
# 3. A forced truncation that HURTS a reader is rolled back
# ===========================================================================


def test_truncation_rolled_back_when_it_hurts_the_reader():
    node, head, loader = _make_low_rank_root(signal_at=0, kept=False)
    predictor = TaskPredictor("grow", head, node=node)
    acc_before = predictor.accuracy(loader, "cpu")
    assert acc_before == 1.0
    original_mlp = node.concept_module.mlp
    was_frozen = node.concept_module.is_frozen
    assert was_frozen is True

    summ, predictor, _ = _consolidate_single_node(node, head, loader, reader_tolerance=0.0)

    # No `truncate` op committed; a `truncate_rejected` one recorded instead.
    assert not any(op["op"] == "truncate" for op in summ["ops"])
    rejected = [op for op in summ["ops"] if op["op"] == "truncate_rejected"]
    assert len(rejected) == 1
    assert summ["params_saved"] == 0            # nothing actually shrank

    entries = [e for e in summ["reader_audit"] if e["kind"] == "truncate"]
    assert len(entries) == 1
    e = entries[0]
    assert e["readers"] == [0]
    assert e["verdict"] == "rolled_back"
    assert e["reader_deltas_test"]["0"] < 0     # the (would-be) hurt that triggered the rollback

    # The module is bit-for-bit restored: same object, same accuracy.
    assert node.concept_module.mlp is original_mlp
    assert node.concept_module.is_frozen is was_frozen
    assert predictor.accuracy(loader, "cpu") == acc_before == 1.0


# ===========================================================================
# 4. No extra cost (A1) when nothing would truncate
# ===========================================================================


def test_no_extra_forward_passes_when_nothing_would_truncate():
    torch.manual_seed(0)
    tasks = make_cls_tasks(n_tasks=2, n_classes=4, feature_dim=24, n_per_class=48, seed=0)

    def _build():
        torch.manual_seed(1)
        node0 = DAGNode(task_id=0, concept_dim=16, cnn_out_dim=16, n_mlp_layers=2,
                        parent_models=None, soft_pca_k=4, use_cnn=False, feature_dim=24,
                        root_family="mlp_cls")
        head0 = LinearHead(16, 4)
        train_node(node0, head0, tasks[0]["train"], 3, 3e-3, "cpu", 1000, name="t0")
        node0.compute_concept_subspace(tasks[0]["train"], "cpu", top_k=4, max_batches=4)
        node0.freeze()

        node1 = DAGNode(task_id=1, concept_dim=16, cnn_out_dim=16, n_mlp_layers=2,
                        parent_models=[node0], soft_pca_k=4, use_cnn=False, feature_dim=24,
                        root_family="mlp_cls")
        head1 = LinearHead(16, 4)
        train_node(node1, head1, tasks[1]["train"], 3, 3e-3, "cpu", 1000, name="t1")
        node1.compute_concept_subspace(tasks[1]["train"], "cpu", top_k=4, max_batches=4)
        node1.freeze()

        nodes = [node0, node1]
        predictors = [TaskPredictor("grow", head0, node=node0),
                     TaskPredictor("grow", head1, node=node1)]
        return nodes, predictors

    # Confirm the premise: at the default energy/max_rel_error neither node's final layer would
    # actually be truncated (matches "0 fires across the whole H8 archive").
    nodes, _ = _build()
    from concept_dag.training.consolidate import low_rank_factorize_final_layer
    for n in nodes:
        probe = low_rank_factorize_final_layer(copy.deepcopy(n.concept_module),
                                                energy=0.99, max_rel_error=0.05)
        assert not probe.get("applied"), "test premise violated: this node WOULD truncate"

    calls = {"n": 0}
    orig_accuracy = TaskPredictor.accuracy

    def _counting_accuracy(self, loader, device):
        calls["n"] += 1
        return orig_accuracy(self, loader, device)

    def _run(truncate_energy):
        calls["n"] = 0
        nodes, predictors = _build()
        accept = make_accuracy_accept_fn(nodes, predictors, tasks, "cpu", tolerance=0.01)
        TaskPredictor.accuracy = _counting_accuracy
        try:
            summ = consolidate_nodes(
                nodes, predictors, tasks, "cpu", accept_fn=accept, similarity_threshold=7.0,
                reader_tolerance=0.01, subspace_k=4, truncate_energy=truncate_energy,
                truncate_max_rel_error=0.05, distill=True, distill_epochs=1,
                merge_tolerance=0.01, functional_threshold=None)
        finally:
            TaskPredictor.accuracy = orig_accuracy
        return summ, calls["n"]

    summ_off, calls_off = _run(truncate_energy=None)      # old behaviour: truncation skipped
    summ_on, calls_on = _run(truncate_energy=0.99)         # new behaviour: probed but inert

    assert summ_off["reader_audit"] == []
    assert summ_on["reader_audit"] == []
    assert not any(op["op"].startswith("truncate") for op in summ_off["ops"])
    assert not any(op["op"].startswith("truncate") for op in summ_on["ops"])
    assert calls_on == calls_off, (
        f"enabling truncation cost {calls_on - calls_off} extra TaskPredictor.accuracy call(s) "
        "on a run where nothing actually truncates"
    )


# ===========================================================================
# 5. An accepted merge: reader_audit entry + test_accs_final actually differs
# ===========================================================================


def test_accepted_merge_contributes_reader_audit_and_moves_final_accs(tmp_path):
    tasks = make_duplicate_cls_tasks()
    cfg = KanExpConfig(
        results_dir=str(tmp_path), device="cpu", backbone="dinov2_vits14",
        feature_dim=24, concept_dim=16, seed=42,
        root_epochs=12, child_epochs=12, gate_epochs=6,
        n_tasks=len(tasks), n_parents=2, routing_batches=4, gate_cache_max=256,
        subspace_k=3, similarity_threshold=1.0, functional_redundancy=False,
        merge_tolerance=0.8,          # generous: accepts a real (non-freeze_keep) drifted merge
        batch_size=16, raw_grow_probe=True, enable_search=True, search_skip=True,
        oracle_rungs=True, consolidate_every=2, distill=True, distill_epochs=8,
        force_grow_ids=(len(tasks) - 1,), log_every=1000,
    )
    res = run_exp3a_kan(cfg, tasks=tasks)

    assert res["consolidation"]["params_saved"] > 0
    assert any(op["op"] == "merge" for op in res["consolidation"]["ops"])

    merge_entries = [e for e in res["reader_audit"] if e["kind"] == "merge"]
    assert merge_entries, "expected an accepted merge to contribute a reader_audit entry"
    for e in merge_entries:
        assert e["verdict"] in ("kept", "rolled_back")
        assert isinstance(e["readers"], list) and e["readers"]
        assert all(0 <= r < len(res["decisions"]) for r in e["readers"])
    kept = [e for e in merge_entries if e["verdict"] == "kept"]
    assert kept
    assert any(e["drift"] > 0.0 for e in kept), "expected a real (non-freeze_keep) weight drift"

    # D: the merge's cost, invisible to `test_accs`/`average_accuracy`, reaches the `_final` pair.
    assert res["test_accs_final"] != res["test_accs"]
    assert res["average_accuracy_final"] != res["average_accuracy"]

    # Every task whose recorded accuracy actually moved is named as a reader SOMEWHERE in the
    # audit trail — i.e. the entries' readers are not just any list, they cover the real effect.
    all_readers = {r for e in res["reader_audit"] for r in e["readers"]}
    moved = {t for t in range(len(tasks)) if res["test_accs_final"][t] != res["test_accs"][t]}
    assert moved, "test premise: expected at least one task's accuracy to actually move"
    assert moved <= all_readers


# ===========================================================================
# 6. A selected update-rung commit contributes a reader_audit entry
# ===========================================================================


def test_update_commit_contributes_reader_audit_entry(tmp_path):
    tasks = _update_stream(feature_dim=24, seed=0)
    cfg = _update_cfg(tmp_path, enable_update=True)
    res = run_exp3a_kan(cfg, tasks)

    d1 = res["decisions"][1]
    assert d1["decision"] == "update" and d1["update"]["selected"] is True

    commits = [e for e in res["reader_audit"] if e["kind"] == "update_commit"]
    assert len(commits) == 1
    e = commits[0]
    assert e["at_task"] == 1
    assert e["verdict"] == "kept"
    assert e["node"] == d1["update"]["parent_task_id"]
    assert e["drift"] > 0.0
    assert e["tolerance"] == cfg.update_tolerance
    assert e["reader_deltas_val"] == d1["update"]["backward_deltas_val"]
    assert e["reader_deltas_test"] == d1["update"]["backward_deltas_test"]
