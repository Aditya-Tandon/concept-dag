"""The linear-CKA merge pre-filter (`--merge_trigger cka`).

ref: [[merge-detector-separability]] (Track C item 1)

The published merge trigger is the mean top-k canonical correlation between two roots' outputs.
CCA is invariant to any invertible linear map, so on a pair whose class-carrying directions merely
*span* each other it saturates near 1 regardless of how much of each representation is actually
shared — which is why the H8 archive's merge rung fires (or does not) almost independently of real
redundancy. Linear CKA keeps the variance weighting and therefore separates those cases, at the
cost of one matmul on features the CCA path already collected.

Four things have to hold:

1. CKA is a similarity: 1 on a representation against itself and against an orthogonal rotation of
   itself (the invariance the merge's recovery adapter can undo for free), well below 1 on
   independent noise.
2. Both statistics are read off ONE draw. `_combined_loader` shuffles, so a second pass would
   score the two triggers on different samples *and* advance the run's RNG stream.
3. The trigger really gates, in both modes: an unreachable threshold stops every candidate, and
   every EXAMINED pair leaves exactly one record per scan — `trigger_rejected` where a bare
   `continue` used to be, which is what makes "the merge rung was silent" distinguishable from
   "no pair was ever considered", and what lets the other trigger's counterfactual be read off
   the archive without re-running the stream.
4. The default (`cca`) arm is untouched: same merges, same `similarity`, same numbers; it only
   gains the two logged statistics and the records for pairs it was already discarding.
"""

from __future__ import annotations

import torch

from concept_dag.experiments.kan_exp import (
    KanExpConfig, _cca_topk, _linear_cka, _paired_root_features, consolidate_nodes,
    run_exp3a_kan,
)
from tests.fixtures.cls_identity_stream import make_duplicate_cls_tasks


# ---------------------------------------------------------------------------
# 1. The statistic itself
# ---------------------------------------------------------------------------


def test_cka_is_one_on_itself_and_on_an_orthogonal_rotation():
    g = torch.Generator().manual_seed(0)
    F = torch.randn(256, 32, generator=g)
    assert abs(_linear_cka(F, F) - 1.0) < 1e-5

    # A random orthogonal 32x32 (QR of a Gaussian) — the map the merge's recovery adapter
    # inverts exactly, so a pair related by one must read as fully redundant.
    Q, _ = torch.linalg.qr(torch.randn(32, 32, generator=g))
    assert abs(_linear_cka(F, F @ Q) - 1.0) < 1e-5

    # ... and a pure rescale, which is not information the merge cares about either.
    assert abs(_linear_cka(F, 3.7 * F) - 1.0) < 1e-5


def test_cka_of_independent_gaussian_features_is_small():
    g = torch.Generator().manual_seed(1)
    Fa = torch.randn(512, 128, generator=g)
    Fb = torch.randn(512, 128, generator=g)
    cka = _linear_cka(Fa, Fb)
    assert 0.0 <= cka < 0.3, cka
    # The contrast the pre-filter exists for: top-k CCA on the SAME independent pair is far from
    # small, because 128 directions of 512 samples span a great deal of each other.
    assert _cca_topk(Fa, Fb, 8) > cka


def test_cka_is_symmetric_and_bounded():
    g = torch.Generator().manual_seed(2)
    Fa = torch.randn(64, 16, generator=g)
    Fb = Fa @ torch.randn(16, 16, generator=g) + 0.3 * torch.randn(64, 16, generator=g)
    assert abs(_linear_cka(Fa, Fb) - _linear_cka(Fb, Fa)) < 1e-6
    assert 0.0 <= _linear_cka(Fa, Fb) <= 1.0 + 1e-6
    # A constant (zero-variance) representation has no similarity to anything, and must not
    # divide by zero.
    assert _linear_cka(Fa, torch.ones(64, 16)) == 0.0


# ---------------------------------------------------------------------------
# 2. Both statistics come off one pass over one draw
# ---------------------------------------------------------------------------


def _two_roots_and_tasks(seed: int = 3):
    """A two-task feature stream and its two frozen roots, built without the gate."""
    from torch.utils.data import DataLoader, TensorDataset

    from concept_dag.experiments.exp3_growing_dag import DAGNode

    g = torch.Generator().manual_seed(seed)
    tasks, nodes = [], []
    for t in range(2):
        X = torch.randn(128, 12, generator=g)
        Y = torch.randint(0, 3, (128,), generator=g)
        loader = DataLoader(TensorDataset(X, Y), batch_size=16, shuffle=True)
        tasks.append({"task_id": t, "n_classes": 3, "train": loader, "test": loader})
        n = DAGNode(task_id=t, concept_dim=8, n_mlp_layers=2, use_cnn=False, feature_dim=12)
        n.eval()
        nodes.append(n)
    return nodes, tasks


def test_paired_root_features_uses_one_draw_for_both_concepts():
    nodes, tasks = _two_roots_and_tasks()
    from concept_dag.experiments.kan_exp import _combined_loader

    loader = _combined_loader(tasks, 0, 1, batch_size=32)
    Fa, Fb = _paired_root_features(nodes[0], nodes[1], loader, "cpu", max_batches=3)
    assert Fa.shape == Fb.shape == (96, 8)      # 3 batches of 32, both concepts, same rows
    assert Fa.dtype == Fb.dtype == torch.float32


# ---------------------------------------------------------------------------
# 3. cka mode gates, and records what it stops
# ---------------------------------------------------------------------------


def _consolidate(tmp_path, **kw):
    """Run the duplicate-task consolidating stream and return its final consolidation summary."""
    tasks = make_duplicate_cls_tasks()
    cfg = KanExpConfig(
        results_dir=str(tmp_path), device="cpu", backbone="dinov2_vits14",
        feature_dim=24, concept_dim=16, seed=42,
        root_epochs=12, child_epochs=12, gate_epochs=6,
        n_tasks=len(tasks), n_parents=2, routing_batches=4, gate_cache_max=256,
        subspace_k=3, batch_size=16, raw_grow_probe=True, enable_search=True, search_skip=True,
        oracle_rungs=True, consolidate_every=2, distill=True, distill_epochs=4,
        force_grow_ids=(len(tasks) - 1,), log_every=100000, **kw)
    return run_exp3a_kan(cfg, tasks)


def test_unreachable_cka_threshold_stops_every_candidate_and_records_it(tmp_path):
    res = _consolidate(tmp_path, merge_trigger="cka", merge_cka_threshold=1.1)

    ops = [op for p in res["consolidation_passes"] for op in p["ops"]]
    assert not any(op["op"] in ("merge", "merge_rejected") for op in ops), (
        "CKA is at most 1, so a threshold of 1.1 must stop every candidate before distill_merge")
    skipped = [op for op in ops if op["op"] == "trigger_rejected"]
    assert skipped, "the stream's duplicate pair must at least be CONSIDERED and recorded"
    for op in skipped:
        assert op["sim_kind"] == "cka"
        assert op["similarity"] == op["cka"]          # the trigger used the CKA value
        assert 0.0 <= op["cka"] <= 1.0
        assert 0.0 <= op["cca_topk"] <= 1.0           # ... and the CCA is recorded alongside
        assert op["threshold"] == 1.1
    assert res["consolidation"]["merge_accepted"] == 0
    assert res["consolidation"]["params_saved"] == 0


def test_every_examined_pair_leaves_exactly_one_record_in_cca_mode(tmp_path):
    """The default arm records the pairs it stops too — the counterfactual depends on it."""
    res = _consolidate(tmp_path, functional_threshold=0.999)   # nothing can clear this

    final = res["consolidation"]
    assert final["merge_attempted"] == 0
    stopped = [op for op in final["ops"] if op["op"] == "trigger_rejected"]
    assert stopped, "pairs were examined and left no trace"
    for op in stopped:
        assert op["sim_kind"] == "functional"
        assert op["similarity"] == op["cca_topk"]     # the trigger used the CCA value
        assert 0.0 <= op["cka"] <= 1.0                # ... and the CKA is recorded alongside
        assert op["threshold"] == 0.999
    # Exactly one record per examined pair: the final pass merges nothing, so it makes a single
    # scan over every unordered pair of the DAG's nodes.
    n_nodes = len(res["params_per_root"])
    assert len(stopped) == n_nodes * (n_nodes - 1) // 2
    assert len({(op["keep"], op["drop"]) for op in stopped}) == len(stopped)


def test_cka_mode_at_a_reachable_threshold_merges_and_carries_both_statistics(tmp_path):
    res = _consolidate(tmp_path, merge_trigger="cka", merge_cka_threshold=0.0)

    ops = [op for p in res["consolidation_passes"] for op in p["ops"]]
    merges = [op for op in ops if op["op"] in ("merge", "merge_rejected")]
    assert merges, "a threshold of 0 must let every candidate through to distill_merge"
    for op in merges:
        assert op["sim_kind"] == "cka"
        assert op["similarity"] == op["cka"]
        assert "cca_topk" in op


# ---------------------------------------------------------------------------
# 4. The default arm is untouched apart from the logged statistic
# ---------------------------------------------------------------------------


def test_default_trigger_ops_are_unchanged_apart_from_the_logged_cka(tmp_path):
    """The cca arm keeps its ops, its `similarity`, and its decisions; it only gains `cka`."""
    import json
    import os

    fixture = os.path.join(os.path.dirname(__file__), "fixtures",
                           "mlp_cls_consolidating_baseline.json")
    with open(fixture) as f:
        expected_ops = json.load(f)["consolidation"]["ops"]

    # Through JSON, as the fixture is: `backward_deltas` is keyed by int in memory.
    res = _consolidate(tmp_path / "cca")
    all_ops = json.loads(json.dumps(res["consolidation"]["ops"]))
    got = [op for op in all_ops if op["op"] != "trigger_rejected"]
    assert len(got) == len(expected_ops)
    for op, exp in zip(got, expected_ops):
        cka = op.pop("cka", None)                    # the two added fields
        cca = op.pop("cca_topk", None)
        assert op == exp
        if exp["op"].startswith("merge"):
            assert cka is not None and 0.0 <= cka <= 1.0
            assert cca is not None and op["similarity"] == cca
            assert exp["sim_kind"] == "functional"   # the trigger is still the CCA
    # Every pair this particular stream examines is accepted, so it has no `trigger_rejected`
    # record to show — which is the point: the new records appear exactly where a pair used to be
    # discarded silently, and nowhere else.
    assert all(op["op"] != "trigger_rejected" for op in all_ops)


def test_consolidate_nodes_defaults_to_the_published_trigger():
    """The new parameters are defaulted, so every existing caller keeps the cca path."""
    import inspect

    sig = inspect.signature(consolidate_nodes)
    assert sig.parameters["merge_trigger"].default == "cca"
    assert sig.parameters["merge_cka_threshold"].default == 0.55
