"""
Tests for [[consolidation-provenance]] (P6 fix): intermediate consolidation passes are no
longer thrown away, and `_resolve_provisional`'s "merge" inference is now backed by the actual
accepted merge op rather than pure absence-inference.

Background: `run_exp3a_kan` calls `consolidate_nodes(...)` once per `cfg.consolidate_every`
tasks, but only the FINAL pass's return value used to reach `results["consolidation"]`. On
`s_interleave` (run with `--consolidate_every 2`) accepted merges happened in the pass after
task 3 and were therefore invisible in the results JSON — see
`concept-dag-results/results_provisional_2026-09-09/int_evalue/seed_49.run.log`, which shows
`[consolidate @ task 3] saved 33280 params, 8 ops` while that run's JSON `consolidation` block
has 3 ops, all `merge_rejected`, `params_saved` 0.

Covers:
  1. `consolidation_passes` collects every pass (>= 2 here), `results["consolidation"]` equals
     the last entry, and every pass carries `merge_attempted`/`merge_accepted`/`merge_rejected`
     derived from its own `ops`.
  2. A forced-duplicate stream where an accepted merge happens in an INTERMEDIATE (non-final)
     pass and drops a provisional root: the op appears in `consolidation_passes`, and the
     dropped root's `provisional_roots` entry carries `merged_into` / `merge_pass_at_task`.
  3. `scripts/eval_provisional.py`'s P6 gate on hand-built synthetic JSONs: a legacy JSON (no
     `consolidation_passes`, matching the whole 2026-09-09 archive) falls back to the final pass
     only, says so in `note`, and correctly flags the resolved_by_merge / accepted-merge-ops
     cross-check mismatch that the archive's `int_evalue` seeds actually exhibit; a new-format
     JSON with multiple passes reports the true merge_attempted/accepted/rejected totals and no
     mismatch.
"""

from __future__ import annotations

import importlib.util
import os

from concept_dag.experiments.kan_exp import KanExpConfig, run_exp3a_kan

from tests.test_provisional_gate import _make_feature_tasks, _base_cfg

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")


def _load_script(name: str, filename: str):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eval_provisional = _load_script("eval_provisional", "eval_provisional.py")


# ===========================================================================
# 1. Every pass is recorded, with per-pass merge counters
# ===========================================================================


def test_consolidation_passes_recorded_with_merge_counters(tmp_path):
    tasks = _make_feature_tasks(n_tasks=4, seed=7)
    cfg = _base_cfg(tmp_path, consolidate_every=1, force_grow_ids=(1, 2, 3))
    res = run_exp3a_kan(cfg, tasks)

    passes = res["consolidation_passes"]
    assert len(passes) >= 2, "expected at least 2 consolidation passes (>2 nodes after task 2)"
    assert passes[-1] == res["consolidation"], (
        "results['consolidation'] must stay the final pass's dict, byte-identical to the last "
        "consolidation_passes entry")
    assert passes[-1]["final"] is True
    assert all(p["final"] is False for p in passes[:-1])
    for p in passes:
        assert {"merge_attempted", "merge_accepted", "merge_rejected", "ops", "at_task"} <= set(p)
        assert p["merge_accepted"] == sum(1 for op in p["ops"] if op["op"] == "merge")
        assert p["merge_rejected"] == sum(1 for op in p["ops"] if op["op"] == "merge_rejected")
        assert p["merge_attempted"] >= p["merge_accepted"] + p["merge_rejected"]


# ===========================================================================
# 2. A provisional root merged away in an intermediate pass carries provenance
# ===========================================================================


def test_provisional_root_merged_in_intermediate_pass_records_provenance(tmp_path):
    """Task 1 mints a provisional root (arm 'always'); task 2 is an EXACT duplicate of task 1's
    distribution (dict-copied, the same technique `tests/fixtures/cls_identity_stream.py`'s
    `make_duplicate_cls_tasks` uses on task 0), force-grown as a parallel root so it can be a
    merge candidate. `raw_grow_probe=True` makes task 1's natural grow ALSO mint a parallel root
    (not a child stacked on the root's output), which is required for the two to be functionally
    comparable at all — see `run_reference`/`run_reference_consolidating` in the same fixture.
    """
    tasks = _make_feature_tasks(n_tasks=3, n_classes=4, feature_dim=24, n_per_class=48, seed=5)
    dup = dict(tasks[1])
    dup["task_id"] = 2
    dup["name"] = "dup-of-1"
    tasks[2] = dup

    cfg = _base_cfg(
        tmp_path, n_tasks=3, feature_dim=24, root_epochs=12, child_epochs=12, gate_epochs=6,
        subspace_k=3, provisional="always", enable_search=True, search_skip=True,
        always_n_max=1000, raw_grow_probe=True, consolidate_every=1, distill=True,
        distill_epochs=4, force_grow_ids=(2,),
    )
    res = run_exp3a_kan(cfg, tasks)

    passes = res["consolidation_passes"]
    assert len(passes) >= 2
    intermediate = [p for p in passes if not p["final"]]
    accepted_intermediate = [op for p in intermediate for op in p["ops"] if op["op"] == "merge"]
    assert accepted_intermediate, "expected an accepted merge in an intermediate (non-final) pass"

    root_1 = next(r for r in res["provisional_roots"] if r["minted_at"] == 1)
    assert root_1["resolution"] == "merge"
    assert root_1["merged_into"] is not None
    assert root_1["merge_pass_at_task"] is not None
    assert root_1["resolved_at"] == root_1["merge_pass_at_task"]

    # The op that actually dropped task 1 is the one `merged_into` must name.
    dropping_ops = [op for op in accepted_intermediate if op["drop"] == 1]
    assert dropping_ops, "no accepted intermediate-pass merge op names task 1 as `drop`"
    assert root_1["merged_into"] == dropping_ops[0]["keep"]


# ===========================================================================
# 3. eval_provisional.py's P6 gate: legacy fallback + cross-check
# ===========================================================================


def _new_format_json():
    """A run whose accepted merge (dropping the minted_at=1 provisional root) happened in an
    INTERMEDIATE pass, correctly recoverable because `consolidation_passes` is present."""
    return {
        "provisional_roots": [
            {"minted_at": 1, "resolution": "merge", "resolved_at": 2},
        ],
        "consolidation_passes": [
            {"at_task": 1, "final": False,
             "ops": [{"op": "merge_rejected", "keep": 0, "drop": 2}],
             "merge_attempted": 1, "merge_accepted": 0, "merge_rejected": 1},
            {"at_task": 2, "final": True,
             "ops": [{"op": "merge", "keep": 0, "drop": 1}],
             "merge_attempted": 1, "merge_accepted": 1, "merge_rejected": 0},
        ],
        "consolidation": {"ops": [{"op": "merge", "keep": 0, "drop": 1}],
                          "merge_attempted": 1, "merge_accepted": 1, "merge_rejected": 0},
        "decisions": [{"task": 3, "decision": "reuse"}],
    }


def _legacy_json_clean():
    """A legacy JSON (no `consolidation_passes`) where the accepted merge IS in the final pass —
    the `ctrl_evalue` case the spec's ground truth describes (each final consolidation contains
    exactly one accepted merge). No cross-check mismatch expected."""
    return {
        "provisional_roots": [
            {"minted_at": 1, "resolution": "merge", "resolved_at": 4},
        ],
        "consolidation": {"ops": [
            {"op": "merge", "keep": 0, "drop": 1},
            {"op": "merge_rejected", "keep": 0, "drop": 2},
        ]},
        "decisions": [{"task": 3, "decision": "reuse"}],
    }


def _legacy_json_mismatched():
    """A legacy JSON reproducing the real `int_evalue seed_49` predicament: the run log showed
    an accepted merge mid-stream (`[consolidate @ task 3] saved 33280 params, 8 ops`) that
    dropped the minted_at=3 provisional root, but the archived JSON's `consolidation` field is
    the FINAL pass only — 3 ops, all `merge_rejected` — so the accepted merge is not recoverable
    from this JSON. `resolved_by_merge` (from provisional_roots) and the ops-derived cross-check
    therefore disagree for a reason that is about the archive's FORMAT, not its content."""
    return {
        "provisional_roots": [
            {"minted_at": 3, "resolution": "merge", "resolved_at": 4},
        ],
        "consolidation": {"ops": [
            {"op": "merge_rejected", "keep": 0, "drop": 1},
            {"op": "merge_rejected", "keep": 0, "drop": 2},
            {"op": "merge_rejected", "keep": 1, "drop": 2},
        ]},
        "decisions": [{"task": 3, "decision": "reuse"}],
    }


def test_p6_new_format_json_reports_totals_across_passes():
    gate = eval_provisional.gate_p6({43: _new_format_json()}, {})
    s = gate["value"]["s_interleave"]
    assert s["legacy"] is False
    assert s["merge_attempted"] == 2          # summed over both passes
    assert s["merge_accepted"] == 1
    assert s["merge_rejected"] == 1
    assert s["resolved_by_merge"] == 1
    assert s["resolved_by_timeout"] == 0
    assert s["resolved_by_merge_cross_check"] == 1
    assert s["cross_check_mismatch"] is False
    assert gate.get("note") is None or "legacy" not in gate["note"]


def test_p6_legacy_json_falls_back_to_final_pass_and_says_so():
    gate = eval_provisional.gate_p6({42: _legacy_json_clean()}, {})
    s = gate["value"]["s_interleave"]
    assert s["legacy"] is True
    assert s["merge_attempted"] is None        # legacy JSONs never recorded the counter
    assert s["merge_accepted"] == 1
    assert s["merge_rejected"] == 1
    assert s["resolved_by_merge"] == 1
    assert s["resolved_by_merge_cross_check"] == 1
    assert s["cross_check_mismatch"] is False
    assert "legacy JSON: final pass only" in gate["note"]


def test_p6_legacy_json_cross_check_is_not_evaluated_at_all():
    """A legacy JSON cannot be cross-checked, so it must not be reported as a mismatch.

    This reverses an earlier reading (PR #8 review, should-fix 10). The counts are still shown,
    and they still disagree — `resolved_by_merge` 1 against 0 accepted merge ops — but the
    disagreement is the guaranteed consequence of the archive recording only the FINAL pass, so
    calling it a mismatch flags the format of every 2026-09-09 run rather than anything about
    its content. The legacy note carries the explanation instead.
    """
    gate = eval_provisional.gate_p6({49: _legacy_json_mismatched()}, {})
    s = gate["value"]["s_interleave"]
    assert s["legacy"] is True
    assert s["resolved_by_merge"] == 1          # provisional_roots says minted_at=3 merged
    assert s["merge_accepted"] == 0             # but the final pass alone has no accepted merge
    assert s["resolved_by_merge_cross_check"] == 0
    assert s["cross_check_mismatch"] is False   # ... and that is expected, not a finding
    assert gate["value"]["cross_check_mismatch"] is False
    assert "[CROSS-CHECK-MISMATCH]" not in gate["headline"]
    assert "legacy JSON: final pass only" in gate["note"]
    assert not s["mismatched_runs"]


def test_p6_new_format_json_cross_check_mismatch_is_still_reported():
    """The cross-check is a real check on an archive that CAN answer it."""
    run = {"provisional_roots": [{"minted_at": 3, "resolution": "merge", "resolved_at": 4}],
           "consolidation_passes": [{"ops": [{"op": "merge_rejected", "keep": 0, "drop": 3}],
                                     "merge_attempted": 1}],
           "decisions": [{"task": 3, "decision": "reuse"}]}
    gate = eval_provisional.gate_p6({49: run}, {})
    s = gate["value"]["s_interleave"]
    assert s["legacy"] is False
    assert s["cross_check_mismatch"] is True
    assert "[CROSS-CHECK-MISMATCH]" in gate["headline"]
    assert s["mismatched_runs"] and s["mismatched_runs"][0]["seed"] == 49


def test_p6_mixed_legacy_and_new_format_pools_correctly():
    int_evalue = {42: _legacy_json_clean(), 43: _new_format_json()}
    gate = eval_provisional.gate_p6(int_evalue, {})
    s = gate["value"]["s_interleave"]
    assert s["legacy"] is True                 # at least one run is legacy
    assert s["merge_attempted"] is None         # can't sum when one run's count is unknown
    assert s["merge_accepted"] == 2
    assert s["merge_rejected"] == 2
    assert s["resolved_by_merge"] == 2
    assert s["cross_check_mismatch"] is False


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
