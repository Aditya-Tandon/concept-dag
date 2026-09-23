"""The H8 evaluator must price the DAG the run ENDS with, and must not invent mismatches.

ref: PR #8 independent review, blocker 2 + should-fixes 9 and 10 (Track C item 5)

Three defects, all in `scripts/eval_provisional.py`:

* P2's "end-of-stream t3 regret" and P5's collateral check read `test_accs`, which is appended
  inside the task loop BEFORE that iteration's consolidation pass — so they priced the DAG before
  the pass that may have resolved the very root the gate is about. They now read
  `test_accs_final` / `average_accuracy_final` where the archive has them, and say `legacy_accs`
  where it does not, so the fallback is never silent.
* P0a/P0b compared decisions, AA and the L_* only. A run whose per-task accuracies or parameter
  curve moved underneath an unchanged mean passed a null check whose entire job is to catch that.
* The `resolved_by_merge` cross-check fired `[CROSS-CHECK-MISMATCH]` on every legacy archive,
  where only the FINAL consolidation pass is recorded and the two bookkeeping paths are EXPECTED
  to disagree — flagging the archive's format as if it were a content defect.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(REPO, "scripts", "eval_provisional.py")
# The 2026-09-09 H8 archive, parked beside the repo (see concept-dag-results/INDEX.md).
ARCHIVE = os.path.abspath(os.path.join(
    REPO, "..", "concept-dag-results", "results_provisional_2026-09-09"))
BASELINE = os.path.abspath(os.path.join(
    REPO, "..", "concept-dag-results", "results_attn_root_2026-09-09"))

sys.path.insert(0, REPO)


# ---------------------------------------------------------------------------
# 1. Unit level: the accessors and the enlarged P0 comparison
# ---------------------------------------------------------------------------


def _load_evaluator():
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_provisional", EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_final_accs_prefers_the_post_consolidation_fields():
    ep = _load_evaluator()

    modern = {"test_accs": [0.1, 0.2], "test_accs_final": [0.3, 0.4],
              "average_accuracy": 0.15, "average_accuracy_final": 0.35}
    assert ep.final_accs(modern) == ([0.3, 0.4], False)
    assert ep.final_aa(modern) == (0.35, False)

    legacy = {"test_accs": [0.1, 0.2], "average_accuracy": 0.15}
    assert ep.final_accs(legacy) == ([0.1, 0.2], True)
    assert ep.final_aa(legacy) == (0.15, True)


def test_regret_at_reads_the_reconsolidated_accuracy():
    ep = _load_evaluator()

    run = {"decisions": [{"task": 3, "oracle_accs": {"grow": 0.9, "reuse": 0.5}}],
           "test_accs": [0, 0, 0, 0.6], "test_accs_final": [0, 0, 0, 0.8]}
    assert ep.regret_at(run, 3, use_final=True) == (pytest.approx(0.1), False)
    assert ep.regret_at(run, 3)[0] == pytest.approx(0.3)       # pre-consolidation, unchanged

    del run["test_accs_final"]
    value, legacy = ep.regret_at(run, 3, use_final=True)
    assert value == pytest.approx(0.3) and legacy is True


def _run_pair(**overrides):
    base = {
        "average_accuracy": 0.8,
        "test_accs": [0.7, 0.9],
        "test_accs_final": [0.7, 0.9],
        "param_curve": [100, 200],
        "decisions": [{"task": 0, "decision": "grow", "L_grow_bits": 1.0},
                      {"task": 1, "decision": "reuse", "L_reuse_bits": 2.0}],
    }
    other = copy.deepcopy(base)
    other.update(overrides)
    return base, other


@pytest.mark.parametrize("field, value", [
    ("param_curve", [100, 201]),
    ("test_accs", [0.7, 0.8]),
    ("test_accs_final", [0.7, 0.8]),
])
def test_p0_comparison_catches_a_sequence_field_and_names_it(field, value):
    ep = _load_evaluator()
    a, b = _run_pair(**{field: value})
    mismatches = ep._compare_runs(a, b, seed=42, stream=None, label_a="ref", label_b="other")
    assert mismatches, f"a differing {field} must not pass a bit-identity check"
    assert mismatches[0]["field"] == field
    assert mismatches[0]["task"] == 1              # the first differing element is named


def test_p0a_fails_on_a_synthetic_pair_that_differs_only_in_param_curve():
    ep = _load_evaluator()
    a, b = _run_pair(param_curve=[100, 999])
    gate = ep.gate_p0a({42: {"s_minus": a}}, {}, {42: {"s_minus": b}}, {})
    assert gate["verdict"] == "fail"
    assert gate["value"]["first_mismatch"]["field"] == "param_curve"


def test_p0_comparison_passes_an_identical_pair():
    ep = _load_evaluator()
    a, b = _run_pair()
    assert ep._compare_runs(a, b, 42, None, "ref", "other") == []


def test_legacy_runs_are_not_cross_check_mismatches():
    """A final-pass-only JSON cannot be cross-checked; it must not be reported as a mismatch."""
    ep = _load_evaluator()

    legacy = {"provisional_roots": [{"minted_at": 1, "resolution": "merge"}],
              "consolidation": {"ops": []}}          # the merge happened in a pass not recorded
    row = ep._p6_row_stats(42, "s_interleave", legacy)
    assert row["legacy"] is True
    assert row["resolved_by_merge"] == 1 and row["accepted_merge_ops_dropping_a_provisional_root"] == 0
    assert row["cross_check_mismatch"] is False

    # ... while a modern JSON with the same disagreement still is one.
    modern = {"provisional_roots": [{"minted_at": 1, "resolution": "merge"}],
              "consolidation_passes": [{"ops": [], "merge_attempted": 0}]}
    row_modern = ep._p6_row_stats(42, "s_interleave", modern)
    assert row_modern["legacy"] is False and row_modern["cross_check_mismatch"] is True


# ---------------------------------------------------------------------------
# 2. End to end on the real 2026-09-09 archive
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.isdir(ARCHIVE) or not os.path.isdir(BASELINE),
                    reason="the 2026-09-09 H8 archive is not parked beside this checkout")
def test_evaluator_still_reports_null_broken_on_the_h8_archive(tmp_path):
    out = str(tmp_path / "gates.json")
    proc = subprocess.run(
        [sys.executable, EVAL, "--ctrl", os.path.join(ARCHIVE, "ctrl"),
         "--interleave", ARCHIVE, "--fivedatasets", ARCHIVE, "--baseline", BASELINE,
         "--out", out],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    with open(out) as f:
        gates = json.load(f)

    # The archived verdict, unchanged: P0b fails on 5-Datasets (the shadow refit's device-RNG
    # leak), so the null is broken and no hypothesis branch is named.
    assert gates["branch"] == "NULL-BROKEN"
    assert gates["P0a"]["verdict"] == "pass"
    assert gates["P0b"]["verdict"] == "fail"

    # The whole archive predates `test_accs_final`, so the two gates that need it say so.
    assert gates["P2"]["legacy_accs"]
    assert gates["P2"]["value"]["a_regret_improvement"]["legacy_accs"]
    assert gates["P5"]["legacy_accs"]

    # ... and the legacy format is reported as a format note, not as a content mismatch.
    assert gates["P6"]["value"]["cross_check_mismatch"] is False
    assert "CROSS-CHECK-MISMATCH" not in gates["P6"]["headline"]
    assert "legacy JSON" in (gates["P6"].get("note") or "")
