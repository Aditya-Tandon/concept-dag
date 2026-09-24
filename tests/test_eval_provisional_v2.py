"""Fixture tests for the H9 V-gate evaluator, written BEFORE the sweep.

ref: Hypotheses/concept-dag/kan-gated-growth/provisional-growth-v2-validity-rerun.md
     (design item 4: "committed and fixture-tested before the sweep, H8's spec-violation
     standard") — Track C item 9

Every gate gets a synthetic pass case and a synthetic fail case, built as the results dicts the
runner writes, so a mis-implemented threshold cannot wait until there are 129 real runs to be
discovered. On top of that the whole script is run end to end against the real 2026-09-09 H8
archive, which is legacy JSON in every respect this evaluator cares about (final pass only, no
`test_accs_final`, no `cka` on any op): it must not crash, and it must name NULL-STILL-BROKEN.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "eval_provisional_v2.py")
ARCHIVE = os.path.abspath(os.path.join(
    REPO, "..", "concept-dag-results", "results_provisional_2026-09-09"))
BASELINE = os.path.abspath(os.path.join(
    REPO, "..", "concept-dag-results", "results_attn_root_2026-09-09"))


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("eval_provisional_v2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic run builder — the shape `run_exp3a_kan` writes
# ---------------------------------------------------------------------------


def run(decisions=None, accs=None, accs_final=None, aa=None, aa_final=None,
        param_curve=None, param_curve_total=None, passes=None, roots=None,
        params_total=100000, ctrl_gt=None, reader_audit=None):
    decisions = decisions if decisions is not None else [
        {"task": 0, "decision": "grow"}, {"task": 1, "decision": "grow"},
        {"task": 2, "decision": "reuse"}, {"task": 3, "decision": "grow"},
        {"task": 4, "decision": "reuse"}]
    accs = accs if accs is not None else [0.9, 0.8, 0.7, 0.6, 0.5]
    accs_final = accs_final if accs_final is not None else list(accs)
    out = {
        "decisions": decisions,
        "test_accs": accs, "test_accs_final": accs_final,
        "average_accuracy": aa if aa is not None else sum(accs) / len(accs),
        "average_accuracy_final": (aa_final if aa_final is not None
                                   else sum(accs_final) / len(accs_final)),
        "param_curve": param_curve if param_curve is not None else [100, 200, 200, 300, 300],
        "param_curve_total": (param_curve_total if param_curve_total is not None
                              else [100, 200, 200, 300, 300]),
        "params_total_pre_consolidation": params_total,
        "consolidation_passes": passes if passes is not None else [
            {"at_task": 4, "ops": [], "merge_attempted": 0, "final": True}],
        "provisional_roots": roots if roots is not None else [],
    }
    if ctrl_gt is not None:
        out["ctrl_ground_truth"] = ctrl_gt
    if reader_audit is not None:
        out["reader_audit"] = reader_audit
    return out


def tree(**streams):
    """{42: {stream: run}} for the CTrL-shaped loaders."""
    return {42: dict(streams)}


# ---------------------------------------------------------------------------
# Bit identity (V0a-i / V0a-ii / V0b all share `compare_runs`)
# ---------------------------------------------------------------------------


def test_identical_runs_compare_clean(ev):
    assert ev.compare_runs(run(), run(), 42, "s_minus", "a", "b") == []


@pytest.mark.parametrize("mutate, field", [
    (lambda r: r.__setitem__("average_accuracy", 0.99), "average_accuracy"),
    (lambda r: r.__setitem__("test_accs", [0.9, 0.8, 0.7, 0.6, 0.4]), "test_accs"),
    (lambda r: r.__setitem__("param_curve", [100, 200, 200, 300, 301]), "param_curve"),
    (lambda r: r.__setitem__("param_curve_total", [1, 2, 3, 4, 5]), "param_curve_total"),
    (lambda r: r.__setitem__("test_accs_final", [0.1, 0.2, 0.3, 0.4, 0.5]), "test_accs_final"),
])
def test_every_bit_identity_field_is_compared(ev, mutate, field):
    b = run()
    mutate(b)
    mism = ev.compare_runs(run(), b, 42, None, "a", "b")
    assert mism and mism[0]["field"] == field


def test_a_field_present_on_one_side_only_is_a_failure(ev):
    b = run()
    del b["param_curve_total"]
    mism = ev.compare_runs(run(), b, 42, None, "a", "b")
    assert mism and mism[0]["detail"] == "present on one side only"


def test_decision_and_L_fields_are_compared(ev):
    a = run(decisions=[{"task": 0, "decision": "grow", "L_grow_bits": 1.0}])
    b = run(decisions=[{"task": 0, "decision": "reuse", "L_grow_bits": 1.0}])
    c = run(decisions=[{"task": 0, "decision": "grow", "L_grow_bits": 1.5}])
    assert ev.compare_runs(a, b, 42, None, "a", "b")[0]["field"] == "decision"
    assert ev.compare_runs(a, c, 42, None, "a", "b")[0]["field"] == "L_grow_bits"


def test_v0b_passes_and_fails(ev):
    off, shadow = tree(s_minus=run()), tree(s_minus=run())
    g = ev.gate_v0b(off, {}, {}, shadow, {}, {})
    assert g["verdict"] == "pass" and g["value"]["n_runs_compared"] == 1

    broken = tree(s_minus=run(accs=[0.9, 0.8, 0.7, 0.6, 0.1]))
    g = ev.gate_v0b(off, {}, {}, broken, {}, {})
    assert g["verdict"] == "fail" and g["value"]["first_mismatch"]["field"] in (
        "average_accuracy", "test_accs")


def test_v0a_i_and_ii_compare_against_their_own_references(ev):
    base, off = tree(s_minus=run()), tree(s_minus=run())
    assert ev.gate_v0a_i(off, {}, base, {})["verdict"] == "pass"
    assert ev.gate_v0a_ii(off, {}, {}, base, {}, {})["verdict"] == "pass"
    # V0a-ii is the only null s_interleave has, so it must actually compare that stream.
    g = ev.gate_v0a_ii({}, {42: run()}, {}, {}, {42: run(accs=[0.1] * 5)}, {})
    assert g["verdict"] == "fail" and g["value"]["n_runs_compared"] == 1


def test_missing_reference_is_not_run_not_pass(ev):
    assert ev.gate_v0a_i(tree(s_minus=run()), {}, {}, {})["verdict"] == "not-run"


# --- expected pair counts (v2 review, blocker 3) --------------------------------------------


def test_an_identity_gate_is_short_changed_not_satisfied_by_a_missing_pair(ev):
    """Comparing only the pairs that happen to exist certifies a null that was never checked."""
    off, shadow = tree(s_minus=run(), s_plus=run()), tree(s_minus=run())
    expected = {("s_minus", 42), ("s_plus", 42)}
    g = ev.gate_v0b(off, {}, {}, shadow, {}, {}, expected=expected)
    assert g["verdict"] == "fail" and g["incomplete"] is True
    assert g["value"]["n_runs_compared"] == 1 and g["value"]["n_expected"] == 2
    assert g["value"]["missing_pairs"] == [{"stream": "s_plus", "seed": 42}]
    assert "INCOMPLETE" in g["note"]

    # ... and with every expected pair present it passes as before.
    g2 = ev.gate_v0b(off, {}, {}, tree(s_minus=run(), s_plus=run()), {}, {}, expected=expected)
    assert g2["verdict"] == "pass" and not g2.get("incomplete")


def test_the_preregistered_expectations_are_the_run_table(ev):
    data = {"baseline_ctrl": {}, "baseline_5ds": {}, "h8_ctrl": {}, "h8_int": {}, "h8_5ds": {}}
    full = ev.expected_pairs(data, preflight=False)
    assert len(full["V0b"]) == 24                       # 20 CTrL + 3 5ds + 1 s_interleave
    assert (None, 44) in full["V0b"] and ("s_interleave", 42) in full["V0b"]

    pre = ev.expected_pairs(data, preflight=True)
    assert pre["V0b"] == {(None, 42), (None, 43), (None, 44), ("s_interleave", 42)}


def test_v0a_expectations_come_from_the_references_own_keys(ev):
    data = {"baseline_ctrl": tree(s_minus=run(), s_plus=run()),
            "baseline_5ds": {42: run(), 43: run()},
            "h8_ctrl": {}, "h8_int": {44: run()}, "h8_5ds": {}}
    full = ev.expected_pairs(data, preflight=False)
    assert full["V0a-i"] == {("s_minus", 42), ("s_plus", 42), (None, 42), (None, 43)}
    assert full["V0a-ii"] == {("s_interleave", 44)}


# ---------------------------------------------------------------------------
# V2
# ---------------------------------------------------------------------------


def _v2_trees(evalue_t3_acc, evalue_aa=None, always_params=200000, evalue_params=150000):
    oracle = {"grow": 0.9, "reuse": 0.5}
    off = tree(s_minus=run(
        decisions=[{"task": 3, "decision": "reuse", "oracle_accs": oracle}],
        accs_final=[0, 0, 0, 0.5, 0], aa_final=0.80, params_total=100000))
    evalue = tree(s_minus=run(
        decisions=[{"task": 3, "decision": "grow", "oracle_accs": oracle}],
        accs_final=[0, 0, 0, evalue_t3_acc, 0],
        aa_final=(evalue_aa if evalue_aa is not None else 0.80),
        params_total=evalue_params))
    always = tree(s_minus=run(aa_final=0.80, params_total=always_params))
    return off, evalue, always


def test_v2_passes_when_regret_closes_and_aa_holds(ev):
    off, evalue, always = _v2_trees(evalue_t3_acc=0.9)
    g = ev.gate_v2(off, evalue, always)
    assert g["value"]["a_regret_improvement"]["verdict"] == "pass"
    assert g["value"]["c_collateral_aa"]["verdict"] == "pass"
    assert g["verdict"] == "pass"


def test_v2a_fails_when_the_regret_improvement_is_too_small(ev):
    off, evalue, always = _v2_trees(evalue_t3_acc=0.51)     # improvement 0.01 < 0.03
    g = ev.gate_v2(off, evalue, always)
    assert g["value"]["a_regret_improvement"]["verdict"] == "fail"
    assert g["verdict"] == "fail"


def test_v2c_fails_on_collateral_aa(ev):
    off, evalue, always = _v2_trees(evalue_t3_acc=0.9, evalue_aa=0.79)   # -0.01 < -0.005
    g = ev.gate_v2(off, evalue, always)
    assert g["value"]["c_collateral_aa"]["verdict"] == "fail"


def test_v2b_fails_when_always_buys_more_per_parameter(ev):
    # evalue: +0.00 AA for +50k; always: +0.00 AA for +100k. Make evalue strictly worse.
    off = tree(s_minus=run(aa_final=0.80, params_total=100000,
                           decisions=[{"task": 3, "decision": "reuse",
                                       "oracle_accs": {"grow": 0.9}}],
                           accs_final=[0, 0, 0, 0.5, 0]))
    evalue = tree(s_minus=run(aa_final=0.81, params_total=200000,
                              decisions=[{"task": 3, "decision": "grow",
                                          "oracle_accs": {"grow": 0.9}}],
                              accs_final=[0, 0, 0, 0.9, 0]))
    always = tree(s_minus=run(aa_final=0.85, params_total=200000))
    g = ev.gate_v2(off, evalue, always)
    assert g["value"]["b_accuracy_per_parameter"]["verdict"] == "fail"
    assert g["verdict"] == "fail"


def test_v2_reads_the_post_consolidation_accuracies(ev):
    """`test_accs` is written before the final pass; using it would price the wrong DAG."""
    oracle = {"grow": 0.9}
    off = tree(s_minus=run(decisions=[{"task": 3, "decision": "reuse", "oracle_accs": oracle}],
                           accs=[0, 0, 0, 0.5, 0], accs_final=[0, 0, 0, 0.5, 0]))
    evalue = tree(s_minus=run(decisions=[{"task": 3, "decision": "grow", "oracle_accs": oracle}],
                              accs=[0, 0, 0, 0.5, 0],          # pre-consolidation: no gain
                              accs_final=[0, 0, 0, 0.9, 0]))   # post: the gain the gate is about
    g = ev.gate_v2(off, evalue, {})
    assert g["value"]["a_regret_improvement"]["mean_improvement"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# V3
# ---------------------------------------------------------------------------


def _int_run(merge_t3_into_t1=True, provisional_t3=True, root_t1=True, extra_ops=()):
    decisions = [
        {"task": 0, "decision": "grow"},
        {"task": 1, "decision": "grow" if root_t1 else "reuse"},
        {"task": 2, "decision": "grow"},
        {"task": 3, "decision": "grow", **({"provisional": True} if provisional_t3 else {})},
        {"task": 4, "decision": "reuse"}]
    ops = list(extra_ops)
    if merge_t3_into_t1:
        ops.append({"op": "merge", "keep": 1, "drop": 3, "similarity": 0.99,
                    "sim_kind": "functional", "cka": 0.9, "cca_topk": 0.99})
    return run(decisions=decisions,
               passes=[{"at_task": 4, "ops": ops, "merge_attempted": len(ops)}],
               roots=[{"minted_at": 3,
                       "resolution": "merge" if merge_t3_into_t1 else "timeout",
                       "resolved_at": 4}])


def test_v3_passes_when_the_object_exists_and_merges(ev):
    g = ev.gate_v3({s: _int_run() for s in range(42, 52)})
    assert g["value"]["n_object_present"] == 10
    assert g["value"]["n_merged"] == 10
    assert g["verdict"] == "pass"


def test_v3_fails_on_rate_with_the_object_present(ev):
    runs = {s: _int_run(merge_t3_into_t1=(s < 46)) for s in range(42, 52)}   # 4/10
    g = ev.gate_v3(runs)
    assert g["value"]["n_object_present"] == 10
    assert g["value"]["frac_merged"] == pytest.approx(0.4)
    assert g["verdict"] == "fail"
    assert g["value"]["object_absent_exceeds_budget"] is False   # -> MERGE-NOT-FOUND


def test_v3_fails_with_the_object_absent(ev):
    runs = {s: _int_run(provisional_t3=(s < 44)) for s in range(42, 52)}     # object in 2/10
    g = ev.gate_v3(runs)
    assert g["value"]["n_object_present"] == 2
    assert g["value"]["object_absent_exceeds_budget"] is True    # -> NO-MERGE-OBJECT
    assert g["verdict"] == "fail"


def test_v3_negative_control_fires_on_a_forbidden_pair(ev):
    bad = {"op": "merge", "keep": 1, "drop": 2}
    runs = {s: _int_run(extra_ops=[bad] if s == 42 else []) for s in range(42, 52)}
    g = ev.gate_v3(runs)
    assert g["value"]["negative_control_ok"] is False
    assert g["value"]["negative_control_hits"][0] == {"seed": 42, "keep": 1, "drop": 2}
    assert g["verdict"] == "fail"


def test_v3_object_needs_three_nodes(ev):
    """Below 3 nodes `consolidate_nodes` is skipped, so the object cannot exist."""
    r = run(decisions=[{"task": 0, "decision": "grow"}, {"task": 1, "decision": "grow"},
                       {"task": 2, "decision": "reuse"},
                       {"task": 3, "decision": "reuse", "provisional": True}],
            passes=[{"at_task": 3, "ops": []}])
    assert ev._v3_object(r)["n_nodes_at_t3"] == 2
    assert ev._v3_object(r)["exists"] is False


# ---------------------------------------------------------------------------
# V4
# ---------------------------------------------------------------------------


def test_v4_passes_on_clean_provenance(ev):
    r = run(param_curve=[100, 200, 300, 300, 200],
            param_curve_total=[100, 200, 300, 300, 200],
            passes=[{"at_task": 3, "ops": [{"op": "merge", "keep": 0, "drop": 1}],
                     "merge_attempted": 1}],
            roots=[{"minted_at": 1, "resolution": "merge", "resolved_at": 3}])
    g = ev.gate_v4([(42, "s_minus", r)])
    assert g["verdict"] == "pass"


def test_v4_fails_on_an_unexplained_parameter_fall(ev):
    r = run(param_curve=[100, 200, 300, 300, 200],
            param_curve_total=[100, 200, 300, 300, 200],
            passes=[{"at_task": 3, "ops": [], "merge_attempted": 0}])
    g = ev.gate_v4([(42, "s_minus", r)])
    assert g["verdict"] == "fail"
    assert g["value"]["violations"][0]["unexplained_curve_falls"][0]["at_task"] == 3


def test_v4_fails_when_a_merge_resolution_has_no_op(ev):
    r = run(roots=[{"minted_at": 1, "resolution": "merge", "resolved_at": 3}],
            passes=[{"at_task": 3, "ops": [{"op": "merge_rejected", "keep": 0, "drop": 1}]}])
    g = ev.gate_v4([(42, "s_minus", r)])
    assert g["verdict"] == "fail"
    assert g["value"]["violations"][0]["merge_resolutions_without_exactly_one_op"]


def test_v4_accepts_a_root_merged_after_crystallisation(ev):
    """Resolution `timeout` + `merged_after_crystallisation` still expects exactly one op."""
    ok = run(roots=[{"minted_at": 1, "resolution": "timeout", "resolved_at": 3,
                     "merged_after_crystallisation": 0}],
             param_curve=[100, 200, 300, 300, 200],
             param_curve_total=[100, 200, 300, 300, 200],
             passes=[{"at_task": 3, "ops": [{"op": "merge", "keep": 0, "drop": 1}]}])
    assert ev.gate_v4([(42, None, ok)])["verdict"] == "pass"

    # ... and it is a violation when no accepted op names it, exactly as a `merge` resolution is.
    bad = run(roots=[{"minted_at": 1, "resolution": "timeout", "resolved_at": 3,
                      "merged_after_crystallisation": 0}],
              passes=[{"at_task": 3, "ops": []}])
    g = ev.gate_v4([(42, None, bad)])
    assert g["verdict"] == "fail"
    assert g["value"]["violations"][0]["merge_resolutions_without_exactly_one_op"][0][
        "merged_after_crystallisation"] == 0


def test_v4_catches_the_opposite_mis_attribution(ev):
    """An accepted merge dropping a provisional root whose record says `timeout` is blocker 1.

    Checking only resolution->op lets that through: the record claims a timeout, so nothing
    "expects an op", and V4 was silent on the very bug H8's P6 cross-check catches.
    """
    bad = run(roots=[{"minted_at": 1, "resolution": "timeout", "resolved_at": 4}],
              param_curve=[100, 200, 300, 300, 200],
              param_curve_total=[100, 200, 300, 300, 200],
              passes=[{"at_task": 3, "ops": [{"op": "merge", "keep": 0, "drop": 1}]}])
    g = ev.gate_v4([(42, None, bad)])
    assert g["verdict"] == "fail"
    orphans = g["value"]["violations"][0]["accepted_ops_with_no_matching_resolution"]
    assert orphans == [{"keep": 0, "drop": 1, "resolution": "timeout",
                        "merged_after_crystallisation": None}]

    # Recording the post-crystallisation merge is what makes the same run clean.
    fixed = json.loads(json.dumps(bad))
    fixed["provisional_roots"][0]["merged_after_crystallisation"] = 0
    assert ev.gate_v4([(42, None, fixed)])["verdict"] == "pass"


def test_v4_ignores_ops_dropping_a_non_provisional_root(ev):
    r = run(roots=[], param_curve=[100, 200, 300, 300, 200],
            param_curve_total=[100, 200, 300, 300, 200],
            passes=[{"at_task": 3, "ops": [{"op": "merge", "keep": 0, "drop": 1}]}])
    assert ev.gate_v4([(42, None, r)])["verdict"] == "pass"


def test_v4_fails_on_removed_unexplained_and_on_an_unresolved_root(ev):
    r1 = run(roots=[{"minted_at": 1, "resolution": "removed_unexplained", "resolved_at": 3}])
    r2 = run(roots=[{"minted_at": 1, "resolution": None, "resolved_at": None}])
    assert ev.gate_v4([(42, None, r1)])["verdict"] == "fail"
    assert ev.gate_v4([(42, None, r2)])["verdict"] == "fail"


def test_v4_checks_both_curves(ev):
    """`param_curve` hides a token root's AttentionPool, so the total curve is checked too."""
    r = run(param_curve=[100, 100, 100, 100, 100],
            param_curve_total=[100, 200, 300, 300, 200],
            passes=[{"at_task": 3, "ops": []}])
    g = ev.gate_v4([(42, None, r)])
    assert g["verdict"] == "fail"
    assert g["value"]["violations"][0]["unexplained_curve_falls"][0]["curve"] == "param_curve_total"


# ---------------------------------------------------------------------------
# V5
# ---------------------------------------------------------------------------


_REVISIT_GT = [None, None, None, None, {"revisit_of": 0}]


def _v5_pair(off_accs, ev_accs, deferred=(3,), ctrl_gt=None,
             off_pre=None, ev_pre=None):
    """`*_accs` are `test_accs_final`; `*_pre` override `test_accs` where a test needs the two
    to differ (the before-first-deferral clause reads the pre-consolidation field)."""
    decisions = [{"task": t, "decision": "grow",
                  **({"provisional": True} if t in deferred else {})} for t in range(5)]
    off = run(accs=list(off_pre if off_pre is not None else off_accs),
              accs_final=list(off_accs))
    evr = run(decisions=decisions, accs=list(ev_pre if ev_pre is not None else ev_accs),
              accs_final=list(ev_accs), ctrl_gt=ctrl_gt)
    return off, evr


def test_v5a_passes_when_nothing_outside_the_deferred_position_moves(ev):
    off, evr = _v5_pair([0.9] * 5, [0.9, 0.9, 0.9, 0.5, 0.9])
    g = ev.gate_v5([(42, "s_minus", off, evr)])
    assert g["value"]["a_non_deferred_positions"]["verdict"] == "pass"


def test_v5a_fails_on_a_leak_before_the_first_deferral(ev):
    """Before the first deferral the arms share every draw — a nonzero delta is a LEAK."""
    off, evr = _v5_pair([0.9] * 5, [0.9, 0.8, 0.9, 0.5, 0.9], deferred=(3,))
    g = ev.gate_v5([(42, "s_minus", off, evr)])
    a = g["value"]["a_non_deferred_positions"]
    assert a["verdict"] == "fail"
    assert a["before_first_deferral_violations"][0]["task"] == 1
    assert g["verdict"] == "fail"


def test_v5a_before_split_reads_test_accs_not_test_accs_final(ev):
    """The final pass may legitimately move `test_accs_final` on one arm and not the other."""
    # Identical pre-consolidation accuracies; the final pass moves t1 on the evalue arm only.
    off, evr = _v5_pair([0.9] * 5, [0.9, 0.7, 0.9, 0.5, 0.9],
                        off_pre=[0.9] * 5, ev_pre=[0.9] * 5, deferred=(3,))
    g = ev.gate_v5([(42, "s_minus", off, evr)])
    assert g["value"]["a_non_deferred_positions"]["before_first_deferral_violations"] == [], (
        "a pre-first-deferral position must be compared on test_accs, where the arms really do "
        "share every draw")

    # ... and a genuine pre-consolidation difference there is still caught.
    off2, evr2 = _v5_pair([0.9] * 5, [0.9] * 5,
                          off_pre=[0.9] * 5, ev_pre=[0.9, 0.8, 0.9, 0.9, 0.9], deferred=(3,))
    g2 = ev.gate_v5([(42, "s_minus", off2, evr2)])
    viol = g2["value"]["a_non_deferred_positions"]["before_first_deferral_violations"]
    assert viol and viol[0]["task"] == 1 and viol[0]["field"] == "test_accs"


def test_v5a_after_split_reads_test_accs_final(ev):
    """After the first deferral the DAG the run ENDS with is the subject."""
    off, evr = _v5_pair([0.9] * 5, [0.9, 0.9, 0.9, 0.5, 0.85],
                        off_pre=[0.9] * 5, ev_pre=[0.9] * 5, deferred=(3,))
    g = ev.gate_v5([(42, "s_minus", off, evr)])
    a = g["value"]["a_non_deferred_positions"]
    assert a["after_first_deferral"][0]["rows"][0]["field"] == "test_accs_final"
    assert a["after_violations"][0]["task"] == 4


def test_v5a_fails_on_a_post_deferral_seed_mean_below_tolerance(ev):
    off, evr = _v5_pair([0.9] * 5, [0.9, 0.9, 0.9, 0.5, 0.85], deferred=(3,))
    g = ev.gate_v5([(42, "s_minus", off, evr)])
    a = g["value"]["a_non_deferred_positions"]
    assert a["verdict"] == "fail" and a["after_violations"][0]["task"] == 4
    # ... and the noise floor is reported beside it, per judge change 6.
    assert "off_across_seed_sd" in a["after_first_deferral"][0]


def test_v5b_fails_at_the_deferred_revisit(ev):
    pairs = []
    for seed in range(42, 45):
        off, evr = _v5_pair([0.9] * 5, [0.9, 0.9, 0.9, 0.9, 0.85], deferred=(3, 4),
                            ctrl_gt=_REVISIT_GT)
        pairs.append((seed, "s_minus", off, evr))
    g = ev.gate_v5(pairs)
    b = g["value"]["b_deferred_revisit"]
    assert b["n"] == 3 and b["mean_delta"] == pytest.approx(-0.05)
    assert b["verdict"] == "fail"


def test_v5b_only_counts_runs_that_defer_at_the_revisit(ev):
    off, evr = _v5_pair([0.9] * 5, [0.9, 0.9, 0.9, 0.5, 0.85], deferred=(3,),
                        ctrl_gt=_REVISIT_GT)
    g = ev.gate_v5([(42, "s_minus", off, evr)])
    assert g["value"]["b_deferred_revisit"]["n"] == 0        # t4 was not deferred here


def test_v5c_counts_rolled_back_audit_entries(ev):
    off, evr = _v5_pair([0.9] * 5, [0.9] * 5, deferred=())
    evr["reader_audit"] = [{"kind": "merge", "verdict": "rolled_back"},
                           {"kind": "truncate", "verdict": "kept"}]
    g = ev.gate_v5([(42, "s_minus", off, evr)])
    assert g["value"]["c_rolled_back_audit_entries"]["n_rolled_back"] == 1


# ---------------------------------------------------------------------------
# V6
# ---------------------------------------------------------------------------


def test_v6_fails_on_an_unresolved_root(ev):
    r = run(roots=[{"minted_at": 1, "resolution": None, "resolved_at": None}])
    g = ev.gate_v6([(42, None, r)], {"dir": None, "files": [], "n_files": 0})
    assert g["verdict"] == "fail"


def test_v6_is_not_run_without_a_dumps_dir_even_when_roots_are_clean(ev):
    r = run(roots=[{"minted_at": 1, "resolution": "timeout", "resolved_at": 4}])
    g = ev.gate_v6([(42, None, r)], {"dir": None, "files": [], "n_files": 0})
    assert g["verdict"] == "not-run"


def test_v6_passes_with_enough_complete_dumps(ev):
    r = run(roots=[{"minted_at": 1, "resolution": "timeout", "resolved_at": 4}])
    inv = {"dir": "/tmp/d", "inspected": True, "n_files": ev.V6_EXPECTED_DUMPS,
           "files": [{"snapshots_carry_reader": True}] * ev.V6_EXPECTED_DUMPS}
    assert ev.gate_v6([(42, None, r)], inv)["verdict"] == "pass"


def test_v6_fails_when_a_dump_has_no_reader_in_its_snapshots(ev):
    r = run(roots=[])
    inv = {"dir": "/tmp/d", "inspected": True, "n_files": ev.V6_EXPECTED_DUMPS,
           "files": ([{"snapshots_carry_reader": True}] * (ev.V6_EXPECTED_DUMPS - 1)
                     + [{"snapshots_carry_reader": False}])}
    assert ev.gate_v6([(42, None, r)], inv)["verdict"] == "fail"


def test_dump_inventory_reads_a_real_token_dump(ev, tmp_path):
    """End to end against a dump this branch's own code writes."""
    torch = pytest.importorskip("torch")
    from concept_dag.experiments.kan_exp import KanExpConfig, run_exp3a_kan
    from tests.test_attn_root_adoption import _make_synthetic_token_tasks

    out = tmp_path / "run"
    cfg = KanExpConfig(
        backbone="dinov2_vits14", feature_dim=24, n_tokens=7, token_pool=2,
        root_family="attn_pool", concept_dim=16, root_epochs=2, child_epochs=2, gate_epochs=2,
        n_parents=1, routing_batches=2, gate_cache_max=128, batch_size=16, raw_grow_probe=True,
        dump_gate_tensors=True, dump_max_per_split=8, log_every=100000, device="cpu",
        results_dir=str(out))
    run_exp3a_kan(cfg, _make_synthetic_token_tasks(n_tasks=3, n_per_class=32, seed=2))

    inv = ev._dump_inventory(str(tmp_path))
    assert inv["n_files"] == 1 and inv["inspected"] is True
    f = inv["files"][0]
    assert f["mode"] == "token" and f["has_roots_at_mint"] and f["n_roots_at_mint"] >= 1
    assert f["snapshots_carry_reader"] is True


# ---------------------------------------------------------------------------
# V1 / V1b
# ---------------------------------------------------------------------------


def _pairs_run(ops):
    return run(passes=[{"at_task": 4, "ops": ops, "merge_attempted": len(ops)}])


def test_v1_passes_when_the_filter_is_free_and_removes_most_vetoed_pairs(ev):
    ops = [{"op": "merge", "keep": 0, "drop": 1, "cka": 0.9, "cca_topk": 0.99},
           {"op": "merge_rejected", "keep": 0, "drop": 2, "cka": 0.2, "cca_topk": 0.95},
           {"op": "merge_rejected", "keep": 1, "drop": 2, "cka": 0.3, "cca_topk": 0.95},
           {"op": "merge_rejected", "keep": 1, "drop": 3, "cka": 0.4, "cca_topk": 0.95},
           {"op": "trigger_rejected", "keep": 2, "drop": 3, "cka": 0.1, "cca_topk": 0.5}]
    g = ev.gate_v1([(42, "s_minus", _pairs_run(ops))])
    assert g["value"]["n_examined_pairs"] == 5
    assert g["value"]["n_accepted_merges_skipped"] == 0
    assert g["value"]["frac_backward_rejected_removed"] == pytest.approx(1.0)
    assert g["verdict"] == "pass"


def test_v1_fails_when_the_filter_would_skip_an_accepted_merge(ev):
    ops = [{"op": "merge", "keep": 0, "drop": 1, "cka": 0.4, "cca_topk": 0.99},
           {"op": "merge_rejected", "keep": 0, "drop": 2, "cka": 0.1, "cca_topk": 0.95}]
    g = ev.gate_v1([(42, "s_minus", _pairs_run(ops))])
    assert g["value"]["n_accepted_merges_skipped"] == 1
    assert g["verdict"] == "fail"


def test_v1_fails_when_it_removes_too_few_vetoed_pairs(ev):
    ops = [{"op": "merge", "keep": 0, "drop": 1, "cka": 0.9, "cca_topk": 0.99}]
    ops += [{"op": "merge_rejected", "keep": 0, "drop": i, "cka": 0.9, "cca_topk": 0.95}
            for i in range(2, 6)]                       # all above threshold: 0 % removed
    g = ev.gate_v1([(42, "s_minus", _pairs_run(ops))])
    assert g["value"]["frac_backward_rejected_removed"] == pytest.approx(0.0)
    assert g["verdict"] == "fail"


def test_v1_is_pooled_not_per_run(ev):
    """A run with no rejected op contributes 0 to both sides, not a 0/0 ratio."""
    a = _pairs_run([{"op": "merge_rejected", "keep": 0, "drop": 1, "cka": 0.1, "cca_topk": 0.9}])
    b = _pairs_run([])
    g = ev.gate_v1([(42, "s_minus", a), (43, "s_minus", b)])
    assert g["value"]["n_backward_rejected"] == 1
    assert g["value"]["frac_backward_rejected_removed"] == pytest.approx(1.0)


def test_a_zero_over_zero_denominator_is_not_decidable_not_a_failure(ev):
    """`fail` here reads as "the pre-filter is not free", which would be a fabrication."""
    ops = [{"op": "merge", "keep": 0, "drop": 1, "cka": 0.9, "cca_topk": 0.99},
           {"op": "trigger_rejected", "keep": 0, "drop": 2, "cka": 0.1, "cca_topk": 0.4}]
    g = ev.gate_v1([(42, "s_minus", _pairs_run(ops))])
    assert g["value"]["n_backward_rejected"] == 0
    assert g["value"]["is_free"] is True
    assert g["verdict"] == "not-decidable" and "0/0" in g["note"]


def test_a_skipped_accepted_merge_still_fails_with_a_zero_denominator(ev):
    """Not-free is demonstrated on the pairs that DO carry the statistic."""
    ops = [{"op": "merge", "keep": 0, "drop": 1, "cka": 0.2, "cca_topk": 0.99}]
    g = ev.gate_v1([(42, "s_minus", _pairs_run(ops))])
    assert g["verdict"] == "fail" and g["value"]["n_accepted_merges_skipped"] == 1


def test_a_partial_cka_pool_is_reported_not_silently_narrowed(ev):
    ops = [{"op": "merge_rejected", "keep": 0, "drop": 1, "cka": 0.1, "cca_topk": 0.9},
           {"op": "merge_rejected", "keep": 0, "drop": 2, "similarity": 0.9}]   # no `cka`
    g = ev.gate_v1([(42, "s_minus", _pairs_run(ops))])
    assert g["value"]["n_missing_cka"] == 1
    assert g["verdict"] == "not-decidable" and "partial denominator" in g["note"]


def test_v1_is_not_run_when_no_pair_was_examined(ev):
    g = ev.gate_v1([(42, "s_minus", _pairs_run([]))])
    assert g["verdict"] == "not-run" and "no pair" in g["note"]


def test_v1_is_not_run_on_an_archive_with_no_cka(ev):
    ops = [{"op": "merge", "keep": 0, "drop": 1, "similarity": 0.99}]
    g = ev.gate_v1([(42, "s_minus", _pairs_run(ops))])
    assert g["verdict"] == "not-run" and "predate" in g["note"]


def test_v1b_reports_the_two_arms_side_by_side_and_never_gates(ev):
    cca = _pairs_run([{"op": "merge", "keep": 0, "drop": 1, "cka": 0.9}])
    cka = _pairs_run([{"op": "merge", "keep": 0, "drop": 2, "cka": 0.9}])
    g = ev.gate_v1b([(42, "s_interleave", cca)], [(42, "s_interleave", cka)])
    assert g["verdict"] == "descriptive"
    assert g["value"]["rows"][0]["only_in_cca"] == [(0, 1)]
    assert g["value"]["rows"][0]["only_in_cka"] == [(0, 2)]


# ---------------------------------------------------------------------------
# R1 — the note's reading, not the shipped evaluator's
# ---------------------------------------------------------------------------


def test_r1_proxy_suffices_below_twenty_percent_not_only_at_zero(ev):
    def arm(decisions):
        return tree(s_minus=run(decisions=decisions))

    same = [{"task": t, "decision": "grow"} for t in range(1, 5)]
    one_diff = [{"task": t, "decision": "reuse" if t == 1 else "grow"} for t in range(1, 5)]
    r = ev.observable_r1(arm(same), arm(one_diff))
    assert r["value"]["frac_diff"] == pytest.approx(0.25)
    assert r["value"]["proxy_suffices"] is False

    two_of_twelve = [{"task": t, "decision": "grow"} for t in range(1, 5)]
    r2 = ev.observable_r1(arm(same), arm(two_of_twelve))
    assert r2["value"]["frac_diff"] == 0.0 and r2["value"]["proxy_suffices"] is True


def test_r2_reports_not_recorded_rather_than_a_fabricated_zero(ev):
    r = ev.observable_r2([(42, None, run())], [])
    assert r["headline"] == "not-recorded" and "not-recorded" in r["note"]


def test_r3_declares_the_unfixed_residual(ev):
    assert ev.observable_r3()["value"]["fixed"] is False


# ---------------------------------------------------------------------------
# Branch precedence
# ---------------------------------------------------------------------------


def _gates(**verdicts):
    base = {k: {"verdict": "pass", "value": {}} for k in
            ("V0a-i", "V0a-ii", "V0b", "V2", "V3", "V4", "V5", "V6", "V1")}
    for k, v in verdicts.items():
        key = k.replace("_", "-")
        base[key] = v if isinstance(v, dict) else {"verdict": v, "value": {}}
    return base


def test_branch_null_still_broken_wins(ev):
    g = _gates(V0b="fail", V2="fail", V3="fail")
    assert ev.determine_branch(g, None)["branch"] == "NULL-STILL-BROKEN"


@pytest.mark.parametrize("clause, branch", [
    ("a_regret_improvement", "REGRET-NOT-REPRODUCED"),
    ("b_accuracy_per_parameter", "NOT-WORTH-THE-PARAMETERS"),
    ("c_collateral_aa", "COLLATERAL-AA"),
])
def test_v2_clauses_select_their_own_branches(ev, clause, branch):
    value = {c: {"verdict": "pass"} for c in
             ("a_regret_improvement", "b_accuracy_per_parameter", "c_collateral_aa")}
    value[clause] = {"verdict": "fail"}
    g = _gates(V2={"verdict": "fail", "value": value})
    assert ev.determine_branch(g, None)["branch"] == branch


def test_v3_object_absent_and_present_select_different_branches(ev):
    absent = _gates(V3={"verdict": "fail", "value": {"object_absent_exceeds_budget": True}})
    present = _gates(V3={"verdict": "fail", "value": {"object_absent_exceeds_budget": False}})
    assert ev.determine_branch(absent, None)["branch"] == "NO-MERGE-OBJECT"
    assert ev.determine_branch(present, None)["branch"] == "MERGE-NOT-FOUND"


@pytest.mark.parametrize("gate, branch", [
    ("V4", "PROVENANCE-BROKEN"), ("V5", "DEFERRAL-COSTS"), ("V6", "ROOTS-OR-DUMPS-MISSING"),
])
def test_later_gates_select_their_branches(ev, gate, branch):
    assert ev.determine_branch(_gates(**{gate: "fail"}), None)["branch"] == branch


def test_v1_failure_without_da_is_unmatched_not_guessed(ev):
    b = ev.determine_branch(_gates(V1="fail"), None)
    assert b["branch"] == "UNMATCHED-COMBINATION" and "DA" in b["reason"]


def test_v1_failure_with_da_transferring_is_prefilter_not_free(ev):
    b = ev.determine_branch(_gates(V1="fail"), {"verdict": "pass", "transfers": True})
    assert b["branch"] == "PREFILTER-NOT-FREE"


def test_da_branches_fire_when_da_is_supplied(ev):
    assert ev.determine_branch(_gates(), {"verdict": "underpowered"})["branch"] == \
        "DA-UNDERPOWERED"
    assert ev.determine_branch(_gates(), {"verdict": "pass", "transfers": False})["branch"] == \
        "THRESHOLDS-DO-NOT-TRANSFER"


def test_all_pass_with_da_is_the_only_way_to_valid_and_reproduces(ev):
    assert ev.determine_branch(_gates(), {"verdict": "pass", "transfers": True})["branch"] == \
        "VALID-AND-REPRODUCES"
    # Without DA it may not be named, and a not-run gate never falls through to a pass.
    assert ev.determine_branch(_gates(), None)["branch"] == "UNMATCHED-COMBINATION"
    assert ev.determine_branch(_gates(V2="not-run"),
                               {"verdict": "pass", "transfers": True})["branch"] == \
        "UNMATCHED-COMBINATION"


# ---------------------------------------------------------------------------
# End to end on the real (legacy) H8 archive
# ---------------------------------------------------------------------------


def _cli(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)


@pytest.mark.skipif(not os.path.isdir(ARCHIVE) or not os.path.isdir(BASELINE),
                    reason="the 2026-09-09 H8 archive is not parked beside this checkout")
def test_evaluator_names_null_still_broken_on_the_h8_archive(tmp_path):
    out = str(tmp_path / "gates_h9.json")
    proc = _cli("--ctrl", os.path.join(ARCHIVE, "ctrl"), "--interleave", ARCHIVE,
                "--fivedatasets", ARCHIVE, "--baseline", BASELINE, "--h8", ARCHIVE,
                "--out", out)
    # 2 = INCONCLUSIVE: the H8 archive is not the H9 run table (it has one 5-Datasets seed, no
    # s_interleave shadow), so pre-registered pairs are missing. The branch is still named.
    assert proc.returncode == 2, proc.stderr[-3000:]
    gates = json.loads(open(out).read())

    assert gates["branch"] == "NULL-STILL-BROKEN"
    assert gates["V0b"]["verdict"] == "fail"
    # The 5-Datasets pair is the one H8 failed; it must be among the mismatches.
    assert any(m["stream"] is None for m in gates["V0b"]["value"]["all_mismatches"])
    # Legacy JSON must not crash anything: no `cka` anywhere, final pass only, no *_final fields.
    assert gates["V1"]["verdict"] == "not-run"
    assert "predate" in gates["V1"]["note"]
    assert gates["V2"].get("legacy_accs")


@pytest.mark.skipif(not os.path.isdir(ARCHIVE) or not os.path.isdir(BASELINE),
                    reason="the 2026-09-09 H8 archive is not parked beside this checkout")
def test_preflight_exits_non_zero_when_the_null_is_broken(tmp_path):
    out = str(tmp_path / "preflight.json")
    proc = _cli("--preflight", "--ctrl", os.path.join(ARCHIVE, "ctrl"), "--interleave", ARCHIVE,
                "--fivedatasets", ARCHIVE, "--baseline", BASELINE, "--h8", ARCHIVE,
                "--out", out)
    # A detected mismatch outranks incompleteness: exit 1, "do not create pod A".
    assert proc.returncode == 1, proc.stdout[-2000:]
    gates = json.loads(open(out).read())
    assert gates["preflight"]["failed_gates"] == ["V0b"]
    # Preflight stops at V0: nothing expensive is computed before the null is certified.
    assert set(gates) == {"V0a-i", "V0a-ii", "V0b", "preflight"}


@pytest.mark.skipif(not os.path.isdir(ARCHIVE), reason="H8 archive not parked beside this tree")
def test_preflight_on_the_ctrl_tree_alone_is_inconclusive_not_ok(tmp_path):
    """The reviewer's case: CTrL alone must not licence the sweep.

    The pre-flight exists for the 5-Datasets and s_interleave shadow pairs — the ones that
    actually fail. Comparing 20 CTrL pairs and printing OK is exactly the failure mode.
    """
    out = str(tmp_path / "pf.json")
    proc = _cli("--preflight", "--ctrl", os.path.join(ARCHIVE, "ctrl"), "--out", out)
    assert proc.returncode == 2, proc.stdout[-2000:]
    assert "PREFLIGHT OK" not in proc.stdout
    assert "INCONCLUSIVE" in proc.stdout
    gates = json.loads(open(out).read())
    missing = gates["preflight"]["incomplete_gates"]["V0b"]
    assert {"stream": "s_interleave", "seed": 42} in missing
    assert {"stream": None, "seed": 42} in missing


def test_a_missing_da_file_is_an_error_not_a_silent_deferral(tmp_path):
    """Ignoring it makes the DA branches and VALID-AND-REPRODUCES unreachable with no message."""
    proc = _cli("--da", str(tmp_path / "nope.json"), "--out", str(tmp_path / "g.json"))
    assert proc.returncode != 0
    assert "--da" in proc.stderr and "does not exist" in proc.stderr

    # A real file is read as before.
    da = tmp_path / "da.json"
    da.write_text(json.dumps({"verdict": "underpowered"}))
    proc2 = _cli("--da", str(da), "--out", str(tmp_path / "g2.json"))
    assert proc2.returncode in (0, 2), proc2.stderr[-2000:]
    assert json.loads(open(str(tmp_path / "g2.json")).read())["da_input"]["verdict"] == \
        "underpowered"


def test_preflight_on_an_empty_tree_is_inconclusive_not_a_pass(tmp_path):
    proc = _cli("--preflight", "--out", str(tmp_path / "p.json"))
    assert proc.returncode == 2, proc.stdout[-2000:]
    assert "INCONCLUSIVE" in proc.stdout
