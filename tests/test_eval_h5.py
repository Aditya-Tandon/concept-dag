"""
Tests for ``scripts/eval_h5_arms.py`` (INTERFACE_SPEC_H5.md §3): the S0-S5 gates, the
``blind_branch``/``collateral_flag`` precedence, and the CLI end to end on hand-built synthetic
``exp3a_kan_results.json`` trees, mirroring the layered style of
``tests/test_preq_cli_oracle.py``/``tests/test_analysis_scripts.py``.

Layer 1 unit-tests each gate function directly against small hand-built ``{seed: {stream: results}}``
dicts. Layer 2 runs the script end to end (subprocess) on synthetic directory trees engineered to
land on TIE-FIXES, OVER-FIRES and NO-ASYMMETRY, plus S0's determinism check.
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _load_script(name: str, filename: str):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eval_h5 = _load_script("eval_h5_arms", "eval_h5_arms.py")

STREAMS = ("s_minus", "s_plus", "s_out", "s_in")
G_STREAMS = ("s_minus", "s_plus")
SEEDS = (42, 43, 44, 45, 46)


# ===========================================================================
# Layer 1 — direct unit tests against hand-built {seed: {stream: results}} dicts
# ===========================================================================


def _oracle_accs():
    return {"reuse": 0.60, "search": 0.63, "grow": 0.65}   # best = 0.65


def _t3_decision(decision, test_acc, novelty=0.5, opt_abs=0.05, tie=None, tie_cf=None):
    meta = {"estimator": "select-score", "novelty": novelty,
            "search_selection_optimism_abs": opt_abs}
    if tie is not None:
        meta["tie"] = tie
    if tie_cf is not None:
        meta["tie_counterfactual"] = tie_cf
    return {"task": 3, "decision": decision, "oracle_accs": _oracle_accs(),
            "estimator_meta": meta}, test_acc


def _results(t3_decision_dict, test_acc_t3, extra_decisions=(), average_accuracy=0.80):
    decisions = [{"task": 0, "decision": "grow", "reason": "root"}, t3_decision_dict, *extra_decisions]
    test_accs = [0.9, 0.0, 0.0, test_acc_t3, 0.0]
    return {"average_accuracy": average_accuracy, "test_accs": test_accs, "decisions": decisions}


def _arm_10_positions(regret, opt_abs, decision="search", tie_fn=None):
    """Build a 5-seed x {s_minus, s_plus} arm with a CONSTANT regret/optimism at t3 (SE == 0, so
    S1 is never accidentally INCONCLUSIVE). `tie_fn(seed, stream, idx) -> (tie, tie_cf) | None`."""
    arm = {}
    idx = 0
    for seed in SEEDS:
        for stream in G_STREAMS:
            tie = tie_cf = None
            if tie_fn is not None:
                tie, tie_cf = tie_fn(seed, stream, idx)
            test_acc = 0.65 - regret
            t3, _ = _t3_decision(decision, test_acc, opt_abs=opt_abs, tie=tie, tie_cf=tie_cf)
            t4 = {"task": 4, "decision": "reuse",
                 "estimator_meta": {"estimator": "select-score", "novelty": 0.5,
                                    "tie": {"fired": False}}}
            arm.setdefault(seed, {})[stream] = _results(t3, test_acc, extra_decisions=[t4])
            idx += 1
    return arm


# --- s1_primary ---


def test_s1_primary_accepts_low_regret():
    q1 = _arm_10_positions(regret=0.01, opt_abs=0.05, decision="grow")
    g = eval_h5.s1_primary(q1)
    assert g["status"] == "OK"
    assert g["n_positions"] == 10
    assert g["mean_regret_Q1"] == pytest.approx(0.01)
    assert g["accept"] is True
    assert g["grow_count"] == 10


def test_s1_primary_rejects_high_regret():
    q1 = _arm_10_positions(regret=0.02, opt_abs=0.05, decision="search")
    g = eval_h5.s1_primary(q1)
    assert g["status"] == "OK"
    assert g["accept"] is False


def test_s1_primary_inconclusive_on_high_se():
    """A mix of very-low and very-high regrets pushes the across-position SE past 0.010."""
    q1 = {}
    for i, seed in enumerate(SEEDS):
        for stream in G_STREAMS:
            regret = 0.001 if i % 2 == 0 else 0.20
            test_acc = 0.65 - regret
            t3, _ = _t3_decision("search", test_acc)
            q1.setdefault(seed, {})[stream] = _results(t3, test_acc)
    g = eval_h5.s1_primary(q1)
    assert g["status"] == "INCONCLUSIVE"
    assert g["accept"] is False


def test_s1_primary_skipped_when_missing():
    assert eval_h5.s1_primary({})["status"] == "SKIPPED"


# --- s2_tie_fixes ---


def test_s2_tie_fixes_accepts_high_fired_fraction_and_low_regret():
    def tie_fn(seed, stream, idx):
        fired = idx < 7   # 7/10 >= 60%
        tie = {"fired": fired, "margin": 0.001, "se": 0.01, "novelty": 0.05}
        tie_cf = {"0.5": fired, "1.0": fired, "2.0": fired}
        return tie, tie_cf

    q2 = _arm_10_positions(regret=0.005, opt_abs=0.05, decision="grow", tie_fn=tie_fn)
    g = eval_h5.s2_tie_fixes(q2)
    assert g["status"] == "OK"
    assert g["tie_count"] == 7
    assert g["tie_frac"] == pytest.approx(0.7)
    assert g["mean_regret"] == pytest.approx(0.005)
    assert g["accept"] is True
    assert g["counterfactual_counts"]["1.0"] == 7


def test_s2_tie_fixes_rejects_low_fired_fraction():
    def tie_fn(seed, stream, idx):
        fired = idx < 3   # 3/10 < 60%
        return {"fired": fired}, {"0.5": fired, "1.0": fired, "2.0": fired}

    q2 = _arm_10_positions(regret=0.005, opt_abs=0.05, decision="search", tie_fn=tie_fn)
    g = eval_h5.s2_tie_fixes(q2)
    assert g["accept"] is False


# --- s3_specificity ---


def test_s3_specificity_accepts_zero_firings_outside_t3():
    q2 = _arm_10_positions(regret=0.005, opt_abs=0.05, decision="search",
                           tie_fn=lambda s, st, i: ({"fired": False}, {"0.5": False, "1.0": False, "2.0": False}))
    g = eval_h5.s3_specificity(q2, {})
    assert g["status"] == "OK"
    assert g["n_firings"] == 0
    assert g["accept"] is True


def test_s3_specificity_rejects_firing_outside_t3():
    q2 = _arm_10_positions(regret=0.005, opt_abs=0.05, decision="search")
    # inject a tie firing at t4 for one (seed, stream)
    q2[SEEDS[0]]["s_minus"]["decisions"][2]["estimator_meta"]["tie"] = {"fired": True}
    g = eval_h5.s3_specificity(q2, {})
    assert g["status"] == "OK"
    assert g["n_firings"] == 1
    assert g["accept"] is False


def test_s3_specificity_inconclusive_on_novelty_gray_zone_with_no_firings():
    q2 = _arm_10_positions(regret=0.005, opt_abs=0.05, decision="search",
                           tie_fn=lambda s, st, i: ({"fired": False}, {"0.5": False, "1.0": False, "2.0": False}))
    # task 4's novelty sits inside [0.05, 0.20] for one run.
    q2[SEEDS[0]]["s_minus"]["decisions"][2]["estimator_meta"]["novelty"] = 0.12
    g = eval_h5.s3_specificity(q2, {})
    assert g["status"] == "INCONCLUSIVE"
    assert g["n_firings"] == 0
    assert g["n_novelty_gray"] == 1
    assert g["accept"] is None


def test_s3_specificity_scans_5ds_too():
    q2 = _arm_10_positions(regret=0.005, opt_abs=0.05, decision="search")
    q2_5ds = {SEEDS[0]: {"decisions": [
        {"task": 0, "decision": "grow"},
        {"task": 1, "decision": "grow",
         "estimator_meta": {"estimator": "select-score", "novelty": 0.5, "tie": {"fired": True}}},
    ]}}
    g = eval_h5.s3_specificity(q2, q2_5ds)
    assert g["n_firings"] == 1
    assert g["accept"] is False


# --- s5_asymmetry_live ---


def test_s5_accepts_high_optimism():
    q1 = _arm_10_positions(regret=0.005, opt_abs=0.05, decision="grow")
    g = eval_h5.s5_asymmetry_live(q1)
    assert g["accept"] is True
    assert g["reject"] is False
    assert g["marginal"] is False


def test_s5_rejects_low_optimism():
    q1 = _arm_10_positions(regret=0.005, opt_abs=0.005, decision="grow")
    g = eval_h5.s5_asymmetry_live(q1)
    assert g["reject"] is True
    assert g["accept"] is False


def test_s5_marginal_between_thresholds():
    q1 = _arm_10_positions(regret=0.005, opt_abs=0.02, decision="grow")
    g = eval_h5.s5_asymmetry_live(q1)
    assert g["accept"] is False
    assert g["reject"] is False
    assert g["marginal"] is True


# --- s0_determinism ---


def test_s0_determinism_matches():
    a0 = _arm_10_positions(regret=0.01, opt_abs=0.05, decision="grow")
    q0 = _arm_10_positions(regret=0.01, opt_abs=0.05, decision="grow")
    g = eval_h5.s0_determinism(a0, q0)
    assert g["status"] == "OK"
    assert g["abort_determinism"] is False


def test_s0_determinism_flags_mismatch():
    a0 = _arm_10_positions(regret=0.01, opt_abs=0.05, decision="grow")
    q0 = _arm_10_positions(regret=0.01, opt_abs=0.05, decision="search")   # decision differs
    g = eval_h5.s0_determinism(a0, q0)
    assert g["status"] == "OK"
    assert g["abort_determinism"] is True


def test_s0_determinism_skipped_when_missing():
    assert eval_h5.s0_determinism({}, {})["status"] == "SKIPPED"


# --- blind_branch / collateral_flag precedence ---


def _gate(status="OK", **kw):
    return {"status": status, **kw}


def test_blind_branch_no_asymmetry_has_top_precedence_over_over_fires():
    s1 = _gate(accept=False)
    s2 = _gate(accept=False)
    s3 = _gate(accept=False)              # would be OVER-FIRES on its own
    s5 = _gate(accept=False, reject=True)  # NO-ASYMMETRY must win
    assert eval_h5.blind_branch(s1, s2, s3, s5) == "NO-ASYMMETRY"


def test_blind_branch_over_fires_beats_tie_and_asymmetry_fixes():
    s1 = _gate(accept=True)               # would be ASYMMETRY-FIXES on its own
    s2 = _gate(accept=True)
    s3 = _gate(accept=False)              # OVER-FIRES must win over S1/S2 accepts
    s5 = _gate(accept=False, reject=False)
    assert eval_h5.blind_branch(s1, s2, s3, s5) == "OVER-FIRES"


def test_blind_branch_asymmetry_fixes_beats_tie_fixes():
    s1 = _gate(accept=True)
    s2 = _gate(accept=True)
    s3 = _gate(accept=True)
    s5 = _gate(accept=False, reject=False)
    assert eval_h5.blind_branch(s1, s2, s3, s5) == "ASYMMETRY-FIXES"


def test_blind_branch_tie_fixes_when_s1_rejects_s2_accepts():
    s1 = _gate(accept=False)
    s2 = _gate(accept=True)
    s3 = _gate(accept=True)
    s5 = _gate(accept=False, reject=False)
    assert eval_h5.blind_branch(s1, s2, s3, s5) == "TIE-FIXES"


def test_blind_branch_no_effect_when_s1_and_s2_reject():
    s1 = _gate(accept=False)
    s2 = _gate(accept=False)
    s3 = _gate(accept=True)
    s5 = _gate(accept=False, reject=False)
    assert eval_h5.blind_branch(s1, s2, s3, s5) == "NO-EFFECT"


def test_blind_branch_inconclusive_when_everything_missing():
    s1 = _gate(status="SKIPPED")
    s2 = _gate(status="SKIPPED")
    s3 = _gate(status="SKIPPED")
    s5 = _gate(status="SKIPPED")
    assert eval_h5.blind_branch(s1, s2, s3, s5) == "INCONCLUSIVE"


def test_collateral_flag_true_only_on_explicit_reject():
    assert eval_h5.collateral_flag(_gate(accept=True), _gate(accept=True)) is False
    assert eval_h5.collateral_flag(_gate(accept=False), _gate(accept=True)) is True
    assert eval_h5.collateral_flag(_gate(status="SKIPPED"), _gate(status="SKIPPED")) is False


# ===========================================================================
# Layer 2 — end-to-end CLI on synthetic directory trees: TIE-FIXES / OVER-FIRES / NO-ASYMMETRY
# ===========================================================================


def _write_ctrl_arm(base_dir, arm):
    for seed, streams in arm.items():
        for stream, results in streams.items():
            d = os.path.join(base_dir, f"seed_{seed}", f"exp_ctrl_{stream}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "exp3a_kan_results.json"), "w") as f:
                json.dump(results, f)


def _run_eval_h5(tmp_path, q1, q2, extra_args=()):
    q1_dir = tmp_path / "q1"; _write_ctrl_arm(str(q1_dir), q1)
    q2_dir = tmp_path / "q2"; _write_ctrl_arm(str(q2_dir), q2)
    out_path = tmp_path / "gates.json"
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "eval_h5_arms.py"),
          "--q1", str(q1_dir), "--q2", str(q2_dir), "--out", str(out_path), *extra_args]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    data = json.loads(out_path.read_text())
    return data, res


def test_eval_h5_scenario_tie_fixes(tmp_path):
    # Q1: S1 rejects (regret 0.02 > 0.015 thresh, SE == 0), S5 accepts (opt_abs 0.05 >= 0.030).
    q1 = _arm_10_positions(regret=0.02, opt_abs=0.05, decision="search")
    # Q2: S2 accepts (7/10 tie fired, mean regret 0.005 <= 0.010); S3 accepts (0 firings outside t3).
    def tie_fn(seed, stream, idx):
        fired = idx < 7
        return ({"fired": fired, "margin": 0.001, "se": 0.01, "novelty": 0.05},
               {"0.5": fired, "1.0": fired, "2.0": fired})
    q2 = _arm_10_positions(regret=0.005, opt_abs=0.05, decision="grow", tie_fn=tie_fn)

    data, _ = _run_eval_h5(tmp_path, q1, q2)
    assert data["s1_primary"]["accept"] is False
    assert data["s2_tie_fixes"]["accept"] is True
    assert data["s3_specificity"]["accept"] is True
    assert data["s5_asymmetry_live"]["reject"] is False
    assert data["blind_branch"] == "TIE-FIXES"


def test_eval_h5_scenario_over_fires(tmp_path):
    q1 = _arm_10_positions(regret=0.02, opt_abs=0.05, decision="search")
    q2 = _arm_10_positions(regret=0.005, opt_abs=0.05, decision="search",
                           tie_fn=lambda s, st, i: ({"fired": False}, {"0.5": False, "1.0": False, "2.0": False}))
    # A real tie firing OUTSIDE t3 (task 4) for one (seed, stream) — must dominate S2's accept.
    q2[SEEDS[0]]["s_minus"]["decisions"][2]["estimator_meta"]["tie"] = {"fired": True}

    data, _ = _run_eval_h5(tmp_path, q1, q2)
    assert data["s3_specificity"]["accept"] is False
    assert data["s5_asymmetry_live"]["reject"] is False
    assert data["blind_branch"] == "OVER-FIRES"


def test_eval_h5_scenario_no_asymmetry(tmp_path):
    # Q1's own search-selection-optimism reads well below the 0.015 reject threshold.
    q1 = _arm_10_positions(regret=0.02, opt_abs=0.005, decision="search")
    q2 = _arm_10_positions(regret=0.005, opt_abs=0.005, decision="grow",
                           tie_fn=lambda s, st, i: ({"fired": True, "margin": 0.0, "se": 0.01, "novelty": 0.05},
                                                    {"0.5": True, "1.0": True, "2.0": True}))

    data, _ = _run_eval_h5(tmp_path, q1, q2)
    assert data["s5_asymmetry_live"]["reject"] is True
    # Even though S2 would independently accept (TIE-FIXES), NO-ASYMMETRY takes precedence.
    assert data["s2_tie_fixes"]["accept"] is True
    assert data["blind_branch"] == "NO-ASYMMETRY"


def test_eval_h5_missing_arms_do_not_crash(tmp_path):
    out_path = tmp_path / "gates.json"
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "eval_h5_arms.py"), "--out", str(out_path)]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    data = json.loads(out_path.read_text())
    assert data["blind_branch"] == "INCONCLUSIVE"
    for key in ("s0_determinism", "s1_primary", "s2_tie_fixes", "s3_specificity",
               "s4a_decisions_identical", "s4b_5ds_aa", "s5_asymmetry_live"):
        assert data[key]["status"] == "SKIPPED"
    assert len(data["notes"]) == 7   # one "not provided" note per arm flag


def test_eval_h5_prints_abort_determinism_on_mismatch(tmp_path):
    a0 = _arm_10_positions(regret=0.01, opt_abs=0.05, decision="grow")
    q0 = _arm_10_positions(regret=0.01, opt_abs=0.05, decision="search")   # deliberately mismatched
    a0_dir = tmp_path / "a0"; _write_ctrl_arm(str(a0_dir), a0)
    q0_dir = tmp_path / "q0"; _write_ctrl_arm(str(q0_dir), q0)
    out_path = tmp_path / "gates.json"
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "eval_h5_arms.py"),
          "--a0", str(a0_dir), "--q0", str(q0_dir), "--out", str(out_path)]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0
    assert "ABORT-DETERMINISM" in res.stdout
    data = json.loads(out_path.read_text())
    assert data["s0_determinism"]["abort_determinism"] is True
