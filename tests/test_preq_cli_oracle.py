"""
Tests for the prequential-arm CLI flags, the ``oracle_val_bits`` instrumentation, and
``scripts/eval_preq_arms.py`` (INTERFACE_SPEC_H4.md §B).

Three layers, mirroring tests/test_ctrl_loader_flags.py and tests/test_analysis_scripts.py:

  1. argparse defaults/choices for --gate_estimator/--preq_blocks/--preq_decide/--preq_exponent.
  2. An end-to-end `run_exp3a_kan` smoke test (tiny synthetic feature-mode tasks, the
     `_make_synth_tasks` generator from tests/test_update_oracle_dump.py) checking every gated
     decision's ``oracle_val_bits`` has three finite keys.
  3. `eval_preq_arms.py` on hand-built synthetic ``exp3a_kan_results.json`` trees: one engineered
     to give TAIL-FIXES, one COLLATERAL, one MISCALIBRATED, plus a missing-``--p2`` run.

This file does NOT depend on `concept_dag/training/kan_gate.py`'s ``estimator="prequential"``
branch (section A, developed concurrently on this branch) — every `estimator_meta` payload here is
a hand-built fixture, not the output of a real prequential run, so these tests pass independent of
whether section A has landed yet.

Runs on CPU only; small enough (feature_dim 20, concept_dim 16, 2 tasks) to finish in seconds.
"""

import importlib.util
import json
import os
import subprocess
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

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


eval_preq_arms = _load_script("eval_preq_arms", "eval_preq_arms.py")

STREAMS = ("s_minus", "s_plus", "s_out", "s_in")
G1_STREAMS = ("s_minus", "s_plus")


# ===========================================================================
# 1. argparse defaults / choices
# ===========================================================================


def test_cli_gate_estimator_accepts_prequential():
    from run_experiment import build_parser

    parser = build_parser()
    args = parser.parse_args(["--gate_estimator", "prequential"])
    assert args.gate_estimator == "prequential"
    # "single"/"crossfit" (pre-existing) must still be accepted.
    assert parser.parse_args(["--gate_estimator", "single"]).gate_estimator == "single"
    assert parser.parse_args(["--gate_estimator", "crossfit"]).gate_estimator == "crossfit"


def test_cli_preq_flag_defaults():
    from run_experiment import build_parser

    args = build_parser().parse_args([])
    assert args.preq_blocks == 5
    assert args.preq_decide == "tail"
    assert args.preq_exponent == 0.5


def test_cli_preq_flag_overrides():
    from run_experiment import build_parser

    args = build_parser().parse_args([
        "--gate_estimator", "prequential", "--preq_blocks", "7",
        "--preq_decide", "total", "--preq_exponent", "0.25",
    ])
    assert args.preq_blocks == 7
    assert args.preq_decide == "total"
    assert args.preq_exponent == 0.25


def test_cli_preq_decide_choices_reject_bad_value():
    from run_experiment import build_parser

    parser = build_parser()
    try:
        parser.parse_args(["--preq_decide", "bogus"])
        assert False, "expected argparse to reject an unknown --preq_decide value"
    except SystemExit:
        pass


# ===========================================================================
# 2. oracle_val_bits — end-to-end run_exp3a_kan smoke test
# ===========================================================================


def _make_synth_tasks(n_tasks=3, feature_dim=20, n_per_class=80, seed=2):
    """Two-Gaussian-blob generator, matching tests/test_update_oracle_dump.py's
    `_make_synth_tasks` (train/test only — no "val" key, so oracle_val_bits must fall back to
    `task["test"]`)."""
    g = torch.Generator().manual_seed(seed)
    tasks = []
    for t in range(n_tasks):
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
        tr = DataLoader(TensorDataset(X[:n_tr], Y[:n_tr]), batch_size=32, shuffle=True)
        te = DataLoader(TensorDataset(X[n_tr:], Y[n_tr:]), batch_size=32)
        tasks.append({"train": tr, "test": te, "n_classes": 2})
    return tasks


def test_oracle_val_bits_three_finite_keys(tmp_path):
    from concept_dag.experiments.kan_exp import KanExpConfig, run_exp3a_kan

    tasks = _make_synth_tasks(n_tasks=3, feature_dim=20, n_per_class=80, seed=5)
    cfg = KanExpConfig(
        backbone="synthetic", feature_dim=20, concept_dim=16, cnn_out_dim=16, n_mlp_layers=2,
        n_tasks=3, n_parents=2, subspace_k=8, soft_pca_k=8, routing_batches=10,
        root_epochs=4, child_epochs=4, gate_epochs=8, gate_lr=3e-3, lr=3e-3,
        eps_rel=0.1, eps_search=0.05, enable_search=True, search_skip=True,
        oracle_rungs=True, similarity_threshold=7.0, results_dir=str(tmp_path), device="cpu",
        log_every=1000,
    )
    res = run_exp3a_kan(cfg, tasks)

    gated = [d for d in res["decisions"] if d["task"] >= 1]
    assert gated, "expected at least one gated (non-root) task"
    for d in gated:
        assert "oracle_val_bits" in d, f"missing oracle_val_bits on task {d['task']}"
        ovb = d["oracle_val_bits"]
        assert set(ovb.keys()) == {"reuse", "search", "grow"}
        for k, v in ovb.items():
            assert isinstance(v, float) and v == v, f"oracle_val_bits[{k!r}] is not a finite float"
        # oracle_accs (existing field) must also still be present, unchanged shape.
        assert set(d["oracle_accs"].keys()) == {"reuse", "search", "grow"}


# ===========================================================================
# 3. eval_preq_arms.py — synthetic result trees for each blind branch
# ===========================================================================


def _mk_gated_decision(task, decision, oracle_accs, chosen_test_acc, oracle_val_bits_grow,
                       L_grow_bits=None, preq_rungs=None):
    """One gated decision dict. `L_grow_bits` (P0/single) xor `preq_rungs` (P1/prequential,
    the section-A `estimator_meta['rungs']` payload) — hand-built, not from a real run."""
    d = {"task": task, "decision": decision, "oracle_accs": oracle_accs,
         "oracle_val_bits": {"reuse": oracle_val_bits_grow - 0.10,
                             "search": oracle_val_bits_grow - 0.05,
                             "grow": oracle_val_bits_grow}}
    if L_grow_bits is not None:
        d["L_grow_bits"] = L_grow_bits
    if preq_rungs is not None:
        d["estimator_meta"] = {"estimator": "prequential", "rungs": preq_rungs}
    return d


def _rung_meta(tail, tail_by_exponent):
    return {"total": tail + 0.01, "tail": tail, "last_block": tail + 0.02,
            "tail_by_exponent": tail_by_exponent}


def _preq_rungs(grow_tail):
    """null/reuse/search/grow rungs whose tail_by_exponent is CONSTANT across the three logged
    exponents (0.25/0.5/1.0), so every exponent recomputes the SAME decision ("grow": the null-grow
    gap is 2.0 bits, search-grow gap is 0.8/2.0 = 0.4 > eps 0.05) — G5 agreement rate = 1.0."""
    tbe = lambda v: {"0.25": v, "0.5": v, "1.0": v}
    return {
        "null": _rung_meta(5.0, tbe(5.0)),
        "reuse": _rung_meta(4.0, tbe(4.0)),
        "search": _rung_meta(3.8, tbe(3.8)),
        "grow": _rung_meta(grow_tail, tbe(grow_tail)),
    }


def _mk_results(t3_decision, t3_test_acc, t4_decision, average_accuracy,
                grow_val_bits, arm, scenario):
    """A full exp3a_kan_results.json content: task 0 (root), task 3 (SVHN@400-analogue, gated,
    oracle-accs'd), task 4 (revisit, gated). `arm` in {"p0", "p1"}."""
    oracle_accs = {"reuse": 0.60, "search": 0.62, "grow": 0.65}
    if arm == "p0":
        t3 = _mk_gated_decision(3, t3_decision, oracle_accs, t3_test_acc, grow_val_bits,
                                L_grow_bits=grow_val_bits + 0.35)
    else:
        err_tail = 0.02 if scenario != "miscalibrated" else 0.50
        t3 = _mk_gated_decision(3, t3_decision, oracle_accs, t3_test_acc, grow_val_bits,
                                preq_rungs=_preq_rungs(grow_val_bits + err_tail))
    t4 = {"task": 4, "decision": t4_decision}
    decisions = [{"task": 0, "decision": "grow", "reason": "root"}, t3, t4]
    test_accs = [0.9, 0.0, 0.0, t3_test_acc, 0.0]
    return {"average_accuracy": average_accuracy, "test_accs": test_accs, "decisions": decisions}


def _write_ctrl_tree(base_dir, seeds, arm, scenario):
    """scenario in {"tail_fixes", "collateral", "miscalibrated"}. P0 always: search @ t3
    (regret 0.05), reuse @ t4, AA 0.80. P1: grow @ t3 (regret 0.0, delta 0.05 >= 0.015, SE 0
    since identical across every position), reuse @ t4 (matches P0) with AA 0.80 — except the
    "collateral" scenario, where ONE seed's s_minus t4 decision and AA both diverge from P0."""
    for seed in seeds:
        for stream in STREAMS:
            d = os.path.join(base_dir, f"seed_{seed}", f"exp_ctrl_{stream}")
            os.makedirs(d, exist_ok=True)
            if arm == "p0":
                results = _mk_results(t3_decision="search", t3_test_acc=0.60, t4_decision="reuse",
                                      average_accuracy=0.80, grow_val_bits=3.00, arm="p0",
                                      scenario=scenario)
            else:
                t4_decision, aa = "reuse", 0.80
                if scenario == "collateral" and seed == seeds[0] and stream == "s_minus":
                    t4_decision, aa = "search", 0.90   # breaks G2: decision differs + |dAA| > 0.005
                results = _mk_results(t3_decision="grow", t3_test_acc=0.65, t4_decision=t4_decision,
                                      average_accuracy=aa, grow_val_bits=3.00, arm="p1",
                                      scenario=scenario)
            with open(os.path.join(d, "exp3a_kan_results.json"), "w") as f:
                json.dump(results, f)


def _run_eval_preq_arms(tmp_path, scenario, with_p2=False):
    seeds = [42, 43, 44, 45, 46]
    p0_dir = tmp_path / "p0"; _write_ctrl_tree(str(p0_dir), seeds, "p0", scenario)
    p1_dir = tmp_path / "p1"; _write_ctrl_tree(str(p1_dir), seeds, "p1", scenario)
    out_path = tmp_path / "gates.json"
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "eval_preq_arms.py"),
           "--p0", str(p0_dir), "--p1", str(p1_dir), "--out", str(out_path)]
    if with_p2:
        p2_dir = tmp_path / "p2"; _write_ctrl_tree(str(p2_dir), seeds, "p1", scenario)  # reuse p1-shaped
        cmd += ["--p2", str(p2_dir)]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    data = json.loads(out_path.read_text())
    return data, res


def test_eval_preq_arms_tail_fixes(tmp_path):
    data, res = _run_eval_preq_arms(tmp_path, "tail_fixes")
    assert data["g1_regret"]["status"] == "OK"
    assert data["g1_regret"]["accept"] is True
    assert data["g1_regret"]["se"] == 0.0
    assert data["g2_collateral"]["accept"] is True
    assert data["g3_calibration"]["accept"] is True
    assert data["g3_calibration"]["median_err_tail"] < 0.10
    assert data["blind_branch"] == "TAIL-FIXES"
    # Every number/branch this script prints is also in the JSON it writes.
    assert "=== blind branch: TAIL-FIXES ===" in res.stdout


def test_eval_preq_arms_collateral(tmp_path):
    data, _ = _run_eval_preq_arms(tmp_path, "collateral")
    # G1 still accepts (same numbers as tail_fixes) — COLLATERAL must override it.
    assert data["g1_regret"]["accept"] is True
    assert data["g2_collateral"]["accept"] is False
    assert data["blind_branch"] == "COLLATERAL"


def test_eval_preq_arms_miscalibrated(tmp_path):
    data, _ = _run_eval_preq_arms(tmp_path, "miscalibrated")
    assert data["g1_regret"]["accept"] is True
    assert data["g2_collateral"]["accept"] is True
    assert data["g3_calibration"]["accept"] is False
    assert data["g3_calibration"]["median_err_tail"] > 0.10
    assert data["blind_branch"] == "MISCALIBRATED"


def test_eval_preq_arms_g5_sensitivity_agreement_on_tail_fixes(tmp_path):
    data, _ = _run_eval_preq_arms(tmp_path, "tail_fixes")
    g5 = data["g5_sensitivity"]
    assert g5["status"] == "OK"
    assert g5["agreement_rate"] == 1.0
    assert g5["accept"] is True


def test_eval_preq_arms_missing_p2_handled(tmp_path):
    data, res = _run_eval_preq_arms(tmp_path, "tail_fixes", with_p2=False)
    assert data["g4_companion"]["status"] == "SKIPPED"
    assert any("P2" in n and "not provided" in n for n in data["notes"])
    assert res.returncode == 0

    # With --p2 provided (reusing the P1-shaped tree), G4 must report descriptive counts.
    data2, res2 = _run_eval_preq_arms(tmp_path, "tail_fixes", with_p2=True)
    assert data2["g4_companion"]["status"] == "OK"
    assert data2["g4_companion"]["decision_counts"]["grow"] >= 1
    assert res2.returncode == 0


def test_eval_preq_arms_all_arms_missing_does_not_crash(tmp_path):
    out_path = tmp_path / "gates.json"
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "eval_preq_arms.py"), "--out", str(out_path)]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    data = json.loads(out_path.read_text())
    for key in ("g1_regret", "g2_collateral", "g3_calibration", "g4_companion", "g5_sensitivity"):
        assert data[key]["status"] == "SKIPPED"
    assert data["blind_branch"] == "INCONCLUSIVE"
    assert len(data["notes"]) == 5  # one "not provided" note per arm flag


# ===========================================================================
# eval_preq_arms.py — unit tests against hand-constructed cases
# ===========================================================================


def test_discover_seeds_missing_dir():
    assert eval_preq_arms.discover_seeds("/no/such/directory/xyz") == {}


def test_decision_under_exponent_grow_denominator_rule():
    tbe = lambda v: {"0.25": v, "0.5": v, "1.0": v}
    rungs = {
        "null": {"tail_by_exponent": tbe(5.0)},
        "reuse": {"tail_by_exponent": tbe(4.0)},
        "search": {"tail_by_exponent": tbe(3.8)},
        "grow": {"tail_by_exponent": tbe(3.0)},
    }
    assert eval_preq_arms._decision_under_exponent(rungs, "0.5") == "grow"

    # rel_grow just under eps -> falls through to rel_search / reuse.
    rungs_reuse = {
        "null": {"tail_by_exponent": tbe(5.0)},
        "reuse": {"tail_by_exponent": tbe(4.0)},
        "search": {"tail_by_exponent": tbe(3.99)},
        "grow": {"tail_by_exponent": tbe(3.98)},
    }
    assert eval_preq_arms._decision_under_exponent(rungs_reuse, "0.5") == "reuse"


def test_decision_under_exponent_missing_key_returns_none():
    assert eval_preq_arms._decision_under_exponent({"null": {"tail_by_exponent": {}}}, "0.5") is None
