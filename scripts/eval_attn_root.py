#!/usr/bin/env python
"""
eval_attn_root.py — the mechanical gate evaluation for H7, the attention-pooling root adoption
(Central Library/Hypotheses/concept-dag/kan-gated-growth/attn-root-adoption.md).

Reads the per-run ``exp3a_kan_results.json`` files written by ``run_experiment.py`` for the two
arms (``C`` = control = ``--root_family mlp_cls``, ``T`` = treatment = ``--root_family
attn_pool``) plus the archived pre-adoption baseline (the null-check reference), and evaluates
every gate the note pre-registers: N1, N2 (null validity, evaluated first), R1a-R1d
(5-Datasets), R2a-R2d (CTrL), R3 (gate noise) and R4 (cost, descriptive) — then names the
"Branches" cell the results select.

Layout expected
----------------
    <ctrl>/ctrl_<family>/seed_<S>/exp_ctrl_<stream>/exp3a_kan_results.json
    <fivedatasets>/5ds_<family>/seed_<S>/exp5ds_kan/exp3a_kan_results.json
    <baseline>/ctrl/seed_<S>/exp_ctrl_<stream>/exp3a_kan_results.json      (no family level)
    <baseline>/5ds/seed_<S>/exp5ds_kan/exp3a_kan_results.json

``family`` in {mlp_cls, attn_pool}; CTrL seeds 42-46 x streams {s_minus, s_out, s_plus, s_in};
5-Datasets seeds 42-44. Any of ``--ctrl``/``--fivedatasets``/``--baseline`` may be omitted (at
least one is required) — missing sections report every gate that needs them as
``"verdict": "not-run"`` rather than crashing, since this script is meant to be run mid-sweep,
before every arm/seed/stream has finished.

Gates (see the note's "Pre-registered gates and blind branches" for the exact wording)
  N1   null validity, 5-Datasets   arm C reproduces the archived baseline (decisions, AA, SVHN
                                    root acc) within tolerance.
  N2   null validity, CTrL         arm C's 80 gated decisions (4 streams x 5 seeds x 4 positions)
                                    match the archived baseline: t4 reuses 20/20, >=90% (72/80)
                                    identical overall, every difference at t3.
  R1a  H7a transfer                arm T mean SVHN-root (t3) test accuracy >= 0.75.
  R1b  H7b no collateral           arm T AA >= 0.9113 and no non-SVHN task down >0.01 vs arm C.
  R1c  H7b decisions               arm T = 4 grow / 0 search / 1 reuse, all 3 seeds, t4 reuse.
  R1d  descriptive                 per-gated-position ladder internals, both arms — no threshold.
  R2a  H7c small-n accuracy        paired (T-C) mean t3 chosen-rung accuracy >= 0.03 and >= 2 SE.
  R2b  H7c regret                  paired (T-C) mean t3 regret <= +0.02 (must not worsen).
  R2c  H7b reuse control           arm T t4 reuse in 20/20 CTrL runs.
  R2d  H7b s_plus                  arm T s_plus t4 reuse 5/5, accuracy within 0.02 of arm C.
  R3   H7d gate noise              across-run SD ratio (T/C) of rel_grow, rel_search at t3.
  R4   cost                        params_per_root, mean gate_seconds, wall time (via --progress).

Blind branches: ADOPT, ADOPT-WITH-NOISE, CAPABLE-BUT-DESTABILISING, NO-TRANSFER, plus the
orthogonal flags COLLATERAL (R1b fails) and SMALL-N-NULL (R2a fails while R1a passes). See
``determine_branch`` for exactly how these are read off the per-gate verdicts, including the
non-registered fallback labels (``NOT-RUN``, ``INVALID-NULL-CHECK-FAILED``, ``INDETERMINATE``)
this script uses so it never has to invent a false positive.

This script does the evaluation ONLY: no plotting, no re-running experiments.

Usage
-----
    python scripts/eval_attn_root.py \\
        --ctrl results/attn_root/ctrl --fivedatasets results/attn_root/5ds \\
        --baseline .../results_rebaseline_2026-09-08 \\
        [--progress results/attn_root/progress_ctrl.log ...] \\
        --out results/attn_root/gates.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from typing import Dict, List, Optional, Tuple

CTRL_STREAMS = ("s_minus", "s_out", "s_plus", "s_in")
FIVEDS_SEEDS = (42, 43, 44)
CTRL_SEEDS = (42, 43, 44, 45, 46)
CONTROL_FAMILY = "mlp_cls"     # arm C
TREATMENT_FAMILY = "attn_pool"  # arm T
T3, T4 = 3, 4
GATED_TASKS = (1, 2, 3, 4)

# --- thresholds, taken verbatim from the pre-registration's gate table -----------------------
N1_AA_TOL = 0.005
N1_ACC_TOL = 0.01
N2_MIN_FRACTION = 72 / 80          # ">= 72 of 80" — read as a fraction so a partial mid-sweep
                                    # run (e.g. 40/40 so far) is judged on the same 90% bar
                                    # rather than an unreachable absolute count. See gate_n2().
R1A_THRESHOLD = 0.75
R1B_AA_THRESHOLD = 0.9113          # = 0.8813 + 0.03 (seed-44 archived AA + margin)
R1B_TASK_TOL = 0.01
R2A_MIN_DELTA = 0.03
R2A_MIN_SE_MULT = 2.0
R2B_MAX_DELTA = 0.02
R2D_ACC_TOL = 0.02
R3_NOISIER_RATIO = 1.5
R3_UNCHANGED_RATIO = 1.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def _discover_seed_dirs(root: Optional[str]) -> Dict[int, str]:
    if not root or not os.path.isdir(root):
        return {}
    out = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if name.startswith("seed_") and os.path.isdir(p):
            try:
                out[int(name[len("seed_"):])] = p
            except ValueError:
                continue
    return out


def load_ctrl_tree(root: Optional[str]) -> Dict[int, Dict[str, Dict]]:
    """{seed: {stream: results}} under <root>/seed_<S>/exp_ctrl_<stream>/exp3a_kan_results.json"""
    out: Dict[int, Dict[str, Dict]] = {}
    for seed, seed_dir in _discover_seed_dirs(root).items():
        for stream in CTRL_STREAMS:
            path = os.path.join(seed_dir, f"exp_ctrl_{stream}", "exp3a_kan_results.json")
            if os.path.isfile(path):
                out.setdefault(seed, {})[stream] = _read_json(path)
    return out


def load_5ds_tree(root: Optional[str]) -> Dict[int, Dict]:
    """{seed: results} under <root>/seed_<S>/exp5ds_kan/exp3a_kan_results.json"""
    out: Dict[int, Dict] = {}
    for seed, seed_dir in _discover_seed_dirs(root).items():
        path = os.path.join(seed_dir, "exp5ds_kan", "exp3a_kan_results.json")
        if os.path.isfile(path):
            out[seed] = _read_json(path)
    return out


def load_ctrl_family(base_dir: Optional[str], family: str) -> Dict[int, Dict[str, Dict]]:
    if not base_dir:
        return {}
    return load_ctrl_tree(os.path.join(base_dir, f"ctrl_{family}"))


def load_5ds_family(base_dir: Optional[str], family: str) -> Dict[int, Dict]:
    if not base_dir:
        return {}
    return load_5ds_tree(os.path.join(base_dir, f"5ds_{family}"))


def load_baseline_ctrl(base_dir: Optional[str]) -> Dict[int, Dict[str, Dict]]:
    if not base_dir:
        return {}
    return load_ctrl_tree(os.path.join(base_dir, "ctrl"))


def load_baseline_5ds(base_dir: Optional[str]) -> Dict[int, Dict]:
    if not base_dir:
        return {}
    return load_5ds_tree(os.path.join(base_dir, "5ds"))


def dec_at(results: Dict, task: int) -> Optional[Dict]:
    for d in results.get("decisions", []):
        if d.get("task") == task:
            return d
    return None


def t3_regret(results: Dict) -> Optional[float]:
    """max(oracle_accs) - chosen (test_accs[t3]) accuracy; skipped (None) when the decision
    carries no oracle_accs, per the note's definition."""
    d = dec_at(results, T3)
    oracle = d.get("oracle_accs") if d else None
    if not oracle:
        return None
    accs = results.get("test_accs")
    if not accs or len(accs) <= T3:
        return None
    return max(oracle.values()) - accs[T3]


def params_per_root(results: Dict) -> Optional[float]:
    """``params_per_root``, falling back to ``param_curve_total[-1]`` then ``param_curve[-1]``
    when the (newer) key is absent, per the spec."""
    v = results.get("params_per_root")
    if v is not None:
        return v
    pct = results.get("param_curve_total")
    if pct:
        return pct[-1]
    pc = results.get("param_curve")
    if pc:
        return pc[-1]
    return None


def mean_gate_seconds(results: Dict) -> Optional[float]:
    vals = [d.get("gate_seconds") for d in results.get("decisions", [])
             if d.get("gate_seconds") is not None]
    return statistics.mean(vals) if vals else None


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def _sample_stdev(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) >= 2 else None


# ---------------------------------------------------------------------------
# N1 / N2 — null validity (evaluated before any R gate)
# ---------------------------------------------------------------------------

def gate_n1(baseline_5ds: Dict[int, Dict], control_5ds: Dict[int, Dict]) -> Dict:
    gate = {"gate": "N1",
            "observable": "arm C (mlp_cls), 5-Datasets, all 3 seeds: decision sequence, AA, "
                          "SVHN root (t3) accuracy vs the archived baseline",
            "threshold": f"decisions == grow*4+reuse; AA within {N1_AA_TOL}; "
                         f"t3 acc within {N1_ACC_TOL}",
            "value": None, "verdict": "not-run"}
    if not baseline_5ds or not control_5ds:
        gate["note"] = "baseline 5-Datasets and/or control(mlp_cls) 5-Datasets arm missing"
        return gate
    rows = []
    for seed in FIVEDS_SEEDS:
        if seed not in baseline_5ds or seed not in control_5ds:
            continue
        b, c = baseline_5ds[seed], control_5ds[seed]
        c_decisions = [(dec_at(c, t) or {}).get("decision") for t in range(5)]
        decisions_match = c_decisions == ["grow", "grow", "grow", "grow", "reuse"]
        aa_c, aa_b = c.get("average_accuracy"), b.get("average_accuracy")
        aa_delta = (aa_c - aa_b) if (aa_c is not None and aa_b is not None) else None
        aa_ok = aa_delta is not None and abs(aa_delta) <= N1_AA_TOL
        t3_c = c.get("test_accs", [None] * 5)[T3]
        t3_b = b.get("test_accs", [None] * 5)[T3]
        t3_delta = (t3_c - t3_b) if (t3_c is not None and t3_b is not None) else None
        t3_ok = t3_delta is not None and abs(t3_delta) <= N1_ACC_TOL
        rows.append({"seed": seed, "decisions": c_decisions, "decisions_match": decisions_match,
                     "control_aa": aa_c, "baseline_aa": aa_b, "aa_delta": aa_delta, "aa_ok": aa_ok,
                     "control_t3_acc": t3_c, "baseline_t3_acc": t3_b, "t3_delta": t3_delta,
                     "t3_ok": t3_ok,
                     "row_ok": bool(decisions_match and aa_ok and t3_ok)})
    if not rows:
        gate["note"] = "no matched seeds between baseline and control(mlp_cls) 5-Datasets"
        return gate
    n_ok = sum(1 for r in rows if r["row_ok"])
    gate["value"] = {"rows": rows}
    gate["verdict"] = "pass" if n_ok == len(rows) else "fail"
    gate["headline"] = f"{n_ok}/{len(rows)} seeds reproduce the baseline"
    if len(rows) < len(FIVEDS_SEEDS):
        gate["note"] = f"partial: {len(rows)}/{len(FIVEDS_SEEDS)} seeds present so far"
    return gate


def gate_n2(baseline_ctrl: Dict[int, Dict[str, Dict]],
            control_ctrl: Dict[int, Dict[str, Dict]]) -> Dict:
    gate = {"gate": "N2",
            "observable": "arm C (mlp_cls), CTrL, gated decisions (indices 1-4) vs the archived "
                          "baseline, up to 80 comparisons (4 streams x 5 seeds x 4 positions)",
            "threshold": ">=90% (72/80) identical; every difference at t3; t4 reuse 20/20",
            "value": None, "verdict": "not-run"}
    if not baseline_ctrl or not control_ctrl:
        gate["note"] = "baseline CTrL and/or control(mlp_cls) CTrL arm missing"
        return gate
    total = identical = off_t3 = t4_reuse = t4_total = 0
    diffs = []
    for seed in CTRL_SEEDS:
        for stream in CTRL_STREAMS:
            b = baseline_ctrl.get(seed, {}).get(stream)
            c = control_ctrl.get(seed, {}).get(stream)
            if b is None or c is None:
                continue
            for t in GATED_TASKS:
                db, dc = dec_at(b, t), dec_at(c, t)
                if db is None or dc is None:
                    continue
                total += 1
                same = db.get("decision") == dc.get("decision")
                if same:
                    identical += 1
                else:
                    diffs.append({"seed": seed, "stream": stream, "task": t,
                                 "baseline": db.get("decision"), "control": dc.get("decision")})
                    if t != T3:
                        off_t3 += 1
                if t == T4:
                    t4_total += 1
                    if dc.get("decision") == "reuse":
                        t4_reuse += 1
    if total == 0:
        gate["note"] = "no matched (seed, stream) pairs between baseline and control CTrL"
        return gate
    fraction = identical / total
    t4_ok = t4_total > 0 and t4_reuse == t4_total
    pass_ = (fraction >= N2_MIN_FRACTION) and (off_t3 == 0) and t4_ok
    gate["value"] = {"total_compared": total, "identical": identical, "fraction_identical": fraction,
                      "differences": diffs, "differences_off_t3": off_t3,
                      "t4_reuse": t4_reuse, "t4_total": t4_total}
    gate["verdict"] = "pass" if pass_ else "fail"
    gate["headline"] = (f"{identical}/{total} identical ({fraction:.0%}), {off_t3} off-t3 diffs, "
                        f"t4 reuse {t4_reuse}/{t4_total}")
    if total < 80:
        gate["note"] = f"partial: {total}/80 gated comparisons available so far"
    return gate


# ---------------------------------------------------------------------------
# R1 — 5-Datasets (3 seeds, paired by seed)
# ---------------------------------------------------------------------------

def gate_r1a(treatment_5ds: Dict[int, Dict]) -> Dict:
    gate = {"gate": "R1a",
            "observable": "arm T SVHN root (t3) test accuracy, mean over seeds",
            "threshold": f">= {R1A_THRESHOLD}", "value": None, "verdict": "not-run"}
    if not treatment_5ds:
        gate["note"] = "treatment(attn_pool) 5-Datasets arm missing"
        return gate
    rows = []
    for seed in FIVEDS_SEEDS:
        if seed not in treatment_5ds:
            continue
        r = treatment_5ds[seed]
        d3 = dec_at(r, T3)
        acc = r.get("test_accs", [None] * 5)[T3]
        rows.append({"seed": seed, "t3_test_acc": acc,
                     "t3_did_not_grow": bool(d3 and d3.get("decision") != "grow")})
    accs = [r["t3_test_acc"] for r in rows if r["t3_test_acc"] is not None]
    if not accs:
        gate["note"] = "no seeds with test_accs found"
        return gate
    mean_acc = statistics.mean(accs)
    gate["value"] = {"mean_t3_test_acc": mean_acc, "n_seeds": len(rows), "rows": rows}
    gate["verdict"] = "pass" if mean_acc >= R1A_THRESHOLD else "fail"
    gate["headline"] = f"mean t3 acc = {mean_acc:.4f} (n={len(rows)})"
    if len(rows) < len(FIVEDS_SEEDS):
        gate["note"] = f"partial: {len(rows)}/{len(FIVEDS_SEEDS)} seeds present so far"
    return gate


def gate_r1b(control_5ds: Dict[int, Dict], treatment_5ds: Dict[int, Dict]) -> Dict:
    gate = {"gate": "R1b",
            "observable": "arm T average_accuracy; per-task (non-SVHN) accuracy vs arm C at "
                          "the same seed",
            # AMBIGUITY: the note states a single AA number (0.8813 + 0.03) without saying
            # whether the bar applies to the seed mean or to every seed individually. We take
            # the conservative reading: EVERY seed's AA must clear it, not just the mean (the
            # mean is still reported). Likewise "within 0.01" per non-SVHN task is applied
            # per-seed (not averaged across seeds) — the stricter, most literal reading.
            "threshold": f"AA >= {R1B_AA_THRESHOLD} in every seed (conservative reading, see "
                         f"comment); no non-SVHN task down > {R1B_TASK_TOL} vs arm C in any seed",
            "value": None, "verdict": "not-run"}
    if not treatment_5ds or not control_5ds:
        gate["note"] = "control(mlp_cls) and/or treatment(attn_pool) 5-Datasets arm missing"
        return gate
    rows, collateral = [], []
    for seed in FIVEDS_SEEDS:
        if seed not in treatment_5ds or seed not in control_5ds:
            continue
        t, c = treatment_5ds[seed], control_5ds[seed]
        aa_t = t.get("average_accuracy")
        aa_ok = aa_t is not None and aa_t >= R1B_AA_THRESHOLD
        task_deltas = []
        for i, (ta, ca) in enumerate(zip(t.get("test_accs", []), c.get("test_accs", []))):
            if i == T3:
                continue
            delta = ta - ca
            within = delta >= -R1B_TASK_TOL
            task_deltas.append({"task": i, "treat_acc": ta, "control_acc": ca, "delta": delta,
                               "within": within})
            if not within:
                collateral.append({"seed": seed, "task": i, "delta": delta})
        rows.append({"seed": seed, "treat_aa": aa_t, "aa_ok": aa_ok, "task_deltas": task_deltas})
    if not rows:
        gate["note"] = "no paired seeds"
        return gate
    aas = [r["treat_aa"] for r in rows if r["treat_aa"] is not None]
    mean_aa = statistics.mean(aas) if aas else None
    aa_ok_all = all(r["aa_ok"] for r in rows)
    pass_ = aa_ok_all and not collateral
    gate["value"] = {"mean_treat_aa": mean_aa, "aa_ok_every_seed": aa_ok_all,
                     "collateral_violations": collateral, "rows": rows}
    gate["verdict"] = "pass" if pass_ else "fail"
    gate["headline"] = (f"mean AA={mean_aa:.4f}" if mean_aa is not None else "mean AA=?") + \
                       f", collateral violations={len(collateral)}"
    if len(rows) < len(FIVEDS_SEEDS):
        gate["note"] = f"partial: {len(rows)}/{len(FIVEDS_SEEDS)} seeds present so far"
    return gate


def gate_r1c(treatment_5ds: Dict[int, Dict]) -> Dict:
    gate = {"gate": "R1c",
            "observable": "arm T decision counts per seed (n_grow/n_search/n_reuse); t4 "
                          "(organic duplicate) decision and rel_grow",
            "threshold": "4 grow / 0 search / 1 reuse in all 3 seeds; t4 reuses with rel_grow<0",
            "value": None, "verdict": "not-run"}
    if not treatment_5ds:
        gate["note"] = "treatment(attn_pool) 5-Datasets arm missing"
        return gate
    rows = []
    for seed in FIVEDS_SEEDS:
        if seed not in treatment_5ds:
            continue
        r = treatment_5ds[seed]
        d4 = dec_at(r, T4)
        counts_ok = (r.get("n_grow") == 4 and r.get("n_search") == 0 and r.get("n_reuse") == 1)
        rel_grow_4 = d4.get("rel_grow") if d4 else None
        t4_ok = bool(d4 and d4.get("decision") == "reuse" and rel_grow_4 is not None
                     and rel_grow_4 < 0)
        rows.append({"seed": seed, "n_grow": r.get("n_grow"), "n_search": r.get("n_search"),
                     "n_reuse": r.get("n_reuse"), "counts_ok": counts_ok,
                     "t4_decision": d4.get("decision") if d4 else None,
                     "t4_rel_grow": rel_grow_4, "t4_ok": t4_ok,
                     "row_ok": bool(counts_ok and t4_ok)})
    if not rows:
        gate["note"] = "no seeds found"
        return gate
    n_ok = sum(1 for r in rows if r["row_ok"])
    gate["value"] = {"rows": rows}
    gate["verdict"] = "pass" if n_ok == len(rows) else "fail"
    gate["headline"] = f"{n_ok}/{len(rows)} seeds match 4/0/1 + t4 reuse<0"
    if len(rows) < len(FIVEDS_SEEDS):
        gate["note"] = f"partial: {len(rows)}/{len(FIVEDS_SEEDS)} seeds present so far"
    return gate


def gate_r1d(control_5ds: Dict[int, Dict], treatment_5ds: Dict[int, Dict]) -> Dict:
    gate = {"gate": "R1d",
            "observable": "L_null/L_reuse/L_search/L_grow, rel_grow, rel_search, gate_cache_n "
                          "at every gated position (tasks 1-4), both arms, 5-Datasets",
            "threshold": "descriptive — recorded, no threshold", "value": None,
            "verdict": "descriptive"}
    rows = []
    for label, data in ((CONTROL_FAMILY, control_5ds), (TREATMENT_FAMILY, treatment_5ds)):
        for seed in sorted(data):
            r = data[seed]
            for t in GATED_TASKS:
                d = dec_at(r, t)
                if d is None:
                    continue
                rows.append({"family": label, "seed": seed, "task": t,
                             "L_null_bits": d.get("L_null_bits"),
                             "L_reuse_bits": d.get("L_reuse_bits"),
                             "L_search_bits": d.get("L_search_bits"),
                             "L_grow_bits": d.get("L_grow_bits"),
                             "rel_grow": d.get("rel_grow"), "rel_search": d.get("rel_search"),
                             "gate_cache_n": d.get("gate_cache_n")})
    gate["value"] = {"rows": rows}
    gate["headline"] = f"{len(rows)} (family, seed, task) records"
    if not rows:
        gate["verdict"] = "not-run"
        gate["note"] = "no 5-Datasets data for either arm"
    return gate


# ---------------------------------------------------------------------------
# R2 — CTrL (4 streams x 5 seeds = 20 paired runs, paired by (seed, stream))
# ---------------------------------------------------------------------------

def paired_ctrl(control_ctrl: Dict[int, Dict[str, Dict]],
                treatment_ctrl: Dict[int, Dict[str, Dict]]
                ) -> List[Tuple[int, str, Dict, Dict]]:
    pairs = []
    for seed in CTRL_SEEDS:
        for stream in CTRL_STREAMS:
            c = control_ctrl.get(seed, {}).get(stream)
            t = treatment_ctrl.get(seed, {}).get(stream)
            if c is None or t is None:
                continue
            pairs.append((seed, stream, c, t))
    return pairs


def gate_r2a(control_ctrl: Dict[int, Dict[str, Dict]],
             treatment_ctrl: Dict[int, Dict[str, Dict]]) -> Dict:
    gate = {"gate": "R2a",
            "observable": "paired (T-C) mean of t3 chosen-rung test accuracy, over "
                          "(seed, stream) pairs",
            "threshold": f">= {R2A_MIN_DELTA} and >= {R2A_MIN_SE_MULT} paired SE",
            "value": None, "verdict": "not-run"}
    pairs = paired_ctrl(control_ctrl, treatment_ctrl)
    if not pairs:
        gate["note"] = "no paired (seed, stream) CTrL runs for both arms"
        return gate
    rows, diffs = [], []
    for seed, stream, c, t in pairs:
        c3, t3 = c.get("test_accs", [None] * 5)[T3], t.get("test_accs", [None] * 5)[T3]
        if c3 is None or t3 is None:
            continue
        diff = t3 - c3
        diffs.append(diff)
        rows.append({"seed": seed, "stream": stream, "control_t3_acc": c3, "treat_t3_acc": t3,
                     "diff": diff})
    if not diffs:
        gate["note"] = "no pairs with t3 test_accs on both arms"
        return gate
    mean_diff = statistics.mean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) >= 2 else None
    pass_ = (se is not None) and (mean_diff >= R2A_MIN_DELTA) and (mean_diff >= R2A_MIN_SE_MULT * se)
    gate["value"] = {"n_pairs": len(diffs), "mean_diff": mean_diff, "paired_se": se, "rows": rows}
    gate["verdict"] = "pass" if pass_ else "fail"
    se_str = f"{se:.4f}" if se is not None else "n/a"
    gate["headline"] = f"mean diff={mean_diff:.4f}, paired SE={se_str}, n={len(diffs)}"
    if len(pairs) < 20:
        gate["note"] = f"partial: {len(pairs)}/20 paired (seed, stream) runs present so far"
    return gate


def gate_r2b(control_ctrl: Dict[int, Dict[str, Dict]],
             treatment_ctrl: Dict[int, Dict[str, Dict]]) -> Dict:
    gate = {"gate": "R2b",
            "observable": "paired (T-C) mean of t3 regret = max(oracle_accs) - chosen accuracy",
            "threshold": f"<= +{R2B_MAX_DELTA} (regret must not worsen)",
            "value": None, "verdict": "not-run"}
    pairs = paired_ctrl(control_ctrl, treatment_ctrl)
    if not pairs:
        gate["note"] = "no paired (seed, stream) CTrL runs for both arms"
        return gate
    rows, diffs, skipped = [], [], []
    for seed, stream, c, t in pairs:
        rc, rt = t3_regret(c), t3_regret(t)
        if rc is None or rt is None:
            skipped.append({"seed": seed, "stream": stream})
            continue
        diff = rt - rc
        diffs.append(diff)
        rows.append({"seed": seed, "stream": stream, "control_regret": rc, "treat_regret": rt,
                     "diff": diff})
    if not diffs:
        gate["note"] = "no pairs with oracle_accs at t3 on both arms"
        return gate
    mean_diff = statistics.mean(diffs)
    pass_ = mean_diff <= R2B_MAX_DELTA
    gate["value"] = {"n_pairs": len(diffs), "mean_diff": mean_diff, "rows": rows,
                     "skipped_no_oracle": skipped}
    gate["verdict"] = "pass" if pass_ else "fail"
    gate["headline"] = f"mean regret diff={mean_diff:.4f}, n={len(diffs)}"
    if len(pairs) < 20:
        gate["note"] = f"partial: {len(pairs)}/20 paired (seed, stream) runs present so far"
    return gate


def gate_r2c(treatment_ctrl: Dict[int, Dict[str, Dict]]) -> Dict:
    gate = {"gate": "R2c",
            "observable": "arm T t4 decision across every CTrL (seed, stream)",
            "threshold": "reuse in 20/20 (any grow = REUSE-BROKEN)",
            "value": None, "verdict": "not-run"}
    rows = []
    for seed in CTRL_SEEDS:
        for stream in CTRL_STREAMS:
            r = treatment_ctrl.get(seed, {}).get(stream)
            if r is None:
                continue
            d4 = dec_at(r, T4)
            rows.append({"seed": seed, "stream": stream,
                        "decision": d4.get("decision") if d4 else None})
    if not rows:
        gate["note"] = "no treatment(attn_pool) CTrL runs found"
        return gate
    n = len(rows)
    n_reuse = sum(1 for r in rows if r["decision"] == "reuse")
    reuse_broken = any(r["decision"] == "grow" for r in rows)
    pass_ = n_reuse == n
    gate["value"] = {"n_runs": n, "n_reuse": n_reuse, "reuse_broken": reuse_broken, "rows": rows}
    gate["verdict"] = "pass" if pass_ else "fail"
    gate["headline"] = f"{n_reuse}/{n} t4 reuse" + (" — REUSE-BROKEN" if reuse_broken else "")
    if n < 20:
        gate["note"] = f"partial: {n}/20 CTrL runs present so far"
    return gate


def gate_r2d(control_ctrl: Dict[int, Dict[str, Dict]],
             treatment_ctrl: Dict[int, Dict[str, Dict]]) -> Dict:
    gate = {"gate": "R2d",
            "observable": "arm T s_plus t4 decision, and t4 accuracy vs arm C at the same seed",
            # AMBIGUITY: "within 0.02 of arm C's s_plus t4" could mean the pooled/mean accuracy
            # across the 5 seeds, or per-seed. We take the conservative (stricter) reading:
            # EVERY seed's |T-C| must clear 0.02, not just the mean of the differences (which
            # is also reported below for reference).
            "threshold": f"reuse 5/5; every seed's |T-C| t4 accuracy within {R2D_ACC_TOL} "
                         "(conservative reading, see comment)",
            "value": None, "verdict": "not-run"}
    rows = []
    for seed in CTRL_SEEDS:
        c = control_ctrl.get(seed, {}).get("s_plus")
        t = treatment_ctrl.get(seed, {}).get("s_plus")
        if c is None or t is None:
            continue
        d4t = dec_at(t, T4)
        acc_c = c.get("test_accs", [None] * 5)[T4]
        acc_t = t.get("test_accs", [None] * 5)[T4]
        if acc_c is None or acc_t is None:
            continue
        diff = abs(acc_t - acc_c)
        rows.append({"seed": seed, "treat_decision": d4t.get("decision") if d4t else None,
                    "control_acc": acc_c, "treat_acc": acc_t, "abs_diff": diff,
                    "within": diff <= R2D_ACC_TOL})
    if not rows:
        gate["note"] = "no paired s_plus CTrL runs"
        return gate
    n = len(rows)
    n_reuse = sum(1 for r in rows if r["treat_decision"] == "reuse")
    all_within = all(r["within"] for r in rows)
    mean_abs_diff = statistics.mean(r["abs_diff"] for r in rows)
    pass_ = (n_reuse == n) and all_within
    gate["value"] = {"n_seeds": n, "n_reuse": n_reuse, "mean_abs_diff": mean_abs_diff,
                     "all_within": all_within, "rows": rows}
    gate["verdict"] = "pass" if pass_ else "fail"
    gate["headline"] = f"{n_reuse}/{n} t4 reuse, all within {R2D_ACC_TOL}={all_within}"
    if n < 5:
        gate["note"] = f"partial: {n}/5 seeds present so far"
    return gate


# ---------------------------------------------------------------------------
# R3 — gate noise (H7d)
# ---------------------------------------------------------------------------

def gate_r3(control_ctrl: Dict[int, Dict[str, Dict]],
            treatment_ctrl: Dict[int, Dict[str, Dict]]) -> Dict:
    gate = {"gate": "R3",
            "observable": "across-run sample SD (ddof=1) of rel_grow and rel_search at t3, "
                          "per arm, over up to 20 CTrL runs; ratio T/C",
            "threshold": f"ratio >= {R3_NOISIER_RATIO} on either margin -> "
                         "GATE-NOISIER-CONFIRMED; ratio <= 1.0 on both -> GATE-NOISE-UNCHANGED; "
                         "else INDETERMINATE",
            "value": None, "verdict": "descriptive"}

    def _collect(data: Dict[int, Dict[str, Dict]]):
        rel_grow, rel_search, hist = [], [], {}
        for seed in CTRL_SEEDS:
            for stream in CTRL_STREAMS:
                r = data.get(seed, {}).get(stream)
                if r is None:
                    continue
                d = dec_at(r, T3)
                if d is None:
                    continue
                if d.get("rel_grow") is not None:
                    rel_grow.append(d["rel_grow"])
                if d.get("rel_search") is not None:
                    rel_search.append(d["rel_search"])
                dec = d.get("decision")
                hist[dec] = hist.get(dec, 0) + 1
        return rel_grow, rel_search, hist

    c_rg, c_rs, c_hist = _collect(control_ctrl)
    t_rg, t_rs, t_hist = _collect(treatment_ctrl)
    sd_c_rg, sd_t_rg = _sample_stdev(c_rg), _sample_stdev(t_rg)
    sd_c_rs, sd_t_rs = _sample_stdev(c_rs), _sample_stdev(t_rs)
    if sd_c_rg is None or sd_t_rg is None or sd_c_rs is None or sd_t_rs is None:
        gate["value"] = {"n_control": len(c_rg), "n_treatment": len(t_rg),
                         "t3_decision_histogram_control": c_hist,
                         "t3_decision_histogram_treatment": t_hist}
        gate["verdict"] = "not-run"
        gate["note"] = "fewer than 2 CTrL runs with rel_grow/rel_search at t3 for one or both arms"
        return gate
    ratio_rg = (sd_t_rg / sd_c_rg) if sd_c_rg else None
    ratio_rs = (sd_t_rs / sd_c_rs) if sd_c_rs else None
    noisier = (ratio_rg is not None and ratio_rg >= R3_NOISIER_RATIO) or \
              (ratio_rs is not None and ratio_rs >= R3_NOISIER_RATIO)
    unchanged = (ratio_rg is not None and ratio_rg <= R3_UNCHANGED_RATIO) and \
                (ratio_rs is not None and ratio_rs <= R3_UNCHANGED_RATIO)
    r3_branch = "GATE-NOISIER-CONFIRMED" if noisier else (
        "GATE-NOISE-UNCHANGED" if unchanged else "INDETERMINATE")
    gate["value"] = {"sd_rel_grow_control": sd_c_rg, "sd_rel_grow_treatment": sd_t_rg,
                     "ratio_rel_grow": ratio_rg, "sd_rel_search_control": sd_c_rs,
                     "sd_rel_search_treatment": sd_t_rs, "ratio_rel_search": ratio_rs,
                     "t3_decision_histogram_control": c_hist,
                     "t3_decision_histogram_treatment": t_hist,
                     "n_control": len(c_rg), "n_treatment": len(t_rg), "r3_branch": r3_branch}
    gate["r3_branch"] = r3_branch
    fmt = lambda x: f"{x:.3f}" if x is not None else "n/a"
    gate["headline"] = f"ratio(grow)={fmt(ratio_rg)}, ratio(search)={fmt(ratio_rs)} -> {r3_branch}"
    if len(c_rg) < 20 or len(t_rg) < 20:
        gate["note"] = f"partial: control n={len(c_rg)}, treatment n={len(t_rg)} (target 20 each)"
    return gate


# ---------------------------------------------------------------------------
# R4 — cost (descriptive, all recorded)
# ---------------------------------------------------------------------------

_PROGRESS_RE = re.compile(
    r"^===\s+(?P<exp>ctrl|5ds)\s+(?P<fam>\S+)\s+seed\s+(?P<seed>\d+)\s+"
    r"(?:(?P<stream>s_minus|s_out|s_plus|s_in)\s+)?"
    r"(?P<phase>start|exit)\s+(?:(?P<code>-?\d+)\s+)?(?P<epoch>\d+(?:\.\d+)?)\s+"
    r"(?P<date>.*?)\s*===\s*$")


def parse_progress(paths: List[str]) -> Optional[Dict]:
    """Parses ``progress_*.log`` lines of the form
    ``=== ctrl <fam> seed <S> <STREAM> start <epoch> <date> ===`` /
    ``=== ctrl <fam> seed <S> <STREAM> exit <code> <epoch> <date> ===``, the only form the
    pre-registration gives verbatim. AMBIGUITY: no 5-Datasets analogue is specified (5-Datasets
    runs have no STREAM); the regex accepts the STREAM group as optional so a ``=== 5ds <fam>
    seed <S> start ...`` line (no stream) also matches, as the conservative generalisation that
    degrades to "not present" rather than crashing if the real format differs. Lines starting
    with ``===`` that still don't match are counted in ``unmatched_lines`` and otherwise ignored.
    """
    events: Dict[Tuple[str, str, int, Optional[str]], Dict[str, float]] = {}
    unmatched = 0
    for path in paths or []:
        if not path or not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("==="):
                    continue
                m = _PROGRESS_RE.match(line)
                if not m:
                    unmatched += 1
                    continue
                key = (m.group("exp"), m.group("fam"), int(m.group("seed")), m.group("stream"))
                events.setdefault(key, {})[m.group("phase")] = float(m.group("epoch"))
    rows = []
    for (exp, fam, seed, stream), ev in sorted(events.items(), key=lambda kv: kv[0]):
        if "start" in ev and "exit" in ev:
            rows.append({"exp": exp, "family": fam, "seed": seed, "stream": stream,
                        "seconds": ev["exit"] - ev["start"]})
    if not rows and unmatched == 0:
        return None
    per_family: Dict[str, List[float]] = {}
    for r in rows:
        per_family.setdefault(r["family"], []).append(r["seconds"])
    return {"rows": rows,
            "mean_seconds_per_family": {f: statistics.mean(v) for f, v in per_family.items()},
            "unmatched_lines": unmatched}


def gate_r4(control_5ds: Dict[int, Dict], treatment_5ds: Dict[int, Dict],
            control_ctrl: Dict[int, Dict[str, Dict]], treatment_ctrl: Dict[int, Dict[str, Dict]],
            progress_stats: Optional[Dict]) -> Dict:
    gate = {"gate": "R4",
            "observable": "params_per_root, mean gate_seconds per gated task, feature-cache "
                          "size on disk, wall time per run — per arm",
            "threshold": "descriptive — all recorded", "value": None, "verdict": "descriptive"}

    def _arm_stats(fivedata: Dict[int, Dict], ctrldata: Dict[int, Dict[str, Dict]]) -> Dict:
        pprs, gsecs = [], []
        for r in fivedata.values():
            p = params_per_root(r)
            if p is not None:
                pprs.append(p)
            g = mean_gate_seconds(r)
            if g is not None:
                gsecs.append(g)
        for streams in ctrldata.values():
            for r in streams.values():
                p = params_per_root(r)
                if p is not None:
                    pprs.append(p)
                g = mean_gate_seconds(r)
                if g is not None:
                    gsecs.append(g)
        return {"mean_params_per_root": statistics.mean(pprs) if pprs else None,
                "params_per_root_values": sorted(set(pprs)),
                "mean_gate_seconds": statistics.mean(gsecs) if gsecs else None,
                "n_runs_with_gate_seconds": len(gsecs)}

    control = _arm_stats(control_5ds, control_ctrl)
    treatment = _arm_stats(treatment_5ds, treatment_ctrl)
    # AMBIGUITY: "feature-cache size on disk per arm" needs the actual cache directories, which
    # are not among this script's inputs (only result-JSON roots are) — omitted rather than
    # guessed, per the CLI contract ("no plotting, no re-running", and no path to guess from).
    value = {"control": control, "treatment": treatment,
             "feature_cache_size_on_disk":
                 "not available: no feature-cache directory is passed to this script "
                 "(only result-JSON roots) — see comment in gate_r4()",
             "wall_time": progress_stats if progress_stats is not None else
                 "omitted (no --progress file given)"}
    gate["value"] = value
    cp, tp = control["mean_params_per_root"], treatment["mean_params_per_root"]
    if cp is None and tp is None:
        gate["verdict"] = "not-run"
        gate["note"] = "no data for either arm"
        gate["headline"] = "no data"
    else:
        fmt = lambda x: f"{x:.0f}" if x is not None else "n/a"
        gate["headline"] = f"params/root C={fmt(cp)}, T={fmt(tp)}"
    return gate


# ---------------------------------------------------------------------------
# per_run tables (raw rows so every number above can be regenerated)
# ---------------------------------------------------------------------------

def build_5ds_rows(family_data: Dict[int, Dict], family_label: str) -> List[Dict]:
    rows = []
    for seed in sorted(family_data):
        r = family_data[seed]
        d3 = dec_at(r, T3)
        d4 = dec_at(r, T4)
        rows.append({
            "family": family_label, "seed": seed,
            "average_accuracy": r.get("average_accuracy"), "test_accs": r.get("test_accs"),
            "n_grow": r.get("n_grow"), "n_search": r.get("n_search"), "n_reuse": r.get("n_reuse"),
            "decisions": [(dec_at(r, t) or {}).get("decision") for t in range(5)],
            "t3_test_acc": (r.get("test_accs") or [None] * 5)[T3],
            "t3_did_not_grow": bool(d3 and d3.get("decision") != "grow"),
            "t4_decision": d4.get("decision") if d4 else None,
            "t4_rel_grow": d4.get("rel_grow") if d4 else None,
            "params_per_root": params_per_root(r),
            "mean_gate_seconds": mean_gate_seconds(r),
        })
    return rows


def build_ctrl_rows(family_data: Dict[int, Dict[str, Dict]], family_label: str) -> List[Dict]:
    rows = []
    for seed in sorted(family_data):
        for stream in CTRL_STREAMS:
            if stream not in family_data[seed]:
                continue
            r = family_data[seed][stream]
            d3 = dec_at(r, T3)
            d4 = dec_at(r, T4)
            rows.append({
                "family": family_label, "seed": seed, "stream": stream,
                "average_accuracy": r.get("average_accuracy"), "test_accs": r.get("test_accs"),
                "decisions": [(dec_at(r, t) or {}).get("decision") for t in range(5)],
                "t3_test_acc": (r.get("test_accs") or [None] * 5)[T3],
                "t3_regret": t3_regret(r),
                "t3_rel_grow": d3.get("rel_grow") if d3 else None,
                "t3_rel_search": d3.get("rel_search") if d3 else None,
                "t4_decision": d4.get("decision") if d4 else None,
                "params_per_root": params_per_root(r),
                "mean_gate_seconds": mean_gate_seconds(r),
            })
    return rows


# ---------------------------------------------------------------------------
# Branch
# ---------------------------------------------------------------------------

def determine_branch(n1: Dict, n2: Dict, r1a: Dict, r1b: Dict, r1c: Dict, r1d: Dict,
                     r2a: Dict, r2b: Dict, r2c: Dict, r2d: Dict, r3: Dict) -> Dict:
    def v(g):
        return g.get("verdict")

    flags = []
    if v(r1b) == "fail":
        flags.append("COLLATERAL")
    if v(r2a) == "fail" and v(r1a) == "pass":
        flags.append("SMALL-N-NULL")

    if v(n1) == "fail" or v(n2) == "fail":
        return {"branch": "INVALID-NULL-CHECK-FAILED",
                "reason": "N1 and/or N2 failed. Per the pre-registration this is a code "
                          "regression, not a hypothesis result: stop, do not trust the R "
                          "gates below (they are still computed for debugging, but none of "
                          "them names a branch until the null check passes).",
                "flags": flags, "null_status": "INVALID"}
    null_status = "OK" if (v(n1) == "pass" and v(n2) == "pass") else "PENDING"

    if v(r1a) == "not-run":
        return {"branch": "NOT-RUN", "reason": "R1a (5-Datasets transfer) has no data yet",
                "flags": flags, "null_status": null_status}
    if v(r1a) == "fail":
        return {"branch": "NO-TRANSFER",
                "reason": "R1a fails — the single-task gain does not survive the DAG's recipe",
                "flags": flags, "null_status": null_status}

    # From here R1a passes.
    needed = {"R1b": r1b, "R1c": r1c, "R2b": r2b, "R2c": r2c, "R2d": r2d}
    missing = [gid for gid, g in needed.items() if v(g) == "not-run"]
    if missing:
        return {"branch": "NOT-RUN",
                "reason": f"R1a passes; awaiting {', '.join(missing)}",
                "flags": flags, "null_status": null_status}

    if v(r1c) == "fail" or v(r2c) == "fail":
        return {"branch": "CAPABLE-BUT-DESTABILISING",
                "reason": "R1a passes but R1c and/or R2c fail — the root works, the gate does "
                          "not survive it",
                "flags": flags, "null_status": null_status}

    core_pass = all(v(g) == "pass" for g in (r1b, r1c, r2b, r2c, r2d))
    if core_pass:
        r3_branch = r3.get("r3_branch")
        branch = "ADOPT-WITH-NOISE" if r3_branch == "GATE-NOISIER-CONFIRMED" else "ADOPT"
        return {"branch": branch,
                "reason": "N passes; R1a, R1b, R1c, R2b, R2c, R2d all pass"
                          + (f"; R3={r3_branch}" if r3_branch else ""),
                "flags": flags, "null_status": null_status}

    return {"branch": "INDETERMINATE",
            "reason": "R1a passes, and R1c/R2c pass (not destabilising), but R1b and/or R2b "
                      "and/or R2d fail without matching a named branch in the pre-registration "
                      "— see per-gate verdicts and the COLLATERAL/SMALL-N-NULL flags",
            "flags": flags, "null_status": null_status}


# ---------------------------------------------------------------------------
# Stdout table
# ---------------------------------------------------------------------------

_GATE_ORDER = ["N1", "N2", "R1a", "R1b", "R1c", "R1d", "R2a", "R2b", "R2c", "R2d", "R3", "R4"]


def print_table(gates: Dict[str, Dict]) -> None:
    print(f"\n{'gate':<5} {'verdict':<11} {'threshold':<58} headline")
    print("-" * 130)
    for gid in _GATE_ORDER:
        g = gates[gid]
        thr = str(g.get("threshold", ""))
        if len(thr) > 56:
            thr = thr[:53] + "..."
        headline = g.get("headline") or g.get("note") or "-"
        print(f"{gid:<5} {str(g.get('verdict', '-')):<11} {thr:<58} {headline}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctrl", default=None,
                    help="dir holding ctrl_mlp_cls/ and ctrl_attn_pool/ (CTrL results)")
    ap.add_argument("--fivedatasets", default=None,
                    help="dir holding 5ds_mlp_cls/ and 5ds_attn_pool/ (5-Datasets results)")
    ap.add_argument("--baseline", default=None,
                    help="archived results_rebaseline_2026-09-08 dir (null-check reference)")
    ap.add_argument("--progress", nargs="+", default=[],
                    help="progress_*.log file(s) for R4 wall time (optional)")
    ap.add_argument("--out", required=True, help="where to write gates.json")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.ctrl or args.fivedatasets or args.baseline):
        raise SystemExit("at least one of --ctrl, --fivedatasets, --baseline is required")

    control_5ds = load_5ds_family(args.fivedatasets, CONTROL_FAMILY)
    treatment_5ds = load_5ds_family(args.fivedatasets, TREATMENT_FAMILY)
    control_ctrl = load_ctrl_family(args.ctrl, CONTROL_FAMILY)
    treatment_ctrl = load_ctrl_family(args.ctrl, TREATMENT_FAMILY)
    baseline_5ds = load_baseline_5ds(args.baseline)
    baseline_ctrl = load_baseline_ctrl(args.baseline)

    notes = []
    if args.fivedatasets and not (control_5ds or treatment_5ds):
        notes.append(f"--fivedatasets given ({args.fivedatasets}) but no exp5ds_kan results "
                     "found under 5ds_mlp_cls/ or 5ds_attn_pool/")
    if args.ctrl and not (control_ctrl or treatment_ctrl):
        notes.append(f"--ctrl given ({args.ctrl}) but no exp_ctrl_* results found under "
                     "ctrl_mlp_cls/ or ctrl_attn_pool/")
    if args.baseline and not (baseline_5ds or baseline_ctrl):
        notes.append(f"--baseline given ({args.baseline}) but no results found under it")

    progress_stats = parse_progress(args.progress) if args.progress else None

    n1 = gate_n1(baseline_5ds, control_5ds)
    n2 = gate_n2(baseline_ctrl, control_ctrl)
    r1a = gate_r1a(treatment_5ds)
    r1b = gate_r1b(control_5ds, treatment_5ds)
    r1c = gate_r1c(treatment_5ds)
    r1d = gate_r1d(control_5ds, treatment_5ds)
    r2a = gate_r2a(control_ctrl, treatment_ctrl)
    r2b = gate_r2b(control_ctrl, treatment_ctrl)
    r2c = gate_r2c(treatment_ctrl)
    r2d = gate_r2d(control_ctrl, treatment_ctrl)
    r3 = gate_r3(control_ctrl, treatment_ctrl)
    r4 = gate_r4(control_5ds, treatment_5ds, control_ctrl, treatment_ctrl, progress_stats)

    gates = {"N1": n1, "N2": n2, "R1a": r1a, "R1b": r1b, "R1c": r1c, "R1d": r1d,
             "R2a": r2a, "R2b": r2b, "R2c": r2c, "R2d": r2d, "R3": r3, "R4": r4}

    branch_info = determine_branch(n1, n2, r1a, r1b, r1c, r1d, r2a, r2b, r2c, r2d, r3)

    per_run = {
        "5ds": {CONTROL_FAMILY: build_5ds_rows(control_5ds, CONTROL_FAMILY),
                TREATMENT_FAMILY: build_5ds_rows(treatment_5ds, TREATMENT_FAMILY)},
        "ctrl": {CONTROL_FAMILY: build_ctrl_rows(control_ctrl, CONTROL_FAMILY),
                 TREATMENT_FAMILY: build_ctrl_rows(treatment_ctrl, TREATMENT_FAMILY)},
        "baseline_5ds": build_5ds_rows(baseline_5ds, "baseline"),
        "baseline_ctrl": build_ctrl_rows(baseline_ctrl, "baseline"),
    }

    print_table(gates)
    print(f"\n=== branch: {branch_info['branch']} ===")
    print(f"  reason: {branch_info['reason']}")
    print(f"  null_status: {branch_info.get('null_status', '-')}")
    if branch_info["flags"]:
        print(f"  flags: {', '.join(branch_info['flags'])}")
    if notes:
        print("\n=== notes ===")
        for n in notes:
            print(f"  - {n}")

    result: Dict = dict(gates)
    result["branch"] = branch_info["branch"]
    result["branch_detail"] = branch_info
    result["per_run"] = per_run
    result["progress"] = progress_stats
    result["notes"] = notes

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
