#!/usr/bin/env python
"""
eval_h5_arms.py — the mechanical gate evaluation for H5, the search-selection-optimism hypothesis
(see the vault note it implements:
Central Library/Hypotheses/concept-dag/kan-gated-growth/search-selection-optimism.md, S0-S5).

Reads ``exp3a_kan_results.json`` files for up to seven arms:

    A0        the historical control this pre-registration reruns against         --a0 DIR
    Q0        control re-run at this branch commit (published single-split)       --q0 DIR
    Q1        H5a arm: select-score estimator                                    --q1 DIR
    Q2        H5b arm: select-score + tie_rule=grow                              --q2 DIR
    Q0_5ds    control on 5-Datasets (S4 collateral check)                        --q0_5ds DIR
    Q1_5ds    Q1 arm on 5-Datasets (S4 collateral check)                         --q1_5ds DIR
    Q2_5ds    Q2 arm on 5-Datasets (S4 collateral check)                         --q2_5ds DIR

CTrL layout:   <DIR>/seed_<S>/exp_ctrl_<stream>/exp3a_kan_results.json  for stream in
               {s_minus, s_plus, s_out, s_in}
5ds layout:    <DIR>/seed_<S>/exp5ds_kan/exp3a_kan_results.json

Every arm must have been run with ``--oracle_rungs`` so each gated decision carries
``oracle_accs``; Q1/Q2 must additionally have been run with ``--gate_estimator select-score``
(Q2 additionally with ``--tie_rule grow``) so each gated decision carries the
``concept_dag.training.kan_gate.decide_reuse_search_grow`` select-score ``estimator_meta`` payload
(``select_bits``/``score_bits``/``search_select_bits``/``search_early_stop_optimism``/
``search_score_bits_selected_on_score``/``search_selection_optimism``/
``search_selection_optimism_abs``/the paired SEs/``se_split_proxy``/``novelty``/
``tie_counterfactual``, and — Q2 only — ``tie``).

Arms may carry DIFFERENT seed sets (e.g. Q1/Q2 seeds 42-51, A0 seeds 42-46, Q0 seed 42 only):
every gate below matches on the INTERSECTION of the seed sets it needs, never assumes they're
equal, and reports how many positions it actually matched.

Gates (exact wording/thresholds from the hypothesis note's S0-S5):

  S0 (null, determinism)   every Q0 (seed, stream) run present in BOTH Q0 and A0 must match A0
                           decision-for-decision (every task, not just t3/t4) and have
                           |delta AA| == 0. Not a gate on the blind branch — reported per run;
                           if ANY run mismatches, "ABORT-DETERMINISM" is printed (the rest of the
                           gates are still computed).
  S1 (H5a primary)         Q1's OWN mean t3 SVHN@400 regret over the 10 independent positions
                           (s_minus + s_plus, 5 seeds x 2 streams — s_out/s_in share t0-t3 with
                           s_minus and are NOT double counted). Accept: mean regret <= 0.015 (the
                           A0 reference regret is 0.030; the drop is reported). grow count is
                           descriptive only (no threshold on it). Status "INCONCLUSIVE" (not
                           OK/reject) if the across-position SE >= 0.010.
  S2 (H5b primary, Q2)     tie fired in >= 60% of the 10 t3 positions AND mean Q2 t3 regret
                           <= 0.010. Firing counts at z in {0.5, 1, 2} are read off each
                           position's ``tie_counterfactual`` and reported (descriptive).
  S3 (specificity)         tie fired at 0 positions outside t3, in Q2 CTrL (any stream, any task
                           != 3) and Q2 5-Datasets (any task). Every such position whose
                           ``novelty`` falls in [0.05, 0.20] (close to the tie_novelty=0.1 guard)
                           is ALSO listed; if any exist (and there were 0 actual firings), S3 is
                           "INCONCLUSIVE" rather than accept (a real firing outside t3 is still an
                           outright reject regardless of the novelty list).
  S4a (collateral)         t4 CTrL decisions and full 5ds decision lists, Q1 vs Q0 AND Q2 vs Q0:
                           identical.
  S4b (collateral)         5-Datasets average_accuracy within 0.005 of Q0, for Q1 and Q2.
  (CTrL |delta AA| at t4 is reported against a 0.015 reference but is NOT a gate.)
  S5 (asymmetry measured live, Q1)  mean ``search_selection_optimism_abs`` over Q1's t3
                           positions. Accept >= 0.030 bits; reject < 0.015 bits; the interval
                           [0.015, 0.030) is "marginal" (neither).
  downstream (descriptive) for every (seed, stream) whose t3 decision flips between Q1/Q2 and the
                           baseline (A0 if given, else Q0), the end-of-stream average_accuracy of
                           each arm — not a gate.

Blind branch precedence (the note's, NOT the same order as H4's eval_preq_arms.py):
  NO-ASYMMETRY  (S5 rejects)
  > OVER-FIRES   (S3 rejects — an actual firing outside t3, NOT the inconclusive novelty-list case)
  > ASYMMETRY-FIXES / TIE-FIXES / NO-EFFECT   (S1 accepts / else S2 accepts / else)
  > INCONCLUSIVE (S1, S2, S3 all unresolvable — e.g. every H5 arm missing)
COLLATERAL is reported as a separate top-level flag (``collateral_flag``), NOT folded into the
branch precedence above (S4a/S4b rejecting no longer overrides S5/S3/S1/S2).

A missing arm directory (or one with no seed_*/... results under it) is skipped with a note; the
script never crashes on a missing arm — gates that need it report status "SKIPPED". Every number
this script prints is also in the JSON it writes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

STREAMS = ("s_minus", "s_plus", "s_out", "s_in")
G_STREAMS = ("s_minus", "s_plus")             # the 10 independent t3 positions (S1/S2/S5)
T3 = 3
T4 = 4
TIE_Z_GRID = ("0.5", "1.0", "2.0")


# ---------------------------------------------------------------------------
# Loading (same layout/robustness as scripts/eval_preq_arms.py)
# ---------------------------------------------------------------------------


def discover_seeds(arm_dir: str) -> dict:
    if not arm_dir or not os.path.isdir(arm_dir):
        return {}
    seeds = {}
    for name in sorted(os.listdir(arm_dir)):
        p = os.path.join(arm_dir, name)
        if name.startswith("seed_") and os.path.isdir(p):
            try:
                seed = int(name[len("seed_"):])
            except ValueError:
                continue
            seeds[seed] = p
    return seeds


def load_ctrl_arm(arm_dir: str) -> dict:
    """{seed: {stream: results_dict}} for every ``exp_ctrl_<stream>/exp3a_kan_results.json`` found."""
    out = {}
    for seed, seed_dir in discover_seeds(arm_dir).items():
        for stream in STREAMS:
            path = os.path.join(seed_dir, f"exp_ctrl_{stream}", "exp3a_kan_results.json")
            if os.path.isfile(path):
                with open(path) as f:
                    out.setdefault(seed, {})[stream] = json.load(f)
    return out


def load_5ds_arm(arm_dir: str) -> dict:
    """{seed: results_dict} for every ``exp5ds_kan/exp3a_kan_results.json`` found."""
    out = {}
    for seed, seed_dir in discover_seeds(arm_dir).items():
        path = os.path.join(seed_dir, "exp5ds_kan", "exp3a_kan_results.json")
        if os.path.isfile(path):
            with open(path) as f:
                out[seed] = json.load(f)
    return out


def dec_at(results: dict, task: int):
    for d in results.get("decisions", []):
        if d.get("task") == task:
            return d
    return None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _sample_se(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 2:
        return float("inf")
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var / n)


# ---------------------------------------------------------------------------
# t3 SVHN@400 positions (S1/S2/S5)
# ---------------------------------------------------------------------------


def _t3_positions(arm: dict, t3: int = T3):
    """Yield (seed, stream, decision_dict, regret, results) for every (seed, stream) in
    G_STREAMS present in `arm` with an oracle-accs'd t3 decision."""
    for seed in sorted(arm):
        for stream in G_STREAMS:
            if stream not in arm[seed]:
                continue
            results = arm[seed][stream]
            d = dec_at(results, t3)
            if d is None:
                continue
            oa = d.get("oracle_accs")
            if not oa:
                continue
            regret = max(oa.values()) - results["test_accs"][t3]
            yield seed, stream, d, regret, results


# ---------------------------------------------------------------------------
# S0 (null, determinism) — Q0 must reproduce A0 decision-for-decision, |dAA| == 0
# ---------------------------------------------------------------------------


def s0_determinism(a0: dict, q0: dict) -> dict:
    if not a0 or not q0:
        return {"status": "SKIPPED", "reason": "A0 or Q0 missing/empty", "abort_determinism": False}
    rows = []
    for seed in sorted(set(a0) & set(q0)):
        for stream in STREAMS:
            if stream not in a0.get(seed, {}) or stream not in q0.get(seed, {}):
                continue
            ra, rq = a0[seed][stream], q0[seed][stream]
            deca = [d.get("decision") for d in ra.get("decisions", [])]
            decq = [d.get("decision") for d in rq.get("decisions", [])]
            dAA = rq.get("average_accuracy", 0.0) - ra.get("average_accuracy", 0.0)
            rows.append({"seed": seed, "stream": stream, "decisions_identical": deca == decq,
                        "delta_AA": dAA, "AA_identical": dAA == 0.0})
    if not rows:
        return {"status": "SKIPPED", "reason": "no matched (seed, stream) runs between A0 and Q0",
                "abort_determinism": False}
    abort = any((not r["decisions_identical"]) or (not r["AA_identical"]) for r in rows)
    return {"status": "OK", "n_runs": len(rows), "abort_determinism": abort, "rows": rows}


# ---------------------------------------------------------------------------
# S1 (H5a primary) — Q1's own mean t3 regret, vs an A0 reference (descriptive)
# ---------------------------------------------------------------------------


def s1_primary(q1: dict, t3: int = T3, regret_thresh: float = 0.015,
              a0_reference_regret: float = 0.030, se_inconclusive_thresh: float = 0.010) -> dict:
    if not q1:
        return {"status": "SKIPPED", "reason": "Q1 missing/empty"}
    positions = list(_t3_positions(q1, t3))
    if not positions:
        return {"status": "SKIPPED", "reason": "no Q1 t3 oracle_accs positions in s_minus/s_plus"}
    rows = [{"seed": s, "stream": st, "decision": d.get("decision"), "regret": r}
            for s, st, d, r, _ in positions]
    regrets = [r["regret"] for r in rows]
    grow_count = sum(1 for r in rows if r["decision"] == "grow")
    mean_regret = _mean(regrets)
    se = _sample_se(regrets)
    inconclusive = se is None or se >= se_inconclusive_thresh
    status = "INCONCLUSIVE" if inconclusive else "OK"
    accept = (not inconclusive) and mean_regret is not None and mean_regret <= regret_thresh
    drop = (a0_reference_regret - mean_regret) if mean_regret is not None else None
    return {"status": status, "n_positions": len(rows), "mean_regret_Q1": mean_regret,
            "se_regret_Q1": se, "se_inconclusive_thresh": se_inconclusive_thresh,
            "a0_reference_regret": a0_reference_regret, "regret_drop_vs_a0_reference": drop,
            "grow_count": grow_count, "regret_thresh": regret_thresh, "accept": accept, "rows": rows}


# ---------------------------------------------------------------------------
# S2 (H5b primary, Q2) — tie-fired fraction + mean regret
# ---------------------------------------------------------------------------


def s2_tie_fixes(q2: dict, t3: int = T3, tie_frac_needed: float = 0.60,
                 regret_ceiling: float = 0.010) -> dict:
    if not q2:
        return {"status": "SKIPPED", "reason": "Q2 missing/empty"}
    positions = list(_t3_positions(q2, t3))
    if not positions:
        return {"status": "SKIPPED", "reason": "no Q2 t3 oracle_accs positions in s_minus/s_plus"}
    rows = []
    counterfactual_counts = {z: 0 for z in TIE_Z_GRID}
    for seed, stream, d, regret, _results in positions:
        meta = d.get("estimator_meta") or {}
        tie = meta.get("tie") or {}
        tf = meta.get("tie_counterfactual") or {}
        for z in TIE_Z_GRID:
            if tf.get(z):
                counterfactual_counts[z] += 1
        rows.append({"seed": seed, "stream": stream, "decision": d.get("decision"),
                    "regret": regret, "tie_fired": bool(tie.get("fired", False))})
    n = len(rows)
    tie_count = sum(1 for r in rows if r["tie_fired"])
    tie_frac = tie_count / n
    mean_regret = _mean([r["regret"] for r in rows])
    accept = (tie_frac >= tie_frac_needed and mean_regret is not None
             and mean_regret <= regret_ceiling)
    return {"status": "OK", "n_positions": n, "tie_count": tie_count, "tie_frac": tie_frac,
            "tie_frac_needed": tie_frac_needed, "mean_regret": mean_regret,
            "regret_ceiling": regret_ceiling, "counterfactual_counts": counterfactual_counts,
            "accept": accept, "rows": rows}


# ---------------------------------------------------------------------------
# S3 (specificity) — tie firings outside t3 (Q2 CTrL + Q2 5ds); novelty gray zone
# ---------------------------------------------------------------------------


def s3_specificity(q2: dict, q2_5ds: dict, t3: int = T3, novelty_lo: float = 0.05,
                   novelty_hi: float = 0.20) -> dict:
    if not q2:
        return {"status": "SKIPPED", "reason": "Q2 missing/empty"}
    firings, novelty_gray = [], []

    def _scan(d, extra, source):
        meta = d.get("estimator_meta") or {}
        tie = meta.get("tie") or {}
        if tie.get("fired"):
            firings.append({**extra, "task": d.get("task"), "source": source})
        nov = meta.get("novelty")
        if nov is not None and novelty_lo <= nov <= novelty_hi:
            novelty_gray.append({**extra, "task": d.get("task"), "novelty": nov, "source": source})

    for seed in sorted(q2):
        for stream in STREAMS:
            if stream not in q2[seed]:
                continue
            for d in q2[seed][stream].get("decisions", []):
                if d.get("task") == t3:
                    continue
                _scan(d, {"seed": seed, "stream": stream}, "ctrl")
    if q2_5ds:
        for seed in sorted(q2_5ds):
            for d in q2_5ds[seed].get("decisions", []):
                _scan(d, {"seed": seed}, "5ds")

    n_firings = len(firings)
    if n_firings > 0:
        status, accept = "OK", False
    elif novelty_gray:
        status, accept = "INCONCLUSIVE", None
    else:
        status, accept = "OK", True
    return {"status": status, "n_firings": n_firings, "accept": accept,
            "n_novelty_gray": len(novelty_gray), "firings": firings, "novelty_gray": novelty_gray}


# ---------------------------------------------------------------------------
# S4a / S4b (collateral) — decisions identical; 5ds AA within 0.005; CTrL AA is descriptive
# ---------------------------------------------------------------------------


def _s4a_one(q0: dict, qX: dict, q0_5ds: dict, qX_5ds: dict, t4: int = T4) -> dict:
    if not q0 or not qX:
        return {"status": "SKIPPED", "reason": "Q0 or arm missing/empty"}
    ctrl_rows = []
    for seed in sorted(set(q0) & set(qX)):
        for stream in STREAMS:
            if stream not in q0.get(seed, {}) or stream not in qX.get(seed, {}):
                continue
            d0, dX = dec_at(q0[seed][stream], t4), dec_at(qX[seed][stream], t4)
            if d0 is None or dX is None:
                continue
            ctrl_rows.append({"seed": seed, "stream": stream, "Q0_decision": d0.get("decision"),
                              "QX_decision": dX.get("decision"),
                              "identical": d0.get("decision") == dX.get("decision")})
    ds_rows = []
    if q0_5ds and qX_5ds:
        for seed in sorted(set(q0_5ds) & set(qX_5ds)):
            dec0 = [d.get("decision") for d in q0_5ds[seed].get("decisions", [])]
            decX = [d.get("decision") for d in qX_5ds[seed].get("decisions", [])]
            ds_rows.append({"seed": seed, "identical": dec0 == decX})
    if not ctrl_rows and not ds_rows:
        return {"status": "SKIPPED", "reason": "no matched t4/5ds decisions"}
    ctrl_identical = all(r["identical"] for r in ctrl_rows) if ctrl_rows else True
    ds_identical = all(r["identical"] for r in ds_rows) if ds_rows else True
    accept = ctrl_identical and ds_identical
    return {"status": "OK", "n_ctrl_runs": len(ctrl_rows), "ctrl_identical": ctrl_identical,
            "n_5ds_seeds": len(ds_rows), "ds_identical": ds_identical, "accept": accept,
            "ctrl_rows": ctrl_rows, "ds_rows": ds_rows}


def s4a_decisions_identical(q0: dict, q1: dict, q2: dict, q0_5ds: dict, q1_5ds: dict,
                            q2_5ds: dict, t4: int = T4) -> dict:
    q1_result = _s4a_one(q0, q1, q0_5ds, q1_5ds, t4)
    q2_result = _s4a_one(q0, q2, q0_5ds, q2_5ds, t4)
    if q1_result.get("status") != "OK" and q2_result.get("status") != "OK":
        return {"status": "SKIPPED", "reason": "Q1 and Q2 both unavailable",
                "q1": q1_result, "q2": q2_result}

    def _blocks(r):
        return r.get("status") == "OK" and not r.get("accept")

    accept = not _blocks(q1_result) and not _blocks(q2_result)
    return {"status": "OK", "accept": accept, "q1": q1_result, "q2": q2_result}


def _s4b_one(q0_5ds: dict, qX_5ds: dict, aa_thresh: float = 0.005) -> dict:
    if not q0_5ds or not qX_5ds:
        return {"status": "SKIPPED", "reason": "Q0_5ds or arm_5ds missing/empty"}
    rows = []
    for seed in sorted(set(q0_5ds) & set(qX_5ds)):
        r0, rX = q0_5ds[seed], qX_5ds[seed]
        dAA = rX.get("average_accuracy", 0.0) - r0.get("average_accuracy", 0.0)
        rows.append({"seed": seed, "delta_AA": dAA, "AA_ok": abs(dAA) <= aa_thresh})
    if not rows:
        return {"status": "SKIPPED", "reason": "no matched 5ds seeds"}
    accept = all(r["AA_ok"] for r in rows)
    return {"status": "OK", "n_seeds": len(rows), "accept": accept, "rows": rows}


def s4b_5ds_aa(q0_5ds: dict, q1_5ds: dict, q2_5ds: dict, aa_thresh: float = 0.005) -> dict:
    q1_result = _s4b_one(q0_5ds, q1_5ds, aa_thresh)
    q2_result = _s4b_one(q0_5ds, q2_5ds, aa_thresh)
    if q1_result.get("status") != "OK" and q2_result.get("status") != "OK":
        return {"status": "SKIPPED", "reason": "Q1_5ds and Q2_5ds both unavailable",
                "q1": q1_result, "q2": q2_result}

    def _blocks(r):
        return r.get("status") == "OK" and not r.get("accept")

    accept = not _blocks(q1_result) and not _blocks(q2_result)
    return {"status": "OK", "accept": accept, "q1": q1_result, "q2": q2_result}


def ctrl_aa_descriptive(q0: dict, q1: dict, q2: dict, t4: int = T4,
                        aa_reference: float = 0.015) -> dict:
    """CTrL t4 |delta AA|, Q1/Q2 vs Q0 — descriptive only, not a gate."""
    rows = []
    for label, arm in (("Q1", q1), ("Q2", q2)):
        for seed in sorted(set(q0) & set(arm)):
            for stream in STREAMS:
                if stream not in q0.get(seed, {}) or stream not in arm.get(seed, {}):
                    continue
                r0, rX = q0[seed][stream], arm[seed][stream]
                if dec_at(r0, t4) is None or dec_at(rX, t4) is None:
                    continue
                dAA = rX.get("average_accuracy", 0.0) - r0.get("average_accuracy", 0.0)
                rows.append({"arm": label, "seed": seed, "stream": stream, "delta_AA": dAA,
                            "within_reference": abs(dAA) <= aa_reference})
    return {"status": "OK" if rows else "SKIPPED", "aa_reference": aa_reference, "rows": rows}


# ---------------------------------------------------------------------------
# S5 (asymmetry measured live, Q1)
# ---------------------------------------------------------------------------


def s5_asymmetry_live(q1: dict, t3: int = T3, accept_thresh: float = 0.030,
                      reject_thresh: float = 0.015) -> dict:
    if not q1:
        return {"status": "SKIPPED", "reason": "Q1 missing/empty"}
    vals = []
    for seed, stream, d, _regret, _results in _t3_positions(q1, t3):
        meta = d.get("estimator_meta") or {}
        v = meta.get("search_selection_optimism_abs")
        if v is not None:
            vals.append(v)
    if not vals:
        return {"status": "SKIPPED", "reason": "no Q1 t3 search_selection_optimism_abs values"}
    mean_v = _mean(vals)
    accept = mean_v is not None and mean_v >= accept_thresh
    reject = mean_v is not None and mean_v < reject_thresh
    marginal = mean_v is not None and not accept and not reject
    return {"status": "OK", "n": len(vals), "mean_search_selection_optimism_abs": mean_v,
            "accept_thresh": accept_thresh, "reject_thresh": reject_thresh,
            "accept": accept, "reject": reject, "marginal": marginal}


# ---------------------------------------------------------------------------
# downstream (descriptive) — end-of-stream AA where the t3 decision flipped
# ---------------------------------------------------------------------------


def downstream_effects(baseline: dict, q1: dict, q2: dict, t3: int = T3) -> dict:
    if not baseline:
        return {"status": "SKIPPED", "reason": "no baseline (A0/Q0) arm available"}
    rows = []
    for seed in sorted(baseline):
        for stream in STREAMS:
            if stream not in baseline.get(seed, {}):
                continue
            r0 = baseline[seed][stream]
            d0 = dec_at(r0, t3)
            if d0 is None:
                continue
            row = {"seed": seed, "stream": stream, "baseline_decision": d0.get("decision"),
                  "baseline_AA": r0.get("average_accuracy")}
            any_flip = False
            for label, arm in (("Q1", q1), ("Q2", q2)):
                if seed in arm and stream in arm.get(seed, {}):
                    rX = arm[seed][stream]
                    dX = dec_at(rX, t3)
                    flipped = dX is not None and dX.get("decision") != d0.get("decision")
                    any_flip = any_flip or flipped
                    row[f"{label}_decision"] = dX.get("decision") if dX else None
                    row[f"{label}_AA"] = rX.get("average_accuracy")
                    row[f"{label}_flipped"] = flipped
            if any_flip:
                rows.append(row)
    return {"status": "OK" if rows else "SKIPPED", "n_flipped": len(rows), "rows": rows}


# ---------------------------------------------------------------------------
# Blind branch — NO-ASYMMETRY > OVER-FIRES > ASYMMETRY-FIXES / TIE-FIXES / NO-EFFECT >
# INCONCLUSIVE. COLLATERAL is a separate flag, not folded into precedence.
# ---------------------------------------------------------------------------


def _rejects(gate: dict) -> bool:
    return gate.get("status") == "OK" and gate.get("accept") is False


def blind_branch(s1: dict, s2: dict, s3: dict, s5: dict) -> str:
    if s5.get("status") == "OK" and s5.get("reject"):
        return "NO-ASYMMETRY"
    if _rejects(s3):
        return "OVER-FIRES"
    if s1.get("status") == "OK" and s1.get("accept"):
        return "ASYMMETRY-FIXES"
    if s2.get("status") == "OK" and s2.get("accept"):
        return "TIE-FIXES"
    if s1.get("status") == "SKIPPED" and s2.get("status") == "SKIPPED" and s3.get("status") == "SKIPPED":
        return "INCONCLUSIVE"
    return "NO-EFFECT"


def collateral_flag(s4a: dict, s4b: dict) -> bool:
    return _rejects(s4a) or _rejects(s4b)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a0", default=None)
    ap.add_argument("--q0", default=None)
    ap.add_argument("--q1", default=None)
    ap.add_argument("--q2", default=None)
    ap.add_argument("--q0_5ds", default=None)
    ap.add_argument("--q1_5ds", default=None)
    ap.add_argument("--q2_5ds", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    notes = []

    def load_arm(path, label, loader):
        if not path:
            notes.append(f"{label}: not provided")
            return {}
        if not os.path.isdir(path):
            notes.append(f"{label}: MISSING directory {path}")
            return {}
        data = loader(path)
        if not data:
            notes.append(f"{label}: no results found under {path}")
        return data

    a0 = load_arm(args.a0, "A0", load_ctrl_arm)
    q0 = load_arm(args.q0, "Q0", load_ctrl_arm)
    q1 = load_arm(args.q1, "Q1", load_ctrl_arm)
    q2 = load_arm(args.q2, "Q2", load_ctrl_arm)
    q0_5ds = load_arm(args.q0_5ds, "Q0_5ds", load_5ds_arm)
    q1_5ds = load_arm(args.q1_5ds, "Q1_5ds", load_5ds_arm)
    q2_5ds = load_arm(args.q2_5ds, "Q2_5ds", load_5ds_arm)

    print("=== S0 (null, determinism): Q0 vs A0, decision-for-decision + |dAA| == 0 ===")
    s0 = s0_determinism(a0, q0)
    print(json.dumps({k: v for k, v in s0.items() if k != "rows"}, indent=2))
    if s0.get("abort_determinism"):
        print("ABORT-DETERMINISM")

    print("\n=== S1 (H5a primary, Q1): mean t3 SVHN@400 regret over 10 positions ===")
    s1 = s1_primary(q1)
    print(json.dumps({k: v for k, v in s1.items() if k != "rows"}, indent=2))

    print("\n=== S2 (H5b primary, Q2): tie-fired fraction + mean regret over 10 positions ===")
    s2 = s2_tie_fixes(q2)
    print(json.dumps({k: v for k, v in s2.items() if k != "rows"}, indent=2))

    print("\n=== S3 (specificity): tie firings outside t3 (Q2 CTrL + Q2 5ds); novelty gray zone ===")
    s3 = s3_specificity(q2, q2_5ds)
    print(json.dumps({k: v for k, v in s3.items() if k not in ("firings", "novelty_gray")}, indent=2))

    print("\n=== S4a (collateral): t4 + 5ds decisions identical, Q1/Q2 vs Q0 ===")
    s4a = s4a_decisions_identical(q0, q1, q2, q0_5ds, q1_5ds, q2_5ds)
    s4a_print = {k: (v if not isinstance(v, dict) else
                     {k2: v2 for k2, v2 in v.items() if k2 not in ("ctrl_rows", "ds_rows")})
                for k, v in s4a.items()}
    print(json.dumps(s4a_print, indent=2))

    print("\n=== S4b (collateral): 5ds AA within 0.005 of Q0, Q1/Q2 ===")
    s4b = s4b_5ds_aa(q0_5ds, q1_5ds, q2_5ds)
    s4b_print = {k: (v if not isinstance(v, dict) else
                     {k2: v2 for k2, v2 in v.items() if k2 != "rows"})
                for k, v in s4b.items()}
    print(json.dumps(s4b_print, indent=2))

    ctrl_aa = ctrl_aa_descriptive(q0, q1, q2)
    print(f"\n=== CTrL |dAA| at t4 (descriptive, 0.015 reference): {len(ctrl_aa.get('rows', []))} rows ===")

    print("\n=== S5 (asymmetry measured live, Q1): mean search_selection_optimism_abs at t3 ===")
    s5 = s5_asymmetry_live(q1)
    print(json.dumps(s5, indent=2))

    baseline = a0 if a0 else q0
    downstream = downstream_effects(baseline, q1, q2)
    print(f"\n=== downstream (descriptive): {downstream.get('n_flipped', 0)} t3-flipped positions ===")

    branch = blind_branch(s1, s2, s3, s5)
    flag = collateral_flag(s4a, s4b)
    print(f"\n=== blind branch: {branch}  (collateral_flag={flag}) ===")

    if notes:
        print("\n=== notes ===")
        for n in notes:
            print(f"  - {n}")

    result = {"s0_determinism": s0, "s1_primary": s1, "s2_tie_fixes": s2, "s3_specificity": s3,
              "s4a_decisions_identical": s4a, "s4b_5ds_aa": s4b, "ctrl_aa_descriptive": ctrl_aa,
              "s5_asymmetry_live": s5, "downstream_effects": downstream,
              "blind_branch": branch, "collateral_flag": flag, "notes": notes}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
