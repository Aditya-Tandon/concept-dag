#!/usr/bin/env python
"""
eval_preq_arms.py — the mechanical gate evaluation for H4, the prequential grow-probe hypothesis
(see the vault note it implements:
Central Library/Hypotheses/concept-dag/kan-gated-growth/prequential-grow-probe.md, G1-G5).

Reads ``exp3a_kan_results.json`` files for up to five arms:

    P0        control (published single-split estimator)                      --p0 DIR
    P1        H4 arm: prequential estimator, decides on the TAIL              --p1 DIR
    P2        H4-MDL companion arm: prequential estimator, decides on TOTAL   --p2 DIR
    P0_5ds    control on 5-Datasets (G2 collateral check)                     --p0_5ds DIR
    P1_5ds    P1 arm on 5-Datasets (G2 collateral check)                      --p1_5ds DIR

CTrL layout:   <DIR>/seed_<S>/exp_ctrl_<stream>/exp3a_kan_results.json  for stream in
               {s_minus, s_plus, s_out, s_in}
5ds layout:    <DIR>/seed_<S>/exp5ds_kan/exp3a_kan_results.json

Every arm must have been run with ``--oracle_rungs`` so each gated decision carries
``oracle_accs`` and ``oracle_val_bits``; P1/P2 must additionally have been run with
``--gate_estimator prequential`` so each gated decision carries
``estimator_meta.rungs.{null,reuse,search,grow}`` (the section-A prequential meta: curve/total/
tail/tail_by_exponent/last_block).

Gates (exact wording/thresholds from the hypothesis note's G1-G5):

  G1 (primary, P1 vs P0)   t3 SVHN@400 regret (oracle-best acc - chosen acc) over the 10
                           independent positions counted as s_minus + s_plus ONLY (5 seeds x 2
                           streams — s_out/s_in share t0-t3 with s_minus and are NOT double
                           counted, matching the result note's "t3 SVHN@400, s_minus family" /
                           "s_plus" split). Accept: mean drop (regret_P0 - regret_P1) >= 0.015,
                           across-seed SE < 0.01, grow count >= 6/10.
  G2 (collateral, P1)     t4 decisions P1 vs P0 identical in all 20 CTrL runs (5 seeds x 4
                           streams); 5ds decisions (the full per-task decision list) identical in
                           the seeds common to --p0_5ds/--p1_5ds; |delta AA| <= 0.005 per CTrL run.
  G3 (calibration, P1)    for every gated task in P1 with oracle_val_bits and a prequential
                           estimator_meta: err_tail = |rungs.grow.tail - oracle_val_bits.grow|;
                           for the SAME (seed, stream, task) position in P0:
                           err_single = |L_grow_bits - oracle_val_bits.grow|. Accept: median
                           err_tail <= 0.10 AND median err_tail <= median err_single.
  G4 (companion, P2)      descriptive only: P2's t3 decisions (search/grow/reuse counts) and
                           regret over the SAME 10 positions as G1, vs P0.
  G5 (sensitivity, P1)    from P1's estimator_meta.rungs.*.tail_by_exponent, recompute the
                           reuse/search/grow decision under each logged exponent (same eps 0.05,
                           grow-denominator rule: reducible = L_null_e - L_grow_e, grow if
                           rel_grow_e > eps else search if rel_search_e > eps else reuse) over the
                           SAME gated-task set as G3. Accept: exponent-0.5's decision agrees with
                           the per-position majority (mode over the logged exponents) on >= 90%
                           of positions.

Blind branch (the note's precedence — INCONCLUSIVE first, then COLLATERAL/MISCALIBRATED override
G1, matching eval_gate_arms.py's H1' precedence style):
  INCONCLUSIVE  if G1's across-seed SE >= 0.01
  COLLATERAL    elif G2 rejects (regardless of G1)
  MISCALIBRATED elif G3 rejects (regardless of G1)
  TAIL-FIXES    elif G1 accepts (and G2, G3 accept, since we got past both above)
  NO-EFFECT     else (G1 rejects, G2 and G3 accept: the tail moves but not enough)

A missing arm directory (or one with no seed_*/... results under it) is skipped with a note; the
script never crashes on a missing arm — gates that need it report status "SKIPPED". Every number
this script prints is also in the JSON it writes.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

STREAMS = ("s_minus", "s_plus", "s_out", "s_in")
G1_STREAMS = ("s_minus", "s_plus")            # the 10 independent t3 positions (G1/G4)
EPS_GROW = 0.05
EPS_SEARCH = 0.05
TAIL_EXPONENTS = ("0.25", "0.5", "1.0")        # str() keys of tail_exponents_logged (spec A1)
DECIDING_EXPONENT = "0.5"


# ---------------------------------------------------------------------------
# Loading (same layout/robustness as scripts/eval_gate_arms.py)
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


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    n = len(xs)
    if n == 0:
        return None
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


# ---------------------------------------------------------------------------
# G1 / G4 — t3 SVHN@400 regret over the 10 s_minus+s_plus positions
# ---------------------------------------------------------------------------


def _t3_positions(arm: dict, t3: int = 3):
    """Yield (seed, stream, decision_dict, regret, results) for every (seed, stream) in
    G1_STREAMS present in `arm` with an oracle-accs'd t3 decision."""
    for seed in sorted(arm):
        for stream in G1_STREAMS:
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


def g1_regret(p0: dict, p1: dict, t3: int = 3, regret_thresh: float = 0.015,
             se_thresh: float = 0.01, grow_needed: int = 6) -> dict:
    if not p0 or not p1:
        return {"status": "SKIPPED", "reason": "P0 or P1 missing/empty"}
    pos0 = {(s, st): (d, r) for s, st, d, r, _ in _t3_positions(p0, t3)}
    pos1 = {(s, st): (d, r) for s, st, d, r, _ in _t3_positions(p1, t3)}
    pairs = []
    for key in sorted(set(pos0) & set(pos1)):
        d0, r0 = pos0[key]
        d1, r1 = pos1[key]
        pairs.append({"seed": key[0], "stream": key[1], "regret_P0": r0, "regret_P1": r1,
                      "delta_regret": r0 - r1, "P1_decision": d1.get("decision")})
    if not pairs:
        return {"status": "SKIPPED", "reason": "no matched (seed, stream) t3 oracle_accs pairs "
                                                "in s_minus/s_plus"}
    deltas = [p["delta_regret"] for p in pairs]
    mean_delta = _mean(deltas)
    se = _sample_se(deltas)
    grow_count = sum(1 for p in pairs if p["P1_decision"] == "grow")
    accept = (mean_delta is not None and mean_delta >= regret_thresh and se < se_thresh
             and grow_count >= grow_needed)
    return {"status": "OK", "n_positions": len(pairs), "mean_delta_regret": mean_delta, "se": se,
            "grow_count": grow_count, "grow_needed": grow_needed, "accept": accept, "pairs": pairs}


def g4_companion(p0: dict, p2: dict, t3: int = 3) -> dict:
    if not p0 or not p2:
        return {"status": "SKIPPED", "reason": "P0 or P2 missing/empty"}
    pos0 = {(s, st): r for s, st, _, r, _ in _t3_positions(p0, t3)}
    pos2 = {(s, st): (d, r) for s, st, d, r, _ in _t3_positions(p2, t3)}
    pairs = []
    for key in sorted(set(pos0) & set(pos2)):
        r0 = pos0[key]
        d2, r2 = pos2[key]
        pairs.append({"seed": key[0], "stream": key[1], "regret_P0": r0, "regret_P2": r2,
                      "delta_regret": r0 - r2, "P2_decision": d2.get("decision")})
    if not pairs:
        return {"status": "SKIPPED", "reason": "no matched (seed, stream) t3 oracle_accs pairs "
                                                "in s_minus/s_plus"}
    counts = collections.Counter(p["P2_decision"] for p in pairs)
    mean_delta = _mean([p["delta_regret"] for p in pairs])
    return {"status": "OK", "n_positions": len(pairs),
            "decision_counts": {"search": counts.get("search", 0), "grow": counts.get("grow", 0),
                                "reuse": counts.get("reuse", 0)},
            "mean_delta_regret_P0_minus_P2": mean_delta, "pairs": pairs}


# ---------------------------------------------------------------------------
# G2 — t4 decisions identical, 5ds decisions identical, |dAA| <= 0.005
# ---------------------------------------------------------------------------


def g2_collateral(p0: dict, p1: dict, p0_5ds: dict, p1_5ds: dict, t4: int = 4,
                  aa_thresh: float = 0.005) -> dict:
    if not p0 or not p1:
        return {"status": "SKIPPED", "reason": "P0 or P1 missing/empty"}
    ctrl_rows = []
    for seed in sorted(set(p0) & set(p1)):
        for stream in STREAMS:
            if stream not in p0.get(seed, {}) or stream not in p1.get(seed, {}):
                continue
            r0, r1 = p0[seed][stream], p1[seed][stream]
            d0, d1 = dec_at(r0, t4), dec_at(r1, t4)
            if d0 is None or d1 is None:
                continue
            dAA = r1.get("average_accuracy", 0.0) - r0.get("average_accuracy", 0.0)
            ctrl_rows.append({"seed": seed, "stream": stream, "P0_decision": d0.get("decision"),
                              "P1_decision": d1.get("decision"),
                              "identical": d0.get("decision") == d1.get("decision"),
                              "delta_AA": dAA, "AA_ok": abs(dAA) <= aa_thresh})
    if not ctrl_rows:
        return {"status": "SKIPPED", "reason": "no matched (seed, stream) t4 decisions"}
    ctrl_identical = all(r["identical"] for r in ctrl_rows)
    ctrl_aa_ok = all(r["AA_ok"] for r in ctrl_rows)

    ds_rows = []
    if p0_5ds and p1_5ds:
        for seed in sorted(set(p0_5ds) & set(p1_5ds)):
            r0, r1 = p0_5ds[seed], p1_5ds[seed]
            dec0 = [d.get("decision") for d in r0.get("decisions", [])]
            dec1 = [d.get("decision") for d in r1.get("decisions", [])]
            dAA = r1.get("average_accuracy", 0.0) - r0.get("average_accuracy", 0.0)
            ds_rows.append({"seed": seed, "identical": dec0 == dec1, "delta_AA": dAA,
                            "AA_ok": abs(dAA) <= aa_thresh})
    ds_identical = all(r["identical"] for r in ds_rows) if ds_rows else None
    ds_aa_ok = all(r["AA_ok"] for r in ds_rows) if ds_rows else None

    accept = ctrl_identical and ctrl_aa_ok and (ds_identical is not False) and (ds_aa_ok is not False)
    return {"status": "OK", "n_ctrl_runs": len(ctrl_rows), "ctrl_identical": ctrl_identical,
            "ctrl_aa_ok": ctrl_aa_ok, "n_5ds_seeds": len(ds_rows), "ds_identical": ds_identical,
            "ds_aa_ok": ds_aa_ok, "accept": accept, "ctrl_rows": ctrl_rows, "ds_rows": ds_rows}


# ---------------------------------------------------------------------------
# G3 / G5 — the gated-task set with a prequential estimator_meta + oracle_val_bits
# ---------------------------------------------------------------------------


def _gated_preq_tasks(p1: dict):
    """Every (seed, stream, task, decision_dict) in P1 with BOTH oracle_val_bits and a
    prequential estimator_meta['rungs'] — the set G3 and G5 both iterate."""
    for seed in sorted(p1):
        for stream, results in p1[seed].items():
            for d in results.get("decisions", []):
                ov = d.get("oracle_val_bits")
                meta = d.get("estimator_meta") or {}
                if ov and isinstance(meta.get("rungs"), dict):
                    yield seed, stream, d.get("task"), d


def g3_calibration(p0: dict, p1: dict, tail_bits_thresh: float = 0.10) -> dict:
    if not p0 or not p1:
        return {"status": "SKIPPED", "reason": "P0 or P1 missing/empty"}
    per_task = []
    for seed, stream, task, d1 in _gated_preq_tasks(p1):
        ov1 = d1["oracle_val_bits"]
        rungs = d1["estimator_meta"]["rungs"]
        if "grow" not in rungs or "tail" not in rungs["grow"] or "grow" not in ov1:
            continue
        err_tail = abs(rungs["grow"]["tail"] - ov1["grow"])
        row = {"seed": seed, "stream": stream, "task": task, "err_tail": err_tail}
        r0 = p0.get(seed, {}).get(stream)
        if r0 is not None:
            d0 = dec_at(r0, task)
            if d0 is not None and d0.get("oracle_val_bits") and d0.get("L_grow_bits") is not None:
                row["err_single"] = abs(d0["L_grow_bits"] - d0["oracle_val_bits"]["grow"])
        per_task.append(row)
    if not per_task:
        return {"status": "SKIPPED", "reason": "no P1 gated tasks with oracle_val_bits + "
                                                "prequential estimator_meta"}
    median_err_tail = _median([r["err_tail"] for r in per_task])
    err_single_vals = [r["err_single"] for r in per_task if "err_single" in r]
    median_err_single = _median(err_single_vals) if err_single_vals else None
    accept = (median_err_tail is not None and median_err_tail <= tail_bits_thresh
             and (median_err_single is None or median_err_tail <= median_err_single))
    return {"status": "OK", "n_tasks": len(per_task), "median_err_tail": median_err_tail,
            "median_err_single": median_err_single, "n_matched_P0": len(err_single_vals),
            "accept": accept, "per_task": per_task}


def _rung_tail_e(rung_meta: dict, e: str):
    tbe = rung_meta.get("tail_by_exponent") or {}
    return tbe.get(e)


def _decision_under_exponent(rungs: dict, e: str):
    try:
        L_null = _rung_tail_e(rungs["null"], e)
        L_reuse = _rung_tail_e(rungs["reuse"], e)
        L_search = _rung_tail_e(rungs["search"], e)
        L_grow = _rung_tail_e(rungs["grow"], e)
    except KeyError:
        return None
    if None in (L_null, L_reuse, L_search, L_grow):
        return None
    reducible = max(L_null - L_grow, 1e-6)   # grow denominator (the recorded rule's normaliser)
    rel_search = (L_reuse - L_search) / reducible
    rel_grow = (L_search - L_grow) / reducible
    if rel_grow > EPS_GROW:
        return "grow"
    elif rel_search > EPS_SEARCH:
        return "search"
    else:
        return "reuse"


def g5_sensitivity(p1: dict, agree_thresh: float = 0.9) -> dict:
    if not p1:
        return {"status": "SKIPPED", "reason": "P1 missing/empty"}
    per_task = []
    for seed, stream, task, d1 in _gated_preq_tasks(p1):
        rungs = d1["estimator_meta"]["rungs"]
        decisions_by_e = {e: _decision_under_exponent(rungs, e) for e in TAIL_EXPONENTS}
        if any(v is None for v in decisions_by_e.values()):
            continue
        counts = collections.Counter(decisions_by_e.values())
        majority = counts.most_common(1)[0][0]
        agrees = decisions_by_e[DECIDING_EXPONENT] == majority
        per_task.append({"seed": seed, "stream": stream, "task": task,
                         "decisions_by_exponent": decisions_by_e, "majority": majority,
                         "agrees": agrees})
    if not per_task:
        return {"status": "SKIPPED", "reason": "no P1 gated tasks with tail_by_exponent for all "
                                                f"of {TAIL_EXPONENTS}"}
    agreement_rate = sum(1 for r in per_task if r["agrees"]) / len(per_task)
    accept = agreement_rate >= agree_thresh
    return {"status": "OK", "n_tasks": len(per_task), "agreement_rate": agreement_rate,
            "accept": accept, "per_task": per_task}


# ---------------------------------------------------------------------------
# Blind branch — the note's precedence
# ---------------------------------------------------------------------------


def blind_branch(g1: dict, g2: dict, g3: dict, se_thresh: float = 0.01) -> str:
    if g1.get("status") != "OK":
        return "INCONCLUSIVE"
    se = g1.get("se", float("inf"))
    if se is None or se >= se_thresh:
        return "INCONCLUSIVE"
    if g2.get("status") != "OK" or not g2.get("accept"):
        return "COLLATERAL"
    if g3.get("status") != "OK" or not g3.get("accept"):
        return "MISCALIBRATED"
    if g1.get("accept"):
        return "TAIL-FIXES"
    return "NO-EFFECT"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p0", default=None)
    ap.add_argument("--p1", default=None)
    ap.add_argument("--p2", default=None)
    ap.add_argument("--p0_5ds", default=None)
    ap.add_argument("--p1_5ds", default=None)
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

    p0 = load_arm(args.p0, "P0", load_ctrl_arm)
    p1 = load_arm(args.p1, "P1", load_ctrl_arm)
    p2 = load_arm(args.p2, "P2", load_ctrl_arm)
    p0_5ds = load_arm(args.p0_5ds, "P0_5ds", load_5ds_arm)
    p1_5ds = load_arm(args.p1_5ds, "P1_5ds", load_5ds_arm)

    print("=== G1 (primary, P1 vs P0): t3 SVHN@400 regret over s_minus+s_plus (10 positions) ===")
    g1 = g1_regret(p0, p1)
    print(json.dumps({k: v for k, v in g1.items() if k != "pairs"}, indent=2))

    print("\n=== G2 (collateral, P1 vs P0): t4 + 5ds decisions identical, |dAA| <= 0.005 ===")
    g2 = g2_collateral(p0, p1, p0_5ds, p1_5ds)
    print(json.dumps({k: v for k, v in g2.items() if k not in ("ctrl_rows", "ds_rows")}, indent=2))

    print("\n=== G3 (calibration, P1): tail-extrapolated L_grow vs oracle val bits ===")
    g3 = g3_calibration(p0, p1)
    print(json.dumps({k: v for k, v in g3.items() if k != "per_task"}, indent=2))

    print("\n=== G4 (companion, P2 vs P0): t3 decisions + regret under the prequential total (descriptive) ===")
    g4 = g4_companion(p0, p2)
    print(json.dumps({k: v for k, v in g4.items() if k != "pairs"}, indent=2))

    print("\n=== G5 (sensitivity, P1): exponent-0.5 agreement with the cross-exponent majority ===")
    g5 = g5_sensitivity(p1)
    print(json.dumps({k: v for k, v in g5.items() if k != "per_task"}, indent=2))

    branch = blind_branch(g1, g2, g3)
    print(f"\n=== blind branch: {branch} ===")

    if notes:
        print("\n=== notes ===")
        for n in notes:
            print(f"  - {n}")

    result = {"g1_regret": g1, "g2_collateral": g2, "g3_calibration": g3, "g4_companion": g4,
              "g5_sensitivity": g5, "blind_branch": branch, "notes": notes}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
