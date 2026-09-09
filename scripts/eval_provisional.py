#!/usr/bin/env python
"""eval_provisional.py — the mechanical gate evaluator for H8, the provisional-growth /
UNDETERMINED-gate-state experiment (Central Library/Hypotheses/concept-dag/kan-gated-growth/
provisional-growth-undetermined-gate.md).

Reads the per-run ``exp3a_kan_results.json`` files written by ``run_experiment.py`` for every arm
of the pre-registration's G4 design (``off``, ``shadow``, ``evalue``, ``se_proxy``, ``always`` on
CTrL; ``off``, ``evalue`` on ``s_interleave``; ``off``, ``shadow``, ``evalue`` on 5-Datasets) plus
the ``attn_pool`` adoption-run baseline (the P0a reference), and evaluates every gate the note
pre-registers: P0a, P0b (null validity, evaluated first), P1 (selectivity, measured on the
``shadow`` arm per the note's 2026-09-09 amendment), P2 (necessity/value), P3a/P3b (decided
positions, clean vs. downstream of a deferral), P4 (merge fires, ``s_interleave``), P5
(collateral), P6 (resolution mode), P7 (cost, descriptive), P8 (wall time, descriptive), P9 (every
provisional root resolved) — then names the branch the results select, per the note's "Branch
precedence" list.

Layout expected
---------------
    <ctrl>/ctrl_<arm>/seed_<S>/exp_ctrl_<stream>/exp3a_kan_results.json
    <interleave>/int_<arm>/seed_<S>/exp_ctrl_s_interleave/exp3a_kan_results.json
    <fivedatasets>/5ds_<arm>/seed_42/exp5ds_kan/exp3a_kan_results.json
    <baseline>/ctrl_attn_pool/seed_<S>/exp_ctrl_<stream>/exp3a_kan_results.json   (adoption run)
    <baseline>/5ds_attn_pool/seed_42/exp5ds_kan/exp3a_kan_results.json           (adoption run)

``arm`` in {off, shadow, evalue, se_proxy, always} on CTrL; {off, evalue} on ``s_interleave``;
{off, shadow, evalue} on 5-Datasets. CTrL seeds 42-46 x streams {s_minus, s_out, s_plus, s_in};
``s_interleave`` seeds 42-51; 5-Datasets seed 42. Any of ``--ctrl``/``--interleave``/
``--fivedatasets``/``--baseline`` may be omitted — missing sections report every gate that needs
them as ``"verdict": "not-run"`` rather than crashing, since this script is meant to be run
mid-sweep, before every arm/seed/stream has finished.

This script does the evaluation ONLY: no plotting, no re-running experiments. Stdlib only.

Usage
-----
    python scripts/eval_provisional.py \\
        --ctrl DIR --interleave DIR --fivedatasets DIR --baseline DIR \\
        [--progress LOG ...] --out gates.json

Ambiguities resolved while implementing this evaluator (conservative readings — see the inline
comments at each site for the reasoning):

  * D2 (branch precedence step 2, "CONSTRUCTION-INVALID") needs rung-swapped/label-permuted A0
    desk-stage units, which are not part of the ``exp3a_kan_results.json`` layout this script
    reads (they live in a separate desk-stage artifact). D2 is reported as a permanent "not-run"
    stub here so the branch precedence chain can still name it and never silently skip it; it can
    never fire CONSTRUCTION-INVALID from this evaluator's inputs alone.
  * P1's "position" granularity: the four CTrL streams share tasks t0-t3 within a seed (only t4
    is a genuine per-stream revisit), so the sign-stability label at t3 is computed on ONE
    position (5 seed-level ``mean_d`` values, collapsed across the streams present, which should
    already agree since the computation is byte-identical up to t3), while t4 gets FOUR positions
    (one per stream, each with up to 5 per-seed values) — this matches the note's own "stated per
    seed at t3 (n=5), per run at t4 (n=20)" phrasing. The 5-Datasets shadow arm is reported
    descriptively alongside (n=1 seed: no stability label is meaningful, so it is not pooled into
    the pass/fail fractions).
  * P2(a)'s "end-of-stream t3 regret": the JSON schema has no post-consolidation re-evaluation of
    task 3 (test_accs[3] is captured once, at the time task 3 is processed, before any later
    consolidation event that might resolve a root minted there) — the literal ``test_accs[T3]``
    value is used, as it is the only accuracy this schema records for that position.
  * P2(b)'s accuracy-per-parameter is computed by first averaging AA and
    params_total_pre_consolidation over the (seed, stream) pairs common to arm off/evalue/always,
    then taking one ratio per arm — not a per-run ratio then averaged — since the denominator can
    be ~0 for an individual run and an aggregate-then-ratio is the standard, more robust reading.
  * P3b: the note's own amendment says reader-level delta accuracy is not recorded per reader in
    Loop 1, so P3b is reported at the task-accuracy level (``test_accs[t]``) with a ``note`` saying
    so, exactly as instructed.
  * P4's "negative control" (no merge may involve the t2 root) is read directly off
    ``consolidation["ops"]`` entries with ``op == "merge"`` whose ``{keep, drop}`` task-id pair is
    ``{1, 2}`` or ``{2, 3}`` — ``merge_rejected`` ops do not count (nothing was merged).
  * Branch precedence step 9 ("MERGE-NEVER-FIRES ... while P6 shows e-value or timeout
    resolutions") references an "e-value crossing" resolution mode that G3's amendment explicitly
    removed from Loop 1 (only merge/timeout exist). Read conservatively as: P4 fails and P6 is
    NOT all-timeout (the all-timeout case is already claimed by step 8, which has higher
    precedence) — i.e. some non-timeout resolution activity exists but merges don't reach 7/10.
  * The precedence list's catch-all is honored literally: an unmatched combination is named
    ``UNMATCHED-COMBINATION`` (never ``INDETERMINATE``), with the exact gate verdicts that failed
    to match recorded in ``reason``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CTRL_STREAMS = ("s_minus", "s_out", "s_plus", "s_in")
CTRL_SEEDS = (42, 43, 44, 45, 46)
INT_SEEDS = tuple(range(42, 52))          # 42..51, 10 seeds
FDS_SEED = 42

CTRL_ARMS = ("off", "shadow", "evalue", "se_proxy", "always")
INT_ARMS = ("off", "evalue")
FDS_ARMS = ("off", "shadow", "evalue")

BASELINE_FAMILY = "attn_pool"             # baseline dir has no "arm" level, only this family

T3, T4 = 3, 4
GATED_TASKS = (1, 2, 3, 4)                # CTrL / 5-Datasets gated positions
INT_GATED_TASKS = (1, 2, 3, 4)            # s_interleave: t1 Fashion, t2 SVHN, t3 dup, t4 revisit

# --- thresholds, taken verbatim from the pre-registration's gate table -----------------------
P1_UNSTABLE_MIN_FRAC = 0.80
P1_STABLE_MAX_FRAC = 0.20
P2A_MIN_IMPROVE = 0.03
P2C_MIN_DIFF_FRAC = 0.20
P3_TAU = 0.01
P4_MIN_MERGE_FRAC = 0.7                   # >= 7/10
P5_AA_TOL = 0.005
P5_TASK_TOL = 0.01
P5_FDS_AA_TOL = 0.005
P6_INT_MIN_MERGE_FRAC = 0.5
NEVER_UNDETERMINED_TOL = 0.01
ALWAYS_UNDETERMINED_TOL = 0.99
NO_MERGE_OBJECT_MIN_FRAC = 0.5


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


def _load_ctrl_tree(root: Optional[str]) -> Dict[int, Dict[str, Dict]]:
    """{seed: {stream: results}} under <root>/seed_<S>/exp_ctrl_<stream>/exp3a_kan_results.json"""
    out: Dict[int, Dict[str, Dict]] = {}
    for seed, seed_dir in _discover_seed_dirs(root).items():
        for stream in CTRL_STREAMS:
            path = os.path.join(seed_dir, f"exp_ctrl_{stream}", "exp3a_kan_results.json")
            if os.path.isfile(path):
                out.setdefault(seed, {})[stream] = _read_json(path)
    return out


def _load_single_stream_tree(root: Optional[str], folder_name: str) -> Dict[int, Dict]:
    """{seed: results} under <root>/seed_<S>/<folder_name>/exp3a_kan_results.json"""
    out: Dict[int, Dict] = {}
    for seed, seed_dir in _discover_seed_dirs(root).items():
        path = os.path.join(seed_dir, folder_name, "exp3a_kan_results.json")
        if os.path.isfile(path):
            out[seed] = _read_json(path)
    return out


def load_ctrl_arm(base_dir: Optional[str], arm: str) -> Dict[int, Dict[str, Dict]]:
    if not base_dir:
        return {}
    return _load_ctrl_tree(os.path.join(base_dir, f"ctrl_{arm}"))


def load_int_arm(base_dir: Optional[str], arm: str) -> Dict[int, Dict]:
    if not base_dir:
        return {}
    return _load_single_stream_tree(os.path.join(base_dir, f"int_{arm}"), "exp_ctrl_s_interleave")


def load_5ds_arm(base_dir: Optional[str], arm: str) -> Dict[int, Dict]:
    if not base_dir:
        return {}
    return _load_single_stream_tree(os.path.join(base_dir, f"5ds_{arm}"), "exp5ds_kan")


def load_baseline_ctrl(base_dir: Optional[str]) -> Dict[int, Dict[str, Dict]]:
    if not base_dir:
        return {}
    return _load_ctrl_tree(os.path.join(base_dir, f"ctrl_{BASELINE_FAMILY}"))


def load_baseline_5ds(base_dir: Optional[str]) -> Dict[int, Dict]:
    if not base_dir:
        return {}
    return _load_single_stream_tree(os.path.join(base_dir, f"5ds_{BASELINE_FAMILY}"), "exp5ds_kan")


# ---------------------------------------------------------------------------
# Small helpers shared across gates
# ---------------------------------------------------------------------------

def dec_at(results: Optional[Dict], task: int) -> Optional[Dict]:
    if not results:
        return None
    for d in results.get("decisions", []):
        if d.get("task") == task:
            return d
    return None


def regret_at(results: Optional[Dict], task: int) -> Optional[float]:
    """max(oracle_accs) - chosen (test_accs[task]) accuracy; skipped (None) when the decision
    carries no oracle_accs, per the note's definition."""
    d = dec_at(results, task)
    oracle = d.get("oracle_accs") if d else None
    if not oracle:
        return None
    accs = (results or {}).get("test_accs")
    if not accs or len(accs) <= task:
        return None
    return max(oracle.values()) - accs[task]


def ev_at(results: Optional[Dict], task: int) -> Optional[Dict]:
    d = dec_at(results, task)
    return d.get("evalue") if d else None


def acc_at(results: Optional[Dict], task: int) -> Optional[float]:
    accs = (results or {}).get("test_accs")
    if not accs or len(accs) <= task:
        return None
    return accs[task]


def _mean(xs: List[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def _stdev(xs: List[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) >= 2 else None


def _seed_level(rows: List[Dict], key: str = "diff") -> Dict:
    """Mean and SE of a per-(seed, stream) difference collapsed to one value PER SEED.

    The four CTrL streams share tasks t0-t3 within a seed, so 20 runs carry only 5 independent
    draws at t3 and a stdev/sqrt(20) understates the uncertainty by ~2x. Every t3 number is
    therefore reported at both levels: the run level (optimistic) and the seed level (honest).
    Logic mirrors ``scripts/eval_attn_root.py``'s ``_seed_level`` verbatim.
    """
    by_seed: Dict[int, List[float]] = {}
    for r in rows:
        if r.get(key) is None:
            continue
        by_seed.setdefault(r["seed"], []).append(float(r[key]))
    per_seed = {s: statistics.mean(v) for s, v in sorted(by_seed.items())}
    vals = list(per_seed.values())
    if not vals:
        return {"n_seeds": 0, "mean": None, "se": None, "per_seed": {}}
    se = (statistics.stdev(vals) / math.sqrt(len(vals))) if len(vals) >= 2 else None
    mean = statistics.mean(vals)
    return {"n_seeds": len(vals), "mean": mean, "se": se,
            "n_se": (mean / se) if se else None,
            "per_seed": {str(k): v for k, v in per_seed.items()}}


def paired_ctrl(a: Dict[int, Dict[str, Dict]], b: Dict[int, Dict[str, Dict]]
                ) -> List[Tuple[int, str, Dict, Dict]]:
    pairs = []
    for seed in CTRL_SEEDS:
        for stream in CTRL_STREAMS:
            ra = a.get(seed, {}).get(stream)
            rb = b.get(seed, {}).get(stream)
            if ra is not None and rb is not None:
                pairs.append((seed, stream, ra, rb))
    return pairs


def _resolution_stats(roots: List[Dict]) -> Dict:
    total = len(roots)
    n_merge = sum(1 for r in roots if r.get("resolution") == "merge")
    n_timeout = sum(1 for r in roots if r.get("resolution") == "timeout")
    n_unresolved = sum(1 for r in roots if r.get("resolution") is None)
    return {"total": total, "n_merge": n_merge, "n_timeout": n_timeout,
            "n_unresolved": n_unresolved,
            "frac_merge": (n_merge / total) if total else None}


# ---------------------------------------------------------------------------
# D2 — desk-stage stub (branch precedence step 2 only; not a required gate here)
# ---------------------------------------------------------------------------

def gate_d2(desk_path: Optional[str] = None, alpha: float = 0.05) -> Dict:
    """The desk stage's null calibration, read in from its artifact so branch step 2 can fire.

    D2 lives entirely in the desk run (rung-swapped / label-permuted archived units, where the
    null is true by construction) and has no counterpart in the run JSONs this evaluator reads.
    Passing ``--desk desk_h8.json`` makes the branch chain adjudicate CONSTRUCTION-INVALID from
    the real numbers instead of stepping over a permanent ``not-run``.
    """
    if desk_path:
        try:
            with open(desk_path) as f:
                desk = json.load(f)
        except Exception as exc:                      # noqa: BLE001 - report, never crash
            return {"gate": "D2", "observable": "desk null calibration", "threshold": "<= alpha",
                    "value": None, "verdict": "not-run",
                    "note": f"could not read --desk {desk_path}: {exc}"}
        # desk_h8.json shape: {family: {task: {"plus": {"rate", "n_cross", "n_total", "ci95"},
        #                                       "minus": {...}}}}
        rates = []
        for fam, per_task in (desk.get("D2") or {}).items():
            if not isinstance(per_task, dict):
                continue
            for task, row in per_task.items():
                if not isinstance(row, dict):
                    continue
                for side in ("plus", "minus"):
                    cell = row.get(side)
                    if isinstance(cell, dict) and isinstance(cell.get("rate"), (int, float)):
                        rates.append({"family": fam, "task": task, "side": side,
                                      "rate": float(cell["rate"]),
                                      "n_cross": cell.get("n_cross"),
                                      "n_total": cell.get("n_total"),
                                      "ci95": cell.get("ci95")})
        if not rates:
            return {"gate": "D2", "observable": "desk null calibration", "threshold": "<= alpha",
                    "value": None, "verdict": "not-run",
                    "note": f"--desk {desk_path} has no readable D2 rates"}
        worst = max(rates, key=lambda r: r["rate"])
        return {"gate": "D2",
                "observable": "crossing rate of the e-process on mean-centred / rung-swapped / "
                              "label-permuted archived units, where the null is TRUE by "
                              "construction (the sensitivity control on the construction itself)",
                "threshold": f"<= alpha = {alpha} on every family and side",
                "value": {"worst": worst, "n_rates": len(rates), "rates": rates,
                          "desk_D2_PASS": desk.get("D2_PASS")},
                "verdict": "pass" if worst["rate"] <= alpha else "fail",
                "headline": (f"worst crossing rate {worst['rate']:.3f} "
                             f"({worst['family']}/task {worst['task']}/{worst['side']}) "
                             f"vs alpha={alpha}, over {len(rates)} rates"),
                "source": desk_path}
    return {"gate": "D2",
            "observable": "crossing rate of the e-process on rung-swapped/label-permuted A0 "
                          "units (the sensitivity control on the construction's null calibration)",
            "threshold": "<= alpha = 0.05 on each side",
            "value": None, "verdict": "not-run",
            "note": "D2 is a desk-stage check over archived A0 units, not part of the "
                    "exp3a_kan_results.json layout this evaluator reads; it is reported here only "
                    "so branch precedence step 2 (CONSTRUCTION-INVALID) is named rather than "
                    "silently skipped. It can never fire from this script's inputs alone — see "
                    "the desk-stage artifact/script for the real D2 result."}


# ---------------------------------------------------------------------------
# P0a / P0b — null validity
# ---------------------------------------------------------------------------

_L_FIELDS = ("L_null_bits", "L_reuse_bits", "L_search_bits", "L_grow_bits")


def _compare_runs(a: Optional[Dict], b: Optional[Dict], seed: int, stream: Optional[str],
                   label_a: str, label_b: str) -> List[Dict]:
    """All (decision, AA, L_*) mismatches between two runs expected to be bit-identical."""
    if a is None or b is None:
        return []
    mismatches = []
    aa_a, aa_b = a.get("average_accuracy"), b.get("average_accuracy")
    if aa_a is not None and aa_b is not None and aa_a != aa_b:
        mismatches.append({"seed": seed, "stream": stream, "task": None,
                            "field": "average_accuracy", label_a: aa_a, label_b: aa_b})
    for da in a.get("decisions", []):
        t = da.get("task")
        db = dec_at(b, t)
        if db is None:
            continue
        if da.get("decision") != db.get("decision"):
            mismatches.append({"seed": seed, "stream": stream, "task": t, "field": "decision",
                                label_a: da.get("decision"), label_b: db.get("decision")})
        for f in _L_FIELDS:
            va, vb = da.get(f), db.get(f)
            if va is not None and vb is not None and va != vb:
                mismatches.append({"seed": seed, "stream": stream, "task": t, "field": f,
                                    label_a: va, label_b: vb})
    return mismatches


def _p0_gate(gate_id: str, ref_ctrl: Dict[int, Dict[str, Dict]], ref_5ds: Dict[int, Dict],
             other_ctrl: Dict[int, Dict[str, Dict]], other_5ds: Dict[int, Dict],
             ref_label: str, other_label: str, observable: str) -> Dict:
    gate = {"gate": gate_id, "observable": observable,
            "threshold": f"decisions, average_accuracy and every L_* bit-identical between "
                         f"{ref_label} and {other_label}",
            "value": None, "verdict": "not-run"}
    if not (ref_ctrl or ref_5ds) or not (other_ctrl or other_5ds):
        gate["note"] = f"{ref_label} and/or {other_label} data missing (CTrL and 5-Datasets)"
        return gate
    all_mismatches: List[Dict] = []
    n_compared = 0
    for seed in CTRL_SEEDS:
        for stream in CTRL_STREAMS:
            ra = ref_ctrl.get(seed, {}).get(stream)
            rb = other_ctrl.get(seed, {}).get(stream)
            if ra is None or rb is None:
                continue
            n_compared += 1
            all_mismatches.extend(_compare_runs(ra, rb, seed, stream, ref_label, other_label))
    for seed in sorted(set(ref_5ds) | set(other_5ds)):
        ra, rb = ref_5ds.get(seed), other_5ds.get(seed)
        if ra is None or rb is None:
            continue
        n_compared += 1
        all_mismatches.extend(_compare_runs(ra, rb, seed, None, ref_label, other_label))
    if n_compared == 0:
        gate["note"] = f"no matched runs between {ref_label} and {other_label}"
        return gate
    gate["value"] = {"n_runs_compared": n_compared, "n_mismatches": len(all_mismatches),
                      "first_mismatch": all_mismatches[0] if all_mismatches else None,
                      "all_mismatches": all_mismatches}
    gate["verdict"] = "pass" if not all_mismatches else "fail"
    gate["headline"] = (f"{n_compared} runs bit-identical" if not all_mismatches else
                         f"{len(all_mismatches)} mismatches over {n_compared} runs "
                         f"(first: {all_mismatches[0]})")
    return gate


def gate_p0a(baseline_ctrl, baseline_5ds, off_ctrl, off_5ds) -> Dict:
    return _p0_gate("P0a", baseline_ctrl, baseline_5ds, off_ctrl, off_5ds,
                     "baseline(attn_pool adoption run)", "arm off",
                     "arm off vs. the attn-root-adoption baseline: CTrL + 5-Datasets, decisions, "
                     "AA and every L_* on every gated decision")


def gate_p0b(off_ctrl, off_5ds, shadow_ctrl, shadow_5ds) -> Dict:
    return _p0_gate("P0b", off_ctrl, off_5ds, shadow_ctrl, shadow_5ds,
                     "arm off", "arm shadow",
                     "arm shadow (e-process on, minting off) vs. arm off: CTrL + 5-Datasets, "
                     "decisions, AA and every L_* on every gated decision — the shadow refit must "
                     "not perturb the ladder")


# ---------------------------------------------------------------------------
# P1 — selectivity against an oracle sign-stability label (measured on arm shadow)
# ---------------------------------------------------------------------------

def _sign_stability(vals: List[Dict]) -> Tuple[Optional[str], int]:
    """vals: [{"mean_d":..., "threshold_bits":...}, ...]. Returns (label, n_signed)."""
    signs = []
    for v in vals:
        md, th = v.get("mean_d"), v.get("threshold_bits")
        if md is None or th is None:
            continue
        d = md - th
        signs.append(0 if d == 0 else (1 if d > 0 else -1))
    if not signs:
        return None, 0
    stable = all(s == signs[0] for s in signs)
    return ("stable" if stable else "unstable"), len(signs)


def gate_p1(shadow_ctrl: Dict[int, Dict[str, Dict]], shadow_5ds: Dict[int, Dict]) -> Dict:
    gate = {"gate": "P1",
            "observable": "sign-stability of the paired (mean_d - threshold_bits) margin across "
                          "the 5 CTrL seeds, in arm shadow, vs. e_state at the same position "
                          "(P1 is measured on shadow, not evalue, per the 2026-09-09 amendment)",
            "threshold": f"undetermined at >= {P1_UNSTABLE_MIN_FRAC:.0%} of sign-UNSTABLE "
                         f"positions and <= {P1_STABLE_MAX_FRAC:.0%} of sign-STABLE ones",
            "value": None, "verdict": "not-run"}
    if not shadow_ctrl:
        gate["note"] = "arm shadow CTrL data missing"
        return gate

    # --- t3: ONE position, streams collapsed within a seed (they share t0-t3 byte-for-byte) ---
    t3_units = []
    t3_disagreements = []
    for seed in CTRL_SEEDS:
        per_stream = [(stream, ev_at(shadow_ctrl.get(seed, {}).get(stream), T3))
                      for stream in CTRL_STREAMS]
        per_stream = [(s, ev) for s, ev in per_stream if ev is not None]
        if not per_stream:
            continue
        mean_ds = [ev.get("mean_d") for _, ev in per_stream if ev.get("mean_d") is not None]
        thrs = [ev.get("threshold_bits") for _, ev in per_stream if ev.get("threshold_bits") is not None]
        states = [ev.get("state") for _, ev in per_stream if ev.get("state") is not None]
        if len(set(states)) > 1:
            t3_disagreements.append({"seed": seed, "states": {s: ev.get("state") for s, ev in per_stream}})
        t3_units.append({"seed": seed, "mean_d": _mean(mean_ds), "threshold_bits": _mean(thrs),
                          "e_state": states[0] if states else None,
                          "n_streams_present": len(per_stream)})

    groups: Dict[str, Dict] = {}
    label, n_signed = _sign_stability(t3_units)
    groups["t3"] = {"label": label, "n_signed": n_signed, "units": t3_units}

    # --- t4: FOUR positions, one per stream (genuinely different revisit per stream) ---
    for stream in CTRL_STREAMS:
        units = []
        for seed in CTRL_SEEDS:
            ev = ev_at(shadow_ctrl.get(seed, {}).get(stream), T4)
            if ev is None:
                continue
            units.append({"seed": seed, "mean_d": ev.get("mean_d"),
                           "threshold_bits": ev.get("threshold_bits"), "e_state": ev.get("state")})
        label, n_signed = _sign_stability(units)
        groups[f"t4:{stream}"] = {"label": label, "n_signed": n_signed, "units": units}

    # --- pool units by label into the 2x2 table ---
    table = {"stable": {"undetermined": 0, "determined": 0},
             "unstable": {"undetermined": 0, "determined": 0}}
    for g in groups.values():
        if g["label"] not in ("stable", "unstable"):
            continue
        for u in g["units"]:
            st = u.get("e_state")
            if st is None:
                continue
            col = "undetermined" if st == "undetermined" else "determined"
            table[g["label"]][col] += 1

    def _frac(row: Dict[str, int]) -> Optional[float]:
        n = row["undetermined"] + row["determined"]
        return (row["undetermined"] / n) if n else None

    unstable_frac = _frac(table["unstable"])
    stable_frac = _frac(table["stable"])
    n_total = sum(table["stable"].values()) + sum(table["unstable"].values())
    n_undet_total = table["stable"]["undetermined"] + table["unstable"]["undetermined"]
    overall_frac = (n_undet_total / n_total) if n_total else None

    # --- 5-Datasets: descriptive only (n=1 seed, no stability label possible) ---
    fds_positions = []
    for seed in sorted(shadow_5ds):
        r = shadow_5ds[seed]
        for t in GATED_TASKS:
            ev = ev_at(r, t)
            if ev is None:
                continue
            fds_positions.append({"seed": seed, "task": t, "mean_d": ev.get("mean_d"),
                                   "threshold_bits": ev.get("threshold_bits"),
                                   "e_state": ev.get("state"), "n_score": ev.get("n")})

    if n_total == 0:
        gate["note"] = "no shadow-arm CTrL evalue records found at t3/t4"
        gate["value"] = {"groups": groups, "table_2x2": table, "fivedatasets": fds_positions}
        return gate

    pass_ = (unstable_frac is not None and unstable_frac >= P1_UNSTABLE_MIN_FRAC and
             stable_frac is not None and stable_frac <= P1_STABLE_MAX_FRAC)
    gate["value"] = {"groups": groups, "table_2x2": table,
                      "unstable_undetermined_fraction": unstable_frac,
                      "stable_undetermined_fraction": stable_frac,
                      "overall_undetermined_fraction": overall_frac,
                      "n_positions": len(groups), "t3_stream_disagreements": t3_disagreements,
                      "fivedatasets": fds_positions}
    gate["verdict"] = "pass" if pass_ else "fail"
    uf = f"{unstable_frac:.0%}" if unstable_frac is not None else "n/a"
    sf = f"{stable_frac:.0%}" if stable_frac is not None else "n/a"
    gate["headline"] = f"unstable-undet={uf}, stable-undet={sf}, positions={len(groups)}"
    if t3_disagreements:
        gate["note"] = (f"{len(t3_disagreements)} seed(s) disagree on e_state across the 4 "
                         f"streams' t3 shadow refit, which should be byte-identical up to t3")
    return gate


# ---------------------------------------------------------------------------
# P2 — necessity and value (T vs C1 se_proxy vs C2 always vs C0 off, on CTrL only:
# se_proxy/always arms are not part of the s_interleave or 5-Datasets layout)
# ---------------------------------------------------------------------------

def gate_p2(off_ctrl, evalue_ctrl, se_proxy_ctrl, always_ctrl) -> Dict:
    gate = {"gate": "P2",
            "observable": "(a) paired t3 regret improvement, T vs C0; (b) accuracy-per-100k-added"
                          "-parameters, T vs C2; (c) fraction of gated positions where T's decision"
                          " differs from C1's",
            "threshold": f"(a) >= {P2A_MIN_IMPROVE}; (b) T's ratio >= C2's; "
                         f"(c) >= {P2C_MIN_DIFF_FRAC:.0%} differ, else PROXY-SUFFICES",
            "value": None, "verdict": "not-run"}
    if not off_ctrl or not evalue_ctrl:
        gate["note"] = "arm off and/or arm evalue CTrL data missing"
        return gate

    # (a) paired t3 regret improvement = regret_off - regret_evalue, must be >= 0.03
    pairs_oe = paired_ctrl(off_ctrl, evalue_ctrl)
    rows_a, diffs_a = [], []
    for seed, stream, off_r, ev_r in pairs_oe:
        r_off, r_ev = regret_at(off_r, T3), regret_at(ev_r, T3)
        if r_off is None or r_ev is None:
            continue
        improve = r_off - r_ev
        diffs_a.append(improve)
        rows_a.append({"seed": seed, "stream": stream, "off_regret": r_off, "evalue_regret": r_ev,
                        "improvement": improve})
    mean_improve = _mean(diffs_a)
    seed_level_a = _seed_level([{**r, "diff": r["improvement"]} for r in rows_a])
    a_verdict = "not-run" if mean_improve is None else (
        "pass" if mean_improve >= P2A_MIN_IMPROVE else "fail")
    a = {"rows": rows_a, "mean_improvement": mean_improve, "seed_level": seed_level_a,
         "verdict": a_verdict}

    # (b) accuracy-per-parameter, evalue vs always, both paired against off over the COMMON
    # (seed, stream) set present in all three (aggregate AA/params first, then one ratio per arm —
    # see the module docstring's ambiguity note).
    b_verdict, b = "not-run", {}
    if always_ctrl:
        common = set()
        for seed in CTRL_SEEDS:
            for stream in CTRL_STREAMS:
                if (off_ctrl.get(seed, {}).get(stream) is not None and
                        evalue_ctrl.get(seed, {}).get(stream) is not None and
                        always_ctrl.get(seed, {}).get(stream) is not None):
                    common.add((seed, stream))
        aa_off = _mean([off_ctrl[s][st].get("average_accuracy") for s, st in common])
        aa_ev = _mean([evalue_ctrl[s][st].get("average_accuracy") for s, st in common])
        aa_al = _mean([always_ctrl[s][st].get("average_accuracy") for s, st in common])
        p_off = _mean([off_ctrl[s][st].get("params_total_pre_consolidation") for s, st in common])
        p_ev = _mean([evalue_ctrl[s][st].get("params_total_pre_consolidation") for s, st in common])
        p_al = _mean([always_ctrl[s][st].get("params_total_pre_consolidation") for s, st in common])

        def _ratio(aa_arm, aa_ref, p_arm, p_ref):
            if None in (aa_arm, aa_ref, p_arm, p_ref):
                return None
            denom = (p_arm - p_ref) / 1e5
            if denom == 0:
                return None
            return (aa_arm - aa_ref) / denom

        ratio_ev = _ratio(aa_ev, aa_off, p_ev, p_off)
        ratio_al = _ratio(aa_al, aa_off, p_al, p_off)
        b = {"n_common_runs": len(common), "mean_aa_off": aa_off, "mean_aa_evalue": aa_ev,
             "mean_aa_always": aa_al, "mean_params_off": p_off, "mean_params_evalue": p_ev,
             "mean_params_always": p_al, "ratio_evalue": ratio_ev, "ratio_always": ratio_al}
        if ratio_ev is not None and ratio_al is not None:
            b_verdict = "pass" if ratio_ev >= ratio_al else "fail"
        b["verdict"] = b_verdict
    else:
        b = {"verdict": "not-run", "note": "arm always CTrL data missing"}

    # (c) fraction of gated positions where T's decision differs from C1(se_proxy)'s
    c_verdict, c = "not-run", {}
    proxy_suffices = False
    if se_proxy_ctrl:
        rows_c, n_diff, n_total = [], 0, 0
        for seed in CTRL_SEEDS:
            for stream in CTRL_STREAMS:
                ev_r = evalue_ctrl.get(seed, {}).get(stream)
                se_r = se_proxy_ctrl.get(seed, {}).get(stream)
                if ev_r is None or se_r is None:
                    continue
                for t in GATED_TASKS:
                    d_ev, d_se = dec_at(ev_r, t), dec_at(se_r, t)
                    if d_ev is None or d_se is None:
                        continue
                    n_total += 1
                    differ = d_ev.get("decision") != d_se.get("decision")
                    if differ:
                        n_diff += 1
                    rows_c.append({"seed": seed, "stream": stream, "task": t,
                                    "evalue_decision": d_ev.get("decision"),
                                    "se_proxy_decision": d_se.get("decision"), "differ": differ})
        frac_diff = (n_diff / n_total) if n_total else None
        c = {"rows": rows_c, "n_total": n_total, "n_diff": n_diff, "frac_diff": frac_diff}
        if frac_diff is not None:
            c_verdict = "pass" if frac_diff >= P2C_MIN_DIFF_FRAC else "fail"
            proxy_suffices = (n_diff == 0)
        c["verdict"] = c_verdict
    else:
        c = {"verdict": "not-run", "note": "arm se_proxy CTrL data missing"}

    verdicts = [a["verdict"], b["verdict"], c["verdict"]]
    if any(v == "fail" for v in verdicts):
        overall = "fail"
    elif all(v == "pass" for v in verdicts):
        overall = "pass"
    else:
        overall = "not-run"
    gate["value"] = {"a_regret_improvement": a, "b_accuracy_per_parameter": b,
                      "c_decision_disagreement": c, "proxy_suffices": proxy_suffices}
    gate["verdict"] = overall
    gate["headline"] = (f"a={a['verdict']}(Δ={mean_improve if mean_improve is not None else 'n/a'})"
                        f", b={b['verdict']}, c={c['verdict']}"
                        + (" [PROXY-SUFFICES]" if proxy_suffices else ""))
    return gate


# ---------------------------------------------------------------------------
# P3a / P3b — decided positions, clean vs. downstream of a deferral (CTrL only)
# ---------------------------------------------------------------------------

def gate_p3(off_ctrl, evalue_ctrl) -> Dict:
    gate_a = {"gate": "P3a",
              "observable": "arm evalue decided positions (evalue.state != undetermined) in "
                            "runs with no EARLIER deferral, vs. arm off's decision at the same "
                            "(seed, stream, task)",
              "threshold": "decision identical to arm off", "value": None, "verdict": "not-run"}
    gate_b = {"gate": "P3b",
              "observable": "the same, downstream of an earlier provisional root in the same run "
                            "(reader-level delta accuracy is not recorded per reader in Loop 1 — "
                            "reported at the task-accuracy level, per the note's own caveat)",
              "threshold": f"identical decisions and test_accs within tau={P3_TAU}",
              "value": None, "verdict": "not-run",
              "note": "Reader-level per-reader deltas are not available in this JSON schema; "
                      "task-accuracy (test_accs[task]) is used as the substitute observable."}
    if not off_ctrl or not evalue_ctrl:
        note = "arm off and/or arm evalue CTrL data missing"
        gate_a["note"] = note
        gate_b["note"] = gate_b["note"] + f"; {note}"
        return {"P3a": gate_a, "P3b": gate_b}

    rows_a, rows_b = [], []
    for seed in CTRL_SEEDS:
        for stream in CTRL_STREAMS:
            r_ev = evalue_ctrl.get(seed, {}).get(stream)
            r_off = off_ctrl.get(seed, {}).get(stream)
            if r_ev is None or r_off is None:
                continue
            deferred_so_far = False
            for t in GATED_TASKS:
                d_ev = dec_at(r_ev, t)
                if d_ev is None:
                    continue
                ev = d_ev.get("evalue")
                state = ev.get("state") if ev else None
                if state == "undetermined":
                    deferred_so_far = True
                    continue
                d_off = dec_at(r_off, t)
                if d_off is None:
                    continue
                same_decision = d_ev.get("decision") == d_off.get("decision")
                at, ao = acc_at(r_ev, t), acc_at(r_off, t)
                acc_delta = (at - ao) if (at is not None and ao is not None) else None
                within_tau = acc_delta is not None and abs(acc_delta) <= P3_TAU
                row = {"seed": seed, "stream": stream, "task": t, "evalue_decision": d_ev.get("decision"),
                       "off_decision": d_off.get("decision"), "same_decision": same_decision,
                       "evalue_test_acc": at, "off_test_acc": ao, "acc_delta": acc_delta,
                       "within_tau": within_tau}
                (rows_b if deferred_so_far else rows_a).append(row)

    if rows_a:
        n_ok = sum(1 for r in rows_a if r["same_decision"])
        gate_a["value"] = {"n_positions": len(rows_a), "n_identical": n_ok, "rows": rows_a}
        gate_a["verdict"] = "pass" if n_ok == len(rows_a) else "fail"
        gate_a["headline"] = f"{n_ok}/{len(rows_a)} decided positions identical to arm off"
    else:
        gate_a["note"] = "no clean (undeferred) decided positions found"

    if rows_b:
        n_dec_ok = sum(1 for r in rows_b if r["same_decision"])
        n_acc_ok = sum(1 for r in rows_b if r["within_tau"])
        gate_b["value"] = {"n_positions": len(rows_b), "n_identical_decision": n_dec_ok,
                            "n_within_tau": n_acc_ok, "rows": rows_b}
        gate_b["verdict"] = "pass" if (n_dec_ok == len(rows_b) and n_acc_ok == len(rows_b)) else "fail"
        gate_b["headline"] = (f"{n_dec_ok}/{len(rows_b)} decisions identical, "
                              f"{n_acc_ok}/{len(rows_b)} within tau (downstream of a deferral)")
    else:
        gate_b["note"] = gate_b["note"] + "; no downstream-of-a-deferral decided positions found"

    return {"P3a": gate_a, "P3b": gate_b}


# ---------------------------------------------------------------------------
# P4 — merge fires, s_interleave (10 seeds), negative control on the (A, B) pair
# ---------------------------------------------------------------------------

def gate_p4(int_evalue: Dict[int, Dict]) -> Dict:
    gate = {"gate": "P4",
            "observable": "the t1-minted provisional root's resolution, arm evalue, s_interleave, "
                          "10 seeds; negative control: no merge op with keep/drop task ids {1,2} "
                          "or {2,3}",
            "threshold": f"merged in >= 7/10 seeds; negative control never fires",
            "value": None, "verdict": "not-run"}
    if not int_evalue:
        gate["note"] = "arm evalue s_interleave data missing"
        return gate
    rows = []
    for seed in INT_SEEDS:
        r = int_evalue.get(seed)
        if r is None:
            continue
        proots = r.get("provisional_roots", []) or []
        root_a = next((p for p in proots if p.get("minted_at") == 1), None)
        resolved_merge = bool(root_a and root_a.get("resolution") == "merge")
        wrong_merge = False
        for op in ((r.get("consolidation") or {}).get("ops", []) or []):
            if op.get("op") == "merge":
                pair = {op.get("keep"), op.get("drop")}
                if pair == {1, 2} or pair == {2, 3}:
                    wrong_merge = True
        t3_decision = (dec_at(r, 3) or {}).get("decision")
        rows.append({"seed": seed, "root_a_present": root_a is not None,
                     "root_a_resolution": root_a.get("resolution") if root_a else None,
                     "resolved_merge": resolved_merge, "wrong_merge": wrong_merge,
                     "t3_decision": t3_decision})
    if not rows:
        gate["note"] = "no matched seeds for arm evalue s_interleave"
        return gate
    n = len(rows)
    n_merge = sum(1 for r in rows if r["resolved_merge"])
    frac_merge = n_merge / n
    any_wrong_merge = any(r["wrong_merge"] for r in rows)
    t3_reuse_fraction = _mean([1.0 if r["t3_decision"] == "reuse" else 0.0 for r in rows])
    pass_ = (frac_merge >= P4_MIN_MERGE_FRAC) and not any_wrong_merge
    gate["value"] = {"n_seeds": n, "n_merge": n_merge, "frac_merge": frac_merge,
                      "any_wrong_merge": any_wrong_merge, "t3_reuse_fraction": t3_reuse_fraction,
                      "rows": rows}
    gate["verdict"] = "pass" if pass_ else "fail"
    gate["headline"] = (f"{n_merge}/{n} merged (t3 reuse fraction={t3_reuse_fraction:.2f})"
                        + (" — WRONG-MERGE" if any_wrong_merge else ""))
    if n < len(INT_SEEDS):
        gate["note"] = f"partial: {n}/{len(INT_SEEDS)} seeds present so far"
    return gate


# ---------------------------------------------------------------------------
# P5 — collateral (CTrL AA + per-task, 5-Datasets AA + decisions)
# ---------------------------------------------------------------------------

def gate_p5(off_ctrl, evalue_ctrl, off_5ds, evalue_5ds) -> Dict:
    gate = {"gate": "P5",
            "observable": "CTrL AA (arm evalue vs off, paired); no non-t3 task down >0.01; "
                          "5-Datasets AA within 0.005 and 4 grow/0 search/1 reuse with the dup "
                          "reusing",
            "threshold": f"AA >= off - {P5_AA_TOL}; per-task tolerance {P5_TASK_TOL}; "
                         f"5-Datasets AA tol {P5_FDS_AA_TOL}", "value": None, "verdict": "not-run"}
    if not off_ctrl or not evalue_ctrl:
        gate["note"] = "arm off and/or arm evalue CTrL data missing"
        return gate
    pairs = paired_ctrl(off_ctrl, evalue_ctrl)
    rows, diffs, collateral = [], [], []
    for seed, stream, off_r, ev_r in pairs:
        aa_off, aa_ev = off_r.get("average_accuracy"), ev_r.get("average_accuracy")
        if aa_off is None or aa_ev is None:
            continue
        diff = aa_ev - aa_off
        diffs.append(diff)
        task_deltas = []
        off_accs, ev_accs = off_r.get("test_accs", []), ev_r.get("test_accs", [])
        for i, (oa, ea) in enumerate(zip(off_accs, ev_accs)):
            if i == T3:
                continue
            delta = ea - oa
            within = delta >= -P5_TASK_TOL
            task_deltas.append({"task": i, "off_acc": oa, "evalue_acc": ea, "delta": delta,
                                "within": within})
            if not within:
                collateral.append({"seed": seed, "stream": stream, "task": i, "delta": delta})
        rows.append({"seed": seed, "stream": stream, "off_aa": aa_off, "evalue_aa": aa_ev,
                    "diff": diff, "task_deltas": task_deltas})
    if not diffs:
        gate["note"] = "no paired (seed, stream) CTrL runs with average_accuracy on both arms"
        return gate
    mean_diff = _mean(diffs)
    seed_level = _seed_level(rows)
    ctrl_aa_ok = mean_diff is not None and mean_diff >= -P5_AA_TOL
    ctrl_ok = ctrl_aa_ok and not collateral

    # 5-Datasets: 1 seed, single comparison
    fds = None
    if off_5ds and evalue_5ds:
        common = sorted(set(off_5ds) & set(evalue_5ds))
        fds_rows = []
        for seed in common:
            ob, eb = off_5ds[seed], evalue_5ds[seed]
            aa_o, aa_e = ob.get("average_accuracy"), eb.get("average_accuracy")
            aa_delta = (aa_e - aa_o) if (aa_o is not None and aa_e is not None) else None
            aa_ok = aa_delta is not None and abs(aa_delta) <= P5_FDS_AA_TOL
            counts_ok = (eb.get("n_grow") == 4 and eb.get("n_search") == 0 and eb.get("n_reuse") == 1)
            d4 = dec_at(eb, T4)
            dup_reuses = bool(d4 and d4.get("decision") == "reuse")
            fds_rows.append({"seed": seed, "off_aa": aa_o, "evalue_aa": aa_e, "aa_delta": aa_delta,
                             "aa_ok": aa_ok, "n_grow": eb.get("n_grow"), "n_search": eb.get("n_search"),
                             "n_reuse": eb.get("n_reuse"), "counts_ok": counts_ok,
                             "dup_reuses": dup_reuses,
                             "row_ok": bool(aa_ok and counts_ok and dup_reuses)})
        fds_ok = bool(fds_rows) and all(r["row_ok"] for r in fds_rows)
        fds = {"rows": fds_rows, "ok": fds_ok}
    pass_ = ctrl_ok and (fds is None or fds["ok"])
    gate["value"] = {"ctrl": {"n_pairs": len(diffs), "mean_aa_diff": mean_diff,
                              "seed_level": seed_level, "aa_ok": ctrl_aa_ok,
                              "collateral_violations": collateral, "rows": rows},
                      "fivedatasets": fds}
    gate["verdict"] = "pass" if pass_ else "fail"
    gate["headline"] = (f"CTrL AA diff={mean_diff:.4f}, collateral={len(collateral)}"
                        + (f", 5ds ok={fds['ok']}" if fds else ", 5ds not-run"))
    if fds is None and off_5ds is not None and evalue_5ds is not None and not (off_5ds and evalue_5ds):
        gate["note"] = "5-Datasets arm off/evalue data missing"
    if len(pairs) < 20:
        note = f"partial: {len(pairs)}/20 paired (seed, stream) runs present so far"
        gate["note"] = (gate.get("note", "") + "; " + note) if gate.get("note") else note
    return gate


# ---------------------------------------------------------------------------
# P6 — resolution mode (>= 50% merge on s_interleave; descriptive on CTrL)
# ---------------------------------------------------------------------------

def gate_p6(int_evalue: Dict[int, Dict], ctrl_evalue: Dict[int, Dict[str, Dict]]) -> Dict:
    gate = {"gate": "P6",
            "observable": "fraction of provisional_roots resolved by merge, s_interleave (arm "
                          "evalue) and CTrL (arm evalue), pooled across runs",
            "threshold": f">= {P6_INT_MIN_MERGE_FRAC:.0%} on s_interleave; no threshold on CTrL",
            "value": None, "verdict": "not-run"}
    int_roots: List[Dict] = []
    for seed in INT_SEEDS:
        r = int_evalue.get(seed)
        if r is not None:
            int_roots.extend(r.get("provisional_roots", []) or [])
    ctrl_roots: List[Dict] = []
    for seed in CTRL_SEEDS:
        for stream in CTRL_STREAMS:
            r = ctrl_evalue.get(seed, {}).get(stream)
            if r is not None:
                ctrl_roots.extend(r.get("provisional_roots", []) or [])
    if not int_roots and not ctrl_roots:
        gate["note"] = "no provisional_roots found for arm evalue on s_interleave or CTrL"
        return gate
    int_stats = _resolution_stats(int_roots)
    ctrl_stats = _resolution_stats(ctrl_roots)
    total_roots = int_stats["total"] + ctrl_stats["total"]
    total_merge = int_stats["n_merge"] + ctrl_stats["n_merge"]
    all_timeout = total_roots > 0 and total_merge == 0
    gate["value"] = {"s_interleave": int_stats, "ctrl": ctrl_stats, "all_timeout": all_timeout}
    if int_stats["total"] == 0:
        gate["verdict"] = "not-run"
        gate["note"] = "no s_interleave provisional_roots to gate on (CTrL reported descriptively)"
    else:
        gate["verdict"] = "pass" if int_stats["frac_merge"] >= P6_INT_MIN_MERGE_FRAC else "fail"
    gate["headline"] = (f"s_interleave merge frac={int_stats['frac_merge']}, "
                        f"CTrL merge frac={ctrl_stats['frac_merge']}"
                        + (" [TIMING-IS-JUST-GROW]" if all_timeout else ""))
    return gate


# ---------------------------------------------------------------------------
# P7 — cost (descriptive)
# ---------------------------------------------------------------------------

def gate_p7(ctrl_by_arm: Dict[str, Dict[int, Dict[str, Dict]]]) -> Dict:
    gate = {"gate": "P7",
            "observable": "param_curve_total (pre- and post-consolidation) and mean gate_seconds "
                          "per gated task, per CTrL arm",
            "threshold": "descriptive — all recorded", "value": None, "verdict": "descriptive"}
    stats = {}
    any_data = False
    for arm, tree in ctrl_by_arm.items():
        runs = [r for streams in tree.values() for r in streams.values()]
        if not runs:
            stats[arm] = None
            continue
        any_data = True
        params_pre = [r.get("params_total_pre_consolidation") for r in runs]
        params_post = [(r.get("consolidation") or {}).get("params_after") for r in runs]
        gate_secs = []
        for r in runs:
            for d in r.get("decisions", []):
                if d.get("gate_seconds") is not None:
                    gate_secs.append(d["gate_seconds"])
        stats[arm] = {"n_runs": len(runs), "mean_params_total_pre": _mean(params_pre),
                      "mean_params_after_consolidation": _mean(params_post),
                      "mean_gate_seconds": _mean(gate_secs)}
    gate["value"] = stats
    if not any_data:
        gate["verdict"] = "not-run"
        gate["note"] = "no CTrL data for any arm"
        gate["headline"] = "no data"
    else:
        parts = []
        for arm in ("off", "shadow", "evalue", "se_proxy", "always"):
            s = stats.get(arm)
            if s and s["mean_params_total_pre"] is not None:
                parts.append(f"{arm}={s['mean_params_total_pre']:.0f}")
        gate["headline"] = "params_total_pre_consolidation: " + ", ".join(parts) if parts else "no data"
    return gate


# ---------------------------------------------------------------------------
# P8 — wall time (descriptive; shadow refit overhead)
# ---------------------------------------------------------------------------

def _shadow_seconds(runs: List[Dict]) -> List[float]:
    out = []
    for r in runs:
        for d in r.get("decisions", []):
            if d.get("shadow_seconds") is not None:
                out.append(d["shadow_seconds"])
    return out


def gate_p8(ctrl_evalue, int_evalue: Dict[int, Dict], fds_evalue: Dict[int, Dict]) -> Dict:
    gate = {"gate": "P8", "observable": "shadow_seconds per gated task, per experiment (arm evalue)",
            "threshold": "descriptive — all recorded", "value": None, "verdict": "descriptive"}
    ctrl_runs = [r for streams in ctrl_evalue.values() for r in streams.values()] if ctrl_evalue else []
    int_runs = list(int_evalue.values()) if int_evalue else []
    fds_runs = list(fds_evalue.values()) if fds_evalue else []
    ctrl_secs, int_secs, fds_secs = (_shadow_seconds(ctrl_runs), _shadow_seconds(int_runs),
                                    _shadow_seconds(fds_runs))
    if not (ctrl_secs or int_secs or fds_secs):
        gate["verdict"] = "not-run"
        gate["note"] = "no shadow_seconds recorded for arm evalue in any experiment"
        gate["headline"] = "no data"
        return gate
    gate["value"] = {"ctrl": {"n": len(ctrl_secs), "mean_seconds": _mean(ctrl_secs)},
                      "s_interleave": {"n": len(int_secs), "mean_seconds": _mean(int_secs)},
                      "fivedatasets": {"n": len(fds_secs), "mean_seconds": _mean(fds_secs)}}
    fmt = lambda xs: f"{_mean(xs):.2f}s" if xs else "n/a"
    gate["headline"] = f"ctrl={fmt(ctrl_secs)}, interleave={fmt(int_secs)}, 5ds={fmt(fds_secs)}"
    return gate


# ---------------------------------------------------------------------------
# P9 — every provisional root resolved by stream end
# ---------------------------------------------------------------------------

def gate_p9(all_trees: List[Tuple[str, Dict, bool]]) -> Dict:
    """``all_trees``: [(label, tree, nested_by_stream), ...]. ``nested_by_stream=True`` means
    ``tree`` is ``{seed: {stream: run}}`` (CTrL-shaped); ``False`` means ``{seed: run}``
    (s_interleave / 5-Datasets-shaped) — passed explicitly rather than inferred, since a run dict
    is itself a ``dict`` and duck-typing on ``isinstance(v, dict)`` cannot tell the two apart."""
    gate = {"gate": "P9", "observable": "every provisional_roots entry, every run, every stream/arm",
            "threshold": "zero roots still flagged unresolved at stream end", "value": None,
            "verdict": "not-run"}
    unresolved = []
    n_runs, n_roots = 0, 0
    for label, tree, nested_by_stream in all_trees:
        if not tree:
            continue
        for seed, v in tree.items():
            runs = list(v.values()) if nested_by_stream else [v]
            for r in runs:
                if r is None or "provisional_roots" not in r:
                    continue
                n_runs += 1
                for root in (r.get("provisional_roots") or []):
                    n_roots += 1
                    if root.get("resolution") is None or root.get("resolved_at") is None:
                        unresolved.append({"source": label, "seed": seed,
                                           "minted_at": root.get("minted_at"),
                                           "resolution": root.get("resolution"),
                                           "resolved_at": root.get("resolved_at")})
    if n_runs == 0:
        gate["note"] = "no runs with a provisional_roots field found"
        return gate
    gate["value"] = {"n_runs_checked": n_runs, "n_roots_checked": n_roots,
                      "n_unresolved": len(unresolved), "unresolved": unresolved}
    gate["verdict"] = "pass" if not unresolved else "fail"
    gate["headline"] = f"{n_roots - len(unresolved)}/{n_roots} roots resolved ({n_runs} runs checked)"
    return gate


# ---------------------------------------------------------------------------
# per_run tables (raw rows so every number above can be regenerated)
# ---------------------------------------------------------------------------

def _decisions_summary(r: Dict) -> List[Dict]:
    out = []
    for t in sorted({d.get("task") for d in r.get("decisions", [])} | set(GATED_TASKS)):
        d = dec_at(r, t)
        if d is None:
            continue
        ev = d.get("evalue")
        out.append({"task": t, "decision": d.get("decision"),
                    "ladder_decision": d.get("ladder_decision"), "provisional": d.get("provisional"),
                    "gate_seconds": d.get("gate_seconds"), "shadow_seconds": d.get("shadow_seconds"),
                    "evalue_state": ev.get("state") if ev else None,
                    "evalue_mean_d": ev.get("mean_d") if ev else None,
                    "evalue_threshold_bits": ev.get("threshold_bits") if ev else None})
    return out


def build_run_row(r: Dict, arm: str, seed: int, stream: Optional[str] = None) -> Dict:
    return {"arm": arm, "seed": seed, "stream": stream,
            "average_accuracy": r.get("average_accuracy"), "test_accs": r.get("test_accs"),
            "params_total_pre_consolidation": r.get("params_total_pre_consolidation"),
            "params_after_consolidation": (r.get("consolidation") or {}).get("params_after"),
            "n_provisional_roots": len(r.get("provisional_roots", []) or []),
            "provisional_roots": r.get("provisional_roots"),
            "decisions_summary": _decisions_summary(r)}


def build_ctrl_rows(tree: Dict[int, Dict[str, Dict]], arm: str) -> List[Dict]:
    rows = []
    for seed in sorted(tree):
        for stream in CTRL_STREAMS:
            if stream in tree[seed]:
                rows.append(build_run_row(tree[seed][stream], arm, seed, stream))
    return rows


def build_single_stream_rows(tree: Dict[int, Dict], arm: str) -> List[Dict]:
    return [build_run_row(tree[seed], arm, seed) for seed in sorted(tree)]


# ---------------------------------------------------------------------------
# Branch
# ---------------------------------------------------------------------------

def determine_branch(gates: Dict[str, Dict], p1_value: Optional[Dict], p2_value: Optional[Dict],
                     p6_value: Optional[Dict], p4_value: Optional[Dict]) -> Dict:
    def v(gid):
        g = gates.get(gid)
        return g.get("verdict") if g else None

    flags = []
    if v("P5") == "fail":
        flags.append("COLLATERAL")
    if p2_value and p2_value.get("proxy_suffices"):
        flags.append("PROXY-SUFFICES")
    if p6_value and p6_value.get("all_timeout"):
        flags.append("TIMING-IS-JUST-GROW")
    if p4_value and p4_value.get("t3_reuse_fraction") is not None and \
            p4_value["t3_reuse_fraction"] >= NO_MERGE_OBJECT_MIN_FRAC:
        flags.append("NO-MERGE-OBJECT")

    def missing(gid):
        return v(gid) in (None, "not-run")

    core_ids = ["P0a", "P0b", "P1", "P2", "P5"]
    if any(missing(g) for g in core_ids):
        missing_ids = [g for g in core_ids if missing(g)]
        return {"branch": "NOT-RUN", "reason": f"awaiting {', '.join(missing_ids)}",
                "flags": flags}

    # 1. NULL-BROKEN
    if v("P0a") == "fail" or v("P0b") == "fail":
        return {"branch": "NULL-BROKEN",
                "reason": "P0a and/or P0b failed — this is a code regression, not a hypothesis "
                          "result; the gates below are still computed for debugging but none of "
                          "them names a branch until the null check passes.", "flags": flags}

    # 2. CONSTRUCTION-INVALID (D2 — never computable from this script's own inputs; see gate_d2)
    if v("D2") == "fail":
        return {"branch": "CONSTRUCTION-INVALID",
                "reason": "D2's crossing rate exceeded alpha under the true null.", "flags": flags}

    # 3. NEVER-UNDETERMINED / ALWAYS-UNDETERMINED (P1's two failure halves)
    if v("P1") == "fail":
        overall = (p1_value or {}).get("overall_undetermined_fraction")
        if overall is not None and overall <= NEVER_UNDETERMINED_TOL:
            return {"branch": "NEVER-UNDETERMINED",
                    "reason": f"P1 fails; the e-process is undetermined at ~{overall:.1%} of "
                              "gated positions overall — it essentially never fires.",
                    "flags": flags}
        if overall is not None and overall >= ALWAYS_UNDETERMINED_TOL:
            return {"branch": "ALWAYS-UNDETERMINED",
                    "reason": f"P1 fails; the e-process is undetermined at ~{overall:.1%} of "
                              "gated positions overall — it fires everywhere, stable or not.",
                    "flags": flags}
        # P1 fails for a reason other than the two named halves — no precedence entry covers
        # this specific mixed failure; fall through toward the catch-all rather than guessing.

    # 4. PROXY-SUFFICES
    if v("P1") == "pass" and "PROXY-SUFFICES" in flags:
        return {"branch": "PROXY-SUFFICES",
                "reason": "P1 passes but arm evalue and arm se_proxy never disagree — the "
                          "e-process apparatus is decorative.", "flags": flags}

    # 5. REGRET-NOT-FIXED
    a_verdict = ((p2_value or {}).get("a_regret_improvement") or {}).get("verdict")
    if a_verdict == "fail":
        return {"branch": "REGRET-NOT-FIXED",
                "reason": "P2(a) fails — deferral does not drive end-of-stream t3 regret down "
                          "by >= 0.03 vs. arm off.", "flags": flags}

    # 6. NOT-WORTH-THE-PARAMETERS
    b_verdict = ((p2_value or {}).get("b_accuracy_per_parameter") or {}).get("verdict")
    if b_verdict == "fail":
        return {"branch": "NOT-WORTH-THE-PARAMETERS",
                "reason": "P2(b) fails — arm evalue's accuracy-per-parameter is below arm "
                          "always's.", "flags": flags}

    # 7. COLLATERAL
    if v("P5") == "fail":
        return {"branch": "COLLATERAL",
                "reason": "P5 fails — deferral costs accuracy the pre-registration did not "
                          "budget for.", "flags": flags}

    # 8. TIMING-IS-JUST-GROW
    if "TIMING-IS-JUST-GROW" in flags:
        return {"branch": "TIMING-IS-JUST-GROW",
                "reason": "Every gate up to here passes, but P6 shows every provisional root "
                          "resolved by timeout — the mechanism is delayed unconditional growth.",
                "flags": flags}

    # 9. MERGE-NEVER-FIRES (P4 fails; the all-timeout case was already claimed by step 8)
    if v("P4") == "fail":
        return {"branch": "MERGE-NEVER-FIRES",
                "reason": "P4 fails (merge rate < 7/10 on s_interleave, or the negative control "
                          "fired) while P6 shows resolution activity other than pure timeout.",
                "flags": flags}

    # 10. NO-MERGE-OBJECT
    if "NO-MERGE-OBJECT" in flags:
        return {"branch": "NO-MERGE-OBJECT",
                "reason": "s_interleave t3 is decided (reuse) rather than undetermined in most "
                          "seeds, so no second provisional root is minted there — not a failure "
                          "of H8, but P4 has no object and must be re-run on a harder pair.",
                "flags": flags}

    # 11. TIMING-WINS
    remaining = ["P3a", "P3b", "P6", "P7", "P8", "P9"]
    if all(v(g) in ("pass", "descriptive", None, "not-run") for g in remaining) and \
            all(v(g) != "fail" for g in ["P0a", "P0b", "P1", "P2", "P4", "P5"]):
        return {"branch": "TIMING-WINS",
                "reason": "P0-P7 pass (P8 descriptive); deferral fixes the regret, earns its "
                          "parameters, and resolves.", "flags": flags}

    failing = [g for g in ["P0a", "P0b", "P1", "P2", "P3a", "P3b", "P4", "P5", "P6", "P9"]
               if v(g) == "fail"]
    return {"branch": "UNMATCHED-COMBINATION",
            "reason": ("No entry in the pre-registration's branch-precedence list matches this "
                       "verdict combination; recorded as a pre-registration gap rather than "
                       f"INDETERMINATE. Failing gates: {failing or 'none'}. "
                       f"Verdicts: {{{', '.join(f'{g}={v(g)}' for g in ['P0a','P0b','P1','P2','P3a','P3b','P4','P5','P6','P9'])}}}"),
            "flags": flags}


# ---------------------------------------------------------------------------
# Stdout table
# ---------------------------------------------------------------------------

_GATE_ORDER = ["P0a", "P0b", "D2", "P1", "P2", "P3a", "P3b", "P4", "P5", "P6", "P7", "P8", "P9"]


def print_table(gates: Dict[str, Dict], branch_info: Dict) -> None:
    print(f"\n{'gate':<5} {'verdict':<11} {'threshold':<58} headline")
    print("-" * 130)
    for gid in _GATE_ORDER:
        g = gates.get(gid)
        if g is None:
            continue
        thr = str(g.get("threshold", ""))
        if len(thr) > 56:
            thr = thr[:53] + "..."
        headline = g.get("headline") or g.get("note") or "-"
        print(f"{gid:<5} {str(g.get('verdict', '-')):<11} {thr:<58} {headline}")
    print(f"\n=== branch: {branch_info['branch']} ===")
    print(f"  reason: {branch_info['reason']}")
    if branch_info["flags"]:
        print(f"  flags: {', '.join(branch_info['flags'])}")


# ---------------------------------------------------------------------------
# Progress log parsing (R4-style, kept for P8 wall-time cross-check)
# ---------------------------------------------------------------------------

_PROGRESS_RE = re.compile(
    r"^===\s+(?P<exp>ctrl|int|5ds)\s+(?P<arm>\S+)\s+seed\s+(?P<seed>\d+)\s+"
    r"(?:(?P<stream>s_minus|s_out|s_plus|s_in|s_interleave)\s+)?"
    r"(?P<phase>start|exit)\s+(?:(?P<code>-?\d+)\s+)?(?P<epoch>\d+(?:\.\d+)?)\s+"
    r"(?P<date>.*?)\s*===\s*$")


def parse_progress(paths: List[str]) -> Optional[Dict]:
    """Parses ``progress_*.log`` lines of the form
    ``=== ctrl <arm> seed <S> <STREAM> start <epoch> <date> ===`` /
    ``=== ctrl <arm> seed <S> <STREAM> exit <code> <epoch> <date> ===``, mirroring
    ``scripts/eval_attn_root.py``'s ``parse_progress``. The STREAM group is optional so 5-Datasets
    lines (no stream) also match. Lines starting with ``===`` that don't match are counted in
    ``unmatched_lines`` and otherwise ignored."""
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
                key = (m.group("exp"), m.group("arm"), int(m.group("seed")), m.group("stream"))
                events.setdefault(key, {})[m.group("phase")] = float(m.group("epoch"))
    rows = []
    for (exp, arm, seed, stream), ev in sorted(events.items(), key=lambda kv: kv[0]):
        if "start" in ev and "exit" in ev:
            rows.append({"exp": exp, "arm": arm, "seed": seed, "stream": stream,
                        "seconds": ev["exit"] - ev["start"]})
    if not rows and unmatched == 0:
        return None
    per_arm: Dict[str, List[float]] = {}
    for r in rows:
        per_arm.setdefault(r["arm"], []).append(r["seconds"])
    return {"rows": rows, "mean_seconds_per_arm": {a: statistics.mean(v) for a, v in per_arm.items()},
            "unmatched_lines": unmatched}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctrl", default=None, help="dir holding ctrl_<arm>/ (CTrL results)")
    ap.add_argument("--interleave", default=None, help="dir holding int_<arm>/ (s_interleave results)")
    ap.add_argument("--fivedatasets", default=None, help="dir holding 5ds_<arm>/ (5-Datasets results)")
    ap.add_argument("--baseline", default=None,
                    help="archived attn-root-adoption results dir (the P0a reference)")
    ap.add_argument("--progress", nargs="+", default=[],
                    help="progress_*.log file(s) for P8 wall-time cross-check (optional)")
    ap.add_argument("--desk", default=None,
                    help="desk_h8.json from the desk stage; supplies D2 (null calibration), which "
                         "has no counterpart in the run JSONs, so branch step 2 "
                         "(CONSTRUCTION-INVALID) can be adjudicated from real numbers")
    ap.add_argument("--out", required=True, help="where to write gates.json")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.ctrl or args.interleave or args.fivedatasets or args.baseline):
        raise SystemExit("at least one of --ctrl, --interleave, --fivedatasets, --baseline is required")

    ctrl_by_arm = {arm: load_ctrl_arm(args.ctrl, arm) for arm in CTRL_ARMS}
    int_by_arm = {arm: load_int_arm(args.interleave, arm) for arm in INT_ARMS}
    fds_by_arm = {arm: load_5ds_arm(args.fivedatasets, arm) for arm in FDS_ARMS}
    baseline_ctrl = load_baseline_ctrl(args.baseline)
    baseline_5ds = load_baseline_5ds(args.baseline)

    notes = []
    if args.ctrl and not any(ctrl_by_arm.values()):
        notes.append(f"--ctrl given ({args.ctrl}) but no exp_ctrl_* results found under any ctrl_<arm>/")
    if args.interleave and not any(int_by_arm.values()):
        notes.append(f"--interleave given ({args.interleave}) but no results found under any int_<arm>/")
    if args.fivedatasets and not any(fds_by_arm.values()):
        notes.append(f"--fivedatasets given ({args.fivedatasets}) but no results found under any 5ds_<arm>/")
    if args.baseline and not (baseline_ctrl or baseline_5ds):
        notes.append(f"--baseline given ({args.baseline}) but no results found under it")

    progress_stats = parse_progress(args.progress) if args.progress else None

    p0a = gate_p0a(baseline_ctrl, baseline_5ds, ctrl_by_arm["off"], fds_by_arm["off"])
    p0b = gate_p0b(ctrl_by_arm["off"], fds_by_arm["off"], ctrl_by_arm["shadow"], fds_by_arm["shadow"])
    d2 = gate_d2(args.desk)
    p1 = gate_p1(ctrl_by_arm["shadow"], fds_by_arm["shadow"])
    p2 = gate_p2(ctrl_by_arm["off"], ctrl_by_arm["evalue"], ctrl_by_arm["se_proxy"], ctrl_by_arm["always"])
    p3 = gate_p3(ctrl_by_arm["off"], ctrl_by_arm["evalue"])
    p4 = gate_p4(int_by_arm["evalue"])
    p5 = gate_p5(ctrl_by_arm["off"], ctrl_by_arm["evalue"], fds_by_arm["off"], fds_by_arm["evalue"])
    p6 = gate_p6(int_by_arm["evalue"], ctrl_by_arm["evalue"])
    p7 = gate_p7(ctrl_by_arm)
    p8 = gate_p8(ctrl_by_arm["evalue"], int_by_arm["evalue"], fds_by_arm["evalue"])
    p9 = gate_p9([("ctrl:" + arm, ctrl_by_arm[arm], True) for arm in CTRL_ARMS] +
                [("interleave:" + arm, int_by_arm[arm], False) for arm in INT_ARMS] +
                [("fivedatasets:" + arm, fds_by_arm[arm], False) for arm in FDS_ARMS])

    gates = {"P0a": p0a, "P0b": p0b, "D2": d2, "P1": p1, "P2": p2, "P3a": p3["P3a"], "P3b": p3["P3b"],
             "P4": p4, "P5": p5, "P6": p6, "P7": p7, "P8": p8, "P9": p9}

    branch_info = determine_branch(gates, p1.get("value"), p2.get("value"), p6.get("value"),
                                    p4.get("value"))

    per_run = {
        "ctrl": {arm: build_ctrl_rows(ctrl_by_arm[arm], arm) for arm in CTRL_ARMS},
        "s_interleave": {arm: build_single_stream_rows(int_by_arm[arm], arm) for arm in INT_ARMS},
        "fivedatasets": {arm: build_single_stream_rows(fds_by_arm[arm], arm) for arm in FDS_ARMS},
        "baseline_ctrl": build_ctrl_rows(baseline_ctrl, "baseline"),
        "baseline_5ds": build_single_stream_rows(baseline_5ds, "baseline"),
    }

    print_table(gates, branch_info)
    if branch_info["flags"]:
        pass  # already printed by print_table
    if notes:
        print("\n=== notes ===")
        for n in notes:
            print(f"  - {n}")

    result: Dict = dict(gates)
    result["branch"] = branch_info["branch"]
    result["branch_detail"] = branch_info
    result["flags"] = branch_info["flags"]
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
