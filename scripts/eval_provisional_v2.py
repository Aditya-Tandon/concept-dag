#!/usr/bin/env python
"""eval_provisional_v2.py — the mechanical V-gate evaluator for H9, the provisional-growth
validity re-run (Central Library/Hypotheses/concept-dag/kan-gated-growth/
provisional-growth-v2-validity-rerun.md).

H8's evaluator (``scripts/eval_provisional.py``, still shipped and still the reader for the
2026-09-09 archive) implements the P-gates and structurally cannot compute four of H9's clauses:
V0a-ii has no H8-archive reference arm, V4's parameter-curve clause did not exist, V5's
before/after-first-deferral split did not exist, and V1's counterfactual needs the per-pair
statistics that only landed with the CKA pre-filter. This file implements the V-gates and the
branch chain verbatim from the note, and is committed and fixture-tested BEFORE the sweep, to
H8's own spec-violation standard.

It does the evaluation ONLY: no plotting, no re-running experiments. Loading helpers and the
seed/stream discovery are imported from ``eval_provisional`` rather than duplicated, so the two
evaluators can never disagree about what a run directory contains.

Layout expected (identical to H8's)
-----------------------------------
    <ctrl>/ctrl_<arm>/seed_<S>/exp_ctrl_<stream>/exp3a_kan_results.json
    <interleave>/int_<arm>/seed_<S>/exp_ctrl_s_interleave/exp3a_kan_results.json
    <fivedatasets>/5ds_<arm>/seed_42/exp5ds_kan/exp3a_kan_results.json
    <baseline>/ctrl_attn_pool/... and <baseline>/5ds_attn_pool/...   (V0a-i reference)
    <h8>/ctrl/ctrl_off/..., <h8>/int_off/..., <h8>/5ds_off/...       (V0a-ii reference)
    <dumps>/**/gate_dump.pt                                          (V6 / DA inventory)

Usage
-----
    python scripts/eval_provisional_v2.py \\
        --ctrl DIR --interleave DIR --fivedatasets DIR --baseline DIR --h8 DIR \\
        [--dumps DIR] [--da DESK_JSON] --out gates_h9.json
    python scripts/eval_provisional_v2.py --preflight --fivedatasets DIR --interleave DIR \\
        --baseline DIR --h8 DIR --out preflight.json          # V0 only; non-zero exit on failure

Ambiguities resolved while implementing (conservative readings, reasoned at each site)
--------------------------------------------------------------------------------------
  * **DA** is the Track A desk gate over the token-mode dumps and is deliberately NOT computed
    here; this evaluator emits the dump inventory V6 needs and nothing else. Its verdict can be
    supplied with ``--da`` (a JSON carrying ``{"verdict": ..., "transfers": bool}``) so the two
    branch steps that depend on it can fire. Without it they are reported ``deferred``, and a V1
    failure cannot be attributed between PREFILTER-NOT-FREE and THRESHOLDS-DO-NOT-TRANSFER — the
    chain then emits UNMATCHED-COMBINATION rather than guessing, per the note's instruction that
    an unmatched combination is never allowed to fall through to a pass.
  * **V3's "at least 3 nodes at t3"**: the JSON records no node count, so it is reconstructed as
    (grow decisions at t0..t3) minus (accepted merge ops in passes recorded at a task before t3).
    That is exactly what `consolidate_nodes`' own `len(nodes) > 2` guard sees.
  * **V4's parameter-curve clause**: a fall between consecutive tasks is "accounted for" when the
    pass recorded at the EARLIER task contains at least one accepted `merge` or `truncate` op.
    The JSON does not record a per-op parameter delta, so an exact arithmetic reconciliation is
    not available from this schema; an unexplained fall (a drop with no accepted op behind it) is
    the failure the clause is for, and that IS decidable.
  * **R2** (consolidation wall time per pass, cca vs cka) is reported from a `seconds` key on
    each pass record when the runner writes one. No runner writes it today, so R2 reports
    ``not-recorded`` rather than a fabricated zero.
  * **R1** uses the note's reading, not the shipped evaluator's: PROXY-SUFFICES fires below 20 %
    differing positions, not only at exactly zero.
  * A gate whose inputs are missing is ``not-run`` and never silently ``pass``; the branch chain
    treats a ``not-run`` gate as unmatched rather than as satisfied.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_provisional as h8       # noqa: E402  (loaders + shared helpers; H8's gates untouched)

CTRL_STREAMS = h8.CTRL_STREAMS
T1, T2, T3, T4 = 1, 2, 3, 4
GATED_TASKS = (1, 2, 3, 4)

# --- thresholds, taken verbatim from the note's gate table ---------------------------------
V2A_MIN_IMPROVE = 0.03            # V2(a) mean paired t3 regret improvement
V2C_AA_TOL = 0.005                # V2(c) average_accuracy_final >= off - this
V3_MIN_MERGE_FRAC = 0.70          # V3 merged in >= 70 % of seeds where the object exists
V3_MIN_MERGE_SEEDS = 6            # ... and >= 6 seeds absolutely
V3_MIN_OBJECT_SEEDS = 8           # ... and the object must exist in >= 8/10 seeds
V3_MAX_OBJECT_ABSENT = 2          # NO-MERGE-OBJECT fires when absent in > 2/10
V5A_POST_TOL = -0.01              # V5(a) seed-mean delta after the first deferral
V5B_MEAN_TOL = -0.02              # V5(b) mean delta at the deferred revisit
V5B_DOWN_BY = 0.02                # V5(b) "down by more than"
V5B_MAX_DOWN = 5                  # ... in at most this many of the runs
V6_EXPECTED_DUMPS = 20            # the `always` arm's 20 CTrL runs
V1_CKA_THRESHOLD = 0.55           # the pre-filter threshold under counterfactual test
V1_MIN_REMOVED_FRAC = 0.70        # >= 70 % of backward-vetoed pairs removed
R1_PROXY_SUFFICES_FRAC = 0.20     # < 20 % differing positions = the proxy suffices

# --- the pre-registered run table, as PAIR EXPECTATIONS -------------------------------------
# A null gate that compares only the pairs that happen to be on disk certifies nothing: a
# missing run is not a mismatch, so the gate passes on a subset and --preflight exits 0 without
# ever having looked at the pairs it exists for (v2 review, blocker 3). Every identity gate is
# therefore told what it MUST compare, and is short-changed, not satisfied, when a pair is absent.
V0B_CTRL_SEEDS = (42, 43, 44, 45, 46)          # x 4 streams = 20 pairs
V0B_5DS_SEEDS = (42, 43, 44)                   # 3 pairs
V0B_INT_SEEDS = (42,)                          # 1 pair  -> 24 in total
#: The V0 pre-flight (judge change 1): 8 runs = 5-Datasets off+shadow seeds 42-44 and
#: s_interleave off+shadow seed 42, i.e. 4 shadow-vs-off pairs, plus whatever V0a references
#: exist for those same runs. Pod A is not created until these pass.
PREFLIGHT_5DS_SEEDS = V0B_5DS_SEEDS
PREFLIGHT_INT_SEEDS = V0B_INT_SEEDS

#: The note's "Bit-identity, operationally" field list. Present-on-one-side-only is a failure.
BIT_IDENTITY_FIELDS = ("average_accuracy", "test_accs", "param_curve", "param_curve_total")
BIT_IDENTITY_OPTIONAL = ("test_accs_final", "average_accuracy_final")


# ---------------------------------------------------------------------------
# Bit-identity, operationally
# ---------------------------------------------------------------------------

def compare_runs(a: Optional[Dict], b: Optional[Dict], seed: int, stream: Optional[str],
                 label_a: str, label_b: str) -> List[Dict]:
    """Field-by-field mismatches between two runs the note requires to be bit-identical.

    Ordered decision list, `average_accuracy`, every `L_*` key in every gate record, `test_accs`,
    `param_curve`, `param_curve_total`, and — when present on BOTH sides — `test_accs_final` and
    `average_accuracy_final`. Floats compare by exact equality on the serialised values. A field
    present on one side and absent on the other is a failure, not a skip.
    """
    if a is None or b is None:
        return []
    out: List[Dict] = []

    def note(field, va, vb, task=None, detail=None):
        row = {"seed": seed, "stream": stream, "task": task, "field": field,
               label_a: va, label_b: vb}
        if detail:
            row["detail"] = detail
        out.append(row)

    for f in BIT_IDENTITY_FIELDS:
        pa, pb = f in a, f in b
        if pa != pb:
            note(f, a.get(f), b.get(f), detail="present on one side only")
            continue
        if not pa:
            continue
        va, vb = a[f], b[f]
        if isinstance(va, list) and isinstance(vb, list):
            if len(va) != len(vb):
                note(f, va, vb, detail="length differs")
            else:
                for i, (x, y) in enumerate(zip(va, vb)):
                    if x != y:
                        note(f, x, y, task=i)
                        break
        elif va != vb:
            note(f, va, vb)

    for f in BIT_IDENTITY_OPTIONAL:
        if f not in a or f not in b:
            continue                      # the note's explicit "when present on both sides"
        va, vb = a[f], b[f]
        if isinstance(va, list) and isinstance(vb, list):
            if len(va) != len(vb):
                note(f, va, vb, detail="length differs")
            else:
                for i, (x, y) in enumerate(zip(va, vb)):
                    if x != y:
                        note(f, x, y, task=i)
                        break
        elif va != vb:
            note(f, va, vb)

    da, db = a.get("decisions", []), b.get("decisions", [])
    if len(da) != len(db):
        note("decisions", len(da), len(db), detail="decision list length differs")
    for x in da:
        t = x.get("task")
        y = h8.dec_at(b, t)
        if y is None:
            note("decisions", x.get("decision"), None, task=t, detail="task missing on one side")
            continue
        if x.get("decision") != y.get("decision"):
            note("decision", x.get("decision"), y.get("decision"), task=t)
        for k in sorted({k for k in list(x) + list(y) if k.startswith("L_")}):
            if (k in x) != (k in y):
                note(k, x.get(k), y.get(k), task=t, detail="present on one side only")
            elif x.get(k) != y.get(k):
                note(k, x.get(k), y.get(k), task=t)
    return out


def _identity_gate(gate_id: str, observable: str, pairs: Sequence[Tuple],
                   label_a: str, label_b: str,
                   expected: Optional[Set[Tuple[Optional[str], int]]] = None) -> Dict:
    """`pairs` is a sequence of (seed, stream, run_a, run_b).

    `expected` is the set of (stream, seed) keys this gate MUST compare. A key with no usable
    pair is `incomplete`, which fails the gate and makes the run INCONCLUSIVE: a null gate that
    silently narrows to whatever is on disk can certify a null it never checked.
    """
    gate = {"gate": gate_id, "observable": observable,
            "threshold": ("decisions, average_accuracy, every L_*, test_accs, param_curve, "
                          "param_curve_total (+ the *_final fields when both carry them) "
                          f"bit-identical between {label_a} and {label_b}"
                          + (f", over all {len(expected)} pre-registered pairs"
                             if expected else "")),
            "value": None, "verdict": "not-run"}
    usable = [(s, st, ra, rb) for (s, st, ra, rb) in pairs if ra is not None and rb is not None]
    compared = {(st, s) for (s, st, _a, _b) in usable}
    missing = sorted(((st or "", s) for (st, s) in (set(expected or ()) - compared)))
    missing_rows = [{"stream": st or None, "seed": s} for st, s in missing]

    mismatches: List[Dict] = []
    for seed, stream, ra, rb in usable:
        mismatches.extend(compare_runs(ra, rb, seed, stream, label_a, label_b))
    gate["value"] = {"n_runs_compared": len(usable),
                     "n_expected": (len(expected) if expected is not None else None),
                     "missing_pairs": missing_rows,
                     "n_mismatches": len(mismatches),
                     "first_mismatch": mismatches[0] if mismatches else None,
                     "all_mismatches": mismatches}
    if not usable and not expected:
        gate["value"] = None
        gate["note"] = f"no matched runs between {label_a} and {label_b}"
        return gate
    if missing_rows:
        gate["incomplete"] = True
        gate["verdict"] = "fail"
        gate["note"] = (f"INCOMPLETE: {len(missing_rows)} of {len(expected)} pre-registered "
                        f"pairs absent — this gate cannot certify a null it never compared")
        gate["headline"] = (f"{len(usable)}/{len(expected)} pairs compared, "
                            f"{len(mismatches)} mismatches; missing {missing_rows}")
        return gate
    gate["verdict"] = "pass" if not mismatches else "fail"
    gate["headline"] = (f"{len(usable)} runs bit-identical" if not mismatches else
                        f"{len(mismatches)} mismatches over {len(usable)} runs "
                        f"(first: {mismatches[0]})")
    return gate


def _ctrl_pairs(a: Dict[int, Dict[str, Dict]], b: Dict[int, Dict[str, Dict]]) -> List[Tuple]:
    out = []
    for seed in sorted(set(a) | set(b)):
        for stream in CTRL_STREAMS:
            ra, rb = a.get(seed, {}).get(stream), b.get(seed, {}).get(stream)
            if ra is not None and rb is not None:
                out.append((seed, stream, ra, rb))
    return out


def _single_pairs(a: Dict[int, Dict], b: Dict[int, Dict], stream: Optional[str]) -> List[Tuple]:
    return [(seed, stream, a[seed], b[seed]) for seed in sorted(set(a) & set(b))]


def reference_keys_ctrl(tree: Dict[int, Dict[str, Dict]]) -> Set[Tuple[Optional[str], int]]:
    """Every (stream, seed) a REFERENCE tree carries — what a V0a gate must reproduce."""
    return {(st, s) for s in tree for st in CTRL_STREAMS if tree[s].get(st) is not None}


def reference_keys_single(tree: Dict[int, Dict],
                          stream: Optional[str]) -> Set[Tuple[Optional[str], int]]:
    return {(stream, s) for s in tree}


def gate_v0a_i(off_ctrl, off_5ds, base_ctrl, base_5ds, expected=None) -> Dict:
    return _identity_gate(
        "V0a-i",
        "arm off on CTrL + 5-Datasets vs the attn-root adoption baseline",
        _ctrl_pairs(base_ctrl, off_ctrl) + _single_pairs(base_5ds, off_5ds, None),
        "baseline(attn_pool adoption run)", "arm off", expected=expected)


def gate_v0a_ii(off_ctrl, off_int, off_5ds, h8_ctrl, h8_int, h8_5ds, expected=None) -> Dict:
    return _identity_gate(
        "V0a-ii",
        "arm off on ALL THREE streams (s_interleave included) vs the archived H8 off arms — "
        "s_interleave has no published baseline, so this is that stream's only null",
        (_ctrl_pairs(h8_ctrl, off_ctrl)
         + _single_pairs(h8_int, off_int, "s_interleave")
         + _single_pairs(h8_5ds, off_5ds, None)),
        "H8 archive arm off", "arm off", expected=expected)


def gate_v0b(off_ctrl, off_int, off_5ds, sh_ctrl, sh_int, sh_5ds, expected=None) -> Dict:
    return _identity_gate(
        "V0b",
        "arm shadow vs arm off — CTrL (20 pairs), 5-Datasets seeds 42-44 (3), s_interleave "
        "seed 42 (1): the shadow refit computes the third state and must act on nothing, "
        "INCLUDING on 5-Datasets, the pair H8 failed",
        (_ctrl_pairs(off_ctrl, sh_ctrl)
         + _single_pairs(off_int, sh_int, "s_interleave")
         + _single_pairs(off_5ds, sh_5ds, None)),
        "arm off", "arm shadow", expected=expected)


# ---------------------------------------------------------------------------
# Shared readers over the post-consolidation fields and the pass records
# ---------------------------------------------------------------------------

def _regret_final(run: Optional[Dict], task: int) -> Tuple[Optional[float], bool]:
    """(max(oracle_accs) - the chosen rung's POST-consolidation accuracy, legacy_fallback)."""
    d = h8.dec_at(run, task)
    oracle = d.get("oracle_accs") if d else None
    if not oracle:
        return None, False
    accs, legacy = h8.final_accs(run)
    if not accs or len(accs) <= task:
        return None, legacy
    return max(oracle.values()) - accs[task], legacy


def _passes(run: Dict) -> Tuple[List[Dict], bool]:
    """(every consolidation pass, oldest first, legacy_fallback). A legacy JSON carries only the
    final pass, which is why V4's provenance clause cannot be evaluated on one."""
    ps = run.get("consolidation_passes")
    if ps:
        return list(ps), False
    final = run.get("consolidation")
    return ([final] if final else []), True


def _ops_with_pass(run: Dict) -> List[Tuple[Optional[int], Dict]]:
    ps, _ = _passes(run)
    return [(p.get("at_task"), op) for p in ps for op in (p.get("ops") or [])]


def _accepted_merges(run: Dict) -> List[Dict]:
    return [op for _, op in _ops_with_pass(run) if op.get("op") == "merge"]


def _deferred_tasks(run: Dict) -> List[int]:
    return sorted(d["task"] for d in run.get("decisions", []) if d.get("provisional") is True)


# ---------------------------------------------------------------------------
# V2 — treatment effect (evalue vs off, CTrL)
# ---------------------------------------------------------------------------

def gate_v2(off_ctrl, evalue_ctrl, always_ctrl) -> Dict:
    gate = {"gate": "V2",
            "observable": "(a) mean paired t3 regret improvement over the 20 (seed, stream) runs; "
                          "(b) evalue's AA gain per 100k added parameters vs always's; "
                          "(c) average_accuracy_final vs off",
            "threshold": f"(a) >= {V2A_MIN_IMPROVE}; (b) ratio_evalue >= ratio_always; "
                         f"(c) >= off - {V2C_AA_TOL}",
            "value": None, "verdict": "not-run"}
    if not off_ctrl or not evalue_ctrl:
        gate["note"] = "arm off and/or arm evalue CTrL data missing"
        return gate
    pairs = h8.paired_ctrl(off_ctrl, evalue_ctrl)
    legacy = False

    # (a) paired t3 regret improvement, on the post-consolidation accuracies.
    rows_a, diffs_a = [], []
    for seed, stream, off_r, ev_r in pairs:
        (r_off, lo), (r_ev, le) = _regret_final(off_r, T3), _regret_final(ev_r, T3)
        legacy = legacy or lo or le
        if r_off is None or r_ev is None:
            continue
        improve = r_off - r_ev
        diffs_a.append(improve)
        rows_a.append({"seed": seed, "stream": stream, "off_regret": r_off,
                       "evalue_regret": r_ev, "diff": improve})
    mean_a = h8._mean(diffs_a)
    a = {"rows": rows_a, "n_runs": len(diffs_a), "mean_improvement": mean_a,
         "seed_level": h8._seed_level(rows_a),
         "verdict": "not-run" if mean_a is None else
                    ("pass" if mean_a >= V2A_MIN_IMPROVE else "fail")}

    # (b) accuracy per 100k added parameters, aggregate-then-ratio over the common runs.
    b: Dict = {"verdict": "not-run", "note": "arm always CTrL data missing"}
    if always_ctrl:
        common = [(s, st) for s in sorted(off_ctrl) for st in CTRL_STREAMS
                  if off_ctrl.get(s, {}).get(st) is not None
                  and evalue_ctrl.get(s, {}).get(st) is not None
                  and always_ctrl.get(s, {}).get(st) is not None]

        def _aa(tree):
            vals = []
            for s, st in common:
                v, lg = h8.final_aa(tree[s][st])
                vals.append(v)
            return h8._mean(vals)

        def _pp(tree):
            return h8._mean([tree[s][st].get("params_total_pre_consolidation")
                             for s, st in common])

        aa_off, aa_ev, aa_al = _aa(off_ctrl), _aa(evalue_ctrl), _aa(always_ctrl)
        p_off, p_ev, p_al = _pp(off_ctrl), _pp(evalue_ctrl), _pp(always_ctrl)
        for s, st in common:
            legacy = legacy or h8.final_aa(off_ctrl[s][st])[1] or h8.final_aa(evalue_ctrl[s][st])[1]

        def _ratio(aa_arm, p_arm):
            if None in (aa_arm, aa_off, p_arm, p_off):
                return None
            denom = (p_arm - p_off) / 1e5
            return None if denom == 0 else (aa_arm - aa_off) / denom

        r_ev, r_al = _ratio(aa_ev, p_ev), _ratio(aa_al, p_al)
        b = {"n_common_runs": len(common), "mean_aa_off": aa_off, "mean_aa_evalue": aa_ev,
             "mean_aa_always": aa_al, "mean_params_off": p_off, "mean_params_evalue": p_ev,
             "mean_params_always": p_al, "ratio_evalue": r_ev, "ratio_always": r_al,
             "verdict": "not-run" if (r_ev is None or r_al is None) else
                        ("pass" if r_ev >= r_al else "fail")}

    # (c) collateral on the mean: average_accuracy_final >= off - 0.005.
    rows_c, diffs_c = [], []
    for seed, stream, off_r, ev_r in pairs:
        (aa_off, lo), (aa_ev, le) = h8.final_aa(off_r), h8.final_aa(ev_r)
        legacy = legacy or lo or le
        if aa_off is None or aa_ev is None:
            continue
        diffs_c.append(aa_ev - aa_off)
        rows_c.append({"seed": seed, "stream": stream, "off_aa": aa_off, "evalue_aa": aa_ev,
                       "diff": aa_ev - aa_off})
    mean_c = h8._mean(diffs_c)
    c = {"rows": rows_c, "mean_aa_diff": mean_c, "seed_level": h8._seed_level(rows_c),
         "verdict": "not-run" if mean_c is None else
                    ("pass" if mean_c >= -V2C_AA_TOL else "fail")}

    verdicts = [a["verdict"], b["verdict"], c["verdict"]]
    gate["value"] = {"a_regret_improvement": a, "b_accuracy_per_parameter": b,
                     "c_collateral_aa": c}
    gate["verdict"] = ("fail" if any(v == "fail" for v in verdicts) else
                       "pass" if all(v == "pass" for v in verdicts) else "not-run")
    if legacy:
        gate["legacy_accs"] = h8.LEGACY_ACCS_NOTE
    gate["headline"] = (f"a={a['verdict']}(mean={mean_a}), b={b['verdict']}, "
                        f"c={c['verdict']}(AA diff={mean_c})")
    return gate


# ---------------------------------------------------------------------------
# V3 — the merge object exists and is found (s_interleave, arm evalue)
# ---------------------------------------------------------------------------

def _v3_object(run: Dict) -> Dict:
    """Does the merge OBJECT exist in this run: a root at t1, a provisional root at t3, and at
    least 3 nodes at t3 (below that `consolidate_nodes` is skipped entirely)?"""
    decisions = run.get("decisions", [])
    d1, d3 = h8.dec_at(run, T1), h8.dec_at(run, T3)
    root_at_t1 = bool(d1 and d1.get("decision") == "grow")
    provisional_at_t3 = bool(d3 and d3.get("provisional") is True)
    grows = sum(1 for d in decisions if d.get("task") is not None and d["task"] <= T3
                and d.get("decision") == "grow")
    merged_before_t3 = sum(1 for at, op in _ops_with_pass(run)
                           if op.get("op") == "merge" and at is not None and at < T3)
    n_nodes_at_t3 = grows - merged_before_t3
    return {"root_at_t1": root_at_t1, "provisional_at_t3": provisional_at_t3,
            "n_nodes_at_t3": n_nodes_at_t3,
            "exists": bool(root_at_t1 and provisional_at_t3 and n_nodes_at_t3 >= 3)}


def gate_v3(int_evalue: Dict[int, Dict]) -> Dict:
    gate = {"gate": "V3",
            "observable": "s_interleave arm evalue, from consolidation_passes: the t3 provisional "
                          "root merged into the t1 root, among the seeds where the OBJECT exists "
                          "(root at t1, provisional root at t3, >= 3 nodes at t3)",
            "threshold": f">= {V3_MIN_MERGE_FRAC:.0%} of object-present seeds and >= "
                         f"{V3_MIN_MERGE_SEEDS} seeds absolutely; no accepted op with "
                         f"{{keep,drop}} == {{1,2}} or {{2,3}}; object present in >= "
                         f"{V3_MIN_OBJECT_SEEDS}/10 seeds",
            "value": None, "verdict": "not-run"}
    if not int_evalue:
        gate["note"] = "s_interleave arm evalue data missing"
        return gate

    rows, negative_hits = [], []
    for seed in sorted(int_evalue):
        run = int_evalue[seed]
        obj = _v3_object(run)
        merges = _accepted_merges(run)
        t3_into_t1 = any(op.get("drop") == T3 and op.get("keep") == T1 for op in merges)
        for op in merges:
            pair = {op.get("keep"), op.get("drop")}
            if pair in ({T1, T2}, {T2, T3}):
                negative_hits.append({"seed": seed, "keep": op.get("keep"),
                                      "drop": op.get("drop")})
        rows.append({"seed": seed, **obj, "t3_merged_into_t1": t3_into_t1,
                     "n_accepted_merges": len(merges),
                     "legacy": _passes(run)[1]})

    n_seeds = len(rows)
    present = [r for r in rows if r["exists"]]
    n_present, n_absent = len(present), n_seeds - len([r for r in rows if r["exists"]])
    n_merged = sum(1 for r in present if r["t3_merged_into_t1"])
    frac = (n_merged / n_present) if n_present else None
    object_ok = n_present >= V3_MIN_OBJECT_SEEDS
    rate_ok = (frac is not None and frac >= V3_MIN_MERGE_FRAC
               and n_merged >= V3_MIN_MERGE_SEEDS)
    negative_ok = not negative_hits

    gate["value"] = {"rows": rows, "n_seeds": n_seeds, "n_object_present": n_present,
                     "n_object_absent": n_absent, "object_present_ok": object_ok,
                     "n_merged": n_merged, "frac_merged": frac, "rate_ok": rate_ok,
                     "negative_control_hits": negative_hits, "negative_control_ok": negative_ok,
                     # Which branch a failure selects: the object being absent is a different
                     # finding (the gate decided, or the pass never ran) from the object being
                     # there and the merge not firing.
                     "object_absent_exceeds_budget": n_absent > V3_MAX_OBJECT_ABSENT}
    gate["verdict"] = "pass" if (object_ok and rate_ok and negative_ok) else "fail"
    gate["headline"] = (f"object {n_present}/{n_seeds}, merged {n_merged}/{n_present}"
                        f" (frac={frac}), negative-control hits={len(negative_hits)}")
    if any(r["legacy"] for r in rows):
        gate["note"] = ("legacy JSON among the runs: only the final pass is recorded, so an "
                        "intermediate-pass merge is invisible and this gate understates")
    return gate


# ---------------------------------------------------------------------------
# V4 — resolution provenance and the parameter curves
# ---------------------------------------------------------------------------

def _v4_run(seed: int, stream: Optional[str], run: Dict) -> Dict:
    ops = _ops_with_pass(run)
    passes, legacy = _passes(run)
    accepted_by_drop: Dict[int, int] = {}
    for _, op in ops:
        if op.get("op") == "merge":
            accepted_by_drop[op.get("drop")] = accepted_by_drop.get(op.get("drop"), 0) + 1

    roots = run.get("provisional_roots", []) or []
    unexplained = [r for r in roots if r.get("resolution") == "removed_unexplained"]
    unresolved = [r for r in roots if r.get("resolution") is None]
    # A root the clock crystallised mid-stream is an ordinary root again and a LATER pass may
    # still merge it: its resolution stays `timeout` (that is what happened to the provisional
    # flag) but it DOES produce an accepted op naming it as `drop`, recorded on the record as
    # `merged_after_crystallisation`. Both kinds must map to exactly one accepted op, or the two
    # bookkeeping paths stop reconciling on a correct run (v2 review, blocker 2).
    def _expects_an_op(r: Dict) -> bool:
        return (r.get("resolution") == "merge"
                or r.get("merged_after_crystallisation") is not None)

    bad_merge_map = [
        {"minted_at": r.get("minted_at"), "resolution": r.get("resolution"),
         "merged_after_crystallisation": r.get("merged_after_crystallisation"),
         "n_accepted_ops_naming_it": accepted_by_drop.get(r.get("minted_at"), 0)}
        for r in roots
        if _expects_an_op(r) and accepted_by_drop.get(r.get("minted_at"), 0) != 1
    ]

    # Parameter curves: every FALL between consecutive tasks needs an accepted merge/truncate in
    # the pass recorded at the earlier task.
    accepted_at: Dict[Optional[int], int] = {}
    for at, op in ops:
        if op.get("op") in ("merge", "truncate"):
            accepted_at[at] = accepted_at.get(at, 0) + 1
    curve_falls = []
    for name in ("param_curve", "param_curve_total"):
        curve = run.get(name) or []
        for t in range(len(curve) - 1):
            if curve[t + 1] < curve[t]:
                explained = accepted_at.get(t, 0) > 0
                curve_falls.append({"curve": name, "at_task": t, "before": curve[t],
                                    "after": curve[t + 1], "explained": explained})
    unexplained_falls = [f for f in curve_falls if not f["explained"]]

    return {"seed": seed, "stream": stream, "legacy": legacy, "n_passes": len(passes),
            "n_provisional_roots": len(roots),
            "removed_unexplained": len(unexplained), "unresolved": len(unresolved),
            "merge_resolutions_without_exactly_one_op": bad_merge_map,
            "curve_falls": curve_falls, "unexplained_curve_falls": unexplained_falls,
            "ok": not (unexplained or unresolved or bad_merge_map or unexplained_falls)}


def gate_v4(all_runs: List[Tuple[int, Optional[str], Dict]]) -> Dict:
    gate = {"gate": "V4",
            "observable": "every provisional root's 'merge' resolution maps to exactly one "
                          "accepted op naming it as drop; zero removed_unexplained; every fall "
                          "in param_curve AND param_curve_total is covered by an accepted "
                          "merge/truncate in the pass recorded at the earlier task",
            "threshold": "zero violations of any clause", "value": None, "verdict": "not-run"}
    if not all_runs:
        gate["note"] = "no runs supplied"
        return gate
    rows = [_v4_run(seed, stream, run) for seed, stream, run in all_runs]
    modern = [r for r in rows if not r["legacy"]]
    bad = [r for r in rows if not r["ok"]]
    gate["value"] = {"n_runs": len(rows), "n_legacy": len(rows) - len(modern),
                     "n_violating_runs": len(bad), "violations": bad, "rows": rows}
    gate["verdict"] = "pass" if not bad else "fail"
    gate["headline"] = f"{len(bad)}/{len(rows)} runs violate provenance or the parameter curves"
    if len(modern) != len(rows):
        gate["note"] = ("legacy JSON among the runs: only the final pass is recorded, so an "
                        "intermediate-pass merge cannot be mapped and this gate is not "
                        "decisive on those runs")
    return gate


# ---------------------------------------------------------------------------
# V5 — deferral cost
# ---------------------------------------------------------------------------

def _first_deferral(run: Dict) -> Optional[int]:
    tasks = _deferred_tasks(run)
    return tasks[0] if tasks else None


def gate_v5(pairs: List[Tuple[int, Optional[str], Dict, Dict]]) -> Dict:
    """`pairs` is (seed, stream, off_run, evalue_run) over CTrL + s_interleave."""
    gate = {"gate": "V5",
            "observable": "(a) per-task deltas, split at the run's FIRST deferral: EXACT "
                          "identity on `test_accs` before it, seed-mean on `test_accs_final` "
                          "after it; "
                          "(b) the deferred reuse-must-win revisit (CTrL t4, revisit_of == 0); "
                          "(c) rolled-back reader_audit entries, counted",
            "threshold": f"(a) before: exact; after: seed-mean >= {V5A_POST_TOL}; "
                         f"(b) mean >= {V5B_MEAN_TOL} and at most {V5B_MAX_DOWN} runs down by "
                         f"more than {V5B_DOWN_BY}; (c) descriptive",
            "value": None, "verdict": "not-run"}
    if not pairs:
        gate["note"] = "no paired off/evalue runs supplied"
        return gate

    legacy = False
    before_violations, after_rows = [], []
    b_rows, rolled_back = [], []
    for seed, stream, off_r, ev_r in pairs:
        (off_accs, lo), (ev_accs, le) = h8.final_accs(off_r), h8.final_accs(ev_r)
        legacy = legacy or lo or le
        off_accs, ev_accs = off_accs or [], ev_accs or []
        # The EXACT-identity clause is read on `test_accs`, not `test_accs_final`. Its whole
        # justification is "the arms share every RNG draw until the first deferral", which is an
        # argument about the accuracy captured inside the task loop — the final consolidation
        # pass runs afterwards and may legitimately move `test_accs_final` on one arm and not the
        # other, so demanding exact equality there reports a correct run as a leak and fires
        # DEFERRAL-COSTS (v2 review, should-fix 4; the note is amended to match). The
        # post-deferral clause stays on `test_accs_final`, which is the DAG the run ends with.
        off_pre = off_r.get("test_accs") or []
        ev_pre = ev_r.get("test_accs") or []
        first = _first_deferral(ev_r)
        deferred = set(_deferred_tasks(ev_r))
        n_tasks = max(len(off_accs), len(ev_accs), len(off_pre), len(ev_pre))
        for t in range(n_tasks):
            if t in deferred:
                continue                       # a deferred position is not "collateral"
            if first is None or t < first:
                if t >= len(off_pre) or t >= len(ev_pre):
                    continue
                oa, ea = off_pre[t], ev_pre[t]
                # Before the first deferral the two arms share every RNG draw, so a nonzero
                # delta is a LEAK, not a cost — reported as such.
                if oa != ea:
                    before_violations.append({"seed": seed, "stream": stream, "task": t,
                                              "field": "test_accs",
                                              "off": oa, "evalue": ea, "delta": ea - oa})
            else:
                if t >= len(off_accs) or t >= len(ev_accs):
                    continue
                oa, ea = off_accs[t], ev_accs[t]
                after_rows.append({"seed": seed, "stream": stream, "task": t,
                                   "field": "test_accs_final",
                                   "off": oa, "evalue": ea, "diff": ea - oa})

        # (b) the deferred revisit: CTrL t4 with ctrl.revisit_of == 0, and only runs that defer
        # there.
        gt = ev_r.get("ctrl_ground_truth") or []
        revisit = (len(gt) > T4 and isinstance(gt[T4], dict)
                   and gt[T4].get("revisit_of") == 0)
        if revisit and T4 in deferred and len(off_accs) > T4 and len(ev_accs) > T4:
            delta = ev_accs[T4] - off_accs[T4]
            b_rows.append({"seed": seed, "stream": stream, "off": off_accs[T4],
                           "evalue": ev_accs[T4], "diff": delta,
                           "down_by_more_than_tol": delta < -V5B_DOWN_BY})

        for entry in (ev_r.get("reader_audit") or []):
            if entry.get("verdict") == "rolled_back":
                rolled_back.append({"seed": seed, "stream": stream, **entry})

    # (a) after the first deferral: seed-mean per task, with the `off` across-seed SD as floor.
    by_task: Dict[int, List[Dict]] = {}
    for r in after_rows:
        by_task.setdefault(r["task"], []).append(r)
    after_summary, a_violations = [], []
    for t in sorted(by_task):
        rows_t = by_task[t]
        seed_level = h8._seed_level(rows_t)
        noise_floor = h8._stdev([r["off"] for r in rows_t])
        entry = {"task": t, "n_runs": len(rows_t), "seed_level": seed_level,
                 "off_across_seed_sd": noise_floor, "rows": rows_t}
        after_summary.append(entry)
        if seed_level.get("mean") is not None and seed_level["mean"] < V5A_POST_TOL:
            a_violations.append({"task": t, "seed_mean": seed_level["mean"]})
    a_verdict = "fail" if (before_violations or a_violations) else (
        "pass" if (after_summary or pairs) else "not-run")
    a = {"before_first_deferral_violations": before_violations,
         "after_first_deferral": after_summary, "after_violations": a_violations,
         "verdict": a_verdict}

    mean_b = h8._mean([r["diff"] for r in b_rows])
    n_down = sum(1 for r in b_rows if r["down_by_more_than_tol"])
    b_verdict = "not-run" if not b_rows else (
        "pass" if (mean_b >= V5B_MEAN_TOL and n_down <= V5B_MAX_DOWN) else "fail")
    b = {"rows": b_rows, "n": len(b_rows), "mean_delta": mean_b,
         "n_down_by_more_than_tol": n_down, "verdict": b_verdict}

    c = {"n_rolled_back": len(rolled_back), "entries": rolled_back, "verdict": "descriptive"}

    gate["value"] = {"a_non_deferred_positions": a, "b_deferred_revisit": b,
                     "c_rolled_back_audit_entries": c}
    gate["verdict"] = ("fail" if "fail" in (a["verdict"], b["verdict"]) else
                       "pass" if a["verdict"] == "pass" and b["verdict"] in ("pass", "not-run")
                       else "not-run")
    if legacy:
        gate["legacy_accs"] = h8.LEGACY_ACCS_NOTE
    gate["headline"] = (f"a={a['verdict']}(leaks={len(before_violations)}, "
                        f"post-violations={len(a_violations)}), "
                        f"b={b['verdict']}(n={len(b_rows)}, mean={mean_b}, down={n_down}), "
                        f"rolled_back={len(rolled_back)}")
    return gate


# ---------------------------------------------------------------------------
# V6 — roots resolved, dumps written
# ---------------------------------------------------------------------------

def _dump_inventory(dumps_dir: Optional[str]) -> Dict:
    """Every gate_dump.pt under `dumps_dir`, with `roots_at_mint` completeness when torch is
    importable. The DA gate itself belongs to the Track A desk script; this is only the
    inventory V6 reads and DA consumes."""
    inv = {"dir": dumps_dir, "files": [], "n_files": 0, "inspected": False,
           "note": None}
    if not dumps_dir or not os.path.isdir(dumps_dir):
        inv["note"] = "no --dumps directory supplied or it does not exist"
        return inv
    paths = []
    for root, _dirs, files in os.walk(dumps_dir):
        for name in sorted(files):
            if name == "gate_dump.pt":
                paths.append(os.path.join(root, name))
    inv["n_files"] = len(paths)
    try:
        import torch                                    # noqa: WPS433 (optional dependency)
    except ImportError:
        inv["files"] = [{"path": p} for p in sorted(paths)]
        inv["note"] = "torch not importable: inventory lists paths only, contents unverified"
        return inv
    inv["inspected"] = True
    required = {"task_id", "node_id", "state_dict", "predictor_kind", "head_state",
                "composer_kind", "composer_state"}
    for p in sorted(paths):
        row = {"path": p}
        try:
            dump = torch.load(p, map_location="cpu")
        except Exception as exc:                        # pragma: no cover - corrupt artifact
            row["error"] = f"{type(exc).__name__}: {exc}"
            inv["files"].append(row)
            continue
        snaps = dump.get("roots_at_mint")
        row["mode"] = dump.get("mode")
        row["has_roots_at_mint"] = snaps is not None
        row["n_roots_at_mint"] = len(snaps) if snaps else 0
        row["n_roots_final"] = len(dump.get("roots") or [])
        row["snapshots_carry_reader"] = bool(
            snaps and all(required <= set(s) for s in snaps))
        inv["files"].append(row)
    return inv


def gate_v6(all_runs: List[Tuple[int, Optional[str], Dict]], inventory: Dict) -> Dict:
    gate = {"gate": "V6",
            "observable": "provisional roots still flagged at stream end (must be zero); and "
                          f"{V6_EXPECTED_DUMPS} token-mode dumps present, each carrying "
                          "roots_at_mint — state dict plus the reader head and composer — for "
                          "every minted root, merged-away roots included",
            "threshold": f"zero unresolved roots; >= {V6_EXPECTED_DUMPS} dumps, all complete",
            "value": None, "verdict": "not-run"}
    if not all_runs:
        gate["note"] = "no runs supplied"
        return gate
    unresolved = []
    n_roots = 0
    for seed, stream, run in all_runs:
        for r in (run.get("provisional_roots") or []):
            n_roots += 1
            if r.get("resolution") is None or r.get("resolved_at") is None:
                unresolved.append({"seed": seed, "stream": stream,
                                   "minted_at": r.get("minted_at"),
                                   "resolution": r.get("resolution")})
    files = inventory.get("files") or []
    complete = [f for f in files if f.get("snapshots_carry_reader")]
    dumps_ok = (inventory.get("n_files", 0) >= V6_EXPECTED_DUMPS
                and (not inventory.get("inspected") or len(complete) == len(files)))
    gate["value"] = {"n_provisional_roots": n_roots, "n_unresolved": len(unresolved),
                     "unresolved": unresolved, "dumps": inventory, "dumps_ok": dumps_ok}
    roots_ok = not unresolved
    if not inventory.get("dir"):
        gate["verdict"] = "not-run" if roots_ok else "fail"
        gate["note"] = "no --dumps directory supplied: the dump clause cannot be evaluated"
    else:
        gate["verdict"] = "pass" if (roots_ok and dumps_ok) else "fail"
    gate["headline"] = (f"{n_roots - len(unresolved)}/{n_roots} roots resolved, "
                        f"{inventory.get('n_files', 0)} dumps "
                        f"({len(complete)} carrying a complete roots_at_mint)")
    return gate


# ---------------------------------------------------------------------------
# V1 / V1b — the CKA pre-filter, offline counterfactual and deployed arm
# ---------------------------------------------------------------------------

_EXAMINED_OPS = ("merge", "merge_rejected", "trigger_rejected")


def _examined_pairs(run: Dict) -> List[Dict]:
    """Every pair the consolidation loop examined, with both statistics, oldest pass first."""
    out = []
    for at, op in _ops_with_pass(run):
        if op.get("op") in _EXAMINED_OPS:
            out.append({"at_task": at, "op": op.get("op"), "keep": op.get("keep"),
                        "drop": op.get("drop"), "cka": op.get("cka"),
                        "cca_topk": op.get("cca_topk"), "similarity": op.get("similarity"),
                        "sim_kind": op.get("sim_kind")})
    return out


def gate_v1(cca_runs: List[Tuple[int, Optional[str], Dict]]) -> Dict:
    gate = {"gate": "V1",
            "observable": "every pair the consolidation loop examined in the evalue CCA arm — "
                          "trigger-passing and trigger-rejected alike — with linear CKA logged "
                          f"for each; the counterfactual of applying CKA >= {V1_CKA_THRESHOLD} "
                          "as a pre-filter",
            "threshold": f"skips ZERO accepted merges, and removes >= "
                         f"{V1_MIN_REMOVED_FRAC:.0%} of the pairs that reached distill_merge "
                         f"and were then rejected by the backward veto (pooled, not per run)",
            "value": None, "verdict": "not-run"}
    if not cca_runs:
        gate["note"] = "no evalue (cca) runs supplied"
        return gate

    examined, missing_cka = 0, 0
    skipped_accepted, rejected_total, rejected_removed = [], 0, 0
    per_run = []
    for seed, stream, run in cca_runs:
        pairs = _examined_pairs(run)
        r_examined = len(pairs)
        r_skipped, r_rej, r_rej_removed = [], 0, 0
        for p in pairs:
            examined += 1
            if p["cka"] is None:
                missing_cka += 1
                continue
            below = p["cka"] < V1_CKA_THRESHOLD
            if p["op"] == "merge" and below:
                row = {"seed": seed, "stream": stream, **p}
                skipped_accepted.append(row)
                r_skipped.append(row)
            if p["op"] == "merge_rejected":
                rejected_total += 1
                r_rej += 1
                if below:
                    rejected_removed += 1
                    r_rej_removed += 1
        per_run.append({"seed": seed, "stream": stream, "n_examined": r_examined,
                        "n_accepted_skipped": len(r_skipped),
                        "n_backward_rejected": r_rej,
                        "n_backward_rejected_removed": r_rej_removed})

    frac_removed = (rejected_removed / rejected_total) if rejected_total else None
    if missing_cka == examined and examined:
        gate["verdict"] = "not-run"
        gate["note"] = ("no examined pair carries a `cka` field: these runs predate the CKA "
                        "pre-filter, so the counterfactual has no denominator")
        gate["value"] = {"n_runs": len(cca_runs), "n_examined_pairs": examined,
                         "n_missing_cka": missing_cka, "per_run": per_run}
        return gate

    free = not skipped_accepted
    removes_enough = frac_removed is not None and frac_removed >= V1_MIN_REMOVED_FRAC
    gate["value"] = {"n_runs": len(cca_runs), "n_examined_pairs": examined,
                     "n_missing_cka": missing_cka,
                     "n_accepted_merges_skipped": len(skipped_accepted),
                     "accepted_merges_skipped": skipped_accepted,
                     "n_backward_rejected": rejected_total,
                     "n_backward_rejected_removed": rejected_removed,
                     "frac_backward_rejected_removed": frac_removed,
                     "is_free": free, "removes_enough": removes_enough, "per_run": per_run}
    gate["verdict"] = ("not-run" if frac_removed is None and not examined else
                       "pass" if (free and removes_enough) else "fail")
    gate["headline"] = (f"{examined} pairs examined; {len(skipped_accepted)} accepted merges "
                        f"would be skipped; {rejected_removed}/{rejected_total} backward-vetoed "
                        f"pairs removed (frac={frac_removed})")
    return gate


def gate_v1b(cca_runs, cka_runs) -> Dict:
    """Confirmatory only — the two arms stop sharing an RNG stream the moment the filter skips a
    pair, so a difference here cannot by itself establish decision-equivalence. Reported side by
    side and never allowed to gate."""
    gate = {"gate": "V1b", "observable": "accepted-merge sets, evalue cka vs evalue cca, paired "
                                         "by (seed, stream)",
            "threshold": "reported side by side; never gates", "value": None,
            "verdict": "descriptive"}
    index = {(s, st): run for s, st, run in cka_runs}
    rows = []
    for seed, stream, run in cca_runs:
        other = index.get((seed, stream))
        if other is None:
            continue
        a = sorted((op.get("keep"), op.get("drop")) for op in _accepted_merges(run))
        b = sorted((op.get("keep"), op.get("drop")) for op in _accepted_merges(other))
        rows.append({"seed": seed, "stream": stream, "cca_accepted": a, "cka_accepted": b,
                     "identical": a == b,
                     "only_in_cca": [p for p in a if p not in b],
                     "only_in_cka": [p for p in b if p not in a]})
    gate["value"] = {"n_pairs": len(rows), "n_identical": sum(1 for r in rows if r["identical"]),
                     "rows": rows}
    gate["headline"] = (f"{gate['value']['n_identical']}/{len(rows)} (seed, stream) pairs agree "
                        f"on the accepted-merge set" if rows else "no paired cca/cka runs")
    if not rows:
        gate["note"] = "the V1b (cka) arm is absent; V1's counterfactual stands alone"
    return gate


# ---------------------------------------------------------------------------
# R1 / R2 / R3 — recorded observables, no failing outcome
# ---------------------------------------------------------------------------

def observable_r1(evalue_ctrl, se_proxy_ctrl) -> Dict:
    rec = {"observable": "R1 PROXY-SUFFICES: fraction of gated positions where se_proxy and "
                         "evalue decide differently",
           "reading": f"< {R1_PROXY_SUFFICES_FRAC:.0%} differing = the proxy suffices (H8's "
                      "reported reading, not the shipped evaluator's zero-difference flag)",
           "value": None, "verdict": "descriptive"}
    if not evalue_ctrl or not se_proxy_ctrl:
        rec["note"] = "arm evalue and/or arm se_proxy CTrL data missing"
        return rec
    rows, n_total, n_diff = [], 0, 0
    for seed in sorted(evalue_ctrl):
        for stream in CTRL_STREAMS:
            ev = evalue_ctrl.get(seed, {}).get(stream)
            se = se_proxy_ctrl.get(seed, {}).get(stream)
            if ev is None or se is None:
                continue
            for t in GATED_TASKS:
                d_ev, d_se = h8.dec_at(ev, t), h8.dec_at(se, t)
                if d_ev is None or d_se is None:
                    continue
                n_total += 1
                differ = d_ev.get("decision") != d_se.get("decision")
                n_diff += bool(differ)
                rows.append({"seed": seed, "stream": stream, "task": t,
                             "evalue": d_ev.get("decision"), "se_proxy": d_se.get("decision"),
                             "differ": differ})
    frac = (n_diff / n_total) if n_total else None
    rec["value"] = {"n_total": n_total, "n_diff": n_diff, "frac_diff": frac,
                    "proxy_suffices": (frac is not None and frac < R1_PROXY_SUFFICES_FRAC),
                    "rows": rows}
    rec["headline"] = (f"{n_diff}/{n_total} positions differ (frac={frac})"
                       + (" [PROXY-SUFFICES]" if rec["value"]["proxy_suffices"] else ""))
    return rec


def observable_r2(cca_runs, cka_runs) -> Dict:
    rec = {"observable": "R2: consolidation wall time per pass, cca vs cka — the pre-filter's "
                         "actual saving",
           "value": None, "verdict": "descriptive"}

    def _secs(runs):
        out = []
        for _s, _st, run in runs:
            for p, _legacy in [(p, None) for p in _passes(run)[0]]:
                v = p.get("seconds")
                if v is not None:
                    out.append(float(v))
        return out

    a, b = _secs(cca_runs), _secs(cka_runs)
    if not a and not b:
        rec["value"] = {"cca": None, "cka": None}
        rec["note"] = ("not-recorded: no runner writes a per-pass `seconds` key, so this "
                       "observable has no data in this archive")
        rec["headline"] = "not-recorded"
        return rec
    rec["value"] = {"cca": {"n": len(a), "mean": h8._mean(a)},
                    "cka": {"n": len(b), "mean": h8._mean(b)}}
    rec["headline"] = f"cca mean={h8._mean(a)}s over {len(a)} passes; cka {h8._mean(b)}s / {len(b)}"
    return rec


def observable_r3() -> Dict:
    return {"observable": "R3: the search_compose device-reseed residual, left unfixed by "
                          "pre-registration",
            "value": {"fixed": False,
                      "consequence": "every arm inherits it, so all seed-level SEs in this loop "
                                     "understate seed variance by an unmeasured amount"},
            "verdict": "descriptive",
            "headline": "search_compose device re-seed NOT fixed (declared, own pre-registration)"}


# ---------------------------------------------------------------------------
# Branch precedence
# ---------------------------------------------------------------------------

def _failed(gate: Dict) -> bool:
    return gate.get("verdict") == "fail"


def determine_branch(gates: Dict[str, Dict], da: Optional[Dict]) -> Dict:
    """The note's chain, first match wins. A combination matching no entry is
    UNMATCHED-COMBINATION with the failing gate list — never INDETERMINATE, and never allowed to
    fall through to a pass."""
    order = ["V0a-i", "V0a-ii", "V0b", "V2", "V3", "V4", "V5", "V6", "V1"]
    verdicts = {k: gates[k]["verdict"] for k in order if k in gates}

    if any(_failed(gates.get(g, {})) for g in ("V0a-i", "V0a-ii", "V0b")):
        return {"branch": "NULL-STILL-BROKEN",
                "reason": "V0a-i, V0a-ii or V0b failed — a code regression, not a hypothesis "
                          "result; later gates are computed for debugging but name no branch.",
                "verdicts": verdicts}

    v2 = gates.get("V2", {})
    v2v = (v2.get("value") or {})
    if _failed(v2):
        if v2v.get("a_regret_improvement", {}).get("verdict") == "fail":
            return {"branch": "REGRET-NOT-REPRODUCED",
                    "reason": "V2(a): the paired t3 regret improvement does not reach 0.03 on "
                              "the corrected RNG stream.", "verdicts": verdicts}
        if v2v.get("b_accuracy_per_parameter", {}).get("verdict") == "fail":
            return {"branch": "NOT-WORTH-THE-PARAMETERS",
                    "reason": "V2(b): evalue buys less accuracy per added parameter than always.",
                    "verdicts": verdicts}
        if v2v.get("c_collateral_aa", {}).get("verdict") == "fail":
            return {"branch": "COLLATERAL-AA",
                    "reason": "V2(c): average_accuracy_final falls more than 0.005 below off.",
                    "verdicts": verdicts}

    v3 = gates.get("V3", {})
    v3v = (v3.get("value") or {})
    if _failed(v3):
        if v3v.get("object_absent_exceeds_budget"):
            return {"branch": "NO-MERGE-OBJECT",
                    "reason": "V3: the merge object is absent in more than 2 of 10 seeds — the "
                              "gate decided, or the consolidation pass never ran.",
                    "verdicts": verdicts}
        return {"branch": "MERGE-NOT-FOUND",
                "reason": "V3: with the object present, the merge rate or the negative control "
                          "fails.", "verdicts": verdicts}

    if _failed(gates.get("V4", {})):
        return {"branch": "PROVENANCE-BROKEN",
                "reason": "V4: a resolution or a parameter-curve fall is unaccounted for.",
                "verdicts": verdicts}

    v5 = gates.get("V5", {})
    if _failed(v5):
        return {"branch": "DEFERRAL-COSTS",
                "reason": "V5(a) or V5(b): deferral costs accuracy at a non-deferred position "
                          "or at the revisit where reuse was right.", "verdicts": verdicts}

    if _failed(gates.get("V6", {})):
        return {"branch": "ROOTS-OR-DUMPS-MISSING",
                "reason": "V6: a provisional root is still flagged at stream end, or the "
                          "token-mode dumps are missing or incomplete.", "verdicts": verdicts}

    # DA belongs to the Track A desk script. Without its verdict the two branches that depend on
    # it cannot fire, and a V1 failure cannot be attributed between them.
    da_verdict = (da or {}).get("verdict")
    da_transfers = (da or {}).get("transfers")
    if da_verdict == "underpowered":
        return {"branch": "DA-UNDERPOWERED",
                "reason": "DA's minimum counts are unmet; V1 is read unconditionally and "
                          "recorded above.", "verdicts": verdicts}
    if da_verdict is not None and da_transfers is False:
        return {"branch": "THRESHOLDS-DO-NOT-TRANSFER",
                "reason": "DA: the CLS backward-delta / CKA threshold does not transfer to "
                          "attention roots.", "verdicts": verdicts}

    if _failed(gates.get("V1", {})):
        if da_verdict is None:
            return {"branch": "UNMATCHED-COMBINATION",
                    "reason": "V1 failed, but DA has not been evaluated (it belongs to the "
                              "Track A desk script; supply it with --da). PREFILTER-NOT-FREE "
                              "may only be named when DA shows the CLS threshold transfers, so "
                              "the failure cannot be attributed yet.",
                    "verdicts": verdicts}
        return {"branch": "PREFILTER-NOT-FREE",
                "reason": "V1: applying CKA >= 0.55 offline skips an accepted merge, or removes "
                          "too few of the backward-vetoed pairs, and DA shows the threshold "
                          "transfers.", "verdicts": verdicts}

    gating = ["V0a-i", "V0a-ii", "V0b", "V2", "V3", "V4", "V5", "V6", "V1"]
    not_run = [g for g in gating if gates.get(g, {}).get("verdict") not in ("pass",)]
    if not_run:
        return {"branch": "UNMATCHED-COMBINATION",
                "reason": f"no gate failed, but these did not pass either: {not_run}. A "
                          f"combination matching no entry is never allowed to fall through to "
                          f"a pass.", "verdicts": verdicts}
    if da_verdict is None:
        return {"branch": "UNMATCHED-COMBINATION",
                "reason": "every evaluable gate passes, but DA (the Track A desk gate over the "
                          "token-mode dumps) has not been evaluated; VALID-AND-REPRODUCES may "
                          "not be named until it is. Supply it with --da.",
                "verdicts": verdicts}
    return {"branch": "VALID-AND-REPRODUCES",
            "reason": "every pre-registered gate passes, DA included.", "verdicts": verdicts}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _flatten_ctrl(tree: Dict[int, Dict[str, Dict]]) -> List[Tuple[int, Optional[str], Dict]]:
    return [(s, st, tree[s][st]) for s in sorted(tree) for st in CTRL_STREAMS
            if tree.get(s, {}).get(st) is not None]


def _flatten_single(tree: Dict[int, Dict], stream: Optional[str]
                    ) -> List[Tuple[int, Optional[str], Dict]]:
    return [(s, stream, tree[s]) for s in sorted(tree)]


def load_all(args) -> Dict:
    return {
        "ctrl": {arm: h8.load_ctrl_arm(args.ctrl, arm)
                 for arm in ("off", "shadow", "evalue", "se_proxy", "always")},
        "int": {arm: h8.load_int_arm(args.interleave, arm)
                for arm in ("off", "shadow", "evalue", "evalue_cka")},
        "5ds": {arm: h8.load_5ds_arm(args.fivedatasets, arm)
                for arm in ("off", "shadow", "evalue")},
        "baseline_ctrl": h8.load_baseline_ctrl(args.baseline),
        "baseline_5ds": h8.load_baseline_5ds(args.baseline),
        "h8_ctrl": h8.load_ctrl_arm(os.path.join(args.h8, "ctrl") if args.h8 else None, "off"),
        "h8_int": h8.load_int_arm(args.h8, "off"),
        "h8_5ds": h8.load_5ds_arm(args.h8, "off"),
    }


def expected_pairs(data: Dict, preflight: bool) -> Dict[str, Set[Tuple[Optional[str], int]]]:
    """What each identity gate MUST compare (v2 review, blocker 3).

    V0b's expectation is the pre-registered run table, so a missing run is short-changed rather
    than quietly excused. V0a's expectations are the REFERENCES' own keys — every archived run
    the new `off` arm has to reproduce — since the references are the authority on what exists.
    In pre-flight mode all three are narrowed to the 8 pre-flight runs (5-Datasets off+shadow
    seeds 42-44, s_interleave off+shadow seed 42), which is exactly the point of judge change 1:
    NULL-STILL-BROKEN must cost 8 runs, not 129, and must not be skippable by an empty directory.
    """
    if preflight:
        v0b = ({(None, s) for s in PREFLIGHT_5DS_SEEDS}
               | {("s_interleave", s) for s in PREFLIGHT_INT_SEEDS})
        ref_i = reference_keys_single(data["baseline_5ds"], None)
        ref_ii = (reference_keys_single(data["h8_5ds"], None)
                  | reference_keys_single(data["h8_int"], "s_interleave"))
        keep = {(None, s) for s in PREFLIGHT_5DS_SEEDS} | {
            ("s_interleave", s) for s in PREFLIGHT_INT_SEEDS}
        return {"V0b": v0b, "V0a-i": ref_i & keep, "V0a-ii": ref_ii & keep}
    return {
        "V0b": ({(st, s) for s in V0B_CTRL_SEEDS for st in CTRL_STREAMS}
                | {(None, s) for s in V0B_5DS_SEEDS}
                | {("s_interleave", s) for s in V0B_INT_SEEDS}),
        "V0a-i": (reference_keys_ctrl(data["baseline_ctrl"])
                  | reference_keys_single(data["baseline_5ds"], None)),
        "V0a-ii": (reference_keys_ctrl(data["h8_ctrl"])
                   | reference_keys_single(data["h8_int"], "s_interleave")
                   | reference_keys_single(data["h8_5ds"], None)),
    }


def evaluate(data: Dict, args) -> Dict:
    gates: Dict[str, Dict] = {}
    exp = expected_pairs(data, args.preflight)
    if args.preflight:
        # Pod B runs first and the pre-flight is about ITS 8 runs; CTrL does not exist yet, and
        # comparing whatever CTrL happens to be on disk would only pad the headline with runs the
        # pre-flight makes no claim about.
        data = dict(data)
        data["ctrl"] = {arm: {} for arm in data["ctrl"]}
        data["baseline_ctrl"] = {}
        data["h8_ctrl"] = {}
    gates["V0a-i"] = gate_v0a_i(data["ctrl"]["off"], data["5ds"]["off"],
                                data["baseline_ctrl"], data["baseline_5ds"],
                                expected=exp["V0a-i"])
    gates["V0a-ii"] = gate_v0a_ii(data["ctrl"]["off"], data["int"]["off"], data["5ds"]["off"],
                                  data["h8_ctrl"], data["h8_int"], data["h8_5ds"],
                                  expected=exp["V0a-ii"])
    gates["V0b"] = gate_v0b(data["ctrl"]["off"], data["int"]["off"], data["5ds"]["off"],
                            data["ctrl"]["shadow"], data["int"]["shadow"], data["5ds"]["shadow"],
                            expected=exp["V0b"])
    if args.preflight:
        return gates

    gates["V2"] = gate_v2(data["ctrl"]["off"], data["ctrl"]["evalue"], data["ctrl"]["always"])
    gates["V3"] = gate_v3(data["int"]["evalue"])

    all_runs = (_flatten_ctrl(data["ctrl"]["off"]) + _flatten_ctrl(data["ctrl"]["evalue"])
                + _flatten_ctrl(data["ctrl"]["always"]) + _flatten_ctrl(data["ctrl"]["shadow"])
                + _flatten_ctrl(data["ctrl"]["se_proxy"])
                + _flatten_single(data["int"]["off"], "s_interleave")
                + _flatten_single(data["int"]["evalue"], "s_interleave")
                + _flatten_single(data["int"]["evalue_cka"], "s_interleave")
                + _flatten_single(data["5ds"]["off"], None)
                + _flatten_single(data["5ds"]["shadow"], None)
                + _flatten_single(data["5ds"]["evalue"], None))
    gates["V4"] = gate_v4(all_runs)

    v5_pairs = [(s, st, a, b) for s, st, a, b in h8.paired_ctrl(data["ctrl"]["off"],
                                                                data["ctrl"]["evalue"])]
    v5_pairs += [(s, "s_interleave", data["int"]["off"][s], data["int"]["evalue"][s])
                 for s in sorted(set(data["int"]["off"]) & set(data["int"]["evalue"]))]
    gates["V5"] = gate_v5(v5_pairs)

    inventory = _dump_inventory(args.dumps)
    gates["V6"] = gate_v6(all_runs, inventory)

    cca_runs = (_flatten_ctrl(data["ctrl"]["evalue"])
                + _flatten_single(data["int"]["evalue"], "s_interleave"))
    cka_runs = _flatten_single(data["int"]["evalue_cka"], "s_interleave")
    gates["V1"] = gate_v1(cca_runs)
    gates["V1b"] = gate_v1b(cca_runs, cka_runs)

    gates["R1"] = observable_r1(data["ctrl"]["evalue"], data["ctrl"]["se_proxy"])
    gates["R2"] = observable_r2(cca_runs, cka_runs)
    gates["R3"] = observable_r3()
    gates["DA"] = {"gate": "DA", "verdict": "deferred",
                   "observable": "backward-delta AUROC and the CKA transfer clause on the "
                                 "token-mode dumps — the Track A desk script's gate, not this "
                                 "evaluator's",
                   "value": {"dump_inventory": inventory},
                   "headline": (f"deferred to the Track A desk script; "
                                f"{inventory.get('n_files', 0)} dumps inventoried")}
    return gates


def _print_table(gates: Dict[str, Dict]) -> None:
    print(f"{'gate':6s}{'verdict':12s}{'headline'}")
    print("-" * 120)
    for key, g in gates.items():
        head = g.get("headline") or g.get("note") or ""
        print(f"{key:6s}{str(g.get('verdict')):12s}{str(head)[:100]}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="H9 V-gate evaluator (provisional growth v2)")
    p.add_argument("--ctrl", default=None, help="CTrL run root (holds ctrl_<arm>/ dirs)")
    p.add_argument("--interleave", default=None, help="s_interleave run root (int_<arm>/)")
    p.add_argument("--fivedatasets", default=None, help="5-Datasets run root (5ds_<arm>/)")
    p.add_argument("--baseline", default=None,
                   help="attn-root adoption run root — the V0a-i reference")
    p.add_argument("--h8", default=None,
                   help="the 2026-09-09 H8 archive — the V0a-ii reference (ctrl/ctrl_off, "
                        "int_off, 5ds_off)")
    p.add_argument("--dumps", default=None,
                   help="root to search for gate_dump.pt files (V6 / DA inventory)")
    p.add_argument("--da", default=None,
                   help="JSON from the Track A desk script carrying DA's verdict "
                        "({'verdict': ..., 'transfers': bool}); without it the DA-dependent "
                        "branches cannot fire")
    p.add_argument("--preflight", action="store_true",
                   help="evaluate V0a-i/V0a-ii/V0b only, on whatever runs exist, and exit "
                        "non-zero if the null is not certified")
    p.add_argument("--out", default="gates_h9.json")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data = load_all(args)
    gates = evaluate(data, args)

    da = None
    if args.da and os.path.isfile(args.da):
        da = h8._read_json(args.da)

    out: Dict = {**gates}
    incomplete = {k: (g.get("value") or {}).get("missing_pairs")
                  for k, g in gates.items() if g.get("incomplete")}
    # A detected mismatch outranks incompleteness: a gate that is BOTH short-changed and
    # already disagreeing has still found a real leak, and the pre-flight must say "failed,
    # do not create pod A" rather than the softer "inconclusive".
    mismatching = [k for k, g in gates.items()
                   if (g.get("value") or {}).get("n_mismatches")]

    if args.preflight:
        checked = sum((g.get("value") or {}).get("n_runs_compared", 0) for g in gates.values())
        out["preflight"] = {"failed_gates": mismatching, "incomplete_gates": incomplete,
                            "n_runs_compared": checked}
        _print_table(gates)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out}")
        if mismatching:
            print(f"\n=== PREFLIGHT FAILED: {mismatching} — do not create pod A ===")
            return 1
        if incomplete:
            # A pre-flight that narrowed itself to whatever was on disk has certified nothing.
            # Saying "OK" here would licence the 129-run sweep on the strength of the runs that
            # happen to exist — including, in the reviewer's case, a ctrl/ tree alone, with the
            # 5-Datasets and s_interleave pairs the pre-flight EXISTS for never compared.
            print(f"\n=== PREFLIGHT INCONCLUSIVE: pre-registered pairs missing: {incomplete} ===")
            return 2
        print(f"\n=== PREFLIGHT OK: {checked} run comparisons, no mismatches ===")
        return 0

    branch = determine_branch(gates, da)
    out["branch"] = branch["branch"]
    out["branch_detail"] = branch
    out["da_input"] = da
    out["incomplete_gates"] = incomplete
    _print_table(gates)
    print(f"\n=== branch: {branch['branch']} ===\n  reason: {branch['reason']}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")
    if incomplete:
        print(f"\n=== INCONCLUSIVE: pre-registered pairs missing: {incomplete} ===")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
