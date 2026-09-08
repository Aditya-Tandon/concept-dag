#!/usr/bin/env python
"""
eval_gate_arms.py — the stream-stage (live-arm) analysis for the three pre-registered gate
hypotheses (see INTERFACE_SPEC.md §8 and the vault notes it links:
small-n-codelength-estimator-stress-test.md, best-rung-denominator-stress-test.md,
reuse-with-update-arm-stress-test.md).

Reads ``exp3a_kan_results.json`` files for up to six arms:

    A0        control (published estimator/denominator/no-update)             --a0 DIR
    A1        H1' arm: cross-fitted codelength estimator                       --a1 DIR
    A2        H2 arm: best-rung denominator decides                            --a2 DIR
    A3        H3' arm: --enable_update                                        --a3 DIR
    A0_5ds    control on 5-Datasets (for the null-validity check)              --a0_5ds DIR
    A3_5ds    update arm on 5-Datasets                                        --a3_5ds DIR

CTrL layout:   <DIR>/seed_<S>/exp_ctrl_<stream>/exp3a_kan_results.json  for stream in
               {s_minus, s_plus, s_out, s_in}
5ds layout:    <DIR>/seed_<S>/exp5ds_kan/exp3a_kan_results.json

Prints and writes (to --out) the mechanical branch for every pre-registered gate:
  H1' stream stage : LOWER-REGRET / NO-EFFECT / ANOMALOUS / INCONCLUSIVE          (A0 vs A1)
  H2               : PRESERVES / FLIPS, plus max |rel_best| and clamp count       (all arms)
  H3'              : U1-U6, blind branch in {FIRES-ON-S-PLUS-ONLY, NEVER-FIRES,
                     OVER-FIRES, FIRES-WITHOUT-GAIN, UNSAFE, INCONCLUSIVE}         (A0 vs A3)
  null-validity    : A0 seed-42 decision pattern vs published, MATCH/MISMATCH per stream

A missing arm directory (or one with no seed_*/... results under it) is skipped with a note;
the script never crashes on a missing arm — gates that need it report status "SKIPPED".
Every number this script prints is also in the JSON it writes.
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


# ---------------------------------------------------------------------------
# Loading
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
# H1' stream stage — A0 (control) vs A1 (cross-fitted estimator)
# ---------------------------------------------------------------------------


def h1_stream_stage(a0: dict, a1: dict, t3: int = 3, t4: int = 4,
                     regret_thresh: float = 0.015, se_thresh: float = 0.01) -> dict:
    if not a0 or not a1:
        return {"status": "SKIPPED", "reason": "A0 or A1 missing/empty"}
    pairs = []
    for seed in sorted(set(a0) & set(a1)):
        for stream in STREAMS:
            if stream not in a0.get(seed, {}) or stream not in a1.get(seed, {}):
                continue
            r0, r1 = a0[seed][stream], a1[seed][stream]
            d0, d1 = dec_at(r0, t3), dec_at(r1, t3)
            if d0 is None or d1 is None:
                continue
            oa0, oa1 = d0.get("oracle_accs"), d1.get("oracle_accs")
            if not oa0 or not oa1:
                continue
            regret0 = max(oa0.values()) - r0["test_accs"][t3]
            regret1 = max(oa1.values()) - r1["test_accs"][t3]
            d0_4, d1_4 = dec_at(r0, t4), dec_at(r1, t4)
            changed = bool(d0_4 and d1_4 and d0_4["decision"] != d1_4["decision"])
            pairs.append({"seed": seed, "stream": stream, "regret_A0": regret0, "regret_A1": regret1,
                          "delta_regret": regret0 - regret1, "t4_decision_changed": changed})
    if not pairs:
        return {"status": "SKIPPED", "reason": "no matched (seed, stream) t3 oracle_accs pairs"}
    deltas = [p["delta_regret"] for p in pairs]
    mean_delta = _mean(deltas)
    se = _sample_se(deltas)
    any_changed = any(p["t4_decision_changed"] for p in pairs)
    if se >= se_thresh:
        branch = "INCONCLUSIVE"
    elif mean_delta < 0 or any_changed:
        branch = "ANOMALOUS"
    elif mean_delta >= regret_thresh:
        branch = "LOWER-REGRET"
    else:
        branch = "NO-EFFECT"
    return {"status": "OK", "n_pairs": len(pairs), "mean_delta_regret": mean_delta, "se": se,
            "any_t4_decision_changed": any_changed, "branch": branch, "pairs": pairs}


# ---------------------------------------------------------------------------
# H2 — best-rung denominator: PRESERVES / FLIPS
# ---------------------------------------------------------------------------


def h2_gate(labeled_results: dict, eps_grow: float = 0.05, eps_search: float = 0.05) -> dict:
    if not labeled_results:
        return {"status": "SKIPPED", "reason": "no arms with decisions provided"}
    per_arm = {}
    flips_total = checked_total = clamp_total = 0
    max_abs = 0.0
    all_flip_details = []
    for label, results in labeled_results.items():
        flips = checked = clamp = 0
        local_max = 0.0
        details = []
        for d in results.get("decisions", []):
            # The "update" rung (§4) is a post-hoc refinement layered AFTER the frozen
            # reuse/search/grow ladder decides; it can overwrite the recorded decision string to
            # "update" without changing which of {reuse, search, grow} the ladder itself picked.
            # H2 is about that ladder's denominator choice, so decisions the best-denominator
            # recompute cannot even express (anything other than grow/search/reuse) are skipped
            # rather than counted as an automatic flip.
            if d.get("decision") not in ("grow", "search", "reuse"):
                continue
            L_null, L_reuse, L_grow = d.get("L_null_bits"), d.get("L_reuse_bits"), d.get("L_grow_bits")
            if L_null is None or L_reuse is None or L_grow is None:
                continue
            L_search = d.get("L_search_bits")
            L_all = [L_reuse, L_grow] + ([L_search] if L_search is not None else [])
            reducible_best = max(L_null - min(L_all), 1e-6)
            is_clamp = (L_null - min(L_all)) <= 1e-6
            rel_search_best = d.get("rel_search_best")
            rel_grow_best = d.get("rel_grow_best")
            if rel_search_best is None and L_search is not None:
                rel_search_best = (L_reuse - L_search) / reducible_best
            base = L_search if L_search is not None else L_reuse
            if rel_grow_best is None:
                rel_grow_best = (base - L_grow) / reducible_best
            if rel_grow_best > eps_grow:
                dec_best = "grow"
            elif L_search is not None and rel_search_best is not None and rel_search_best > eps_search:
                dec_best = "search"
            else:
                dec_best = "reuse"
            checked += 1
            clamp += int(is_clamp)
            local_max = max(local_max, abs(rel_grow_best), abs(rel_search_best or 0.0))
            if dec_best != d.get("decision"):
                flips += 1
                details.append({"task": d.get("task"), "recorded": d.get("decision"), "best_denominator": dec_best})
        per_arm[label] = {"checked": checked, "flips": flips, "clamp_count": clamp,
                          "max_abs_rel_best": local_max, "flip_details": details,
                          "branch": ("PRESERVES" if flips == 0 and checked > 0 else
                                     ("FLIPS" if flips > 0 else "SKIPPED"))}
        flips_total += flips
        checked_total += checked
        clamp_total += clamp
        max_abs = max(max_abs, local_max)
        all_flip_details.extend({"arm": label, **fd} for fd in details)
    overall = ("PRESERVES" if flips_total == 0 and checked_total > 0 else
               ("FLIPS" if flips_total > 0 else "SKIPPED"))
    return {"status": "OK", "checked": checked_total, "flips": flips_total, "clamp_count": clamp_total,
            "max_abs_rel_best": max_abs, "branch": overall, "per_arm": per_arm,
            "flip_details": all_flip_details}


# ---------------------------------------------------------------------------
# H3' — U1..U6, A0 (control) vs A3 (--enable_update)
# ---------------------------------------------------------------------------


def h3_gates(a0: dict, a3: dict, h3_task: int = 4, mid_tasks=(1, 2, 3),
             specificity_streams=("s_minus", "s_out"), descriptive_stream="s_in",
             positive_stream="s_plus") -> dict:
    if not a0 or not a3:
        return {"status": "SKIPPED", "reason": "A0 or A3 missing/empty"}
    seeds = sorted(set(a0) & set(a3))

    # --- U1 (positive): s_plus t4. ---
    fires, gains, worst_backward = [], [], []
    for seed in seeds:
        if positive_stream not in a0.get(seed, {}) or positive_stream not in a3.get(seed, {}):
            continue
        d0, d3 = dec_at(a0[seed][positive_stream], h3_task), dec_at(a3[seed][positive_stream], h3_task)
        if d0 is None or d3 is None:
            continue
        fires.append(d3.get("decision") == "update")
        acc3 = a3[seed][positive_stream]["test_accs"][h3_task]
        oracle0 = (d0.get("oracle_accs") or {}).get("reuse")
        if oracle0 is None and d0.get("decision") == "reuse":
            oracle0 = a0[seed][positive_stream]["test_accs"][h3_task]
        if oracle0 is not None:
            gains.append(acc3 - oracle0)
        upd = d3.get("update") or {}
        if upd.get("backward_deltas_val"):
            worst_backward.append(min(upd["backward_deltas_val"].values()))
    n_u1 = len(fires)
    n_fire_u1 = sum(fires)
    mean_gain, se_gain = _mean(gains), _sample_se(gains)
    worst_backward_val = min(worst_backward) if worst_backward else None
    u1_needed = max(1, math.ceil(0.8 * n_u1)) if n_u1 else 0  # generalises the pre-registered "4/5"
    u1_fires_any = n_fire_u1 > 0
    u1_accept = (n_u1 > 0 and n_fire_u1 >= u1_needed and mean_gain is not None and mean_gain >= 0.02
                 and (worst_backward_val is None or worst_backward_val >= -0.01))

    # --- U2 (no masking, structural check): t1-t3 identical to A0, every seed/stream. ---
    mismatches = []
    for seed in seeds:
        for stream in STREAMS:
            if stream not in a0.get(seed, {}) or stream not in a3.get(seed, {}):
                continue
            for t in mid_tasks:
                d0, d3 = dec_at(a0[seed][stream], t), dec_at(a3[seed][stream], t)
                if d0 is None or d3 is None:
                    continue
                if d0["decision"] != d3["decision"]:
                    mismatches.append({"seed": seed, "stream": stream, "task": t,
                                       "A0": d0["decision"], "A3": d3["decision"]})
    u2_identical = len(mismatches) == 0

    # --- U3 (specificity): s_minus / s_out t4. ---
    u3_per_stream = {}
    u3_accept = True
    for stream in specificity_streams:
        fires_s, diffs = [], []
        for seed in seeds:
            if stream not in a0.get(seed, {}) or stream not in a3.get(seed, {}):
                continue
            d3 = dec_at(a3[seed][stream], h3_task)
            if d3 is None:
                continue
            fires_s.append(d3.get("decision") == "update")
            diffs.append(abs(a3[seed][stream]["test_accs"][h3_task] - a0[seed][stream]["test_accs"][h3_task]))
        n = len(fires_s)
        n_fire = sum(fires_s)
        needed_max = max(1, math.ceil(0.2 * n)) if n else 0  # generalises the pre-registered "<=1/5"
        mean_diff = _mean(diffs)
        accept = (n == 0) or (n_fire <= needed_max and (mean_diff is None or mean_diff <= 0.01))
        u3_per_stream[stream] = {"n_seeds": n, "n_fire": n_fire, "mean_abs_acc_diff": mean_diff, "accept": accept}
        if n > 0:
            u3_accept = u3_accept and accept

    # --- U4 (safety): every accepted update, all streams/tasks. ---
    worst_val, worst_test, violations = [], [], []
    for seed in seeds:
        for stream in STREAMS:
            if stream not in a3.get(seed, {}):
                continue
            for d in a3[seed][stream].get("decisions", []):
                if d.get("decision") != "update":
                    continue
                upd = d.get("update") or {}
                vd, td = upd.get("backward_deltas_val") or {}, upd.get("backward_deltas_test") or {}
                if vd:
                    worst_val.append(min(vd.values()))
                if td:
                    w = min(td.values())
                    worst_test.append(w)
                    if w < -0.02:
                        violations.append({"seed": seed, "stream": stream, "task": d.get("task"),
                                           "worst_test_delta": w})
    worst_val_overall = min(worst_val) if worst_val else None
    worst_test_overall = min(worst_test) if worst_test else None
    u4_accept = (worst_val_overall is None or worst_val_overall >= -0.01) and not violations

    # --- U5 (diagnostic): logged-only update probes on grow decisions. ---
    u5_hits = u5_total = 0
    for seed in seeds:
        for stream in STREAMS:
            if stream not in a3.get(seed, {}):
                continue
            for d in a3[seed][stream].get("decisions", []):
                if d.get("decision") != "grow":
                    continue
                upd = d.get("update")
                if not upd or not upd.get("logged_only"):
                    continue
                u5_total += 1
                if (upd.get("L_update") is not None
                        and upd["L_update"] <= d.get("L_grow_bits", float("inf"))
                        and upd.get("backward_safe")):
                    u5_hits += 1

    # --- U6 (descriptive): s_in t4. ---
    u6_records = []
    for seed in seeds:
        if descriptive_stream not in a3.get(seed, {}):
            continue
        d3 = dec_at(a3[seed][descriptive_stream], h3_task)
        if d3 is None:
            continue
        u6_records.append({"seed": seed, "decision": d3.get("decision"),
                           "acc": a3[seed][descriptive_stream]["test_accs"][h3_task],
                           "update": d3.get("update")})

    if se_gain >= 0.015:
        blind = "INCONCLUSIVE"
    elif u1_accept and u3_accept and u4_accept:
        blind = "FIRES-ON-S-PLUS-ONLY"
    elif (not u1_fires_any) and u3_accept:
        blind = "NEVER-FIRES"
    elif not u3_accept:
        blind = "OVER-FIRES"
    elif u1_fires_any and mean_gain is not None and mean_gain < 0.02:
        blind = "FIRES-WITHOUT-GAIN"
    elif not u4_accept:
        blind = "UNSAFE"
    else:
        blind = "INCONCLUSIVE"

    return {
        "status": "OK",
        "U1": {"n_seeds": n_u1, "n_fire": n_fire_u1, "needed": u1_needed, "mean_gain": mean_gain,
               "se_gain": se_gain, "worst_backward_delta_val": worst_backward_val, "fires_any": u1_fires_any,
               "accept": u1_accept},
        "U2": {"identical": u2_identical, "mismatches": mismatches},
        "U3": {"per_stream": u3_per_stream, "accept": u3_accept},
        "U4": {"worst_backward_delta_val": worst_val_overall, "worst_backward_delta_test": worst_test_overall,
               "violations": violations, "accept": u4_accept},
        "U5": {"hits": u5_hits, "total": u5_total},
        "U6": {"records": u6_records},
        "blind_branch": blind,
    }


# ---------------------------------------------------------------------------
# Null-validity — A0 seed-42 decision pattern vs published
# ---------------------------------------------------------------------------


def null_validity(a0: dict, a0_5ds: dict, seed: int = 42) -> dict:
    out = {}
    if a0_5ds and seed in a0_5ds:
        counts = {"grow": 0, "search": 0, "reuse": 0}
        for d in a0_5ds[seed].get("decisions", []):
            if d.get("decision") in counts:
                counts[d["decision"]] += 1
        published = {"grow": 4, "search": 0, "reuse": 1}
        out["5ds"] = {"observed": counts, "published": published,
                      "match": "MATCH" if counts == published else "MISMATCH"}
    else:
        out["5ds"] = {"status": "SKIPPED", "reason": "A0_5ds seed-42 results missing"}

    expected_ctrl = {1: "grow", 2: "grow", 3: "search", 4: "reuse"}
    for stream in STREAMS:
        if a0 and seed in a0 and stream in a0[seed]:
            observed = {t: (dec_at(a0[seed][stream], t) or {}).get("decision") for t in expected_ctrl}
            match = all(observed[t] == exp for t, exp in expected_ctrl.items())
            out[stream] = {"observed": observed, "expected": expected_ctrl,
                           "match": "MATCH" if match else "MISMATCH"}
        else:
            out[stream] = {"status": "SKIPPED", "reason": "A0 seed-42 stream missing"}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a0", default=None)
    ap.add_argument("--a1", default=None)
    ap.add_argument("--a2", default=None)
    ap.add_argument("--a3", default=None)
    ap.add_argument("--a0_5ds", default=None)
    ap.add_argument("--a3_5ds", default=None)
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
    a1 = load_arm(args.a1, "A1", load_ctrl_arm)
    a2 = load_arm(args.a2, "A2", load_ctrl_arm)
    a3 = load_arm(args.a3, "A3", load_ctrl_arm)
    a0_5ds = load_arm(args.a0_5ds, "A0_5ds", load_5ds_arm)
    a3_5ds = load_arm(args.a3_5ds, "A3_5ds", load_5ds_arm)

    print("=== H1' stream stage (A0 vs A1): LOWER-REGRET / NO-EFFECT / ANOMALOUS / INCONCLUSIVE ===")
    h1 = h1_stream_stage(a0, a1)
    print(json.dumps({k: v for k, v in h1.items() if k != "pairs"}, indent=2))

    print("\n=== H2 (best-rung denominator): PRESERVES / FLIPS ===")
    labeled = {}
    for label, arm in (("A0", a0), ("A1", a1), ("A2", a2), ("A3", a3)):
        for seed, streams in arm.items():
            for stream, results in streams.items():
                labeled[f"{label}/seed_{seed}/{stream}"] = results
    for label, arm in (("A0_5ds", a0_5ds), ("A3_5ds", a3_5ds)):
        for seed, results in arm.items():
            labeled[f"{label}/seed_{seed}"] = results
    h2 = h2_gate(labeled)
    print(json.dumps({k: v for k, v in h2.items() if k not in ("per_arm", "flip_details")}, indent=2))

    print("\n=== H3' (reuse-with-update arm), A0 vs A3: U1-U6 ===")
    h3 = h3_gates(a0, a3)
    if h3.get("status") == "OK":
        print(json.dumps({k: v for k, v in h3.items() if k not in ("U2", "U6")}, indent=2))
        print(f"U2 (structural, reported not gated): identical={h3['U2']['identical']} "
              f"mismatches={len(h3['U2']['mismatches'])}")
        print(f"U6 (descriptive, s_in t4): {len(h3['U6']['records'])} seed(s) recorded")
    else:
        print(json.dumps(h3, indent=2))

    print("\n=== null-validity: A0 seed-42 decision pattern vs published ===")
    nv = null_validity(a0, a0_5ds)
    print(json.dumps(nv, indent=2))

    if notes:
        print("\n=== notes ===")
        for n in notes:
            print(f"  - {n}")

    result = {"h1_stream_stage": h1, "h2_denominator": h2, "h3_update_arm": h3,
              "null_validity": nv, "notes": notes}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
