#!/usr/bin/env python
"""
eval_module_family.py — the mechanical gate evaluation for H6, the module-family ablation
(Central Library/Hypotheses/concept-dag/kan-gated-growth/module-family-ablation.md,
revised 2026-09-08 arms/gates: M1, M2, M3, M3b, M4, M5).

Reads the per-run JSONs written by ``scripts/module_family_ablation.py``:

    <results>/<dataset>_<n_train>_pool<p>/family_<family>_seed_<seed>.json

(any depth under ``--results`` is scanned, so several datasets, regimes, pool values and
invocations can share one tree) and, for M5, the probe JSONs written by
``scripts/module_family_probe_noise.py`` under ``--probe``.

Arms: A=mlp_cls, B=mlp_cls_meanpool (CLS ⊕ mean patches), C0=proj_uniform_pool,
C=attn_pool, D=self_attn, E=cnn.

``token_pool`` is a treatment, not a nuisance: runs are aggregated per (regime, family,
pool) and every accuracy gate reads each family's BEST POOL, which is reported alongside
the number. Arm E has no pool (it never reads tokens).

Gates

  M1 (H6a, information)   SVHN full AND @400: acc(B) - acc(A) >= 0.02 in BOTH regimes.
                          B contains A (concatenation), so this is the patch tokens'
                          contribution and nothing else.
  M2 (H6b, attention)     SVHN full AND @400: acc(C) - acc(C0) and acc(D) - acc(C0),
                          reported SEPARATELY (no max). Accept if at least one of the two
                          clears 0.02 in both regimes. C0 is C with the learned query
                          removed, so this isolates attention from capacity.
  M3 (H6c, full data)     SVHN full: acc(E) - best(A..D). DESCRIPTIVE — a from-scratch CNN
                          at ~0.90 against 0.59 is a foregone conclusion, so it names no
                          branch.
  M3b (H6c, small n)      SVHN@400: acc(E) - acc(A) >= 0.05 -> SUBSTRATE-AT-SMALL-N. This is
                          the open question: two estimator loops may have been fighting a
                          substrate problem.
  M4 (control)            KMNIST@400: every arm within +/-0.02 of A. (Fashion is ceiling-
                          censored at ~0.921 and is reported as a smoke check only.)
  M5 (gate noise)         From --probe: a family whose held-out-bits split SD or selection
                          optimism exceeds 1.5x arm A's is GATE-NOISIER.

  INCONCLUSIVE            if any family's SVHN@400 across-seed SE of test accuracy exceeds
                          0.02, or if M1/M2 lack data.

  Null validity           arm A must reproduce the archived DAG-root numbers, SVHN full
                          0.594 +/- 0.005 and SVHN@400 0.303 +/- 0.004. Reported (with a
                          3-SD "matches" flag), never folded into the branch: a mismatch
                          means the harness, not the hypothesis, needs attention.

Blind branches

    ATTENTION-WINS    M1 and M2 accept
    INFORMATION-WINS  M1 accepts, M2 rejects
    NO-EFFECT         M1 and M2 reject
    ATTENTION-ONLY    M1 rejects, M2 accepts — the cell the pre-registration did not name;
                      reported under this label rather than folded into another.

each combined with SUBSTRATE-AT-SMALL-N (M3b accepts) or SUBSTRATE-OK (M3b rejects).
COLLATERAL (M4 fails) and GATE-NOISIER (M5) are separate top-level flags.

Usage
-----
    python scripts/eval_module_family.py --results results/h6 \\
        --probe results/h6_probe --out results/h6/gates.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional

ARM_A, ARM_B = "mlp_cls", "mlp_cls_meanpool"
ARM_C0, ARM_C, ARM_D, ARM_E = "proj_uniform_pool", "attn_pool", "self_attn", "cnn"
DINO_ARMS = (ARM_A, ARM_B, ARM_C0, ARM_C, ARM_D)
ARM_LETTER = {ARM_A: "A", ARM_B: "B", ARM_C0: "C0", ARM_C: "C", ARM_D: "D", ARM_E: "E",
              "mlp_meanpool": "B0"}

ACC_THRESHOLD = 0.02        # M1, M2
SUBSTRATE_THRESHOLD = 0.05  # M3, M3b
CONTROL_TOL = 0.02          # M4
SE_LIMIT = 0.02             # INCONCLUSIVE
NOISE_RATIO = 1.5           # M5
SVHN_REGIMES = ("svhn_full", "svhn_400")
NULL_VALIDITY = {"svhn_full": (0.594, 0.005), "svhn_400": (0.303, 0.004)}
FASHION_SMOKE = 0.921


# ---------------------------------------------------------------------------
# Loading / aggregation
# ---------------------------------------------------------------------------

def _read_json_tree(directory: str, prefix: str) -> List[Dict]:
    out = []
    for root, _dirs, files in os.walk(directory):
        for fn in sorted(files):
            if fn.startswith(prefix) and fn.endswith(".json") and "summary" not in fn:
                with open(os.path.join(root, fn)) as f:
                    out.append(json.load(f))
    return out


def load_results(results_dir: str) -> Dict[str, Dict[str, Dict]]:
    return aggregate(_read_json_tree(results_dir, "family_"))


def load_probe(probe_dir: str) -> List[Dict]:
    return _read_json_tree(probe_dir, "probe_")


def aggregate(runs: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    """{"<dataset>_<n_train>": {family: {"pools": {...}, "best_pool": p, **best-pool stats}}}.

    Pool values are folded by taking each family's best-accuracy pool, which the note calls
    for ("C - C0 and D - C0 ... best pool"); every pool's numbers stay in "pools".
    """
    groups: Dict[str, Dict[str, Dict[str, List[Dict]]]] = {}
    for r in runs:
        key = f"{r['dataset']}_{r['n_train']}"
        pool = r.get("token_pool")
        pool_key = "none" if pool is None else str(pool)
        groups.setdefault(key, {}).setdefault(r["family"], {}).setdefault(pool_key, []).append(r)

    out: Dict[str, Dict[str, Dict]] = {}
    for key, fams in sorted(groups.items()):
        out[key] = {}
        for fam, pools in sorted(fams.items()):
            per_pool = {p: _stats(rs) for p, rs in sorted(pools.items())}
            best_pool = max(per_pool, key=lambda p: per_pool[p]["acc_mean"])
            entry = dict(per_pool[best_pool])
            entry["pools"] = per_pool
            entry["best_pool"] = best_pool
            entry["arm"] = ARM_LETTER.get(fam)
            out[key][fam] = entry
    return out


def _stats(rs: List[Dict]) -> Dict:
    accs = [float(r["test_acc"]) for r in rs]
    bits = [float(r["val_bits"]) for r in rs]
    tbits = [float(r["test_bits"]) for r in rs if r.get("test_bits") is not None]
    return {"n_seeds": len(rs), "seeds": sorted(int(r["seed"]) for r in rs),
            "acc_mean": _mean(accs), "acc_se": _se(accs), "acc_sd": _sd(accs),
            "val_bits_mean": _mean(bits), "val_bits_sd": _sd(bits),
            "test_bits_mean": _mean(tbits) if tbits else None,
            "params": int(rs[0].get("params", 0)),
            "split_hashes": sorted({r.get("split_hash") for r in rs}),
            "test_accs": accs, "val_bits": bits}


def _mean(v: List[float]) -> Optional[float]:
    return float(sum(v) / len(v)) if v else None


def _sd(v: List[float]) -> Optional[float]:
    if len(v) < 2:
        return 0.0 if v else None
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def _se(v: List[float]) -> Optional[float]:
    sd = _sd(v)
    return None if sd is None else sd / math.sqrt(len(v))


def _acc(summary: Dict, key: str, family: str) -> Optional[float]:
    return summary.get(key, {}).get(family, {}).get("acc_mean")


def _pool(summary: Dict, key: str, family: str) -> Optional[str]:
    return summary.get(key, {}).get(family, {}).get("best_pool")


# ---------------------------------------------------------------------------
# Pairing check — every arm at a (regime, seed) must have seen the same images
# ---------------------------------------------------------------------------

def pairing_check(summary: Dict) -> Dict:
    rows = []
    for key, fams in sorted(summary.items()):
        hashes = set()
        for fam, agg in fams.items():
            hashes.update(h for h in agg.get("split_hashes", []) if h)
        rows.append({"regime": key, "n_distinct_split_hashes": len(hashes)})
    # One hash per seed is expected, so "paired" means: no more distinct hashes than seeds.
    n_seeds = max((agg["n_seeds"] for fams in summary.values() for agg in fams.values()),
                  default=0)
    unpaired = [r for r in rows if r["n_distinct_split_hashes"] > max(n_seeds, 1)]
    return {"check": "pairing", "status": "OK" if rows else "SKIPPED", "rows": rows,
            "max_seeds": n_seeds, "paired": not unpaired, "unpaired": unpaired}


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _delta_rows(summary: Dict, treat: str, ref: str, threshold: float,
                regimes=SVHN_REGIMES) -> Dict:
    rows, missing = [], []
    for key in regimes:
        t, r = _acc(summary, key, treat), _acc(summary, key, ref)
        if t is None or r is None:
            missing.append(f"{key}:" + ",".join(
                f for f, v in ((treat, t), (ref, r)) if v is None))
            continue
        rows.append({"regime": key, "treat": treat, "ref": ref, "treat_acc": t,
                     "ref_acc": r, "delta": t - r, "accept": (t - r) >= threshold,
                     "treat_pool": _pool(summary, key, treat),
                     "ref_pool": _pool(summary, key, ref)})
    if missing or len(rows) < len(regimes):
        return {"status": "SKIPPED", "missing": missing, "rows": rows,
                "threshold": threshold}
    return {"status": "OK", "threshold": threshold, "rows": rows,
            "accept": all(r["accept"] for r in rows)}


def m1_information(summary: Dict, threshold: float = ACC_THRESHOLD) -> Dict:
    """acc(B) - acc(A) >= threshold in BOTH SVHN regimes."""
    out = _delta_rows(summary, ARM_B, ARM_A, threshold)
    out["gate"] = "M1"
    return out


def m2_attention(summary: Dict, threshold: float = ACC_THRESHOLD) -> Dict:
    """acc(C) - acc(C0) and acc(D) - acc(C0), separately, in BOTH SVHN regimes."""
    c = _delta_rows(summary, ARM_C, ARM_C0, threshold)
    d = _delta_rows(summary, ARM_D, ARM_C0, threshold)
    status = "OK" if (c["status"] == "OK" and d["status"] == "OK") else "SKIPPED"
    out = {"gate": "M2", "status": status, "threshold": threshold,
           "attn_pool_vs_uniform": c, "self_attn_vs_uniform": d}
    if status == "OK":
        out["accept_attn_pool"] = c["accept"]
        out["accept_self_attn"] = d["accept"]
        out["accept"] = bool(c["accept"] or d["accept"])   # "at least one, in both regimes"
    return out


def m3_substrate_full(summary: Dict, threshold: float = SUBSTRATE_THRESHOLD) -> Dict:
    """SVHN full: acc(E) - best(A..D). Descriptive — names no branch."""
    key = "svhn_full"
    cnn = _acc(summary, key, ARM_E)
    present = {f: _acc(summary, key, f) for f in DINO_ARMS
               if _acc(summary, key, f) is not None}
    if cnn is None or not present:
        return {"gate": "M3", "status": "SKIPPED", "descriptive": True,
                "threshold": threshold}
    best = max(present, key=lambda f: present[f])
    delta = cnn - present[best]
    return {"gate": "M3", "status": "OK", "descriptive": True, "regime": key,
            "threshold": threshold, "cnn_acc": cnn, "best_dino_family": best,
            "best_dino_acc": present[best], "delta": delta,
            "exceeds_threshold": delta >= threshold}


def m3b_substrate_small_n(summary: Dict, threshold: float = SUBSTRATE_THRESHOLD) -> Dict:
    """SVHN@400: acc(E) - acc(A) >= threshold -> SUBSTRATE-AT-SMALL-N."""
    out = _delta_rows(summary, ARM_E, ARM_A, threshold, regimes=("svhn_400",))
    out["gate"] = "M3b"
    return out


def m4_control(summary: Dict, tol: float = CONTROL_TOL, dataset: str = "kmnist",
               n_train: str = "400") -> Dict:
    """Control dataset: every arm within +/-tol of A."""
    key = f"{dataset}_{n_train}"
    a = _acc(summary, key, ARM_A)
    if a is None:
        return {"gate": "M4", "status": "SKIPPED", "tolerance": tol, "regime": key,
                "rows": []}
    rows = []
    for fam, agg in sorted(summary[key].items()):
        if fam == ARM_A or agg.get("acc_mean") is None:
            continue
        delta = agg["acc_mean"] - a
        rows.append({"regime": key, "family": fam, "arm": ARM_LETTER.get(fam),
                     "acc": agg["acc_mean"], "delta": delta, "within": abs(delta) <= tol})
    if not rows:
        return {"gate": "M4", "status": "SKIPPED", "tolerance": tol, "regime": key,
                "rows": []}
    return {"gate": "M4", "status": "OK", "tolerance": tol, "regime": key,
            "arm_a_acc": a, "accept": all(r["within"] for r in rows),
            "violations": [r for r in rows if not r["within"]], "rows": rows}


def m5_gate_noise(probe_runs: List[Dict], ratio_limit: float = NOISE_RATIO) -> Dict:
    """From the probe-noise script: split SD / optimism per family, as a ratio to arm A's."""
    if not probe_runs:
        return {"gate": "M5", "status": "SKIPPED", "ratio_limit": ratio_limit, "rows": []}
    fams: Dict[str, List[Dict]] = {}
    for r in probe_runs:
        fams.setdefault(r["family"], []).extend(r.get("splits", []))
    stats = {}
    for fam, splits in fams.items():
        mins = [s["min_held_out_bits"] for s in splits]
        opts = [s["optimism"] for s in splits]
        stats[fam] = {"n_splits": len(splits), "sd_held_out_bits": _sd(mins),
                      "mean_min_held_out_bits": _mean(mins), "mean_optimism": _mean(opts)}
    if ARM_A not in stats:
        return {"gate": "M5", "status": "SKIPPED", "ratio_limit": ratio_limit,
                "note": f"no probe runs for the reference arm {ARM_A}",
                "rows": sorted(stats)}
    base = stats[ARM_A]
    rows = []
    for fam in sorted(stats):
        s = stats[fam]
        sd_ratio = (s["sd_held_out_bits"] / base["sd_held_out_bits"]
                    if base["sd_held_out_bits"] else None)
        opt_ratio = (s["mean_optimism"] / base["mean_optimism"]
                     if base["mean_optimism"] else None)
        noisier = bool((sd_ratio is not None and sd_ratio > ratio_limit)
                       or (opt_ratio is not None and opt_ratio > ratio_limit))
        rows.append({"family": fam, "arm": ARM_LETTER.get(fam), **s,
                     "sd_ratio_vs_A": sd_ratio, "optimism_ratio_vs_A": opt_ratio,
                     "gate_noisier": noisier})
    return {"gate": "M5", "status": "OK", "ratio_limit": ratio_limit, "rows": rows,
            "gate_noisier_arms": [r["family"] for r in rows if r["gate_noisier"]]}


def power_check(summary: Dict, se_limit: float = SE_LIMIT) -> Dict:
    """INCONCLUSIVE condition: a family's SVHN@400 across-seed SE above se_limit."""
    key = "svhn_400"
    rows = []
    for fam, agg in sorted(summary.get(key, {}).items()):
        se = agg.get("acc_se")
        if se is None:
            continue
        rows.append({"family": fam, "acc_se": se, "n_seeds": agg["n_seeds"],
                     "over": se > se_limit})
    if not rows:
        return {"check": "power", "status": "SKIPPED", "se_limit": se_limit, "rows": []}
    return {"check": "power", "status": "OK", "se_limit": se_limit, "rows": rows,
            "inconclusive": any(r["over"] for r in rows),
            "over_arms": [r["family"] for r in rows if r["over"]]}


def null_validity(summary: Dict) -> Dict:
    """Arm A must reproduce the archived DAG-root SVHN numbers (descriptive)."""
    rows = []
    for key, (ref, sd) in NULL_VALIDITY.items():
        a = _acc(summary, key, ARM_A)
        if a is None:
            continue
        rows.append({"regime": key, "arm_a_acc": a, "reference": ref, "reference_sd": sd,
                     "delta": a - ref, "matches_within_3sd": abs(a - ref) <= 3 * sd})
    fashion = _acc(summary, "fashion_full", ARM_A)
    return {"check": "null_validity", "status": "OK" if rows else "SKIPPED", "rows": rows,
            "matches": all(r["matches_within_3sd"] for r in rows) if rows else None,
            "fashion_full_smoke": {"arm_a_acc": fashion, "reference": FASHION_SMOKE,
                                   "delta": None if fashion is None else fashion - FASHION_SMOKE}}


# ---------------------------------------------------------------------------
# Blind branch
# ---------------------------------------------------------------------------

def _accepts(gate: Dict) -> bool:
    return gate.get("status") == "OK" and bool(gate.get("accept"))


def _rejects(gate: Dict) -> bool:
    return gate.get("status") == "OK" and gate.get("accept") is False


def blind_branch(m1: Dict, m2: Dict, m3b: Dict, power: Dict) -> str:
    if power.get("status") == "OK" and power.get("inconclusive"):
        return "INCONCLUSIVE"
    if m1.get("status") != "OK" or m2.get("status") != "OK":
        return "INCONCLUSIVE"
    if _accepts(m1) and _accepts(m2):
        core = "ATTENTION-WINS"
    elif _accepts(m1):
        core = "INFORMATION-WINS"
    elif _accepts(m2):
        core = "ATTENTION-ONLY"   # unregistered cell — named, not hidden
    else:
        core = "NO-EFFECT"
    if m3b.get("status") != "OK":
        return f"{core} + SUBSTRATE-UNKNOWN"
    return f"{core} + " + ("SUBSTRATE-AT-SMALL-N" if _accepts(m3b) else "SUBSTRATE-OK")


def collateral_flag(m4: Dict) -> bool:
    return _rejects(m4)


def gate_noisier_flag(m5: Dict) -> bool:
    return bool(m5.get("status") == "OK" and m5.get("gate_noisier_arms"))


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def print_table(summary: Dict) -> None:
    for key in sorted(summary):
        print(f"\n=== {key} ===")
        print(f"{'arm':<4}{'family':<20}{'pool':>5}{'n':>3}  {'test acc (mean ± SE)':>22}  "
              f"{'val bits':>9}  {'test bits':>10}  {'params':>10}")
        for fam in sorted(summary[key], key=lambda f: str(ARM_LETTER.get(f, f))):
            agg = summary[key][fam]
            se = agg["acc_se"] if agg["acc_se"] is not None else float("nan")
            acc = f"{agg['acc_mean']:.4f} ± {se:.4f}"
            tb = agg.get("test_bits_mean")
            print(f"{str(agg.get('arm') or '-'):<4}{fam:<20}{str(agg['best_pool']):>5}"
                  f"{agg['n_seeds']:>3}  {acc:>22}  {agg['val_bits_mean']:>9.4f}  "
                  f"{(float('nan') if tb is None else tb):>10.4f}  {agg['params']:>10,}")
            if len(agg["pools"]) > 1:
                for p, st in agg["pools"].items():
                    print(f"      pool {p:<4} acc={st['acc_mean']:.4f} "
                          f"val_bits={st['val_bits_mean']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="directory holding the ablation JSONs")
    ap.add_argument("--probe", default=None,
                    help="directory holding module_family_probe_noise.py output (M5)")
    ap.add_argument("--out", required=True, help="where to write gates.json")
    ap.add_argument("--acc_threshold", type=float, default=ACC_THRESHOLD)
    ap.add_argument("--substrate_threshold", type=float, default=SUBSTRATE_THRESHOLD)
    ap.add_argument("--control_tol", type=float, default=CONTROL_TOL)
    ap.add_argument("--control_dataset", default="kmnist")
    ap.add_argument("--control_n_train", default="400")
    ap.add_argument("--se_limit", type=float, default=SE_LIMIT)
    ap.add_argument("--noise_ratio", type=float, default=NOISE_RATIO)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not os.path.isdir(args.results):
        raise SystemExit(f"--results directory not found: {args.results}")

    summary = load_results(args.results)
    if not summary:
        print(f"[h6-gate] no family_*_seed_*.json found under {args.results}")
    probe_runs = load_probe(args.probe) if args.probe else []

    print_table(summary)

    m1 = m1_information(summary, args.acc_threshold)
    m2 = m2_attention(summary, args.acc_threshold)
    m3 = m3_substrate_full(summary, args.substrate_threshold)
    m3b = m3b_substrate_small_n(summary, args.substrate_threshold)
    m4 = m4_control(summary, args.control_tol, args.control_dataset, args.control_n_train)
    m5 = m5_gate_noise(probe_runs, args.noise_ratio)
    power = power_check(summary, args.se_limit)
    pairing = pairing_check(summary)
    null = null_validity(summary)

    labels = {
        "M1": "M1 (H6a, information): acc(B) - acc(A), both SVHN regimes",
        "M2": "M2 (H6b, attention): acc(C) - acc(C0) and acc(D) - acc(C0), separately",
        "M3": "M3 (H6c, full data, DESCRIPTIVE): acc(E) - best DINO arm",
        "M3b": "M3b (H6c, small n): acc(E) - acc(A) on SVHN@400",
        "M4": f"M4 (control): {args.control_dataset}@{args.control_n_train}, arms within tolerance of A",
        "M5": "M5 (gate noise): probe held-out-bits SD / optimism vs arm A",
    }
    for gate in (m1, m2, m3, m3b, m4, m5):
        print(f"\n=== {labels[gate['gate']]} ===")
        print(json.dumps(gate, indent=2))

    print("\n=== null validity (arm A vs the archived DAG-root numbers) ===")
    print(json.dumps(null, indent=2))
    print("\n=== pairing (identical split_hash per seed across arms) ===")
    print(json.dumps(pairing, indent=2))
    print("\n=== power (INCONCLUSIVE if any SVHN@400 seed SE > se_limit) ===")
    print(json.dumps(power, indent=2))

    branch = blind_branch(m1, m2, m3b, power)
    flag = collateral_flag(m4)
    noisier = gate_noisier_flag(m5)
    print(f"\n=== blind branch: {branch}  (collateral_flag={flag}, "
          f"gate_noisier={noisier}) ===")

    result = {"summary": summary, "m1_information": m1, "m2_attention": m2,
              "m3_substrate_full": m3, "m3b_substrate_small_n": m3b, "m4_control": m4,
              "m5_gate_noise": m5, "power": power, "pairing": pairing,
              "null_validity": null, "blind_branch": branch, "collateral_flag": flag,
              "gate_noisier_flag": noisier,
              "gate_noisier_arms": m5.get("gate_noisier_arms", [])}
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
