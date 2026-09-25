#!/usr/bin/env python
"""
desk_tie_se.py — per-example paired standard error of the search-vs-grow margin (H5), and how a
``|margin| <= z * se_paired`` tie rule compares to the resampling SD ``desk_fraction.py`` measured
for the same margin.

Question (pre-registered by the caller, not tuned here): at CTrL SVHN@400 (task 3 of the s_minus
stream), a proposed tie rule would treat the search and grow rungs as indistinguishable — and fall
back to grow rather than let the search-side candidate-selection optimism decide — whenever
``|L_search - L_grow| <= z * se_paired``, where ``se_paired`` is a WITHIN-RESAMPLE paired standard
error built from the per-example score-set bits of the two rungs. For the rule to be worth having,
``z * median(se_paired)`` must be in the neighbourhood of the BETWEEN-RESAMPLE resampling SD of the
margin that ``desk_fraction.py`` already measured directly (~0.109 bits, mean of the per-dump
stdev of the 12 D(1.0) = L_grow - L_search values, search selected on SCORE). This script measures
``se_paired`` and checks whether some small-integer z gets it there, using search selected on
SELECT (the ``desk_fraction.py`` addendum protocol, so the winner is not chosen on the very set the
margin is scored on).

Protocol (fixed; do not tune)
------------------------------
For each of the 5 gate dumps, task 3, R = 12 resamples (private ``torch.Generator`` seeds 0..R-1):
  * hold out a fixed 20% SCORE set and a fixed 15% SELECT set (same split protocol as
    ``desk_fraction.py``: ``torch.randperm(n, generator=Generator().manual_seed(r))``, SCORE first,
    SELECT next, the remaining 65% is the training pool);
  * train fraction f = 1.0 of the pool (the whole 65%), 40 epochs, lr = 1e-3;
  * reuse  = ReuseComposer, early-stopped on SELECT, scored on SCORE;
  * search = best-of-6 SearchComposer (rank=16, skip=True), candidate SELECTED on SELECT bits,
    winner's bits read off SCORE (``search_compose(..., select_on="select")`` — the
    ``desk_fraction.py`` addendum protocol; this removes the candidate-selection optimism the main
    ``desk_fraction.py`` table carries: the search winner there is chosen on the same SCORE set the
    comparison is reported on);
  * grow   = _RootGrowModel(ConceptModule(in_dim=F, hidden=D, out=D, n_layers, n_parents=0)),
    early-stopped on SELECT, scored on SCORE;
  * null   = head on a zero embedding, early-stopped on SELECT, scored on SCORE.

Per-example score bits and the best-select-epoch state
--------------------------------------------------------
The task brief asked us to check whether ``_held_out_codelength`` leaves the model in its LAST-epoch
state (so that a naive re-evaluation on SCORE after the call would score the wrong epoch) and, if
so, to reimplement a local training loop that checkpoints the best-SELECT state. We checked:
``_held_out_codelength(..., return_both=True)`` (kan_gate.py, ~L339-397) does NOT need that. It
already computes and stashes the per-example SCORE-set ``spec.nll_bits`` vector INSIDE the training
loop, at the exact moment ``val_bits`` (the SELECT loss) sets a new best — i.e. at the best-select
epoch itself, before any further training moves the weights away from it. So the returned
``best_score_bits`` tensor is correct regardless of what epoch the model object is left at when
training ends; no external checkpoint/restore is needed. We used ``return_both=True`` directly
(reuse and grow) and ``search_compose(..., select_on="select", ...)`` (search — internally the same
mechanism, per-candidate, tracking the winning candidate's vector; it currently returns a 6-tuple
``(L, cfg, trace, winner_select_bits, winner_per_example_score_bits, score_argmin_L)`` — the last
element is a select_on="score"-style counterfactual this protocol does not use) rather than
reimplementing training. No changes were made to ``concept_dag/training/kan_gate.py`` or ``scripts/desk_stage.py``;
this script only calls their existing, already-committed machinery.

When the trivial composition (plain reuse) wins the search race (no candidate beats it on SELECT),
``search_compose`` returns ``winner_per_example_score_bits=None``; per its docstring we then reuse
the REUSE rung's own per-example SCORE vector, exactly as ``baseline_L`` already stands in for the
trivial composition's reported scalar bits elsewhere in this codebase.

Per unit (one dump, one resample), over the n_score examples in that resample's SCORE set:
  d_i          = bits_search_i - bits_grow_i                       (paired per-example difference)
  margin       = mean_i(d_i)                                        (== L_search - L_grow)
  sd_per_example = sample sd_i(d_i)
  se_paired    = sd_per_example / sqrt(n_score)

The un-tied decision (grow / search / reuse) and the novelty guard use the same rule and eps=0.05
denominator as ``desk_fraction.decide`` (``L_null - L_grow``), fed the rungs' SCORE-set means.

Outputs (printed and written to --out as JSON)
  * median and IQR (25/75 pct) of se_paired over the 60 (dump x resample) units;
  * the resampling SD of the margin across the 12 resamples WITHIN each dump, averaged over the 5
    dumps — the quantity the tie rule's ``z * se_paired`` is being asked to reproduce;
  * the ratio (that resampling SD) / median(se_paired);
  * the tie firing fraction at z in {0.5, 1, 2, 3}: the fraction of the 60 units where
    novelty = (L_null - L_reuse) / L_null < 0.1  AND  the un-tied decision is not "grow"  AND
    |margin| <= z * se_paired.

CPU only. Reuses ``concept_dag.training.kan_gate`` and ``scripts/desk_stage.py`` /
``scripts/desk_fraction.py`` machinery; adds nothing to the library.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from concept_dag.training.kan_gate import (  # noqa: E402
    ReuseComposer,
    _held_out_codelength,
    _RootGrowModel,
    classification_task,
    search_compose,
)
from concept_dag.modules.concept_module import ConceptModule  # noqa: E402
from desk_stage import load_dump, rebuild_parent_stack  # noqa: E402
from desk_fraction import _fit_null, decide  # noqa: E402

DEVICE = "cpu"

_DEFAULT_DUMPS = [
    f"/Users/adityatandon/.claude/jobs/8297b3b2/tmp/arms/a0/seed_{s}/exp_ctrl_s_minus/gate_dump.pt"
    for s in (42, 43, 44, 45, 46)
]
_DEFAULT_OUT = "/Users/adityatandon/.claude/jobs/8297b3b2/tmp/h5-desk-tie-se.json"

# Matches desk_fraction.py's per-(resample, fraction) fit seed: seed = r*1000 + fi*10, with
# fi = index of f=1.0 in its default fractions list [0.25, 0.5, 0.75, 1.0] -> fi = 3. Reusing this
# exact formula means the reuse/grow/null fits here are BYTE-IDENTICAL runs to desk_fraction.py's
# own f=1.0 fits (same seed -> same init, same minibatch order), so the SCORE-set scalars this
# script reports at f=1.0 reproduce desk_fraction's own numbers as a sanity check.
_F1_FI = 3


def _quantile(xs, q):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    if n == 1:
        return xs[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _fit_reuse_both(parent_stack, y, tr_idx, sel_idx, sc_idx, D, P, spec, epochs, seed, lr=1e-3):
    Xtr, ytr = parent_stack[tr_idx], y[tr_idx]
    Xsel, ysel = parent_stack[sel_idx], y[sel_idx]
    Xsc, ysc = parent_stack[sc_idx], y[sc_idx]
    torch.manual_seed(seed)
    model = ReuseComposer(parent_dim=D, n_parents=max(P, 1), head=spec.make_head(D))
    sel_bits, sc_bits, per_example = _held_out_codelength(
        model, lambda m, xb: m(xb), spec, Xtr, ytr, Xsel, ysel,
        n_epochs=epochs, lr=lr, device=DEVICE, score_X=Xsc, score_y=ysc, return_both=True)
    return sel_bits, sc_bits, per_example


def _fit_grow_both(raw, y, tr_idx, sel_idx, sc_idx, F, D, n_layers, spec, epochs, seed, lr=1e-3,
                    tag="tie_grow"):
    Rtr, ytr = raw[tr_idx], y[tr_idx]
    Rsel, ysel = raw[sel_idx], y[sel_idx]
    Rsc, ysc = raw[sc_idx], y[sc_idx]
    torch.manual_seed(seed)
    module = ConceptModule(module_id=tag, in_dim=F, hidden_dim=D, out_dim=D, n_layers=n_layers,
                            n_parents=0)
    model = _RootGrowModel(module, spec.make_head(D))
    sel_bits, sc_bits, per_example = _held_out_codelength(
        model, lambda m, xb: m(xb), spec, Rtr, ytr, Rsel, ysel,
        n_epochs=epochs, lr=lr, device=DEVICE, score_X=Rsc, score_y=ysc, return_both=True)
    return sel_bits, sc_bits, per_example


def run_unit(parent_stack, raw, y, F, D, P, n_layers, spec, epochs, lr, r, n_score, n_select,
             budget, rank, eps):
    n = raw.shape[0]
    gen = torch.Generator().manual_seed(r)
    perm = torch.randperm(n, generator=gen)
    score_idx = perm[:n_score]
    select_idx = perm[n_score:n_score + n_select]
    pool = perm[n_score + n_select:]
    tr_idx = pool  # f = 1.0 of the 65% pool

    seed = r * 1000 + _F1_FI * 10

    sel_reuse, sc_reuse, pe_reuse = _fit_reuse_both(parent_stack, y, tr_idx, select_idx, score_idx,
                                                     D, P, spec, epochs, seed=seed, lr=lr)

    # search_compose(select_on="select", score_X given, no estimator_fn) returns a 6-tuple:
    # (L, cfg, trace, winner_select_bits, winner_per_example_score_bits, score_argmin_L). The last
    # element (the select_on="score"-style counterfactual) is not part of this protocol; unused here.
    L_search, cfg, _trace, sel_search, pe_search, _score_argmin_L = search_compose(
        parent_stack[tr_idx], y[tr_idx], parent_stack[select_idx], y[select_idx], spec,
        concept_dim=D, n_parents=max(P, 1), device=DEVICE, n_epochs=epochs, lr=lr,
        budget=budget, rank=rank, skip=True, baseline_L=sc_reuse, baseline_select_L=sel_reuse,
        select_on="select", score_X=parent_stack[score_idx], score_y=y[score_idx])
    if pe_search is None:
        # Trivial composition (plain reuse) won the search race: no candidate per-example vector
        # exists, so — per search_compose's own docstring — reuse the reuse rung's SCORE-set vector,
        # exactly as baseline_L already stands in for its reported scalar.
        pe_search = pe_reuse

    sel_grow, sc_grow, pe_grow = _fit_grow_both(raw, y, tr_idx, select_idx, score_idx, F, D,
                                                 n_layers, spec, epochs, seed=seed, lr=lr,
                                                 tag=f"tie_grow_{r}")

    L_null = _fit_null(parent_stack, y, tr_idx, select_idx, D, spec, epochs, seed=seed, lr=lr,
                       score_idx=score_idx)

    d = (pe_search - pe_grow).double()
    n_sc = int(d.numel())
    margin = float(d.mean().item())
    sd_pe = float(d.std(unbiased=True).item()) if n_sc > 1 else float("nan")
    se_paired = sd_pe / math.sqrt(n_sc) if n_sc > 1 else float("nan")

    untied, rel_grow, rel_search = decide(sc_reuse, L_search, sc_grow, L_null, eps)
    novelty = (L_null - sc_reuse) / L_null if L_null else float("nan")

    return {
        "resample": r, "n_score": n_sc,
        "L_reuse": sc_reuse, "L_search": L_search, "L_grow": sc_grow, "L_null": L_null,
        "margin": margin, "sd_per_example": sd_pe, "se_paired": se_paired,
        "novelty": novelty, "decision": untied, "search_trivial": bool(cfg.get("trivial", False)),
    }


def run_dump(path, task, R, epochs, lr, score_frac, select_frac, budget, rank, eps, verbose=True):
    dump = load_dump(path)
    parent_stack, raw, y, parent_idx = rebuild_parent_stack(dump, task, split="train")
    F, D = dump["feature_dim"], dump["concept_dim"]
    n_layers = dump["config"]["n_mlp_layers"]
    n_classes = dump["tasks"][task]["n_classes"]
    spec = classification_task(n_classes)
    P = max(parent_stack.shape[1], 1)
    n = raw.shape[0]

    n_score = int(round(score_frac * n))
    n_select = int(round(select_frac * n))

    units = []
    for r in range(R):
        t0 = time.time()
        u = run_unit(parent_stack, raw, y, F, D, P, n_layers, spec, epochs, lr, r, n_score,
                     n_select, budget, rank, eps)
        units.append(u)
        if verbose:
            print(f"    r={r:2d}  margin={u['margin']:+.4f}  se_paired={u['se_paired']:.4f}  "
                  f"decision={u['decision']:6s} novelty={u['novelty']:.4f}  "
                  f"[{time.time() - t0:.1f}s]")
    return {"dump": path, "task": task, "n": int(n), "feature_dim": F, "concept_dim": D,
            "n_score": n_score, "n_select": n_select, "n_pool": n - n_score - n_select,
            "units": units}


def aggregate(per_dump, z_grid, eps):
    units = [(d, u) for d in per_dump for u in d["units"]]
    se_list = [u["se_paired"] for _, u in units]

    resample_sds = []
    for d in per_dump:
        margins = [u["margin"] for u in d["units"]]
        if len(margins) > 1:
            resample_sds.append(statistics.stdev(margins))
    resample_sd_mean = (sum(resample_sds) / len(resample_sds)) if resample_sds else None

    med_se = _quantile(se_list, 0.5)
    q1_se = _quantile(se_list, 0.25)
    q3_se = _quantile(se_list, 0.75)
    ratio = (resample_sd_mean / med_se) if (resample_sd_mean is not None and med_se) else None

    tie_fire = {}
    for z in z_grid:
        n_fire = 0
        for _, u in units:
            guard = (u["novelty"] < 0.1) and (u["decision"] != "grow")
            fires = guard and (abs(u["margin"]) <= z * u["se_paired"])
            if fires:
                n_fire += 1
        tie_fire[str(z)] = {"n_fire": n_fire, "n_units": len(units),
                            "fraction": n_fire / len(units) if units else None}

    # what z makes z * median(se_paired) ~= resample_sd_mean
    z_match = (resample_sd_mean / med_se) if (resample_sd_mean is not None and med_se) else None

    return {
        "n_units": len(units),
        "se_paired": {"median": med_se, "q1": q1_se, "q3": q3_se, "iqr": (q3_se - q1_se)
                      if (q1_se is not None and q3_se is not None) else None},
        "resample_sd_of_margin": {"per_dump": resample_sds, "mean": resample_sd_mean},
        "ratio_resample_sd_over_median_se": ratio,
        "z_that_matches_resample_sd": z_match,
        "tie_firing": tie_fire,
        "eps": eps,
    }


def print_report(agg, per_dump):
    print("\n" + "=" * 96)
    print(f"se_paired over {agg['n_units']} (dump x resample) units")
    print("=" * 96)
    se = agg["se_paired"]
    print(f"  median = {se['median']:.4f}   IQR = [{se['q1']:.4f}, {se['q3']:.4f}]  "
          f"(width {se['iqr']:.4f})")

    print("\n" + "=" * 96)
    print("resampling SD of margin (grow - search... reported as search - grow == margin), "
          "within-dump across 12 resamples")
    print("=" * 96)
    for path, sd in zip([d["dump"] for d in per_dump], agg["resample_sd_of_margin"]["per_dump"]):
        tag = os.path.basename(os.path.dirname(os.path.dirname(path))) or path
        print(f"  {tag:<10s} sd = {sd:.4f}")
    print(f"  mean over dumps = {agg['resample_sd_of_margin']['mean']:.4f}   "
          f"(desk_fraction.py found ~0.109 for the search-selected-on-SCORE margin)")

    print(f"\n  ratio  SD_resample / median(se_paired) = {agg['ratio_resample_sd_over_median_se']:.3f}")

    print("\n" + "=" * 96)
    print("tie firing fraction: novelty<0.1 AND decision!=grow AND |margin| <= z*se_paired")
    print("=" * 96)
    print(f"  {'z':<6s} {'n_fire':>8s} {'n_units':>8s} {'fraction':>10s}")
    for z, v in agg["tie_firing"].items():
        print(f"  {z:<6s} {v['n_fire']:>8d} {v['n_units']:>8d} {v['fraction']:>10.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", nargs="+", default=_DEFAULT_DUMPS)
    ap.add_argument("--out", default=_DEFAULT_OUT)
    ap.add_argument("--task", type=int, default=3)
    ap.add_argument("--resamples", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--score_frac", type=float, default=0.20)
    ap.add_argument("--select_frac", type=float, default=0.15)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--z_grid", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0])
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    t_start = time.time()
    per_dump = []
    for path in args.dumps:
        tag = os.path.basename(os.path.dirname(os.path.dirname(path))) or path
        print(f"[{tag}] task {args.task}, R={args.resamples}")
        per_dump.append(run_dump(path, args.task, args.resamples, args.epochs, args.lr,
                                 args.score_frac, args.select_frac, args.budget, args.rank,
                                 args.eps))
    print(f"\ntotal runtime: {(time.time() - t_start) / 60:.1f} min")

    agg = aggregate(per_dump, args.z_grid, args.eps)
    print_report(agg, per_dump)

    result = {
        "protocol": {
            "task": args.task, "resamples": args.resamples, "epochs": args.epochs, "lr": args.lr,
            "score_frac": args.score_frac, "select_frac": args.select_frac,
            "search_budget": args.budget, "search_rank": args.rank, "search_skip": True,
            "search_select_on": "select", "eps": args.eps, "z_grid": args.z_grid,
            "fit_seed_formula": "r*1000 + 30  (matches desk_fraction.py's f=1.0 seed)",
            "note": "per-example SCORE bits taken at the best-SELECT epoch via "
                    "_held_out_codelength(..., return_both=True) / "
                    "search_compose(..., select_on='select'); no external best-state "
                    "checkpoint/restore needed -- see module docstring.",
        },
        "dumps": args.dumps,
        "aggregate": agg,
        "per_dump": per_dump,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
