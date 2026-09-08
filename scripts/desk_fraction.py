#!/usr/bin/env python
"""
desk_fraction.py — training-fraction (learning-curve) desk measurement for the SVHN@400 gate
decision, with the early-stopping optimism removed from the score.

Question (pre-registered by the caller, not tuned here): at CTrL SVHN@400 (task 3 of the s_minus
stream), how much of the code-length gap between the GROW rung and the SEARCH / REUSE rungs is a
function of the TRAINING FRACTION the probe sees, once the min-over-evaluations (early-stopping)
optimism has been removed from the reported bits?

Two competing readings are on the table and this script takes neither side:
  * `prequential-grow-probe.md` (H4): the single-split estimator trains grow on a strict subset of
    the data it will be deployed on, so L_grow is biased UP relative to reuse/search by a
    training-fraction (learning-curve) effect;
  * `h4-critique.md` (§1-3): the A0->A1 shift is early-stopping optimism, and the SVHN decision is
    variance-limited rather than fraction-limited.

Protocol (fixed; do not tune)
-----------------------------
For each gate dump, task 3, R resamples (private ``torch.Generator`` seeds 0..R-1):
  * hold out a fixed 20 % SCORE set and a fixed 15 % SELECT set;
  * early stopping selects the epoch on SELECT, the reported bits are measured on SCORE
    (``_held_out_codelength(..., score_X=SCORE)``) — this removes the min-over-epochs optimism;
  * from the remaining 65 % pool, train each rung on nested training FRACTIONS
    f in {0.25, 0.5, 0.75, 1.0} (same order, nested subsets), 40 epochs, lr 1e-3:
        reuse  = ReuseComposer
        search = search_compose(..., skip=True, budget=6, rank=16,
                                baseline_L = reuse's SELECT-early-stopped SCORE bits)
        grow   = _RootGrowModel(ConceptModule(in_dim=F, hidden=D, out=D, n_layers, n_parents=0))
        null   = head on a zero embedding
  * at f = 1.0 only, each rung is ALSO run with no score set, giving the SELECT-set
    min-over-epochs bits (the optimistic, single-split-style value);
    optimism_r = select_bits_r - score_bits_r.

Outputs (printed and written to --out as JSON)
  (a) per rung: mean +/- SE over resamples of L(f) at each f, and the slope L(1.0) - L(0.5)
      (n goes 130 -> 260, exactly one doubling, so this is "bits per doubling of n");
  (b) D(f) = L_grow(f) - L_search(f) with SE, and a power-law extrapolation
      L(n) ~ a + c * n^(-e) from the four fractions to n_deploy = the full task n (400), with the
      exponent fitted by grid search and also fixed at 0.5 and 1;
  (c) the early-stopping optimism per rung (mean, SE);
  (d) decision counts under (i) single-split-style optimistic bits, (ii) honest SCORE bits at
      f = 1.0, (iii) extrapolated bits, using
        grow   if (L_search - L_grow) / (L_null - L_grow) > eps
        search if (L_reuse  - L_search) / (L_null - L_grow) > eps
        reuse  otherwise.

CPU only. Reuses the machinery of ``scripts/desk_stage.py`` (stack rebuild + rung fits) and
``concept_dag.training.kan_gate``; adds nothing to the library.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from concept_dag.training.kan_gate import (  # noqa: E402
    SearchComposer,
    _enumerate_subsets,
    _held_out_codelength,
    classification_task,
)
from desk_stage import (  # noqa: E402
    _NullModel,
    _find_decision,
    _fit_grow,
    _fit_reuse,
    _fit_search,
    load_dump,
    rebuild_parent_stack,
)

DEVICE = "cpu"
RUNGS = ("reuse", "search", "grow", "null")


# ---------------------------------------------------------------------------
# small stats helpers
# ---------------------------------------------------------------------------


def _mean(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return (sum(xs) / len(xs)) if xs else None


def _se(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var / len(xs))


def _ms(xs):
    return {"mean": _mean(xs), "se": _se(xs), "n": len([x for x in xs if x is not None])}


# ---------------------------------------------------------------------------
# power-law extrapolation:  L(n) ~ a + c * n^(-e)
# ---------------------------------------------------------------------------


def _fit_powerlaw(ns, bits, exponent):
    """Closed-form OLS of ``bits`` on {1, n^(-e)}; returns (a, c, sse) or None if singular."""
    e = float(exponent)
    u = [float(n) ** (-e) for n in ns]
    m = len(u)
    if m < 2:
        return None
    um = sum(u) / m
    bm = sum(bits) / m
    s_uu = sum((x - um) ** 2 for x in u)
    if s_uu <= 1e-24 * max(1.0, sum(x * x for x in u)):
        return None
    c = sum((u[i] - um) * (bits[i] - bm) for i in range(m)) / s_uu
    a = bm - c * um
    sse = sum((bits[i] - (a + c * u[i])) ** 2 for i in range(m))
    return a, c, sse


def extrapolate(ns, bits, n_deploy, exponent=None, grid=None):
    """Predict the code length at ``n_deploy``.

    ``exponent=None`` fits the exponent over ``grid`` by minimum SSE; otherwise the exponent is
    fixed. Returns ``{"pred", "exponent", "a", "c", "sse"}`` (pred = last observed bits when the
    design is singular). No clamping: unlike ``kan_gate._fit_tail`` this is a measurement, and the
    caller wants the raw extrapolation, including where it is wild.
    """
    if exponent is not None:
        fit = _fit_powerlaw(ns, bits, exponent)
        if fit is None:
            return {"pred": float(bits[-1]), "exponent": float(exponent), "a": None,
                    "c": None, "sse": None, "singular": True}
        a, c, sse = fit
        return {"pred": a + c * float(n_deploy) ** (-float(exponent)),
                "exponent": float(exponent), "a": a, "c": c, "sse": sse, "singular": False}

    grid = grid if grid is not None else [round(0.05 * k, 4) for k in range(1, 61)]  # 0.05 .. 3.0
    best = None
    for e in grid:
        fit = _fit_powerlaw(ns, bits, e)
        if fit is None:
            continue
        a, c, sse = fit
        if best is None or sse < best[3]:
            best = (e, a, c, sse)
    if best is None:
        return {"pred": float(bits[-1]), "exponent": None, "a": None, "c": None,
                "sse": None, "singular": True}
    e, a, c, sse = best
    return {"pred": a + c * float(n_deploy) ** (-e), "exponent": e, "a": a, "c": c,
            "sse": sse, "singular": False}


# ---------------------------------------------------------------------------
# the decision rule (grow denominator, as pre-registered)
# ---------------------------------------------------------------------------


def decide(L_reuse, L_search, L_grow, L_null, eps=0.05):
    reducible = max(L_null - L_grow, 1e-6)
    rel_grow = (L_search - L_grow) / reducible
    rel_search = (L_reuse - L_search) / reducible
    if rel_grow > eps:
        return "grow", rel_grow, rel_search
    if rel_search > eps:
        return "search", rel_grow, rel_search
    return "reuse", rel_grow, rel_search


# ---------------------------------------------------------------------------
# null rung with an optional score set (desk_stage's _fit_null has neither, and the protocol
# wants every rung on the same 40 epochs and the same SCORE set)
# ---------------------------------------------------------------------------


def _fit_null(parent_stack, y, tr_idx, sel_idx, D, spec, epochs, seed, lr=1e-3, score_idx=None):
    Xtr, ytr = parent_stack[tr_idx], y[tr_idx]
    Xsel, ysel = parent_stack[sel_idx], y[sel_idx]
    Xsc, ysc = (parent_stack[score_idx], y[score_idx]) if score_idx is not None else (None, None)
    torch.manual_seed(seed)
    model = _NullModel(spec.make_head(D), D)
    return _held_out_codelength(model, lambda m, xb: m(xb), spec, Xtr, ytr, Xsel, ysel,
                                n_epochs=epochs, lr=lr, device=DEVICE,
                                score_X=Xsc, score_y=ysc)


# ---------------------------------------------------------------------------
# ADDENDUM (not part of the pre-registered protocol; reported separately)
#
# ``search_compose`` with a score set selects its best-of-``budget`` candidate by the value
# ``_held_out_codelength`` RETURNS, which with ``score_X`` set is the SCORE-set bits. So the search
# rung keeps a candidate-selection optimism on the very set the comparison is scored on, while
# reuse/grow/null (single candidates) do not. That biases D = L_grow - L_search upward, in search's
# favour, by an amount the main protocol cannot see. This addendum re-runs search at f = 1.0 with
# the candidate chosen on the SELECT set and only the winner reported on SCORE, and reports the
# difference. It changes nothing in the main tables.
# ---------------------------------------------------------------------------


def search_select_on_select(parent_stack, y, tr_idx, sel_idx, sc_idx, D, P, spec, epochs, seed,
                            baseline_sel, baseline_sc, lr=1e-3, budget=6, rank=16):
    """Best-of-``budget`` SearchComposer, SELECTED on the select set, REPORTED on the score set.

    Each candidate is fitted twice with the same global seed (identical trajectories, since the
    score-set evaluation consumes no RNG): once reading the select set, once reading the score set.
    ``baseline_sel/baseline_sc`` are the reuse rung's select/score bits — the trivial composition,
    exactly as ``search_compose``'s ``baseline_L`` seeds the search space.
    """
    Xtr, ytr = parent_stack[tr_idx], y[tr_idx]
    Xsel, ysel = parent_stack[sel_idx], y[sel_idx]
    Xsc, ysc = parent_stack[sc_idx], y[sc_idx]
    subsets = _enumerate_subsets(P)
    candidates, sd = [], 0
    while len(candidates) < budget:
        for sub in subsets:
            candidates.append((sub, sd))
            if len(candidates) >= budget:
                break
        sd += 1

    def _fit(sub, init_seed, score):
        torch.manual_seed(seed)                       # training-permutation stream
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            model = SearchComposer(parent_dim=D, n_parents=P, head=spec.make_head(D),
                                   rank=rank, subset=sub, skip=True)
        return _held_out_codelength(model, lambda m, xb: m(xb), spec, Xtr, ytr, Xsel, ysel,
                                    n_epochs=epochs, lr=lr, device=DEVICE,
                                    score_X=(Xsc if score else None),
                                    score_y=(ysc if score else None))

    best_sel, best_sc, best_sub = float(baseline_sel), float(baseline_sc), None
    for sub, init_seed in candidates:
        L_sel = _fit(sub, init_seed, score=False)
        if L_sel < best_sel:
            best_sel, best_sc, best_sub = L_sel, _fit(sub, init_seed, score=True), sub
    return best_sc, best_sel, (list(best_sub) if best_sub is not None else None)


# ---------------------------------------------------------------------------
# one dump
# ---------------------------------------------------------------------------


def run_dump(path, task, R, fractions, epochs, lr, score_frac, select_frac,
             budget, rank, eps, n_deploy_override=None, verbose=True, addendum=False):
    dump = load_dump(path)
    parent_stack, raw, y, parent_idx = rebuild_parent_stack(dump, task, split="train")
    F, D = dump["feature_dim"], dump["concept_dim"]
    n_layers = dump["config"]["n_mlp_layers"]
    n_classes = dump["tasks"][task]["n_classes"]
    spec = classification_task(n_classes)
    P = max(parent_stack.shape[1], 1)
    n = raw.shape[0]
    n_deploy = int(n_deploy_override) if n_deploy_override else int(n)

    dec = _find_decision(dump, task)

    n_score = int(round(score_frac * n))
    n_select = int(round(select_frac * n))

    resamples = []
    for r in range(R):
        gen = torch.Generator().manual_seed(r)
        perm = torch.randperm(n, generator=gen)
        score_idx = perm[:n_score]
        select_idx = perm[n_score:n_score + n_select]
        pool = perm[n_score + n_select:]
        n_pool = int(pool.numel())

        by_f = {}
        for fi, f in enumerate(fractions):
            k = max(2, int(round(f * n_pool)))
            tr_idx = pool[:k]                      # nested subsets, same order
            seed = r * 1000 + fi * 10              # deterministic, rung-shared init seed

            L_reuse = _fit_reuse(parent_stack, y, tr_idx, select_idx, D, P, spec, epochs,
                                 seed=seed, lr=lr, score_idx=score_idx)
            L_search, cfg = _fit_search(parent_stack, y, tr_idx, select_idx, D, P, spec, epochs,
                                        seed=seed, baseline_L=L_reuse, lr=lr, score_idx=score_idx)
            L_grow = _fit_grow(raw, y, tr_idx, select_idx, F, D, n_layers, spec, epochs,
                               seed=seed, lr=lr, score_idx=score_idx, tag=f"frac_grow_{r}_{fi}")
            L_null = _fit_null(parent_stack, y, tr_idx, select_idx, D, spec, epochs,
                               seed=seed, lr=lr, score_idx=score_idx)

            entry = {"f": f, "n_train": int(k),
                     "score": {"reuse": L_reuse, "search": L_search, "grow": L_grow, "null": L_null},
                     "search_trivial": bool(cfg.get("trivial", False)),
                     "search_subset": list(cfg.get("subset", ()))}

            if abs(f - 1.0) < 1e-9:
                # Same seeds, no score set -> the SELECT-set min-over-epochs bits: the optimistic,
                # single-split-style value. Trajectories are identical (the score-set evaluation
                # consumes no RNG), so this is the same fit read on the selection set.
                O_reuse = _fit_reuse(parent_stack, y, tr_idx, select_idx, D, P, spec, epochs,
                                     seed=seed, lr=lr, score_idx=None)
                O_search, _ = _fit_search(parent_stack, y, tr_idx, select_idx, D, P, spec, epochs,
                                          seed=seed, baseline_L=O_reuse, lr=lr, score_idx=None)
                O_grow = _fit_grow(raw, y, tr_idx, select_idx, F, D, n_layers, spec, epochs,
                                   seed=seed, lr=lr, score_idx=None, tag=f"frac_growO_{r}_{fi}")
                O_null = _fit_null(parent_stack, y, tr_idx, select_idx, D, spec, epochs,
                                   seed=seed, lr=lr, score_idx=None)
                entry["select"] = {"reuse": O_reuse, "search": O_search,
                                   "grow": O_grow, "null": O_null}
                entry["optimism"] = {k2: entry["select"][k2] - entry["score"][k2] for k2 in RUNGS}

                if addendum:
                    sc_bits, sel_bits, sub = search_select_on_select(
                        parent_stack, y, tr_idx, select_idx, score_idx, D, P, spec, epochs,
                        seed=seed, baseline_sel=O_reuse, baseline_sc=L_reuse, lr=lr,
                        budget=budget, rank=rank)
                    entry["search_sel_on_select"] = {
                        "score_bits": sc_bits, "select_bits": sel_bits, "subset": sub,
                        "candidate_selection_optimism": L_search - sc_bits,
                        "D_grow_minus_search": L_grow - sc_bits,
                    }

            by_f[f] = entry

        # --- per-resample extrapolation of every rung ---
        ns = [by_f[f]["n_train"] for f in fractions]
        extrap = {}
        for rung in RUNGS:
            bits = [by_f[f]["score"][rung] for f in fractions]
            extrap[rung] = {
                "fitted": extrapolate(ns, bits, n_deploy, exponent=None),
                "e0.5": extrapolate(ns, bits, n_deploy, exponent=0.5),
                "e1.0": extrapolate(ns, bits, n_deploy, exponent=1.0),
            }

        # --- decisions under the three bit sets ---
        s1 = by_f[1.0]["score"]
        o1 = by_f[1.0]["select"]
        decisions = {
            "optimistic_f1": decide(o1["reuse"], o1["search"], o1["grow"], o1["null"], eps)[0],
            "honest_f1": decide(s1["reuse"], s1["search"], s1["grow"], s1["null"], eps)[0],
        }
        for key in ("fitted", "e0.5", "e1.0"):
            decisions[f"extrap_{key}"] = decide(
                extrap["reuse"][key]["pred"], extrap["search"][key]["pred"],
                extrap["grow"][key]["pred"], extrap["null"][key]["pred"], eps)[0]

        resamples.append({"resample": r, "n_pool": n_pool,
                          "n_score": int(score_idx.numel()), "n_select": int(select_idx.numel()),
                          "by_f": {str(f): by_f[f] for f in fractions},
                          "extrap": extrap, "decisions": decisions})
        if verbose:
            d10 = by_f[1.0]["score"]["grow"] - by_f[1.0]["score"]["search"]
            d05 = by_f[0.5]["score"]["grow"] - by_f[0.5]["score"]["search"]
            print(f"    r={r:2d}  D(0.5)={d05:+.4f}  D(1.0)={d10:+.4f}  "
                  f"honest={decisions['honest_f1']:6s} optimistic={decisions['optimistic_f1']:6s} "
                  f"extrap={decisions['extrap_fitted']}")

    return {
        "dump": path, "task": task, "n": int(n), "n_deploy": n_deploy,
        "feature_dim": F, "concept_dim": D, "n_parents": int(parent_stack.shape[1]),
        "recorded_decision": dec.get("decision"),
        "recorded_bits": {"reuse": dec.get("L_reuse_bits"), "search": dec.get("L_search_bits"),
                          "grow": dec.get("L_grow_bits"), "null": dec.get("L_null_bits")},
        "resamples": resamples,
    }


# ---------------------------------------------------------------------------
# aggregation over (dump, resample) units
# ---------------------------------------------------------------------------


def aggregate(per_dump, fractions, n_deploy, eps):
    units = [(d, rs) for d in per_dump for rs in d["resamples"]]

    # (a) per rung, per fraction
    curves = {}
    for rung in RUNGS:
        curves[rung] = {}
        for f in fractions:
            curves[rung][str(f)] = _ms([rs["by_f"][str(f)]["score"][rung] for _, rs in units])
        slope = [rs["by_f"]["1.0"]["score"][rung] - rs["by_f"]["0.5"]["score"][rung]
                 for _, rs in units]
        curves[rung]["slope_0.5_to_1.0_per_doubling"] = _ms(slope)

    # (b) the gap D(f) = L_grow - L_search  (and, for context, L_grow - L_reuse)
    gap = {}
    for f in fractions:
        gap[str(f)] = _ms([rs["by_f"][str(f)]["score"]["grow"] - rs["by_f"][str(f)]["score"]["search"]
                           for _, rs in units])
    d_slope = [(rs["by_f"]["1.0"]["score"]["grow"] - rs["by_f"]["1.0"]["score"]["search"])
               - (rs["by_f"]["0.5"]["score"]["grow"] - rs["by_f"]["0.5"]["score"]["search"])
               for _, rs in units]
    gap["slope_0.5_to_1.0_per_doubling"] = _ms(d_slope)
    gap_reuse = {str(f): _ms([rs["by_f"][str(f)]["score"]["grow"] - rs["by_f"][str(f)]["score"]["reuse"]
                              for _, rs in units]) for f in fractions}
    gap_reuse["slope_0.5_to_1.0_per_doubling"] = _ms(
        [(rs["by_f"]["1.0"]["score"]["grow"] - rs["by_f"]["1.0"]["score"]["reuse"])
         - (rs["by_f"]["0.5"]["score"]["grow"] - rs["by_f"]["0.5"]["score"]["reuse"])
         for _, rs in units])

    # extrapolated bits and gap
    extrap = {}
    for key in ("fitted", "e0.5", "e1.0"):
        extrap[key] = {rung: _ms([rs["extrap"][rung][key]["pred"] for _, rs in units])
                       for rung in RUNGS}
        extrap[key]["D_grow_minus_search"] = _ms(
            [rs["extrap"]["grow"][key]["pred"] - rs["extrap"]["search"][key]["pred"]
             for _, rs in units])
        extrap[key]["exponent"] = {rung: _ms([rs["extrap"][rung][key]["exponent"] for _, rs in units])
                                   for rung in RUNGS}

    # (c) early-stopping optimism
    optimism = {rung: _ms([rs["by_f"]["1.0"]["optimism"][rung] for _, rs in units])
                for rung in RUNGS}
    optimism["search_minus_grow"] = _ms(
        [rs["by_f"]["1.0"]["optimism"]["search"] - rs["by_f"]["1.0"]["optimism"]["grow"]
         for _, rs in units])

    # (d) decision counts
    keys = ["optimistic_f1", "honest_f1", "extrap_fitted", "extrap_e0.5", "extrap_e1.0"]
    counts = {}
    for k in keys:
        c = {"grow": 0, "search": 0, "reuse": 0}
        for _, rs in units:
            c[rs["decisions"][k]] += 1
        counts[k] = c

    per_dump_counts = {}
    for d in per_dump:
        tag = os.path.basename(os.path.dirname(os.path.dirname(d["dump"]))) or d["dump"]
        per_dump_counts[tag] = {k: {"grow": 0, "search": 0, "reuse": 0} for k in keys}
        for rs in d["resamples"]:
            for k in keys:
                per_dump_counts[tag][k][rs["decisions"][k]] += 1

    add = None
    if all("search_sel_on_select" in rs["by_f"]["1.0"] for _, rs in units):
        a_sc = [rs["by_f"]["1.0"]["search_sel_on_select"]["score_bits"] for _, rs in units]
        a_D = [rs["by_f"]["1.0"]["search_sel_on_select"]["D_grow_minus_search"] for _, rs in units]
        a_opt = [rs["by_f"]["1.0"]["search_sel_on_select"]["candidate_selection_optimism"]
                 for _, rs in units]
        a_dec = {"grow": 0, "search": 0, "reuse": 0}
        for _, rs in units:
            s1 = rs["by_f"]["1.0"]["score"]
            a_dec[decide(s1["reuse"],
                         rs["by_f"]["1.0"]["search_sel_on_select"]["score_bits"],
                         s1["grow"], s1["null"], eps)[0]] += 1
        add = {"L_search_f1": _ms(a_sc), "D_grow_minus_search_f1": _ms(a_D),
               "candidate_selection_optimism": _ms(a_opt), "decision_counts": a_dec}

    return {"addendum_search_selected_on_select": add,
            "n_units": len(units), "curves": curves, "gap_grow_minus_search": gap,
            "gap_grow_minus_reuse": gap_reuse, "extrap": extrap, "optimism": optimism,
            "decision_counts": counts, "decision_counts_per_dump": per_dump_counts,
            "n_deploy": n_deploy, "eps": eps}


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------


def _fmt(ms, w=8, p=4):
    if ms is None or ms["mean"] is None:
        return " " * w + "  n/a   "
    se = ms["se"]
    return f"{ms['mean']:+{w}.{p}f} +/-{(se if se is not None else float('nan')):.4f}"


def print_report(agg, fractions, n_deploy):
    fr = [str(f) for f in fractions]
    print("\n" + "=" * 96)
    print(f"(a) L(f) on the SCORE set, bits/sample — mean +/- SE over {agg['n_units']} "
          f"(dump x resample) units")
    print("=" * 96)
    header = "rung    " + "".join(f"  f={f:<20s}" for f in fr) + "  slope 0.5->1.0/doubling"
    print(header)
    for rung in RUNGS:
        row = f"{rung:<8s}"
        for f in fr:
            row += "  " + _fmt(agg["curves"][rung][f])
        row += "     " + _fmt(agg["curves"][rung]["slope_0.5_to_1.0_per_doubling"])
        print(row)

    print("\n" + "=" * 96)
    print("(b) gap D(f) = L_grow(f) - L_search(f)  [>0 favours search]")
    print("=" * 96)
    row = f"{'D(f)':<8s}"
    for f in fr:
        row += "  " + _fmt(agg["gap_grow_minus_search"][f])
    row += "     " + _fmt(agg["gap_grow_minus_search"]["slope_0.5_to_1.0_per_doubling"])
    print(row)
    row = f"{'G-R(f)':<8s}"
    for f in fr:
        row += "  " + _fmt(agg["gap_grow_minus_reuse"][f])
    row += "     " + _fmt(agg["gap_grow_minus_reuse"]["slope_0.5_to_1.0_per_doubling"])
    print(row)

    print(f"\n  power-law extrapolation  L(n) ~ a + c*n^(-e)  to n_deploy = {n_deploy}")
    print(f"  {'exponent':<12s} {'L_grow':<22s} {'L_search':<22s} {'D = grow - search':<22s} e_fit(grow/search)")
    for key, lbl in (("fitted", "fitted"), ("e0.5", "fixed 0.5"), ("e1.0", "fixed 1.0")):
        e = agg["extrap"][key]
        eg, es = e["exponent"]["grow"]["mean"], e["exponent"]["search"]["mean"]
        etxt = (f"{eg:.2f}/{es:.2f}" if eg is not None and es is not None else "-")
        print(f"  {lbl:<12s} {_fmt(e['grow'])}  {_fmt(e['search'])}  "
              f"{_fmt(e['D_grow_minus_search'])}  {etxt}")

    print("\n" + "=" * 96)
    print("(c) early-stopping optimism at f=1.0:  select_min_bits - score_bits\n      [NEGATIVE = optimistic: the min-over-epochs value on SELECT sits BELOW the honest SCORE bits]")
    print("=" * 96)
    for rung in RUNGS:
        print(f"  {rung:<8s} {_fmt(agg['optimism'][rung])}")
    print(f"  {'srch-grw':<8s} {_fmt(agg['optimism']['search_minus_grow'])}")

    print("\n" + "=" * 96)
    print(f"(d) decision counts over {agg['n_units']} units (eps = {agg['eps']}, grow denominator)")
    print("=" * 96)
    print(f"  {'bit set':<22s} {'grow':>6s} {'search':>7s} {'reuse':>7s}")
    for k, v in agg["decision_counts"].items():
        print(f"  {k:<22s} {v['grow']:>6d} {v['search']:>7d} {v['reuse']:>7d}")
    print("\n  per dump (grow/search/reuse):")
    print(f"  {'dump':<10s}" + "".join(f"{k:>22s}" for k in agg["decision_counts"]))
    for tag, cc in agg["decision_counts_per_dump"].items():
        print(f"  {tag:<10s}" + "".join(
            f"{cc[k]['grow']:>8d}/{cc[k]['search']:d}/{cc[k]['reuse']:<11d}"
            for k in agg["decision_counts"]))

    add = agg.get("addendum_search_selected_on_select")
    if add:
        print("\n" + "=" * 96)
        print("ADDENDUM (not pre-registered): search with its best-of-6 candidate chosen on SELECT")
        print("=" * 96)
        print(f"  L_search(f=1.0)                        {_fmt(add['L_search_f1'])}")
        print(f"  candidate-selection optimism it removes{_fmt(add['candidate_selection_optimism'])}")
        print(f"  D = L_grow - L_search at f=1.0         {_fmt(add['D_grow_minus_search_f1'])}")
        c = add["decision_counts"]
        print(f"  decisions  grow/search/reuse           {c['grow']}/{c['search']}/{c['reuse']}")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", type=int, default=3)
    ap.add_argument("--resamples", type=int, default=12)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--score_frac", type=float, default=0.20)
    ap.add_argument("--select_frac", type=float, default=0.15)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--n_deploy", type=int, default=None,
                    help="deployed sample size for the extrapolation (default: the task's full n)")
    ap.add_argument("--addendum", action="store_true",
                    help="also measure the search rung with its candidate chosen on SELECT "
                         "instead of SCORE (see the ADDENDUM note); reported separately")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    # desk_stage._fit_search hard-codes the published search settings; the flags exist to make the
    # protocol explicit and are checked, not silently ignored.
    if (args.budget, args.rank) != (6, 16):
        raise SystemExit("--budget/--rank are fixed at 6/16 by scripts/desk_stage._fit_search "
                         "(skip=True); change desk_stage.py to vary them.")

    per_dump = []
    for path in args.dumps:
        tag = os.path.basename(os.path.dirname(os.path.dirname(path))) or path
        print(f"[{tag}] task {args.task}, R={args.resamples}, f={args.fractions}")
        per_dump.append(run_dump(path, args.task, args.resamples, args.fractions, args.epochs,
                                 args.lr, args.score_frac, args.select_frac, args.budget,
                                 args.rank, args.eps, args.n_deploy,
                                 addendum=args.addendum))

    n_deploy = per_dump[0]["n_deploy"]
    agg = aggregate(per_dump, args.fractions, n_deploy, args.eps)
    print_report(agg, args.fractions, n_deploy)

    result = {
        "protocol": {
            "task": args.task, "resamples": args.resamples, "fractions": args.fractions,
            "epochs": args.epochs, "lr": args.lr, "score_frac": args.score_frac,
            "select_frac": args.select_frac, "search_budget": args.budget,
            "search_rank": args.rank, "search_skip": True, "eps": args.eps,
            "denominator": "L_null - L_grow", "n_deploy": n_deploy,
            "scoring": "early stopping on SELECT, bits reported on disjoint SCORE set",
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
