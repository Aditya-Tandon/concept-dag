#!/usr/bin/env python
"""
desk_stage.py — the desk-stage (no-GPU, cached-tensor) analysis for the three pre-registered
gate-estimator / denominator / update-arm hypotheses (see INTERFACE_SPEC.md §8 and the vault
notes it links: small-n-codelength-estimator-stress-test.md, best-rung-denominator-stress-test.md,
reuse-with-update-arm-stress-test.md).

Reads one or more ``gate_dump.pt`` files (§6 format) and, without any GPU / re-encoding, rebuilds
the frozen parent stacks from the recorded ConceptModule states (all nodes are roots in these runs,
so ``emb_i = ConceptModule(state_i)(train_raw)``) to run:

  H1'a — split variance: resample the SVHN@400 (t=3) held-out split ``n_splits`` times and see how
         often sign(L_search - L_grow) flips relative to the recorded decision.
  H1'b — selection bias: on the same splits, score each rung on a set disjoint from the one used for
         early stopping / candidate selection, and compare the resulting optimism per rung.
  H3' (i)  — t=1 (Fashion@400, MNIST root parent): update_probe vs a fresh grow/reuse on one split,
             plus the backward-safety signal (t0 validation accuracy before/after swapping the copy in).
  H3' (ii) — same on the --s_plus dump(s) at t=4: rel_update, delta_t0_val.
  Sensitivity — fine-tune a copy of the MNIST root on Fashion with NO backward-safety check, to show
                the un-gated failure mode (delta_t0_val_broken, expected < -0.01).

Every number this script prints is also written to --out as JSON. CPU-only; torch + numpy + argparse
+ json only (no GPU, no other project modules besides the gate core and ConceptModule).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn

# Make the repo root importable when this file is run directly (``python scripts/desk_stage.py``).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from concept_dag.training.kan_gate import (  # noqa: E402
    ReuseComposer,
    SearchComposer,  # noqa: F401  (re-exported for callers / completeness; not used directly here)
    _held_out_codelength,
    _RootGrowModel,
    classification_task,
    pack_update_input,  # noqa: F401  (used implicitly by update_probe; kept for parity with the spec)
    search_compose,
    update_probe,
)
from concept_dag.modules.concept_module import ConceptModule  # noqa: E402
from concept_dag.models.baselines import LinearHead  # noqa: E402

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Small local pieces the gate core deliberately keeps private (not in the
# allowed import list): the null (marginal) model.
# ---------------------------------------------------------------------------


class _NullModel(nn.Module):
    """Best input-independent predictor — mirrors kan_gate.py's private ``_NullModel``."""

    def __init__(self, head: nn.Module, concept_dim: int):
        super().__init__()
        self.head = head
        self.concept_dim = concept_dim

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        return self.head(torch.zeros(xb.shape[0], self.concept_dim, device=xb.device))


# ---------------------------------------------------------------------------
# Dump loading / rebuilding
# ---------------------------------------------------------------------------


def load_dump(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def _find_decision(dump: dict, task: int) -> dict:
    for d in dump["decisions"]:
        if d["task"] == task:
            return d
    raise KeyError(f"no decision recorded for task {task} in dump")


def build_root_module(dump: dict, node_index: int) -> ConceptModule:
    """Rebuild a frozen root ConceptModule from its recorded state (§6: all nodes are roots)."""
    node = dump["nodes"][node_index]
    n_layers = dump["config"]["n_mlp_layers"]
    m = ConceptModule(
        module_id=f"reload_{node_index}",
        in_dim=dump["feature_dim"],
        hidden_dim=dump["concept_dim"],
        out_dim=dump["concept_dim"],
        n_layers=n_layers,
        n_parents=0,
    )
    m.load_state_dict(node["state_dict"])
    m.eval()
    return m


def rebuild_parent_stack(dump: dict, task: int, split: str = "train"):
    """(parent_stack (N,P,D), raw (N,F), y (N,)) for ``task``, split in {train, val, test}."""
    dec = _find_decision(dump, task)
    parent_idx = dec.get("parents", [])
    tdata = dump["tasks"][task]
    raw = tdata[f"{split}_raw"]
    y = tdata[f"{split}_y"]
    embs = []
    with torch.no_grad():
        for idx in parent_idx:
            module = build_root_module(dump, idx)
            embs.append(module(raw))
    if embs:
        stack = torch.stack(embs, dim=1)
    else:
        stack = torch.zeros(raw.shape[0], 0, dump["concept_dim"])
    return stack, raw, y, parent_idx


def find_predictor(dump: dict, task: int) -> dict:
    for p in dump["predictors"]:
        if p["task"] == task:
            return p
    raise KeyError(f"no predictor recorded for task {task} in dump")


def build_head(dump: dict, task: int) -> LinearHead:
    pred = find_predictor(dump, task)
    n_classes = dump["tasks"][task]["n_classes"]
    head = LinearHead(dump["concept_dim"], n_classes)
    head.load_state_dict(pred["head_state"])
    head.eval()
    return head


@torch.no_grad()
def eval_head_acc(module: ConceptModule, head: LinearHead, x: torch.Tensor, y: torch.Tensor) -> float:
    logits = head(module(x))
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


# ---------------------------------------------------------------------------
# H1'a / H1'b — one fresh split
# ---------------------------------------------------------------------------


def _fresh_split(n: int, val_fraction: float, seed: int):
    gen = torch.Generator().manual_seed(seed)
    n_val = max(1, int(round(val_fraction * n)))
    perm = torch.randperm(n, generator=gen)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    return tr_idx, val_idx


def _fit_reuse(parent_stack, y, tr_idx, val_idx, D, P, spec, epochs, seed, lr=1e-3,
               score_idx=None):
    Xtr, ytr = parent_stack[tr_idx], y[tr_idx]
    Xval, yval = parent_stack[val_idx], y[val_idx]
    Xsc, ysc = (parent_stack[score_idx], y[score_idx]) if score_idx is not None else (None, None)
    torch.manual_seed(seed)
    model = ReuseComposer(parent_dim=D, n_parents=max(P, 1), head=spec.make_head(D))
    return _held_out_codelength(model, lambda m, xb: m(xb), spec, Xtr, ytr, Xval, yval,
                                 n_epochs=epochs, lr=lr, device=DEVICE, score_X=Xsc, score_y=ysc)


def _fit_search(parent_stack, y, tr_idx, val_idx, D, P, spec, epochs, seed, baseline_L, lr=1e-3,
                 score_idx=None):
    Xtr, ytr = parent_stack[tr_idx], y[tr_idx]
    Xval, yval = parent_stack[val_idx], y[val_idx]
    Xsc, ysc = (parent_stack[score_idx], y[score_idx]) if score_idx is not None else (None, None)
    torch.manual_seed(seed)
    L, cfg, _trace = search_compose(Xtr, ytr, Xval, yval, spec, concept_dim=D, n_parents=max(P, 1),
                                     device=DEVICE, n_epochs=epochs, lr=lr, budget=6, rank=16,
                                     skip=True, baseline_L=baseline_L, score_X=Xsc, score_y=ysc)
    return L, cfg


def _fit_grow(raw, y, tr_idx, val_idx, F, D, n_layers, spec, epochs, seed, lr=1e-3,
              score_idx=None, tag="desk_grow"):
    Rtr, ytr = raw[tr_idx], y[tr_idx]
    Rval, yval = raw[val_idx], y[val_idx]
    Rsc, ysc = (raw[score_idx], y[score_idx]) if score_idx is not None else (None, None)
    torch.manual_seed(seed)
    module = ConceptModule(module_id=tag, in_dim=F, hidden_dim=D, out_dim=D, n_layers=n_layers, n_parents=0)
    model = _RootGrowModel(module, spec.make_head(D))
    return _held_out_codelength(model, lambda m, xb: m(xb), spec, Rtr, ytr, Rval, yval,
                                 n_epochs=epochs, lr=lr, device=DEVICE, score_X=Rsc, score_y=ysc)


def _fit_null(parent_stack, y, tr_idx, val_idx, D, spec, epochs, seed, lr=1e-3):
    Xtr, ytr = parent_stack[tr_idx], y[tr_idx]
    Xval, yval = parent_stack[val_idx], y[val_idx]
    torch.manual_seed(seed)
    model = _NullModel(spec.make_head(D), D)
    return _held_out_codelength(model, lambda m, xb: m(xb), spec, Xtr, ytr, Xval, yval,
                                 n_epochs=max(epochs // 2, 10), lr=lr, device=DEVICE)


def h1_one_dump(dump: dict, task: int, n_splits: int, epochs: int, val_fraction: float,
                 eps_grow: float, eps_search: float) -> dict:
    parent_stack, raw, y, parent_idx = rebuild_parent_stack(dump, task, split="train")
    F, D = dump["feature_dim"], dump["concept_dim"]
    n_layers = dump["config"]["n_mlp_layers"]
    n_classes = dump["tasks"][task]["n_classes"]
    spec = classification_task(n_classes)
    P = parent_stack.shape[1]
    n = raw.shape[0]

    dec = _find_decision(dump, task)
    recorded_L_search = dec.get("L_search_bits")
    recorded_L_grow = dec.get("L_grow_bits")
    recorded_sign = None
    if recorded_L_search is not None and recorded_L_grow is not None:
        d = recorded_L_search - recorded_L_grow
        recorded_sign = 1 if d > 0 else (-1 if d < 0 else 0)

    splits = []
    flips = 0
    n_flip_checked = 0
    bias_reuse_list, bias_search_list, bias_grow_list = [], [], []

    for s in range(n_splits):
        tr_idx, val_idx = _fresh_split(n, val_fraction, seed=s)

        # --- H1'a: single-estimator four rungs on this split. ---
        L_reuse = _fit_reuse(parent_stack, y, tr_idx, val_idx, D, P, spec, epochs, seed=s)
        L_search, search_cfg = _fit_search(parent_stack, y, tr_idx, val_idx, D, P, spec, epochs,
                                            seed=s, baseline_L=L_reuse)
        L_grow = _fit_grow(raw, y, tr_idx, val_idx, F, D, n_layers, spec, epochs, seed=s)
        L_null = _fit_null(parent_stack, y, tr_idx, val_idx, D, spec, epochs, seed=s)

        reducible = max(L_null - L_grow, 1e-6)
        rel_search = (L_reuse - L_search) / reducible
        rel_grow = (L_search - L_grow) / reducible
        if rel_grow > eps_grow:
            decision = "grow"
        elif rel_search > eps_search:
            decision = "search"
        else:
            decision = "reuse"

        sign = 1 if (L_search - L_grow) > 0 else (-1 if (L_search - L_grow) < 0 else 0)
        if recorded_sign is not None:
            n_flip_checked += 1
            if sign != recorded_sign:
                flips += 1

        # --- H1'b: same split, held-out half split into select / score. ---
        n_val = val_idx.shape[0]
        half = max(1, n_val // 2)
        sel_idx, sco_idx = val_idx[:half], val_idx[half:]
        if sco_idx.shape[0] == 0:
            sco_idx = sel_idx  # degenerate tiny-n fallback; bias is then ~0, reported honestly

        reuse_sel = _fit_reuse(parent_stack, y, tr_idx, sel_idx, D, P, spec, epochs, seed=1000 + s)
        reuse_sco = _fit_reuse(parent_stack, y, tr_idx, sel_idx, D, P, spec, epochs, seed=1000 + s,
                                score_idx=sco_idx)
        search_sel, _ = _fit_search(parent_stack, y, tr_idx, sel_idx, D, P, spec, epochs,
                                     seed=1000 + s, baseline_L=reuse_sel)
        search_sco, _ = _fit_search(parent_stack, y, tr_idx, sel_idx, D, P, spec, epochs,
                                     seed=1000 + s, baseline_L=reuse_sco, score_idx=sco_idx)
        grow_sel = _fit_grow(raw, y, tr_idx, sel_idx, F, D, n_layers, spec, epochs, seed=1000 + s,
                              tag="desk_grow_bias_sel")
        grow_sco = _fit_grow(raw, y, tr_idx, sel_idx, F, D, n_layers, spec, epochs, seed=1000 + s,
                              score_idx=sco_idx, tag="desk_grow_bias_sco")

        bias_reuse_list.append(reuse_sco - reuse_sel)
        bias_search_list.append(search_sco - search_sel)
        bias_grow_list.append(grow_sco - grow_sel)

        splits.append({
            "seed": s, "L_reuse": L_reuse, "L_search": L_search, "L_grow": L_grow, "L_null": L_null,
            "decision": decision, "sign_search_minus_grow": sign,
        })

    flip_frac = (flips / n_flip_checked) if n_flip_checked else None
    bias_reuse = sum(bias_reuse_list) / len(bias_reuse_list)
    bias_search = sum(bias_search_list) / len(bias_search_list)
    bias_grow = sum(bias_grow_list) / len(bias_grow_list)

    return {
        "task": task, "n_parents": P,
        "recorded_decision": dec.get("decision"),
        "recorded_L_search_bits": recorded_L_search, "recorded_L_grow_bits": recorded_L_grow,
        "recorded_sign_search_minus_grow": recorded_sign,
        "flip_frac": flip_frac, "n_flip_checked": n_flip_checked,
        "bias_reuse": bias_reuse, "bias_search": bias_search, "bias_grow": bias_grow,
        "splits": splits,
    }


def h1_branch(flip_frac: float, bias_search: float, bias_reuse: float, bias_grow: float) -> str:
    split_limited = (flip_frac is not None) and flip_frac >= 0.3
    selection_limited = (bias_search - max(bias_reuse, bias_grow)) >= 0.07
    if split_limited and selection_limited:
        return "BOTH"
    if split_limited:
        return "SPLIT-LIMITED"
    if selection_limited:
        return "SELECTION-LIMITED"
    return "NEITHER"


# ---------------------------------------------------------------------------
# H3' kills
# ---------------------------------------------------------------------------


def _find_root_parent(dump: dict, parent_idx: list, mnist_task_id: int):
    """Local index (into ``parent_idx``) of the parent that is a root belonging to
    ``mnist_task_id``; falls back to the first root parent, then to index 0."""
    for local, node_idx in enumerate(parent_idx):
        node = dump["nodes"][node_idx]
        if node.get("is_root") and node.get("task_id") == mnist_task_id:
            return local
    for local, node_idx in enumerate(parent_idx):
        if dump["nodes"][node_idx].get("is_root"):
            return local
    return 0


def h3_kill(dump: dict, task: int, mnist_task_id: int, epochs: int, val_fraction: float,
            update_lr: float = 1e-4, composer_lr: float = 1e-3, seed: int = 0) -> dict:
    parent_stack, raw, y, parent_idx = rebuild_parent_stack(dump, task, split="train")
    F, D = dump["feature_dim"], dump["concept_dim"]
    n_layers = dump["config"]["n_mlp_layers"]
    n_classes = dump["tasks"][task]["n_classes"]
    spec = classification_task(n_classes)
    P = max(parent_stack.shape[1], 1)
    n = raw.shape[0]

    p_local = _find_root_parent(dump, parent_idx, mnist_task_id)
    parent_module = build_root_module(dump, parent_idx[p_local])

    tr_idx, val_idx = _fresh_split(n, val_fraction, seed=seed)

    L_update, module_copy = update_probe(
        parent_stack, raw, y, parent_module, p_local, spec,
        concept_dim=D, n_parents=P, tr_idx=tr_idx, val_idx=val_idx, device=DEVICE,
        n_epochs=epochs, update_lr=update_lr, composer_lr=composer_lr,
    )
    L_reuse = _fit_reuse(parent_stack, y, tr_idx, val_idx, D, P, spec, epochs, seed=seed)
    L_grow = _fit_grow(raw, y, tr_idx, val_idx, F, D, n_layers, spec, epochs, seed=seed,
                        tag="h3_grow")
    rel_update = (L_reuse - L_update) / max(L_reuse, 1e-6)

    # --- backward-safety signal: t0 (mnist_task_id) validation accuracy before/after the swap. ---
    t0 = dump["tasks"][mnist_task_id]
    node0_idx = find_predictor(dump, mnist_task_id)["node_index"]
    head0 = build_head(dump, mnist_task_id)

    node0_before = build_root_module(dump, node0_idx)
    acc_before = eval_head_acc(node0_before, head0, t0["val_raw"], t0["val_y"])

    node0_after = build_root_module(dump, node0_idx)
    node0_after.load_state_dict(module_copy.state_dict())
    node0_after.eval()
    acc_after = eval_head_acc(node0_after, head0, t0["val_raw"], t0["val_y"])

    delta_t0_val = acc_after - acc_before

    return {
        "task": task, "mnist_task_id": mnist_task_id, "parent_local_index": p_local,
        "parent_node_index": parent_idx[p_local],
        "L_update": L_update, "L_reuse": L_reuse, "L_grow": L_grow, "rel_update": rel_update,
        "acc_before": acc_before, "acc_after": acc_after, "delta_t0_val": delta_t0_val,
    }


def h3_broken_sensitivity(dump: dict, fashion_task: int, mnist_task_id: int, epochs: int,
                           lr: float = 1e-3, batch_size: int = 128) -> dict:
    """Fine-tune a copy of the MNIST root on Fashion with NO backward-safety gating; report the
    t0 validation-accuracy delta (expected < -0.01: catastrophic forgetting)."""
    F, D = dump["feature_dim"], dump["concept_dim"]

    node0_idx = find_predictor(dump, mnist_task_id)["node_index"]
    head0 = build_head(dump, mnist_task_id)
    t0 = dump["tasks"][mnist_task_id]
    node0_before = build_root_module(dump, node0_idx)
    acc_before = eval_head_acc(node0_before, head0, t0["val_raw"], t0["val_y"])

    broken = build_root_module(dump, node0_idx)
    for p in broken.parameters():
        p.requires_grad_(True)
    broken._frozen = False
    fdata = dump["tasks"][fashion_task]
    raw, y = fdata["train_raw"], fdata["train_y"]
    n_classes = fdata["n_classes"]
    scratch_head = nn.Linear(D, n_classes)
    opt = torch.optim.Adam(list(broken.parameters()) + list(scratch_head.parameters()), lr=lr)
    n = raw.shape[0]
    torch.manual_seed(0)
    for _ in range(epochs):
        broken.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = scratch_head(broken(raw[idx]))
            loss = torch.nn.functional.cross_entropy(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    broken.eval()
    acc_after = eval_head_acc(broken, head0, t0["val_raw"], t0["val_y"])
    delta = acc_after - acc_before
    return {"fashion_task": fashion_task, "mnist_task_id": mnist_task_id,
            "acc_before": acc_before, "acc_after_broken": acc_after, "delta_t0_val_broken": delta}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", nargs="+", required=True, help="s_minus (or equivalent) gate_dump.pt path(s)")
    ap.add_argument("--s_plus", nargs="+", default=None, help="s_plus gate_dump.pt path(s) for the H3' kill (ii)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_splits", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--h1_task", type=int, default=3, help="SVHN@400 position in the CTrL stream")
    ap.add_argument("--h3_task", type=int, default=1, help="Fashion@400 position (MNIST root parent)")
    ap.add_argument("--splus_task", type=int, default=4, help="revisit task position in the s_plus stream")
    ap.add_argument("--mnist_task_id", type=int, default=0)
    ap.add_argument("--val_fraction", type=float, default=0.3)
    ap.add_argument("--eps_grow", type=float, default=0.05)
    ap.add_argument("--eps_search", type=float, default=0.05)
    ap.add_argument("--update_lr", type=float, default=1e-4)
    ap.add_argument("--composer_lr", type=float, default=1e-3)
    ap.add_argument("--broken_lr", type=float, default=1e-3)
    args = ap.parse_args()

    per_dump = []
    for path in args.dumps:
        dump = load_dump(path)
        h1 = h1_one_dump(dump, args.h1_task, args.n_splits, args.epochs, args.val_fraction,
                          args.eps_grow, args.eps_search)
        h3 = h3_kill(dump, args.h3_task, args.mnist_task_id, args.epochs, args.val_fraction,
                     args.update_lr, args.composer_lr)
        broken = h3_broken_sensitivity(dump, args.h3_task, args.mnist_task_id, args.epochs,
                                        args.broken_lr)
        entry = {"dump": path, "h1": h1, "h3_t1": h3, "sensitivity_broken": broken}
        per_dump.append(entry)
        print(f"[{os.path.basename(os.path.dirname(path)) or path}] "
              f"flip_frac={h1['flip_frac']} bias(reuse/search/grow)="
              f"{h1['bias_reuse']:.4f}/{h1['bias_search']:.4f}/{h1['bias_grow']:.4f} "
              f"t1_L_update={h3['L_update']:.4f} t1_L_grow={h3['L_grow']:.4f} "
              f"t1_delta_t0_val={h3['delta_t0_val']:.4f} "
              f"delta_t0_val_broken={broken['delta_t0_val_broken']:.4f}")

    flip_frac = _mean([e["h1"]["flip_frac"] for e in per_dump])
    bias_reuse = _mean([e["h1"]["bias_reuse"] for e in per_dump])
    bias_search = _mean([e["h1"]["bias_search"] for e in per_dump])
    bias_grow = _mean([e["h1"]["bias_grow"] for e in per_dump])
    branch = h1_branch(flip_frac, bias_search, bias_reuse, bias_grow)
    t1_L_update = _mean([e["h3_t1"]["L_update"] for e in per_dump])
    t1_L_grow = _mean([e["h3_t1"]["L_grow"] for e in per_dump])
    t1_delta_t0_val = _mean([e["h3_t1"]["delta_t0_val"] for e in per_dump])
    delta_t0_val_broken = _mean([e["sensitivity_broken"]["delta_t0_val_broken"] for e in per_dump])

    print(f"\nH1' desk stage: flip_frac={flip_frac} bias_search-max(bias_reuse,bias_grow)="
          f"{(bias_search - max(bias_reuse, bias_grow)) if None not in (bias_search, bias_reuse, bias_grow) else None} "
          f"=> {branch}")
    print(f"H3' t1 kill: L_update={t1_L_update} L_grow={t1_L_grow} delta_t0_val={t1_delta_t0_val}")
    print(f"Sensitivity (broken, no safety): delta_t0_val_broken={delta_t0_val_broken} (expect < -0.01)")

    result = {
        "dumps": args.dumps, "n_splits": args.n_splits, "epochs": args.epochs,
        "h1_task": args.h1_task, "h3_task": args.h3_task,
        "per_dump": per_dump,
        "flip_frac": flip_frac, "bias_reuse": bias_reuse, "bias_search": bias_search,
        "bias_grow": bias_grow, "h1_branch": branch,
        "t1_L_update": t1_L_update, "t1_L_grow": t1_L_grow, "t1_delta_t0_val": t1_delta_t0_val,
        "delta_t0_val_broken": delta_t0_val_broken,
    }

    if args.s_plus:
        s_plus_entries = []
        for path in args.s_plus:
            dump = load_dump(path)
            h3 = h3_kill(dump, args.splus_task, args.mnist_task_id, args.epochs, args.val_fraction,
                         args.update_lr, args.composer_lr)
            s_plus_entries.append({"dump": path, **h3})
            print(f"[s_plus {os.path.basename(os.path.dirname(path)) or path}] "
                  f"rel_update={h3['rel_update']:.4f} delta_t0_val={h3['delta_t0_val']:.4f}")

        rel_updates = [e["rel_update"] for e in s_plus_entries]
        s_plus_delta_t0_val = _mean([e["delta_t0_val"] for e in s_plus_entries])
        s_plus_rel_update = _mean(rel_updates)
        n_fire = sum(1 for r in rel_updates if r >= 0.1)
        needed = max(1, math.ceil(0.6 * len(rel_updates)))  # generalises the pre-registered "3/5"
        h3_desk_branch = "ARM-CAN-FIRE" if n_fire >= needed else "ARM-DEAD"

        print(f"H3' s_plus kill: rel_update={s_plus_rel_update} "
              f"({n_fire}/{len(rel_updates)} dumps >= 0.1, need {needed}) => {h3_desk_branch}")

        result["s_plus"] = s_plus_entries
        result["s_plus_rel_updates"] = rel_updates
        result["s_plus_rel_update"] = s_plus_rel_update
        result["s_plus_delta_t0_val"] = s_plus_delta_t0_val
        result["s_plus_n_fire"] = n_fire
        result["s_plus_n_dumps"] = len(rel_updates)
        result["h3_desk_branch"] = h3_desk_branch

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
