"""
Experiment 3a-KAN — Kan-gated growing Concept DAG on Split-CIFAR-100.

Same protocol as exp3a (route → build → train → freeze), but each task passes through the
**Kan gate** first: grow a new concept node only on a certified obstruction, otherwise solve the task
by *reusing* existing concepts (a ReuseComposer over the routed parents + head) and add **no node**.
After growth, an optional **consolidation pass** reclaims parameters (low-rank re-crystallisation +
gated subspace-redundancy merge). Together these make the parameter count track *distinct concepts*
rather than task count.

Predictors are heterogeneous: a task is served either by a DAGNode (grow) or a ReuseComposer over
frozen parent nodes (reuse). :class:`TaskPredictor` unifies evaluation over both.

This module is task-agnostic through the TaskSpec (bits / code length); the default is
classification, but any TaskSpec works — that is the answer to "the metric may differ per task".
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..modules.concept_module import ConceptModule
from ..models.baselines import LinearHead
from ..training.kan_gate import (
    TaskSpec, classification_task, decide_reuse_vs_grow, decide_reuse_search_grow,
    ReuseComposer, SearchComposer, _fit_full, update_probe,
)
from ..training.consolidate import low_rank_factorize_final_layer
from ..utils.metrics import principal_angles_between
from .exp3_growing_dag import (
    Exp3Config, DAGNode, forward_dag_memoized, train_node, eval_node, route_for_task, _flush,
)


# ---------------------------------------------------------------------------
# Heterogeneous per-task predictor (grow: DAGNode | reuse: composer over parents)
# ---------------------------------------------------------------------------


class TaskPredictor:
    """Unifies evaluation over a grow-task (owns a DAGNode) and a reuse-task (composer over parents)."""

    def __init__(self, kind: str, head: nn.Module,
                 node: Optional[DAGNode] = None,
                 parents: Optional[List[DAGNode]] = None,
                 composer: Optional[ReuseComposer] = None):
        assert kind in ("grow", "reuse", "search", "update")   # search: like reuse but a
                                                               # SearchComposer; update: like reuse
                                                               # but over a refined parent
        self.kind, self.head, self.node, self.parents, self.composer = kind, head, node, parents, composer
        self.parent_adapters = None   # per-parent recovery adapters (reuse), installed by a merge
        self.node_adapter = None      # recovery adapter on the grown node's output, installed by a merge

    @torch.no_grad()
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "grow":
            emb = forward_dag_memoized(self.node, x)
            if self.node_adapter is not None:
                emb = self.node_adapter(emb)
            return self.head(emb)
        outs = [forward_dag_memoized(p, x) for p in self.parents]
        adapters = getattr(self, "parent_adapters", None)
        if adapters:
            outs = [o if a is None else a(o) for o, a in zip(outs, adapters)]
        stack = torch.stack(outs, dim=1)  # (B,P,D)
        return self.composer(stack)   # ReuseComposer already applies the task head

    @torch.no_grad()
    def accuracy(self, loader, device: str) -> float:
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = self.logits(x).argmax(-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Cache parent embeddings from DAGNodes (frozen) for the gate / final fit
# ---------------------------------------------------------------------------


@torch.no_grad()
def _cache_parent_stack(parents: List[DAGNode], loader, device: str,
                        max_batches: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (parent_stack (N,P,D), raw_inputs (N,...), targets). The raw inputs are the same
    batches the parent stack was computed from, so a raw-root grow probe shares the gate's split."""
    stacks, raws, ys = [], [], []
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device)
        outs = [forward_dag_memoized(p, x) for p in parents]      # each (B, D)
        stacks.append(torch.stack(outs, dim=1).cpu())             # (B, P, D)
        raws.append(x.cpu())
        ys.append(y.cpu())
    return torch.cat(stacks, 0), torch.cat(raws, 0), torch.cat(ys, 0)


# ---------------------------------------------------------------------------
# Consolidation over the DAGNode list (reduction / "sleep" pass)
# ---------------------------------------------------------------------------


def _node_children(nodes: List[DAGNode], node: DAGNode) -> List[DAGNode]:
    return [c for c in nodes if node in c.parent_models]


def _identity_linear(D: int, device: str) -> nn.Linear:
    W = nn.Linear(D, D, bias=True).to(device)
    with torch.no_grad():
        W.weight.copy_(torch.eye(D, device=device)); W.bias.zero_()
    for p in W.parameters():
        p.requires_grad_(False)
    return W.eval()


def _lstsq_linear(src: torch.Tensor, tgt: torch.Tensor, device: str) -> Tuple[nn.Linear, float]:
    """Least-squares affine map W: src -> tgt (closed form). Returns (frozen Linear, recon MSE)."""
    D_in, D_out = src.shape[1], tgt.shape[1]
    A = torch.cat([src, torch.ones(src.shape[0], 1)], dim=1)        # augment for bias
    sol = torch.linalg.lstsq(A, tgt).solution                       # (D_in+1, D_out)
    W = nn.Linear(D_in, D_out, bias=True).to(device)
    with torch.no_grad():
        W.weight.copy_(sol[:D_in].T.to(device)); W.bias.copy_(sol[D_in].to(device))
    for p in W.parameters():
        p.requires_grad_(False)
    recon = float(((A @ sol) - tgt).pow(2).mean().item())
    return W.eval(), recon


def distill_merge(keep: DAGNode, drop: DAGNode, loader, device: str,
                  epochs: int = 30, lr: float = 1e-3, freeze_keep: bool = False) -> Dict[str, nn.Module]:
    """
    Functional merge for DAGNodes with *different but overlapping* subspaces.

    Trains ``keep`` to be a **sufficient statistic of the pair {keep, drop}**: a vector from which
    both concepts' outputs are linearly recoverable. Returns per-concept linear *recovery adapters*
    ``{"keep": W_keep, "drop": W_drop}`` — the transport maps a re-pointed child inserts before its
    aggregator so it still sees (approximately) the activations it was trained on. When the two
    subspaces are near-identical, W_keep ≈ W_drop ≈ I and this degrades to a structural merge; when
    their joint rank exceeds concept_dim the reconstruction error stays high and the caller's gate
    rejects the merge (they are not actually redundant). See module/keep docs.

    ``freeze_keep=True`` — the correct mode when ``drop`` is already linearly recoverable from ``keep``
    (high canonical correlation): keep is left **untouched** (W_keep = I) and only W_drop is fit by
    least squares (keep's frozen output → drop's output). This is essential when keep has *other-task*
    reuse consumers: retraining keep shifts the concept those tasks depend on and the forgetting gate
    then rightly rejects the merge (observed: FashionMNIST −0.078 when two MNIST concepts were merged
    by retraining keep). Freezing keep makes W_keep exactly identity, so keep's consumers are unchanged.
    """
    D = keep.concept_module.out_dim
    keep.eval(); drop.eval()

    # Cache inputs + ORIGINAL reference targets (keep's and drop's outputs BEFORE keep is retrained).
    xs, tgt_keep_ref, tgt_drop_ref = [], [], []
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(device)
            xs.append(x.cpu())
            tgt_keep_ref.append(forward_dag_memoized(keep, x).cpu())
            tgt_drop_ref.append(forward_dag_memoized(drop, x).cpu())

    if freeze_keep:
        # Keep is a sufficient statistic already: don't perturb it. W_keep = I (its consumers are
        # untouched); W_drop = least-squares map from keep's frozen output to drop's output.
        keep_out = torch.cat(tgt_keep_ref); drop_out = torch.cat(tgt_drop_ref)
        keep.freeze()
        W_keep = _identity_linear(D, device)
        W_drop, recon = _lstsq_linear(keep_out, drop_out, device)
        return {"keep": W_keep, "drop": W_drop, "recon_loss": recon}

    # Train ONLY keep's concept module so both original outputs are linearly recoverable from the new
    # keep output; W_keep, W_drop are the recovery (transport) maps.
    for p in keep.concept_module.parameters():
        p.requires_grad_(True)
    keep.concept_module.train()
    W_keep = nn.Linear(D, D, bias=True).to(device)
    W_drop = nn.Linear(D, D, bias=True).to(device)
    params = list(keep.concept_module.parameters()) + list(W_keep.parameters()) + list(W_drop.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    last = 0.0
    for _ in range(epochs):
        for xb, tk, td in zip(xs, tgt_keep_ref, tgt_drop_ref):
            xb, tk, td = xb.to(device), tk.to(device), td.to(device)
            out = forward_dag_memoized(keep, xb)                   # (B, D), grad flows into keep only
            loss = ((W_keep(out) - tk) ** 2).mean() + ((W_drop(out) - td) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            last = float(loss.item())
    keep.concept_module.eval(); keep.freeze()
    for W in (W_keep, W_drop):
        for p in W.parameters():
            p.requires_grad_(False)
        W.eval()
    return {"keep": W_keep, "drop": W_drop, "recon_loss": last}


def _combined_loader(tasks: List[Dict], ta: int, tb: int, batch_size: int = 128):
    ds = torch.utils.data.ConcatDataset([tasks[ta]["train"].dataset, tasks[tb]["train"].dataset])
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)


def _snapshot_topology(nodes: List[DAGNode], predictors: List[TaskPredictor], keep: DAGNode) -> dict:
    """Everything a tentative merge can mutate: the node list, every edge, and keep's own weights."""
    return {
        "nodes": list(nodes),
        "node_edges": {id(n): (list(n.parent_models),
                               list(getattr(n, "parent_adapters", None) or []),
                               n.concept_module.n_parents) for n in nodes},
        "pred_edges": {id(p): (list(p.parents or []), list(getattr(p, "parent_adapters", None) or []))
                       for p in predictors if p.kind == "reuse"},
        "grow_preds": {id(p): (p.node, p.node_adapter) for p in predictors if p.kind == "grow"},
        "keep_state": {k: v.detach().clone() for k, v in keep.concept_module.state_dict().items()},
        "keep_id": id(keep),
    }


def _restore_topology(nodes: List[DAGNode], predictors: List[TaskPredictor], snap: dict):
    nodes[:] = snap["nodes"]
    for n in nodes:
        pm, pa, npar = snap["node_edges"][id(n)]
        n.parent_models = list(pm)
        n.parent_adapters = list(pa) if pa else None
        n.concept_module.n_parents = npar
    for p in predictors:
        if id(p) in snap["pred_edges"]:
            par, pa = snap["pred_edges"][id(p)]
            p.parents = list(par)
            p.parent_adapters = list(pa) if pa else None
        if id(p) in snap["grow_preds"]:
            p.node, p.node_adapter = snap["grow_preds"][id(p)]
    for n in nodes:
        if id(n) == snap["keep_id"]:
            n.concept_module.load_state_dict(snap["keep_state"])
            n.concept_module.eval()


def consolidate_nodes(
    nodes: List[DAGNode],
    predictors: List[TaskPredictor],
    tasks: List[Dict],
    device: str,
    *,
    accept_fn: Callable[[DAGNode, DAGNode, Set[int]], bool],
    similarity_threshold: float,
    subspace_k: int = 8,
    truncate_energy: Optional[float] = 0.99,
    truncate_max_rel_error: float = 0.05,
    distill: bool = True,
    distill_epochs: int = 20,
    merge_tolerance: float = 0.01,
    functional_threshold: Optional[float] = None,
) -> Dict[str, object]:
    """
    One consolidation pass over the grown DAGNode list.

    (1) Low-rank re-crystallise every node's concept module. (2) Merge pairs whose concept subspaces
    are near-identical (principal-angle similarity ≥ threshold), re-pointing children's parent lists,
    each merge gated by ``accept_fn(keep, drop, affected_task_ids)`` which MUST re-evaluate the
    affected tasks (backward-interference check).
    """
    def pcount():
        return sum(p.numel() for n in nodes for p in n.concept_module.parameters())

    params_before = pcount()
    ops: List[dict] = []

    # (1) Low-rank re-crystallisation.
    if truncate_energy is not None:
        for n in nodes:
            rec = low_rank_factorize_final_layer(
                n.concept_module, energy=truncate_energy, max_rel_error=truncate_max_rel_error)
            if rec.get("applied"):
                ops.append({"op": "truncate", "task": n.task_id, **rec})

    # (2) Subspace-redundancy merges (re-scan after each accepted merge).
    def affected_task_ids(node: DAGNode) -> Set[int]:
        # task indices (predictor positions) whose predictor routes through `node`.
        ids: Set[int] = set()
        for ti, p in enumerate(predictors):
            chain = [p.node] if (p.kind == "grow" and p.node is not None) else list(p.parents or [])
            reach: Set[int] = set()
            frontier = list(chain)
            while frontier:
                nd = frontier.pop()
                if id(nd) in reach:
                    continue
                reach.add(id(nd)); frontier.extend(nd.parent_models)
            if id(node) in reach:
                ids.add(ti)
        return ids

    merged = True
    while merged:
        merged = False
        subspaced = [n for n in nodes if n.concept_module.get_concept_subspace() is not None]
        for i, a in enumerate(subspaced):
            for b in subspaced[i + 1:]:
                # skip ancestor/descendant pairs (stacked, not parallel)
                if _is_ancestor(a, b) or _is_ancestor(b, a):
                    continue
                # Redundancy trigger. Default (geometric) principal-angle overlap misses concepts
                # that are functionally identical but sit in different bases; when a functional
                # threshold is set, use mean canonical correlation instead (basis-invariant).
                if functional_threshold is not None:
                    sim = _functional_similarity(
                        a, b, _combined_loader(tasks, a.task_id, b.task_id), device, subspace_k)
                    sim_kind = "functional"
                    if sim < functional_threshold:
                        continue
                else:
                    sim = _subspace_similarity(a, b, subspace_k)
                    sim_kind = "subspace"
                    if sim < similarity_threshold:
                        continue
                affected = affected_task_ids(a) | affected_task_ids(b)

                if not distill:
                    # Structural merge (safe only for near-identical subspaces); gate decides.
                    if not accept_fn(a, b, affected):
                        continue
                    _merge_nodes(nodes, predictors, keep=a, drop=b)
                    ops.append({"op": "merge", "keep": a.task_id, "drop": b.task_id,
                                "similarity": sim, "sim_kind": sim_kind, "distilled": False})
                    merged = True
                    break

                # Distilled merge: snapshot → distill keep → tentatively apply adapted merge →
                # forgetting check → commit or roll back (topology + keep weights).
                base = {t: predictors[t].accuracy(tasks[t]["test"], device)
                        for t in affected if t < len(predictors)}
                snap = _snapshot_topology(nodes, predictors, keep=a)
                loader = _combined_loader(tasks, a.task_id, b.task_id)
                W = distill_merge(a, b, loader, device, epochs=distill_epochs,
                                  freeze_keep=(functional_threshold is not None))
                _merge_nodes(nodes, predictors, keep=a, drop=b, W_keep=W["keep"], W_drop=W["drop"])
                post = {t: predictors[t].accuracy(tasks[t]["test"], device)
                        for t in affected if t < len(predictors)}
                deltas = {t: round(post[t] - base[t], 4) for t in post}
                ok = all(post[t] >= base.get(t, 0.0) - merge_tolerance for t in post)
                if ok:
                    ops.append({"op": "merge", "keep": a.task_id, "drop": b.task_id,
                                "similarity": sim, "sim_kind": sim_kind, "distilled": True,
                                "recon_loss": W["recon_loss"], "backward_deltas": deltas})
                    merged = True
                    break
                else:
                    # Forgetting gate rejected the merge — roll back, but RECORD it (a silent
                    # rollback is invisible to the research loop and looks identical to "no
                    # candidate found"). worst_delta says how far the backward check was missed.
                    _restore_topology(nodes, predictors, snap)
                    ops.append({"op": "merge_rejected", "keep": a.task_id, "drop": b.task_id,
                                "similarity": sim, "sim_kind": sim_kind, "recon_loss": W["recon_loss"],
                                "backward_deltas": deltas, "worst_delta": min(deltas.values()),
                                "merge_tolerance": merge_tolerance})
            if merged:
                break

    return {"params_before": params_before, "params_after": pcount(),
            "params_saved": params_before - pcount(), "n_ops": len(ops), "ops": ops}


def _is_ancestor(anc: DAGNode, desc: DAGNode) -> bool:
    frontier = list(desc.parent_models)
    while frontier:
        n = frontier.pop()
        if n is anc:
            return True
        frontier.extend(n.parent_models)
    return False


def _subspace_similarity(a: DAGNode, b: DAGNode, top_k: int) -> float:
    ca = a.concept_module.get_concept_subspace()
    cb = b.concept_module.get_concept_subspace()
    angles = principal_angles_between(ca, cb)
    return float(torch.cos(angles).sum().item())


def _functional_similarity(a: DAGNode, b: DAGNode, loader, device: str, top_k: int,
                           max_batches: int = 8) -> float:
    """Mean of the top-k canonical correlations between the two concepts' outputs over `loader`.

    This is a *functional* redundancy signal, in [0, 1], and — unlike principal-angle subspace
    overlap — it is invariant to the basis each concept happens to have learned. Two concepts trained
    independently on the same data are functionally near-identical (CCA ≈ 0.99) yet occupy nearly
    orthogonal subspaces (principal-angle sim ≈ 1.9/8); the geometric detector misses them entirely,
    so this is the correct trigger for the distill+recovery-adapter merge (which linearly re-aligns
    the surviving concept anyway). See Central Library: five-datasets-kan-merge-detector.
    """
    a_out, b_out = [], []
    a.eval(); b.eval()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            x = batch[0].to(device)
            a_out.append(a(x).detach().cpu().float())
            b_out.append(b(x).detach().cpu().float())
    Ha = torch.cat(a_out); Hb = torch.cat(b_out)
    if Ha.shape[0] <= Ha.shape[1]:               # too few samples for a stable CCA
        return 0.0
    Ha = Ha - Ha.mean(0, keepdim=True)
    Hb = Hb - Hb.mean(0, keepdim=True)
    qa, _ = torch.linalg.qr(Ha)
    qb, _ = torch.linalg.qr(Hb)
    sv = torch.linalg.svdvals(qa.T @ qb).clamp(0.0, 1.0)
    k = min(top_k, sv.numel())
    return float(sv[:k].mean().item())


def _compose(existing: Optional[nn.Module], recovery: Optional[nn.Module]) -> Optional[nn.Module]:
    """
    Compose an edge's existing adapter with a new recovery map. `recovery` reconstructs the OLD parent
    output from the merged parent's output; `existing` (if any) is what the child already applied to
    the old output. Applied order on the merged output is recovery THEN existing, so the child ends up
    with (approximately) the activation it was trained on.
    """
    if recovery is None:
        return existing
    if existing is None:
        return recovery
    return nn.Sequential(recovery, existing)


def _repoint_with_adapters(models, adapters, keep, drop, W_keep, W_drop):
    """Return (new_models, new_adapters): drop→keep gets W_drop, keep→keep gets W_keep. No dedup, so
    parent counts (and thus aggregator / composer input dims) stay fixed."""
    adapters = adapters or [None] * len(models)
    new_models, new_adapters = [], []
    for p, ad in zip(models, adapters):
        if p is drop:
            new_models.append(keep); new_adapters.append(_compose(ad, W_drop))
        elif p is keep:
            new_models.append(keep); new_adapters.append(_compose(ad, W_keep))
        else:
            new_models.append(p); new_adapters.append(ad)
    return new_models, new_adapters


def _merge_nodes(nodes: List[DAGNode], predictors: List[TaskPredictor], keep: DAGNode, drop: DAGNode,
                 W_keep: Optional[nn.Module] = None, W_drop: Optional[nn.Module] = None):
    """
    Merge `drop` into `keep`: re-point every child/predictor edge drop→keep (and keep→keep, whose
    function changed under distillation) installing the recovery adapters, then remove `drop`.
    Parent counts are preserved (no dedup), so a child that had BOTH keep and drop keeps two edges to
    keep with distinct adapters — exactly the two signals it was trained on.
    """
    for c in nodes:
        if keep in c.parent_models or drop in c.parent_models:
            c.parent_models, adapters = _repoint_with_adapters(
                c.parent_models, getattr(c, "parent_adapters", None), keep, drop, W_keep, W_drop)
            c.parent_adapters = adapters if any(a is not None for a in adapters) else None
            c.concept_module.n_parents = len(c.parent_models)
    for pred in predictors:
        if pred.kind == "reuse" and pred.parents and (keep in pred.parents or drop in pred.parents):
            pred.parents, adapters = _repoint_with_adapters(
                pred.parents, getattr(pred, "parent_adapters", None), keep, drop, W_keep, W_drop)
            pred.parent_adapters = adapters if any(a is not None for a in adapters) else None
        elif pred.kind == "grow" and pred.node is not None:
            # The dropped node's OWN task must now read `keep` through the recovery map (else the task
            # loses its predictor and `drop` is never actually freed). keep's own task reads through
            # W_keep because keep's function changed under distillation.
            if pred.node is drop:
                pred.node_adapter = _compose(pred.node_adapter, W_drop)
                pred.node = keep
            elif pred.node is keep:
                pred.node_adapter = _compose(pred.node_adapter, W_keep)
    nodes.remove(drop)


def _repoint(models: List[DAGNode], keep: DAGNode, drop: DAGNode):
    """Replace `drop`→`keep` in a parent list, de-duplicating."""
    out, seen = [], set()
    for p in models:
        q = keep if p is drop else p
        if id(q) not in seen:
            seen.add(id(q)); out.append(q)
    return out


def make_accuracy_accept_fn(nodes: List[DAGNode], predictors: List[TaskPredictor],
                            tasks: List[Dict], device: str, tolerance: float = 0.01):
    """
    Default gate: accept a merge iff no affected task's accuracy regresses by more than `tolerance`
    (the backward-interference / forgetting check that makes reduction safe).

    The trial applies the *full* structural re-point (drop→keep across every node and predictor that
    references drop, anywhere in the ancestry), measures affected tasks, then restores — so the check
    faithfully reflects what the real merge will do.
    """
    def accept(keep: DAGNode, drop: DAGNode, affected: Set[int]) -> bool:
        base = {t: predictors[t].accuracy(tasks[t]["test"], device)
                for t in affected if t < len(predictors)}
        snap_nodes = {id(n): list(n.parent_models) for n in nodes}
        snap_preds = {id(p): list(p.parents) for p in predictors if p.parents}
        for n in nodes:
            if drop in n.parent_models:
                n.parent_models = _repoint(n.parent_models, keep, drop)
        for p in predictors:
            if p.parents and drop in p.parents:
                p.parents = _repoint(p.parents, keep, drop)
        ok = all(predictors[t].accuracy(tasks[t]["test"], device) >= base.get(t, 0.0) - tolerance
                 for t in affected if t < len(predictors))
        for n in nodes:
            if id(n) in snap_nodes:
                n.parent_models = snap_nodes[id(n)]
        for p in predictors:
            if id(p) in snap_preds:
                p.parents = snap_preds[id(p)]
        return ok
    return accept


# ---------------------------------------------------------------------------
# The gated growth run
# ---------------------------------------------------------------------------


@dataclass
class KanExpConfig(Exp3Config):
    eps_rel:             float = 0.05
    gate_epochs:         int   = 40
    gate_lr:             float = 1e-3
    consolidate_every:   int   = 0        # 0 = only at end; K = every K tasks
    similarity_threshold: float = 7.0     # principal-angle sim (max = subspace_k) for a merge
    merge_tolerance:     float = 0.01
    distill:             bool  = True     # functional (distilled) merge with recovery adapters
    distill_epochs:      int   = 20
    force_grow_ids:      tuple = ()       # stream positions to grow unconditionally (merge stress-test:
                                          # forces a redundant concept the consolidation pass must merge)
    functional_redundancy: bool = True    # detect merge candidates by canonical correlation (basis-
                                          # invariant), not principal-angle subspace overlap
    functional_threshold: float = 0.9     # mean top-k canonical correlation to trigger a merge
    enable_search:       bool  = False    # three-way reuse/search/grow gate (test-time-compute rung)
    eps_search:          float = 0.05     # min reducible-info fraction bounded search must add over reuse
    search_budget:       int   = 6        # trained candidates the Search level may spend
    search_rank:         int   = 16       # bottleneck rank of the SearchComposer (≪ concept_dim)
    search_skip:         bool  = False    # give the SearchComposer a full-rank linear skip so it
                                          # NESTS reuse (L_search ≤ L_reuse by construction). Without
                                          # it the rank-16 bottleneck is narrower than reuse's
                                          # full-rank linear map, rel_search goes negative and the
                                          # ladder is non-monotone — see search-on-raw-probe-result.
    raw_grow_probe:      bool  = False    # grow probe sees the raw encoder features (a real grown
                                          # root's view) instead of the frozen parent stack; an
                                          # organic grow then mints a ROOT node. Feature-mode only
                                          # (use_cnn=False) — the parents-only probe cannot certify
                                          # obstructions on domains absent from the parents.
    reducible_mode:      str   = "grow"   # "grow" | "best" — which normaliser DECIDES (both always
                                          # recorded on the KanGateRecord).
    gate_estimator:      str   = "single" # "single" | "crossfit" — how the gate's held-out code
                                          # lengths are measured (decide_reuse_search_grow only).
    gate_splits:         int   = 5        # crossfit folds (gate_estimator == "crossfit").
    oracle_rungs:        bool  = False    # after the decision, ALSO train+eval the other rungs'
                                          # predictors (reuse/search/grow) on task["test"], without
                                          # altering the DAG — for post-hoc regret analysis.
    dump_gate_tensors:   bool  = False    # write gate_dump.pt (feature mode only) — raw per-task
                                          # tensors + node/predictor state, for offline desk-stage
                                          # re-analysis without re-running training.
    enable_update:       bool  = False    # the "update" rung: refine an existing root parent's
                                          # concept in place instead of reuse/search/grow, gated by
                                          # backward safety on earlier tasks.
    update_lr:           float = 1e-4     # fine-tune rate for the copied concept in update_probe.
    eps_update:          float = 0.1      # rel_update = (L_reuse - L_update)/L_reuse must exceed
                                          # this for the update candidate to be eligible.
    update_tolerance:    float = 0.01     # backward-safety tolerance on earlier tasks' VAL accuracy.


# ---------------------------------------------------------------------------
# Oracle rungs — post-hoc, without altering the DAG (§5)
# ---------------------------------------------------------------------------


def _oracle_rungs(rec, X: torch.Tensor, y: torch.Tensor, parents: List[DAGNode], task: Dict, t: int,
                  cfg: "KanExpConfig", spec: TaskSpec, use_cnn: bool, device: str,
                  chosen_acc: float) -> Dict[str, float]:
    """Train + evaluate the rungs NOT chosen by the frozen ladder, on the same cache, WITHOUT
    altering the DAG. The chosen rung's entry reuses `chosen_acc` rather than retraining. Wrapped in
    `torch.random.fork_rng()` so it does not perturb the main run's RNG stream."""
    accs: Dict[str, float] = {}
    with torch.random.fork_rng():
        if rec.decision == "reuse":
            accs["reuse"] = chosen_acc
        else:
            oh = LinearHead(cfg.concept_dim, task["n_classes"])
            oc = ReuseComposer(parent_dim=cfg.concept_dim, n_parents=len(parents), head=oh)
            _fit_full(oc, lambda m, xb: m(xb), spec, X, y, cfg.child_epochs, cfg.lr, device)
            accs["reuse"] = TaskPredictor("reuse", oh, parents=parents, composer=oc).accuracy(
                task["test"], device)

        if rec.decision == "search":
            accs["search"] = chosen_acc
        else:
            meta = rec.search_meta or {}
            oh = LinearHead(cfg.concept_dim, task["n_classes"])
            if meta.get("trivial"):
                oc = ReuseComposer(parent_dim=cfg.concept_dim, n_parents=len(parents), head=oh)
            else:
                oc = SearchComposer(parent_dim=cfg.concept_dim, n_parents=len(parents), head=oh,
                                    rank=meta.get("rank", cfg.search_rank), subset=meta.get("subset"),
                                    skip=meta.get("skip", cfg.search_skip))
            _fit_full(oc, lambda m, xb: m(xb), spec, X, y, cfg.child_epochs, cfg.lr, device)
            accs["search"] = TaskPredictor("search", oh, parents=parents, composer=oc).accuracy(
                task["test"], device)

        if rec.decision == "grow":
            accs["grow"] = chosen_acc
        else:
            onode = DAGNode(task_id=t, concept_dim=cfg.concept_dim, cnn_out_dim=cfg.cnn_out_dim,
                            n_mlp_layers=cfg.n_mlp_layers, parent_models=None,
                            soft_pca_k=cfg.soft_pca_k, use_cnn=use_cnn, feature_dim=cfg.feature_dim)
            ohead = LinearHead(cfg.concept_dim, task["n_classes"])
            train_node(onode, ohead, task["train"], cfg.child_epochs, cfg.lr, device, cfg.log_every,
                       name=f"t{t}-oracle-grow", orth_weight=cfg.orth_weight)
            accs["grow"] = eval_node(onode, ohead, task["test"], device)
            del onode, ohead
            _flush(device)
    return accs


# ---------------------------------------------------------------------------
# gate_dump.pt — raw per-task tensors + node/predictor state (§6, feature mode only)
# ---------------------------------------------------------------------------


def _cache_raw_capped(loader, max_batches: int) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        xs.append(x.cpu())
        ys.append(y.cpu())
    return torch.cat(xs, 0), torch.cat(ys, 0)


def _cache_raw_full(loader) -> Tuple[torch.Tensor, torch.Tensor]:
    return _cache_raw_capped(loader, max_batches=0)


def _build_gate_dump(cfg: "KanExpConfig", tasks: List[Dict], nodes: List[DAGNode],
                     predictors: List[TaskPredictor], decisions: List[dict]) -> dict:
    gate_dump_tasks = []
    for t, task in enumerate(tasks):
        train_raw, train_y = _cache_raw_capped(task["train"], cfg.routing_batches)
        val_raw, val_y = _cache_raw_full(task.get("val", task["test"]))
        test_raw, test_y = _cache_raw_full(task["test"])
        gate_dump_tasks.append({
            "task": t, "n_classes": task["n_classes"], "ctrl": task.get("ctrl"),
            "train_raw": train_raw.float(), "train_y": train_y.long(),
            "val_raw": val_raw.float(), "val_y": val_y.long(),
            "test_raw": test_raw.float(), "test_y": test_y.long(),
        })

    node_index = {id(n): i for i, n in enumerate(nodes)}
    gate_dump_nodes = []
    for i, n in enumerate(nodes):
        gate_dump_nodes.append({
            "index": i, "task_id": n.task_id, "is_root": n.is_root,
            "parent_indices": [node_index[id(p)] for p in n.parent_models],
            "state_dict": {k: v.cpu() for k, v in n.concept_module.state_dict().items()},
        })

    gate_dump_preds = []
    for t, pred in enumerate(predictors):
        head_state = {k: v.cpu() for k, v in pred.head.state_dict().items()}
        if pred.kind == "grow":
            gate_dump_preds.append({
                "task": t, "kind": pred.kind, "head_state": head_state,
                "node_index": node_index.get(id(pred.node)), "parent_indices": None,
                "composer_kind": None, "composer_state": None, "search_meta": None,
            })
        else:
            composer = pred.composer
            if isinstance(composer, SearchComposer):
                composer_kind = "search"
            elif isinstance(composer, ReuseComposer):
                composer_kind = "reuse"
            else:
                composer_kind = None
            composer_state = ({k: v.cpu() for k, v in composer.state_dict().items()}
                              if composer is not None else None)
            search_meta = decisions[t].get("search_meta") if pred.kind == "search" else None
            gate_dump_preds.append({
                "task": t, "kind": pred.kind, "head_state": head_state,
                "node_index": None,
                "parent_indices": [node_index.get(id(p)) for p in (pred.parents or [])],
                "composer_kind": composer_kind, "composer_state": composer_state,
                "search_meta": search_meta,
            })

    return {
        "feature_mode": True,
        "concept_dim": cfg.concept_dim,
        "feature_dim": cfg.feature_dim,
        "n_parents": cfg.n_parents,
        "seed": cfg.seed,
        "config": {k: getattr(cfg, k) for k in (
            "gate_epochs", "gate_lr", "eps_rel", "eps_search", "search_budget", "search_rank",
            "search_skip", "routing_batches", "child_epochs", "lr", "reducible_mode",
            "gate_estimator", "gate_splits", "update_lr", "eps_update", "update_tolerance",
            "subspace_k", "n_mlp_layers",
        )},
        "tasks": gate_dump_tasks,
        "nodes": gate_dump_nodes,
        "predictors": gate_dump_preds,
        "decisions": decisions,
    }


def run_exp3a_kan(
    cfg: KanExpConfig,
    tasks: List[Dict],
    spec_factory: Optional[Callable[[Dict], TaskSpec]] = None,
) -> Dict:
    """
    Kan-gated growth. `spec_factory(task) -> TaskSpec` lets each task carry its own code-length
    functional (defaults to classification). Returns metrics incl. the params-vs-tasks curve, the
    grow/reuse decisions, per-task accuracy, and consolidation savings.
    """
    torch.manual_seed(cfg.seed)
    device = cfg.device
    os.makedirs(cfg.results_dir, exist_ok=True)
    spec_factory = spec_factory or (lambda task: classification_task(task["n_classes"]))
    use_cnn = (cfg.backbone == "smallcnn")

    nodes: List[DAGNode] = []
    predictors: List[TaskPredictor] = []
    decisions: List[dict] = []
    param_curve: List[int] = []
    test_accs: List[float] = []

    def new_module_factory(parents: List[DAGNode]):
        def factory():
            return ConceptModule(
                module_id="__probe__", in_dim=cfg.concept_dim, hidden_dim=cfg.concept_dim,
                out_dim=cfg.concept_dim, n_layers=cfg.n_mlp_layers, n_parents=len(parents),
                aggregation="soft_pca", agg_kwargs={"top_k": min(cfg.soft_pca_k, cfg.concept_dim)},
            )
        return factory

    # Raw-root grow probe: capacity-matched to a real grown root DAGNode (ConceptModule on the raw
    # encoder features, n_parents=0). Feature-mode only — in CNN mode the raw input is an image and
    # the probe would need its own backbone, so we fall back to the parents-only probe there.
    use_raw_probe = cfg.raw_grow_probe and not use_cnn and cfg.feature_dim is not None

    def root_module_factory():
        return ConceptModule(
            module_id="__root_probe__", in_dim=cfg.feature_dim, hidden_dim=cfg.concept_dim,
            out_dim=cfg.concept_dim, n_layers=cfg.n_mlp_layers, n_parents=0,
        )

    for t, task in enumerate(tasks):
        n_par = min(cfg.n_parents, t)
        spec = spec_factory(task)
        head = LinearHead(cfg.concept_dim, task["n_classes"])

        if n_par == 0:
            # Root: growth forced.
            node = DAGNode(task_id=t, concept_dim=cfg.concept_dim, cnn_out_dim=cfg.cnn_out_dim,
                           n_mlp_layers=cfg.n_mlp_layers, parent_models=None,
                           soft_pca_k=cfg.soft_pca_k, use_cnn=use_cnn, feature_dim=cfg.feature_dim)
            train_node(node, head, task["train"], cfg.root_epochs, cfg.lr, device, cfg.log_every,
                       name=f"t{t}-root", orth_weight=cfg.orth_weight)
            node.compute_concept_subspace(task["train"], device, top_k=cfg.subspace_k,
                                          max_batches=cfg.routing_batches)
            node.freeze()
            nodes.append(node)
            predictors.append(TaskPredictor("grow", head, node=node))
            decisions.append({"task": t, "decision": "grow", "reason": "root"})
        else:
            sel_idx, _scores = route_for_task(nodes, task["train"], n_par, cfg.subspace_k,
                                              device, cfg.routing_batches)
            parents = [nodes[i] for i in sel_idx]
            # --- Kan gate on cached parent embeddings (+ raw features for the root grow probe) ---
            X, Xraw, y = _cache_parent_stack(parents, task["train"], device,
                                             max_batches=cfg.routing_batches)
            raw_kwargs = ({"raw_stack": Xraw, "root_module_factory": root_module_factory}
                          if use_raw_probe else {})
            force = t in cfg.force_grow_ids
            split_gen = torch.Generator().manual_seed(cfg.seed * 1000 + t)
            if cfg.enable_search and not force:
                # Three-way reuse/search/grow escalation (test-time-compute rung).
                rec = decide_reuse_search_grow(
                    X, y, new_module_factory(parents), spec,
                    concept_dim=cfg.concept_dim, n_parents=len(parents), device=device,
                    n_epochs=cfg.gate_epochs, lr=cfg.gate_lr, eps_grow=cfg.eps_rel,
                    eps_search=cfg.eps_search, search_budget=cfg.search_budget, search_rank=cfg.search_rank,
                    search_skip=cfg.search_skip, reducible_mode=cfg.reducible_mode,
                    estimator=cfg.gate_estimator, n_splits=cfg.gate_splits, split_generator=split_gen,
                    **raw_kwargs,
                )
            else:
                rec = decide_reuse_vs_grow(
                    X, y, new_module_factory(parents), spec,
                    concept_dim=cfg.concept_dim, n_parents=len(parents), device=device,
                    n_epochs=cfg.gate_epochs, lr=cfg.gate_lr, eps_rel=cfg.eps_rel,
                    reducible_mode=cfg.reducible_mode,
                    **raw_kwargs,
                )
            if force:
                # Merge stress-test: skip the gate and grow unconditionally, so a redundant
                # concept exists for the consolidation pass to detect and merge. NOT a claim
                # the gate would grow here — the injected duplicate would correctly reuse.
                # Grow it as a ROOT (parent_models=None) so it is PARALLEL to the concept it
                # duplicates, not stacked on it: find_redundant_pairs excludes ancestor/descendant
                # pairs, so a child of the original could never be a merge candidate.
                rec.decision = "grow"
                d = {"task": t, "decision": "grow", "parents": [], "reason": "force-grow(dup-stress,parallel-root)",
                     **{k: getattr(rec, k) for k in ("rel_improvement", "L_reuse_bits", "L_grow_bits")}}
            else:
                d = {"task": t, "decision": rec.decision, "parents": sel_idx,
                     "grow_probe_input": rec.grow_probe_input,
                     **{k: getattr(rec, k) for k in ("rel_improvement", "L_reuse_bits", "L_grow_bits")}}
                for k in ("L_search_bits", "rel_search", "rel_grow", "search_meta", "search_trace"):
                    v = getattr(rec, k, None)
                    if v is not None:
                        d[k] = v
                for k in ("L_null_bits", "reducible_grow", "reducible_best", "reducible_mode",
                          "rel_search_best", "rel_grow_best", "rel_improvement_best",
                          "n_rungs_above_null", "estimator_meta"):
                    v = getattr(rec, k, None)
                    if v is not None:
                        d[k] = v
            decisions.append(d)

            if rec.decision == "grow":
                # A raw-probe grow certified a ROOT's view (raw encoder features), so mint a root —
                # composition over existing concepts is reuse/search's job, not the grown node's.
                grow_as_root = force or use_raw_probe
                grow_parents = None if grow_as_root else parents
                node = DAGNode(task_id=t, concept_dim=cfg.concept_dim, cnn_out_dim=cfg.cnn_out_dim,
                               n_mlp_layers=cfg.n_mlp_layers, parent_models=grow_parents,
                               soft_pca_k=cfg.soft_pca_k, use_cnn=use_cnn, feature_dim=cfg.feature_dim)
                train_node(node, head, task["train"], cfg.child_epochs, cfg.lr, device, cfg.log_every,
                           name=f"t{t}-grow{'-root' if grow_as_root else ''}", orth_weight=cfg.orth_weight)
                node.compute_concept_subspace(task["train"], device, top_k=cfg.subspace_k,
                                              max_batches=cfg.routing_batches)
                node.freeze()
                nodes.append(node)
                predictors.append(TaskPredictor("grow", head, node=node))
            elif rec.decision == "search":
                # Search: keep the best bounded-search composition over frozen parents; add NO node.
                meta = rec.search_meta or {}
                # "trivial" = the search space's no-compute member (plain linear recombination) won,
                # so the predictor IS a ReuseComposer; only a genuine non-linear winner needs one.
                if meta.get("trivial"):
                    composer = ReuseComposer(parent_dim=cfg.concept_dim, n_parents=len(parents), head=head)
                else:
                    composer = SearchComposer(parent_dim=cfg.concept_dim, n_parents=len(parents), head=head,
                                              rank=meta.get("rank", cfg.search_rank), subset=meta.get("subset"),
                                              skip=meta.get("skip", cfg.search_skip))
                _fit_full(composer, lambda m, xb: m(xb), spec, X, y, cfg.child_epochs, cfg.lr, device)
                predictors.append(TaskPredictor("search", head, parents=parents, composer=composer))
            else:
                # Reuse: train a linear composer over frozen parents; add NO node.
                composer = ReuseComposer(parent_dim=cfg.concept_dim, n_parents=len(parents), head=head)
                _fit_full(composer, lambda m, xb: m(xb), spec, X, y, cfg.child_epochs, cfg.lr, device)
                predictors.append(TaskPredictor("reuse", head, parents=parents, composer=composer))

        acc = predictors[t].accuracy(task["test"], device)

        if n_par > 0 and not force:
            # --- Oracle rungs (§5): computed on the FROZEN state, before any update commit. ---
            if cfg.oracle_rungs:
                d["oracle_accs"] = _oracle_rungs(rec, X, y, parents, task, t, cfg, spec, use_cnn,
                                                 device, acc)

            # --- Update rung (§4): refinement placement, cannot mask a grow decision. ---
            if cfg.enable_update and not use_cnn and cfg.feature_dim is not None:
                root_parent_idxs = [i for i, p in enumerate(parents) if p.is_root]
                if root_parent_idxs:
                    # The probe (randperm in update_probe's training loop, deepcopy init, etc.)
                    # consumes RNG state. Forked + deterministically reseeded so it neither leaks
                    # into nor depends on the main run's RNG stream — otherwise every later task's
                    # init/batch order would differ from a control run with enable_update=False,
                    # and borderline decisions on later tasks would flip for an RNG reason rather
                    # than because of the update rung. Matches _oracle_rungs' fork_rng usage above.
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(cfg.seed * 1000 + t + 500)
                        best = None
                        for i in root_parent_idxs:
                            p = parents[i]
                            L_upd, mcopy = update_probe(
                                X, Xraw, y, p.concept_module, i, spec,
                                concept_dim=cfg.concept_dim, n_parents=len(parents),
                                tr_idx=rec.split_meta["tr_idx"], val_idx=rec.split_meta["val_idx"],
                                device=device, n_epochs=cfg.gate_epochs, update_lr=cfg.update_lr,
                                composer_lr=cfg.gate_lr,
                            )
                            if best is None or L_upd < best[0]:
                                best = (L_upd, mcopy, i, p)
                        L_update, module_copy, best_i, best_p = best
                        rel_update = (rec.L_reuse_bits - L_update) / max(rec.L_reuse_bits, 1e-6)

                        # Affected = every earlier task whose predictor reads `best_p`, directly or via
                        # an ancestor chain (parent_models).
                        affected: List[int] = []
                        for tprime, pred in enumerate(predictors[:t]):
                            reads = False
                            if pred.kind == "grow" and pred.node is not None:
                                if pred.node is best_p or _is_ancestor(best_p, pred.node):
                                    reads = True
                            if not reads and pred.parents:
                                if best_p in pred.parents or any(_is_ancestor(best_p, par)
                                                                 for par in pred.parents):
                                    reads = True
                            if reads:
                                affected.append(tprime)

                        orig_module = best_p.concept_module
                        try:
                            before_val = {tp: predictors[tp].accuracy(tasks[tp].get("val", tasks[tp]["test"]),
                                                                       device) for tp in affected}
                            before_test = {tp: predictors[tp].accuracy(tasks[tp]["test"], device)
                                           for tp in affected}
                            best_p.concept_module = module_copy
                            after_val = {tp: predictors[tp].accuracy(tasks[tp].get("val", tasks[tp]["test"]),
                                                                      device) for tp in affected}
                            after_test = {tp: predictors[tp].accuracy(tasks[tp]["test"], device)
                                         for tp in affected}
                        finally:
                            best_p.concept_module = orig_module

                        backward_deltas_val = {str(tp): after_val[tp] - before_val[tp] for tp in affected}
                        backward_deltas_test = {str(tp): after_test[tp] - before_test[tp] for tp in affected}
                        backward_safe = all(after_val[tp] >= before_val[tp] - cfg.update_tolerance
                                            for tp in affected)

                        selected = (rec.decision in ("reuse", "search") and rel_update > cfg.eps_update
                                   and backward_safe)
                        logged_only = rec.decision == "grow"

                        d["update"] = {
                            "probed": True,
                            "parent": sel_idx[best_i],
                            "parent_task_id": best_p.task_id,
                            "L_update": L_update,
                            "rel_update": rel_update,
                            "backward_safe": backward_safe,
                            "backward_deltas_val": backward_deltas_val,
                            "backward_deltas_test": backward_deltas_test,
                            "selected": selected,
                            "logged_only": logged_only,
                        }

                    # Commit step (parent state mutation) stays in the main RNG stream — it only
                    # runs when `selected`, and must not be shielded from it.
                    if selected:
                        best_p.concept_module.load_state_dict(module_copy.state_dict())
                        best_p.compute_concept_subspace(tasks[best_p.task_id]["train"], device,
                                                        top_k=cfg.subspace_k,
                                                        max_batches=cfg.routing_batches)
                        best_p.freeze()
                        X2, Xraw2, y2 = _cache_parent_stack(parents, task["train"], device,
                                                            max_batches=cfg.routing_batches)
                        composer2 = ReuseComposer(parent_dim=cfg.concept_dim, n_parents=len(parents),
                                                  head=head)
                        _fit_full(composer2, lambda m, xb: m(xb), spec, X2, y2, cfg.child_epochs,
                                 cfg.lr, device)
                        predictors[t] = TaskPredictor("update", head, parents=parents,
                                                      composer=composer2)
                        d["decision"] = "update"
                        acc = predictors[t].accuracy(task["test"], device)

        test_accs.append(acc)
        param_curve.append(sum(p.numel() for n in nodes for p in n.concept_module.parameters()))
        print(f"  task {t:2d}: decision={decisions[t]['decision']:5s}  acc={acc:.4f}  "
              f"nodes={len(nodes)}  params={param_curve[-1]}")

        if cfg.consolidate_every and (t + 1) % cfg.consolidate_every == 0 and len(nodes) > 2:
            accept = make_accuracy_accept_fn(nodes, predictors, tasks, device, cfg.merge_tolerance)
            summ = consolidate_nodes(nodes, predictors, tasks, device, accept_fn=accept,
                                     similarity_threshold=cfg.similarity_threshold,
                                     subspace_k=cfg.subspace_k, distill=cfg.distill,
                                     distill_epochs=cfg.distill_epochs, merge_tolerance=cfg.merge_tolerance,
                                     functional_threshold=(cfg.functional_threshold
                                                           if cfg.functional_redundancy else None))
            print(f"  [consolidate @ task {t}] saved {summ['params_saved']} params, {summ['n_ops']} ops")
        _flush(device)

    # Final consolidation.
    accept = make_accuracy_accept_fn(nodes, predictors, tasks, device, cfg.merge_tolerance)
    consolidation = consolidate_nodes(nodes, predictors, tasks, device, accept_fn=accept,
                                      similarity_threshold=cfg.similarity_threshold,
                                      subspace_k=cfg.subspace_k, distill=cfg.distill,
                                      distill_epochs=cfg.distill_epochs, merge_tolerance=cfg.merge_tolerance,
                                      functional_threshold=(cfg.functional_threshold
                                                            if cfg.functional_redundancy else None))

    n_grow = sum(1 for d in decisions if d["decision"] == "grow")
    n_search = sum(1 for d in decisions if d["decision"] == "search")
    n_reuse = sum(1 for d in decisions if d["decision"] == "reuse")
    n_update = sum(1 for d in decisions if d["decision"] == "update")
    results = {
        "average_accuracy": float(np.mean(test_accs)),
        "test_accs": test_accs,
        "n_grow": n_grow,
        "n_search": n_search,
        "n_reuse": n_reuse,
        "n_update": n_update,
        "reuse_rate": (len(tasks) - n_grow) / len(tasks),   # non-grow fraction (reuse+search+update)
        "param_curve": param_curve,
        "params_final_pre_consolidation": param_curve[-1] if param_curve else 0,
        "consolidation": consolidation,
        "decisions": decisions,
    }
    out_path = os.path.join(cfg.results_dir, "exp3a_kan_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAA={results['average_accuracy']:.4f}  grow={n_grow}/{len(tasks)}  "
          f"reuse_rate={results['reuse_rate']:.2f}  "
          f"params {results['params_final_pre_consolidation']}→{consolidation['params_after']}")
    print(f"Results saved to {out_path}")

    if cfg.dump_gate_tensors and not use_cnn:
        dump = _build_gate_dump(cfg, tasks, nodes, predictors, decisions)
        dump_path = os.path.join(cfg.results_dir, "gate_dump.pt")
        torch.save(dump, dump_path)
        print(f"Gate tensor dump saved to {dump_path}")

    return results
