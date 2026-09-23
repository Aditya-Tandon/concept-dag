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

import copy
import gc
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..modules.concept_module import ConceptModule
from ..modules.token_roots import AttnPoolRoot
from ..models.baselines import LinearHead
from ..training.kan_gate import (
    TaskSpec, classification_task, decide_reuse_vs_grow, decide_reuse_search_grow,
    ReuseComposer, SearchComposer, _fit_full, update_probe,
)
from ..training.consolidate import low_rank_factorize_final_layer
from ..utils.metrics import principal_angles_between
from ..utils.rng import fork_rng_all_devices
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
            # Token inputs (N, 1 + T, D) are cached half-precision: the merge caches BOTH tasks'
            # full train splits, which at 65 x 384 float32 is ~100 kB/image (12 GB for two
            # 60k-image tasks). The token feature cache is itself float16 round-tripped
            # (feature_cache.cache_features), so the cast is exact. 2-D CLS features are
            # float32 on disk and pass through unchanged.
            xs.append(x.cpu().half() if x.ndim == 3 else x.cpu())
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
            if xb.dtype == torch.float16:
                xb = xb.float()
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


def _flat_theta(module: nn.Module) -> torch.Tensor:
    """Every parameter of `module`, flattened and concatenated into one CPU vector — the
    weight-space drift metric's raw material (reader-audit gate, [[post-mint-audit-gap]])."""
    return torch.cat([p.detach().cpu().reshape(-1) for p in module.parameters()])


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
    reader_tolerance: float,
    subspace_k: int = 8,
    truncate_energy: Optional[float] = 0.99,
    truncate_max_rel_error: float = 0.05,
    distill: bool = True,
    distill_epochs: int = 20,
    merge_tolerance: float = 0.01,
    functional_threshold: Optional[float] = None,
    merge_trigger: str = "cca",
    merge_cka_threshold: float = 0.55,
) -> Dict[str, object]:
    """
    One consolidation pass over the grown DAGNode list.

    (1) Low-rank re-crystallise every node's concept module. (2) Merge pairs whose concept subspaces
    are near-identical (principal-angle similarity ≥ threshold), re-pointing children's parent lists,
    each merge gated by ``accept_fn(keep, drop, affected_task_ids)`` which MUST re-evaluate the
    affected tasks (backward-interference check).

    Every post-mint change this pass makes to a node — a truncation or an accepted/rejected merge —
    is also appended, in order, to the returned ``reader_audit`` list (the caller adds an
    ``update_commit`` entry of its own and stamps ``at_task`` on all of them): the audit gate that
    covers every way a node can change after it was minted, not just the merge path's own forgetting
    check ([[post-mint-audit-gap]]).
    """
    def pcount():
        return sum(p.numel() for n in nodes for p in n.concept_module.parameters())

    params_before = pcount()
    ops: List[dict] = []
    reader_audit: List[dict] = []
    merge_attempted = 0

    # Task indices (predictor positions) whose predictor routes through `node`. Hoisted above step
    # (1) so the truncation audit can use it too, rather than duplicating it.
    def affected_task_ids(node: DAGNode) -> Set[int]:
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

    # (1) Low-rank re-crystallisation, audited: a truncation that hurts a reader is rolled back.
    #
    # A1: `low_rank_factorize_final_layer` applies zero times across the whole H8 archive, so a
    # run in which nothing truncates must pay NOTHING extra (no reader forward passes, no RNG
    # draws) versus today. We first probe on a `copy.deepcopy` of the module to learn whether it
    # would apply; only when it would do we pay for reader accuracies and the real (mutating) call.
    if truncate_energy is not None:
        for n in nodes:
            # The probe is a dry run and must cost the run nothing but time. An APPLIED
            # factorisation builds two fresh `nn.Linear`s, and their default init draws from the
            # CPU generator — so without this fork a node that truncates pays that init TWICE
            # (once for the throwaway deepcopy, once for real) and every later draw in the run
            # shifts. Zero runs in the H8 archive truncate, which is exactly why it went unseen.
            with fork_rng_all_devices():
                probe_rec = low_rank_factorize_final_layer(
                    copy.deepcopy(n.concept_module),
                    energy=truncate_energy, max_rel_error=truncate_max_rel_error)
            if not probe_rec.get("applied"):
                continue

            readers = sorted(affected_task_ids(n))
            before_val = {t: predictors[t].accuracy(tasks[t].get("val", tasks[t]["test"]), device)
                          for t in readers}
            before_test = {t: predictors[t].accuracy(tasks[t]["test"], device) for t in readers}
            old_mlp = n.concept_module.mlp
            old_final_weight = old_mlp[-1].weight.detach().clone()
            was_frozen = n.concept_module.is_frozen

            rec = low_rank_factorize_final_layer(
                n.concept_module, energy=truncate_energy, max_rel_error=truncate_max_rel_error)

            new_mlp = n.concept_module.mlp
            lin1, lin2 = new_mlp[-2], new_mlp[-1]
            recon = lin2.weight.detach().cpu() @ lin1.weight.detach().cpu()
            drift = float((recon - old_final_weight.cpu()).norm().item())

            after_val = {t: predictors[t].accuracy(tasks[t].get("val", tasks[t]["test"]), device)
                         for t in readers}
            after_test = {t: predictors[t].accuracy(tasks[t]["test"], device) for t in readers}
            ok = all(after_val[t] >= before_val[t] - reader_tolerance for t in readers)

            audit_entry = {
                "at_task": None, "kind": "truncate", "node": n.task_id, "readers": readers,
                "drift": drift, "rel_error": rec.get("rel_error"),
                "reader_deltas_val": {str(t): round(after_val[t] - before_val[t], 4) for t in readers},
                "reader_deltas_test": {str(t): round(after_test[t] - before_test[t], 4) for t in readers},
                "tolerance": reader_tolerance,
            }
            if ok:
                ops.append({"op": "truncate", "task": n.task_id, **rec})
                audit_entry["verdict"] = "kept"
            else:
                n.concept_module.mlp = old_mlp
                if was_frozen:
                    n.concept_module.freeze()
                ops.append({"op": "truncate_rejected", "task": n.task_id, **rec})
                audit_entry["verdict"] = "rolled_back"
            reader_audit.append(audit_entry)

    # (2) Subspace-redundancy merges (re-scan after each accepted merge).
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
                #
                # `merge_trigger="cka"` swaps the DECIDING statistic for linear CKA (the merge
                # pre-filter). Both statistics are read off ONE pass over ONE draw
                # (`_paired_root_features`): `_combined_loader` shuffles and therefore consumes the
                # global RNG, so scoring the two triggers on two passes would compare different
                # samples AND move the run's stream.
                sim_extra: Dict[str, float] = {}
                if functional_threshold is not None:
                    Fa, Fb = _paired_root_features(
                        a, b, _combined_loader(tasks, a.task_id, b.task_id), device)
                    # BOTH statistics, off the one draw, in BOTH modes. The one the trigger did
                    # not use is logging — and it is the logging that makes an offline
                    # counterfactual ("what would the other trigger have merged?") possible at
                    # all, which is the whole reason the pre-filter can be evaluated without
                    # re-running the stream. Neither statistic draws RNG and neither builds a
                    # loader of its own, so the default path's stream is untouched.
                    cca = _cca_topk(Fa, Fb, subspace_k)
                    cka = _linear_cka(Fa, Fb)
                    sim_extra = {"cka": cka, "cca_topk": cca}
                    if merge_trigger == "cka":
                        sim, sim_kind, thr = cka, "cka", merge_cka_threshold
                    else:
                        sim, sim_kind, thr = cca, "functional", functional_threshold
                    if sim < thr:
                        # A pair the trigger stops used to vanish: a bare `continue` is
                        # indistinguishable from "no pair was ever considered", so the archive
                        # could not say whether the merge rung was silent because nothing was
                        # redundant or because the trigger was mis-set. Every EXAMINED pair now
                        # leaves exactly one record per scan. Appended to the op list only — no
                        # decision, no loader, no draw moves.
                        ops.append({"op": "trigger_rejected", "keep": a.task_id, "drop": b.task_id,
                                    "similarity": sim, "sim_kind": sim_kind, "threshold": thr,
                                    **sim_extra})
                        continue
                else:
                    sim = _subspace_similarity(a, b, subspace_k)
                    sim_kind = "subspace"
                    if sim < similarity_threshold:
                        continue
                affected = affected_task_ids(a) | affected_task_ids(b)
                merge_attempted += 1

                if not distill:
                    # Structural merge (safe only for near-identical subspaces); gate decides.
                    if not accept_fn(a, b, affected):
                        continue
                    _merge_nodes(nodes, predictors, keep=a, drop=b)
                    ops.append({"op": "merge", "keep": a.task_id, "drop": b.task_id,
                                "similarity": sim, "sim_kind": sim_kind, "distilled": False,
                                **sim_extra})
                    merged = True
                    break

                # Distilled merge: snapshot → distill keep → tentatively apply adapted merge →
                # forgetting check → commit or roll back (topology + keep weights).
                base = {t: predictors[t].accuracy(tasks[t]["test"], device)
                        for t in affected if t < len(predictors)}
                snap = _snapshot_topology(nodes, predictors, keep=a)
                freeze_keep = functional_threshold is not None
                # keep is provably untouched when freeze_keep=True (W_keep is the identity, fit
                # entirely on the frozen keep output); only measure the weight-space drift when
                # distill_merge actually retrains keep.concept_module in place.
                theta_before = None if freeze_keep else _flat_theta(a.concept_module)
                loader = _combined_loader(tasks, a.task_id, b.task_id)
                W = distill_merge(a, b, loader, device, epochs=distill_epochs, freeze_keep=freeze_keep)
                drift = (0.0 if freeze_keep
                         else float((_flat_theta(a.concept_module) - theta_before).norm().item()))
                _merge_nodes(nodes, predictors, keep=a, drop=b, W_keep=W["keep"], W_drop=W["drop"])
                post = {t: predictors[t].accuracy(tasks[t]["test"], device)
                        for t in affected if t < len(predictors)}
                deltas = {t: round(post[t] - base[t], 4) for t in post}
                ok = all(post[t] >= base.get(t, 0.0) - merge_tolerance for t in post)
                audit_entry = {
                    "at_task": None, "kind": "merge", "node": a.task_id,
                    "readers": sorted(affected), "drift": drift, "rel_error": None,
                    "reader_deltas_val": {}, "reader_deltas_test": {str(t): d for t, d in deltas.items()},
                    "tolerance": merge_tolerance,
                }
                if ok:
                    ops.append({"op": "merge", "keep": a.task_id, "drop": b.task_id,
                                "similarity": sim, "sim_kind": sim_kind, "distilled": True,
                                "recon_loss": W["recon_loss"], "backward_deltas": deltas,
                                **sim_extra})
                    audit_entry["verdict"] = "kept"
                    reader_audit.append(audit_entry)
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
                                "merge_tolerance": merge_tolerance, **sim_extra})
                    audit_entry["verdict"] = "rolled_back"
                    reader_audit.append(audit_entry)
            if merged:
                break

    return {"params_before": params_before, "params_after": pcount(),
            "params_saved": params_before - pcount(), "n_ops": len(ops), "ops": ops,
            "merge_attempted": merge_attempted,
            "merge_accepted": sum(1 for op in ops if op["op"] == "merge"),
            "merge_rejected": sum(1 for op in ops if op["op"] == "merge_rejected"),
            "reader_audit": reader_audit}


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
    Fa, Fb = _paired_root_features(a, b, loader, device, max_batches)
    return _cca_topk(Fa, Fb, top_k)


def _paired_root_features(a: DAGNode, b: DAGNode, loader, device: str,
                          max_batches: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Both concepts' outputs over the SAME sample batch of `loader`, in the same order.

    Factored out of `_functional_similarity` so every redundancy statistic is read off one pass
    over one draw. That matters for more than cost: `_combined_loader` shuffles, so a second pass
    would both score the two triggers on different samples and advance the run's RNG stream.
    Returns raw (uncentred) CPU float32 (N, concept_dim) matrices.
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
    return torch.cat(a_out), torch.cat(b_out)


def _cca_topk(Fa: torch.Tensor, Fb: torch.Tensor, top_k: int) -> float:
    """Mean of the top-k canonical correlations between two feature matrices, in [0, 1]."""
    Ha, Hb = Fa, Fb
    if Ha.shape[0] <= Ha.shape[1]:               # too few samples for a stable CCA
        return 0.0
    Ha = Ha - Ha.mean(0, keepdim=True)
    Hb = Hb - Hb.mean(0, keepdim=True)
    qa, _ = torch.linalg.qr(Ha)
    qb, _ = torch.linalg.qr(Hb)
    sv = torch.linalg.svdvals(qa.T @ qb).clamp(0.0, 1.0)
    k = min(top_k, sv.numel())
    return float(sv[:k].mean().item())


def _linear_cka(Fa: torch.Tensor, Fb: torch.Tensor) -> float:
    """Linear CKA between two column-centred feature matrices (Kornblith 2019), in [0, 1].

    ``cka = ||Fa^T Fb||_F^2 / (||Fa^T Fa||_F ||Fb^T Fb||_F)`` — 1 exactly when the two
    representations agree up to an orthogonal map (and any isotropic scale), so like CCA it is
    basis-invariant, but unlike CCA it is NOT invariant to an arbitrary invertible map: it keeps
    reading the variance each direction carries. That is what makes it usable as a merge
    PRE-FILTER — the top-k CCA of two roots saturates near 1 on any pair whose class-carrying
    directions happen to span, which is why the H8 archive's merge rung fires (or does not) almost
    independently of how redundant the pair really is. Needs no QR/SVD, so it costs one matmul per
    side on features `_paired_root_features` already collected.
    """
    Ha = Fa - Fa.mean(0, keepdim=True)
    Hb = Fb - Fb.mean(0, keepdim=True)
    cross = float((Ha.T @ Hb).pow(2).sum().item())
    na = float((Ha.T @ Ha).norm().item())
    nb = float((Hb.T @ Hb).norm().item())
    if na <= 0.0 or nb <= 0.0:                   # a constant representation has no similarity
        return 0.0
    return cross / (na * nb)


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
    merge_trigger:       str   = "cca"    # "cca" | "cka" — which statistic DECIDES a merge
                                          # candidate. "cca" is the published trigger; "cka" is the
                                          # linear-CKA pre-filter (merge-detector separability).
                                          # Both are always RECORDED on every merge op record.
    merge_cka_threshold: float = 0.55     # linear CKA a pair must reach (merge_trigger == "cka")
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
    reducible_mode:      str   = "best"   # "grow" | "best" — which normaliser DECIDES (both always
                                          # recorded on the KanGateRecord). Published runs up to
                                          # 2026-09-03 used "grow"; "best" is the default now —
                                          # validated equal decisions in best-rung-denominator-
                                          # stress-test / gate-arms-multiseed-ctrl-result. "grow"
                                          # stays reachable via --reducible grow.
    gate_cache_max:       int   = 16384   # samples the GATE's probe cache sees (0 = unlimited /
                                          # full task). Separate from `routing_batches`, which still
                                          # caps `route_for_task` / `compute_concept_subspace`: a
                                          # cache capped at routing_batches * batch_size (2,560 by
                                          # default) is far below the 50-70k a 5-Datasets root
                                          # deploys on, so the grow probe was starved on data-rich
                                          # tasks (5-Datasets SVHN, seed 42).
    gate_estimator:      str   = "single" # "single" | "crossfit" | "prequential" — how the gate's
                                          # held-out code lengths are measured (decide_reuse_search_grow
                                          # only).
    gate_splits:         int   = 5        # crossfit folds (gate_estimator == "crossfit").
    preq_blocks:         int   = 5        # prequential block count B (gate_estimator == "prequential").
    preq_decide:         str   = "tail"   # "tail" | "total" — which prequential quantity DECIDES
                                          # (gate_estimator == "prequential").
    preq_exponent:       float = 0.5      # tail power-law exponent (gate_estimator == "prequential").
    tie_rule:            str   = "none"   # "none" | "grow" — the select-score tie rule
                                          # (gate_estimator == "select-score"); "grow" resolves a
                                          # variance-limited search-vs-grow tie to grow on a novel task.
    tie_z:               float = 1.0      # tie band width in SEs of the paired SCORE-bit difference
                                          # (gate_estimator == "select-score", tie_rule == "grow").
    tie_novelty:         float = 0.1      # novelty guard: (L_null - L_reuse)/L_null must be BELOW
                                          # this for the tie rule to fire (gate_estimator ==
                                          # "select-score", tie_rule == "grow").
    # --- decision timing ([[provisional-growth-undetermined-gate]]) -----------------------------
    provisional:         str   = "off"   # "off" | "shadow" | "evalue" | "se_proxy" | "always"
                                          # off      = published behaviour, nothing computed
                                          # shadow   = compute the third state, ACT ON NOTHING (P0b)
                                          # evalue   = UNDETERMINED -> mint a provisional root (arm T)
                                          # se_proxy = |L_alt - L_grow| <= z*paired_SE -> mint (arm C1)
                                          # always   = mint unconditionally at every gated position
                                          #            whose gate cache is <= `always_n_max` (arm C2)
    provisional_alpha:   float = 0.05     # the e-process level; decide at 1/alpha
    provisional_z:       float = 1.0      # se_proxy arm: margin band in paired SEs
    always_n_max:        int   = 1000     # "always" arm: only data-poor positions (CTrL t3 has 400)
    crystallise_after:   int   = 3        # tasks after which a still-flagged provisional root freezes
    oracle_rungs:        bool  = False    # after the decision, ALSO train+eval the other rungs'
                                          # predictors (reuse/search/grow) on task["test"], without
                                          # altering the DAG — for post-hoc regret analysis.
    dump_gate_tensors:   bool  = False    # write gate_dump.pt (feature mode only) — raw per-task
                                          # tensors + node/predictor state, for offline desk-stage
                                          # re-analysis without re-running training.
    dump_max_per_split:  int   = 1024     # token mode only: examples per split the dump keeps
                                          # (the FIRST n in loader order; a split SHORTER than
                                          # this is dumped whole). A token set is ~100 kB an
                                          # image, so the CLS dump's "everything" is tens of GB.
                                          # 1024 > `always_n_max` (1000) on purpose: every
                                          # data-poor position the provisional hypothesis is
                                          # about — CTrL t3's 400 — is dumped COMPLETE.
    dump_max_bytes:      float = 2e9      # refuse to write a gate_dump.pt bigger than this
    token_pool:          Optional[int] = None   # patch-grid average-pool factor the token cache
                                          # was built with; recorded in the dump so a desk script
                                          # can tell a 64-token run from a 256-token one.
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


@torch.no_grad()
def _mean_nll_bits(predictor: "TaskPredictor", loader, spec: TaskSpec, device: str) -> float:
    """Mean `spec.nll_bits` (bits/sample) of `predictor` over every batch in `loader`, weighted by
    batch size (spec B3) — the calibration target for the prequential tail estimate."""
    total_bits = 0.0
    total_n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        bits = spec.nll_bits(predictor.logits(x), y)
        total_bits += bits.sum().item()
        total_n += y.size(0)
    return total_bits / max(total_n, 1)


def _oracle_rungs(rec, X: torch.Tensor, y: torch.Tensor, parents: List[DAGNode], task: Dict, t: int,
                  cfg: "KanExpConfig", spec: TaskSpec, use_cnn: bool, device: str,
                  chosen_acc: float, chosen_predictor: Optional["TaskPredictor"] = None,
                  ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Train + evaluate the rungs NOT chosen by the frozen ladder, on the same cache, WITHOUT
    altering the DAG. The chosen rung's entry reuses `chosen_acc`/`chosen_predictor` rather than
    retraining. Wrapped in `torch.random.fork_rng()` so it does not perturb the main run's RNG
    stream. Returns `(oracle_accs, oracle_val_bits)`: for every rung (reuse/search/grow), also the
    predictor's mean `spec.nll_bits` on `task["val"]` (fallback `task["test"]` if no `"val"`) —
    the calibration target for the prequential tail estimate (spec B3)."""
    accs: Dict[str, float] = {}
    val_bits: Dict[str, float] = {}
    val_loader = task.get("val", task["test"])
    with torch.random.fork_rng():
        if rec.decision == "reuse":
            accs["reuse"] = chosen_acc
            val_bits["reuse"] = _mean_nll_bits(chosen_predictor, val_loader, spec, device)
        else:
            oh = LinearHead(cfg.concept_dim, task["n_classes"])
            oc = ReuseComposer(parent_dim=cfg.concept_dim, n_parents=len(parents), head=oh)
            _fit_full(oc, lambda m, xb: m(xb), spec, X, y, cfg.child_epochs, cfg.lr, device)
            pred = TaskPredictor("reuse", oh, parents=parents, composer=oc)
            accs["reuse"] = pred.accuracy(task["test"], device)
            val_bits["reuse"] = _mean_nll_bits(pred, val_loader, spec, device)

        if rec.decision == "search":
            accs["search"] = chosen_acc
            val_bits["search"] = _mean_nll_bits(chosen_predictor, val_loader, spec, device)
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
            pred = TaskPredictor("search", oh, parents=parents, composer=oc)
            accs["search"] = pred.accuracy(task["test"], device)
            val_bits["search"] = _mean_nll_bits(pred, val_loader, spec, device)

        if rec.decision == "grow":
            accs["grow"] = chosen_acc
            val_bits["grow"] = _mean_nll_bits(chosen_predictor, val_loader, spec, device)
        else:
            onode = DAGNode(task_id=t, concept_dim=cfg.concept_dim, cnn_out_dim=cfg.cnn_out_dim,
                            n_mlp_layers=cfg.n_mlp_layers, parent_models=None,
                            soft_pca_k=cfg.soft_pca_k, use_cnn=use_cnn, feature_dim=cfg.feature_dim,
                           root_family=cfg.root_family)
            ohead = LinearHead(cfg.concept_dim, task["n_classes"])
            train_node(onode, ohead, task["train"], cfg.child_epochs, cfg.lr, device, cfg.log_every,
                       name=f"t{t}-oracle-grow", orth_weight=cfg.orth_weight)
            accs["grow"] = eval_node(onode, ohead, task["test"], device)
            # DAGNode root + LinearHead: logits = head(node(x)) — TaskPredictor's "grow" kind
            # computes exactly that via forward_dag_memoized (a root has no ancestors to memoize).
            opred = TaskPredictor("grow", ohead, node=onode)
            val_bits["grow"] = _mean_nll_bits(opred, val_loader, spec, device)
            del onode, ohead, opred
            _flush(device)
    return accs, val_bits


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


def _cache_raw_first_n(loader, n_max: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The FIRST `n_max` examples in loader order (0 = all).

    Deterministic by construction: it selects a prefix rather than drawing a subset, so the token
    dump owes the run no RNG state. The loader's own shuffling still decides which examples a
    prefix contains, which is why the caller forks around the iteration.
    """
    xs, ys = [], []
    got = 0
    for x, y in loader:
        take = len(y) if not n_max else min(len(y), n_max - got)
        xs.append(x[:take].cpu())
        ys.append(y[:take].cpu())
        got += take
        if n_max and got >= n_max:
            break
    if not xs:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(xs, 0), torch.cat(ys, 0)


def _root_mint_snapshot(node: DAGNode, predictor: TaskPredictor, t: int) -> dict:
    """A CPU float32 copy of a root AND its reader, the moment the root finished training.

    Consolidation merges roots away and rolls truncations back, so the end-of-run `nodes` list is
    not enough to recover what any given task actually minted — the merged ones are simply gone.
    The root alone is not enough either: re-scoring a dropped root's task off the dump needs the
    head (and, for a non-grow reader, the composer) that read it, which is exactly what
    `distill_merge`'s backward audit compares against. Plain tensor copies, taken outside anything
    that draws: the snapshot cannot move the run.
    """
    composer = getattr(predictor, "composer", None)
    return {
        "task_id": t, "_node": node,
        "state_dict": {k: v.detach().cpu().float().clone()
                       for k, v in node.state_dict().items()},
        "predictor_kind": predictor.kind,
        "head_state": {k: v.detach().cpu().float().clone()
                       for k, v in predictor.head.state_dict().items()},
        "composer_kind": (None if composer is None else type(composer).__name__),
        "composer_state": (None if composer is None else
                           {k: v.detach().cpu().float().clone()
                            for k, v in composer.state_dict().items()}),
    }


def _dump_nbytes(obj) -> int:
    """Bytes of tensor payload in a (nested) dump structure — the size guard's estimate."""
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_dump_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_dump_nbytes(v) for v in obj)
    return 0


def _build_gate_dump(cfg: "KanExpConfig", tasks: List[Dict], nodes: List[DAGNode],
                     predictors: List[TaskPredictor], decisions: List[dict],
                     gate_cache: Dict[int, Tuple[Optional[torch.Tensor], torch.Tensor]],
                     roots_at_mint: Optional[List[dict]] = None) -> dict:
    # Token mode keeps a bounded PREFIX of train and test only; the CLS dump below is unchanged,
    # field for field, because every desk script that reads a published dump depends on it.
    token_mode = (cfg.root_family == "attn_pool")
    # Same GATE cache sizing as run_exp3a_kan's live decisions — separate from `routing_batches`.
    gate_batches = math.ceil(cfg.gate_cache_max / cfg.batch_size) if cfg.gate_cache_max else 0
    gate_dump_tasks = []
    if token_mode:
        # A (1 + T, D) token set is ~100 kB an image, so "every split in full" is tens of GB on a
        # 5-Datasets stream. Keep a bounded PREFIX of train and test, in fp16 (the cache's own
        # storage dtype — see test_token_feature_cache_stays_half_in_memory). Val is dropped
        # outright: the desk analyses this dump exists for score train-fit against test, and the
        # `meta` block below says so rather than leaving a reader to infer it from a missing key.
        for t, task in enumerate(tasks):
            entry = {"task": t, "n_classes": task["n_classes"], "ctrl": task.get("ctrl")}
            for split in ("train", "test"):
                # `_cache_raw_first_n` draws nothing itself, but iterating a shuffled train loader
                # does; fork so a dump can never move a stream the run has already been scored on.
                with fork_rng_all_devices():
                    raw, y = _cache_raw_first_n(task[split], cfg.dump_max_per_split)
                entry[f"{split}_raw"] = raw.half()
                entry[f"{split}_y"] = y.long()
            gate_dump_tasks.append(entry)
    else:
        for t, task in enumerate(tasks):
            if t in gate_cache:
                # SAME cap/order as the (Xraw, y) that fed this task's live gate decision
                # (INTERFACE_SPEC.md §6) — captured once at decision time in the main loop, not
                # re-sampled here (re-sampling from the shuffled loader after the run would draw a
                # different `gate_cache_max`-sized subset than `_cache_parent_stack` used).
                train_raw, train_y = gate_cache[t]
            else:
                # No gate decision was made for this task (root: growth forced, nothing to
                # reproduce) — re-sampling here is harmless, but fork the RNG so it doesn't perturb
                # the main stream that later tasks' training/decisions already consumed.
                with torch.random.fork_rng(devices=[]):
                    train_raw, train_y = _cache_raw_capped(task["train"], gate_batches)
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

    dump = {
        "feature_mode": True,
        "concept_dim": cfg.concept_dim,
        "feature_dim": cfg.feature_dim,
        "n_parents": cfg.n_parents,
        "seed": cfg.seed,
        "config": {k: getattr(cfg, k) for k in (
            "gate_epochs", "gate_lr", "eps_rel", "eps_search", "search_budget", "search_rank",
            "search_skip", "routing_batches", "gate_cache_max", "child_epochs", "lr", "reducible_mode",
            "gate_estimator", "gate_splits", "update_lr", "eps_update", "update_tolerance",
            "subspace_k", "n_mlp_layers",
        )},
        "tasks": gate_dump_tasks,
        "nodes": gate_dump_nodes,
        "predictors": gate_dump_preds,
        "decisions": decisions,
    }
    if not token_mode:
        # A published CLS dump is what every existing desk script reads; it gains nothing here.
        return dump

    # `nodes` holds each node's CONCEPT MODULE only, which in token mode leaves the root's
    # AttentionPool — the thing the whole family ablation is about — out of the dump entirely.
    dump["mode"] = "token"
    dump["token_pool"] = cfg.token_pool
    dump["roots"] = [
        {"index": i, "task_id": n.task_id,
         "state_dict": {k: v.detach().cpu().float() for k, v in n.state_dict().items()}}
        for i, n in enumerate(nodes) if n.is_root
    ]
    # ... and consolidation merges roots away, so the end-of-run list cannot answer "what did
    # task t actually mint?" for a root that was later dropped. `node_id` is the index into
    # `nodes`/`roots` if the snapshotted root survived to the end of the run, and None if a
    # merge took it.
    dump["roots_at_mint"] = [
        {k: v for k, v in r.items() if k != "_node"}
        | {"node_id": node_index.get(id(r["_node"]))}
        for r in (roots_at_mint or [])
    ]
    dump["meta"] = {
        "splits": ["train", "test"],
        "val_omitted": True,
        "val_omitted_reason": (
            "token mode keeps a bounded prefix per split; val is dropped because the desk "
            "analyses this dump exists for fit on train and score on test"),
        "max_per_split": cfg.dump_max_per_split,
        "selection": ("first n examples in loader order (no RNG draw of its own); a split with "
                      "fewer than max_per_split examples is dumped WHOLE, and the default 1024 "
                      "is above always_n_max (1000) so every data-poor gated position is"
                      " complete"),
        "raw_dtype": "float16",
        "raw_shape": "(n, 1 + T, feature_dim)",
        "n_tokens": cfg.n_tokens,
        "roots_at_mint_keys": {
            "task_id": "stream position that minted this root",
            "node_id": "index into `nodes`/`roots` if it survived to the end of the run, else None"
                       " (a consolidation merge dropped it)",
            "state_dict": "the ROOT's full state at mint, CPU float32 (AttentionPool included)",
            "predictor_kind": "the reader's rung at mint — 'grow' for every freshly minted root",
            "head_state": "that task's reader head at mint, CPU float32",
            "composer_kind": "class name of the reader's composer, or None for a grow reader",
            "composer_state": "the composer's state at mint, CPU float32, or None",
        },
    }
    return dump


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
    # The GATE's probe cache is sized to the task (gate_cache_max samples), separate from
    # `routing_batches` which still caps `route_for_task` / `compute_concept_subspace`. 0 means
    # unlimited (no cap passed through to `_cache_parent_stack` / `_cache_raw_capped`).
    gate_batches = math.ceil(cfg.gate_cache_max / cfg.batch_size) if cfg.gate_cache_max else 0
    use_cnn = (cfg.backbone == "smallcnn")

    # --- token-mode preconditions -------------------------------------------------------
    if cfg.root_family == "attn_pool":
        if use_cnn:
            raise ValueError("root_family='attn_pool' requires a feature backbone, not smallcnn.")
        if cfg.enable_update:
            raise NotImplementedError(
                "--enable_update is not supported with root_family='attn_pool': the update probe "
                "packs the raw features into a flat (N, P*D + F) tensor (pack_update_input), which "
                "a (N, 1 + T, D) token set has no 2-D form for. Run the update rung under "
                "root_family='mlp_cls', or extend _UpdateModel to carry a token root."
            )
    # `--dump_gate_tensors` used to be refused in token mode, because the CLS dump keeps every
    # split of every task in full and a token set is ~100 kB an image (65 x 384 float32) — tens of
    # GB for a 5-Datasets stream. The token dump instead keeps the FIRST `dump_max_per_split`
    # examples of train and test only, and is guarded by `dump_max_bytes` at write time.
    dump_gate_tensors = cfg.dump_gate_tensors

    nodes: List[DAGNode] = []
    predictors: List[TaskPredictor] = []
    decisions: List[dict] = []
    param_curve: List[int] = []
    param_curve_total: List[int] = []
    test_accs: List[float] = []
    # (Xraw, y) actually fed to each gated task's live decision, captured at decision time so
    # `_build_gate_dump` can dump the SAME rows/order instead of re-sampling post hoc (§6).
    gate_cache: Dict[int, Tuple[Optional[torch.Tensor], torch.Tensor]] = {}
    # Every ROOT's full state the moment it finished training, before any consolidation pass could
    # merge or truncate it (token-mode dump only — see `_root_mint_snapshot`).
    roots_at_mint: List[dict] = []
    # One record per provisional root, from mint to resolution. `resolution` is the quantity the
    # loop turns on: a mechanism whose roots are ALL resolved by timeout is delayed unconditional
    # growth, not decision timing ([[provisional-growth-undetermined-gate]] P6).
    provisional_log: Dict[int, dict] = {}
    # Every consolidation pass's summary dict, in call order (see `consolidation_passes` in
    # `results`); `consolidate_every` intermediate passes are otherwise thrown away except for a
    # one-line print, which hid every accepted merge on `s_interleave`
    # ([[consolidation-provenance]] P6).
    consolidation_passes: List[dict] = []
    # Every post-mint change to any node, in chronological order: truncations and merges (stamped
    # from each pass's own `reader_audit`, below) plus `update_commit` entries appended in place
    # at commit time — the audit trail the reader-audit gate covers ([[post-mint-audit-gap]] P5).
    reader_audit: List[dict] = []

    def _resolve_provisional(current_t: int, end_of_stream: bool = False,
                             pass_result: Optional[dict] = None) -> None:
        """Mark merged provisional roots, and crystallise the ones that have run out of time.

        A provisional root that consolidation merged away is no longer in `nodes` (`_merge_nodes`
        removes it), which is the merge signal that it resolved by merge — but WHICH merge, and
        into what, has to come from `pass_result["ops"]` (the pass that just ran), not inference:
        map the vanished node's `task_id` to the accepted `{"op": "merge", ...}` whose `drop`
        names it. If no such op exists, the node vanished for an unexplained reason — a bug
        signal, not a merge — and is recorded as `"removed_unexplained"` rather than silently
        called a merge.
        """
        live = {id(n): n for n in nodes}
        merge_by_drop = {op["drop"]: op for op in (pass_result or {}).get("ops", [])
                         if op["op"] == "merge"}
        for task_id, rec_p in provisional_log.items():
            if rec_p["resolution"] is not None:
                continue
            node = rec_p["_node"]
            if id(node) not in live:
                merge_op = merge_by_drop.get(task_id)
                if merge_op is not None:
                    rec_p["resolution"] = "merge"
                    rec_p["merged_into"] = merge_op["keep"]
                    rec_p["merge_pass_at_task"] = current_t
                else:
                    rec_p["resolution"] = "removed_unexplained"
                rec_p["resolved_at"] = current_t
                continue
            aged_out = (current_t - rec_p["minted_at"]) >= cfg.crystallise_after
            if end_of_stream or aged_out:
                node.provisional = False
                node.freeze()
                rec_p["resolution"] = "timeout"
                rec_p["resolved_at"] = current_t
                rec_p["params_at_crystallisation"] = sum(p.numel() for p in node.parameters())

    def new_module_factory(parents: List[DAGNode]):
        def factory():
            return ConceptModule(
                module_id="__probe__", in_dim=cfg.concept_dim, hidden_dim=cfg.concept_dim,
                out_dim=cfg.concept_dim, n_layers=cfg.n_mlp_layers, n_parents=len(parents),
                aggregation="soft_pca", agg_kwargs={"top_k": min(cfg.soft_pca_k, cfg.concept_dim)},
            )
        return factory

    # Raw-root grow probe: capacity-matched to a real grown root DAGNode. Feature-mode only — in
    # CNN mode the raw input is an image and the probe would need its own backbone, so we fall back
    # to the parents-only probe there.
    #
    # The probe must be the SAME FAMILY as the root the decision would deploy, or L_grow prices a
    # module the DAG would never build. In token mode that is `AttnPoolRoot` — AttentionPool
    # (feature_dim → concept_dim) + the identical 2-layer ConceptModule the token-mode DAGNode
    # root carries, in the same construction order; `tests/test_attn_root_adoption.py` asserts the
    # parameter-count and structural parity. In CLS mode it is the published ConceptModule probe,
    # unchanged.
    use_raw_probe = cfg.raw_grow_probe and not use_cnn and cfg.feature_dim is not None
    token_root_mode = (not use_cnn) and cfg.root_family == "attn_pool"

    def root_module_factory():
        if token_root_mode:
            return AttnPoolRoot(feature_dim=cfg.feature_dim, concept_dim=cfg.concept_dim)
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
                           soft_pca_k=cfg.soft_pca_k, use_cnn=use_cnn, feature_dim=cfg.feature_dim,
                           root_family=cfg.root_family)
            train_node(node, head, task["train"], cfg.root_epochs, cfg.lr, device, cfg.log_every,
                       name=f"t{t}-root", orth_weight=cfg.orth_weight)
            node.compute_concept_subspace(task["train"], device, top_k=cfg.subspace_k,
                                          max_batches=cfg.routing_batches)
            node.freeze()
            nodes.append(node)
            predictors.append(TaskPredictor("grow", head, node=node))
            if dump_gate_tensors and node.is_root:
                roots_at_mint.append(_root_mint_snapshot(node, predictors[-1], t))
            decisions.append({"task": t, "decision": "grow", "reason": "root"})
        else:
            sel_idx, _scores = route_for_task(nodes, task["train"], n_par, cfg.subspace_k,
                                              device, cfg.routing_batches)
            parents = [nodes[i] for i in sel_idx]
            # --- Kan gate on cached parent embeddings (+ raw features for the root grow probe) ---
            # Sized to `gate_cache_max`, NOT `routing_batches` — the gate's probe cache is the
            # task's own budget, decoupled from the (much smaller) routing subspace cap.
            X, Xraw, y = _cache_parent_stack(parents, task["train"], device,
                                             max_batches=gate_batches)
            gate_cache_n = y.shape[0]
            if dump_gate_tensors:
                # Only retained when a dump will be written: Xraw is the full gate cache
                # (16,384 x 1+T x D in token mode ≈ 1.6 GB float32) and holding one per task
                # would carry the whole stream's caches to the end of the run for nothing.
                # The token dump reads a bounded prefix straight off the loaders and never looks
                # at this cache, so there it keeps the labels alone (`results["_gate_cache_y"]`).
                gate_cache[t] = (None if token_root_mode else Xraw, y)
            raw_kwargs = ({"raw_stack": Xraw, "root_module_factory": root_module_factory}
                          if use_raw_probe else {})
            # Prequential-estimator kwargs are passed ONLY when selected, so the call signature
            # for "single"/"crossfit" is unchanged even before decide_reuse_search_grow grows a
            # `preq_*` branch (spec B1) — this file's tests must pass independent of that landing.
            preq_kwargs = ({"preq_blocks": cfg.preq_blocks, "preq_decide": cfg.preq_decide,
                            "preq_exponent": cfg.preq_exponent}
                          if cfg.gate_estimator == "prequential" else {})
            # select-score tie-rule kwargs are passed ONLY when selected, mirroring preq_kwargs —
            # the call signature for every other gate_estimator is unchanged.
            ss_kwargs = ({"tie_rule": cfg.tie_rule, "tie_z": cfg.tie_z, "tie_novelty": cfg.tie_novelty}
                        if cfg.gate_estimator == "select-score" else {})
            force = t in cfg.force_grow_ids
            split_gen = torch.Generator().manual_seed(cfg.seed * 1000 + t)
            gate_t0 = time.perf_counter()
            if cfg.enable_search and not force:
                # Three-way reuse/search/grow escalation (test-time-compute rung).
                rec = decide_reuse_search_grow(
                    X, y, new_module_factory(parents), spec,
                    concept_dim=cfg.concept_dim, n_parents=len(parents), device=device,
                    n_epochs=cfg.gate_epochs, lr=cfg.gate_lr, eps_grow=cfg.eps_rel,
                    eps_search=cfg.eps_search, search_budget=cfg.search_budget, search_rank=cfg.search_rank,
                    search_skip=cfg.search_skip, reducible_mode=cfg.reducible_mode,
                    estimator=cfg.gate_estimator, n_splits=cfg.gate_splits, split_generator=split_gen,
                    **raw_kwargs, **preq_kwargs, **ss_kwargs,
                )
            else:
                rec = decide_reuse_vs_grow(
                    X, y, new_module_factory(parents), spec,
                    concept_dim=cfg.concept_dim, n_parents=len(parents), device=device,
                    n_epochs=cfg.gate_epochs, lr=cfg.gate_lr, eps_rel=cfg.eps_rel,
                    reducible_mode=cfg.reducible_mode,
                    **raw_kwargs,
                )
            gate_seconds = time.perf_counter() - gate_t0

            # --- the third gate state, on a SHADOW refit (G1) -------------------------------
            # The live ladder above is untouched: this repeats it under the `select-score`
            # estimator inside a forked RNG, purely to obtain per-example score bits whose set
            # never selected an epoch. The shadow's own decision is discarded; only its
            # `estimator_meta["evalue"]` and paired SEs are read. Cost: one extra ladder fit.
            #
            # The fork must cover the DEVICE generator, not just the CPU one (P0b). Dropout masks
            # are drawn on the training device, and `torch.manual_seed` below re-seeds every
            # device generator; `torch.random.fork_rng(devices=[])` puts back only the CPU half,
            # so the shadow used to leave the device generator wherever its own refit ended.
            # Whether that showed up depended on arithmetic: the live rung's split ('single',
            # val_fraction 0.3) and the shadow's ('select-score', ss_fracs 0.65/0.15/0.20) give
            # the same minibatch count per epoch at CTrL's n (200 -> 2/2, 2,000 -> 11/11) and a
            # different one at the 16,384-row 5-Datasets gate cache (90 vs 84), so the arm that
            # computes and acts on nothing changed the 5-Datasets run and not the CTrL ones.
            ev = None
            shadow_seconds = 0.0
            if cfg.provisional != "off" and cfg.enable_search and not force:
                sh_t0 = time.perf_counter()
                with fork_rng_all_devices():
                    torch.manual_seed(cfg.seed * 1000 + t + 900)
                    shadow = decide_reuse_search_grow(
                        X, y, new_module_factory(parents), spec,
                        concept_dim=cfg.concept_dim, n_parents=len(parents), device=device,
                        n_epochs=cfg.gate_epochs, lr=cfg.gate_lr, eps_grow=cfg.eps_rel,
                        eps_search=cfg.eps_search, search_budget=cfg.search_budget,
                        search_rank=cfg.search_rank, search_skip=cfg.search_skip,
                        reducible_mode=cfg.reducible_mode, estimator="select-score",
                        split_generator=torch.Generator().manual_seed(cfg.seed * 1000 + t + 900),
                        evalue_bits=math.log2(max(task["n_classes"], 2)),
                        evalue_alpha=cfg.provisional_alpha,
                        **raw_kwargs,
                    )
                sm = shadow.estimator_meta or {}
                ev = dict(sm.get("evalue") or {})
                ev["shadow_decision"] = shadow.decision
                ev["se_search_minus_grow"] = sm.get("se_search_minus_grow")
                ev["se_reuse_minus_grow"] = sm.get("se_reuse_minus_grow")
                # The cheap rule, recorded for the necessity arm: is the live margin inside z SEs?
                alt_is_search = ev.get("alt_rung") == "search"
                se_alt = (sm.get("se_search_minus_grow") if alt_is_search
                          else sm.get("se_reuse_minus_grow")) or 0.0
                live_margin = abs((rec.L_search_bits if alt_is_search else rec.L_reuse_bits)
                                  - rec.L_grow_bits)
                ev["se_proxy_margin"] = live_margin
                ev["se_proxy_se"] = se_alt
                ev["se_proxy_fires"] = bool(se_alt > 0 and live_margin <= cfg.provisional_z * se_alt)
                shadow_seconds = time.perf_counter() - sh_t0
            _flush(device)

            if force:
                # Merge stress-test: skip the gate and grow unconditionally, so a redundant
                # concept exists for the consolidation pass to detect and merge. NOT a claim
                # the gate would grow here — the injected duplicate would correctly reuse.
                # Grow it as a ROOT (parent_models=None) so it is PARALLEL to the concept it
                # duplicates, not stacked on it: find_redundant_pairs excludes ancestor/descendant
                # pairs, so a child of the original could never be a merge candidate.
                rec.decision = "grow"
                d = {"task": t, "decision": "grow", "parents": [], "reason": "force-grow(dup-stress,parallel-root)",
                     "gate_cache_n": gate_cache_n,
                     **{k: getattr(rec, k) for k in ("rel_improvement", "L_reuse_bits", "L_grow_bits")}}
            else:
                d = {"task": t, "decision": rec.decision, "parents": sel_idx,
                     "grow_probe_input": rec.grow_probe_input, "gate_cache_n": gate_cache_n,
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
            # Wall time of the ladder itself (probe training + scoring) — the cost axis a token
            # root moves most, since every rung's grow probe now reads 1 + T tokens per sample.
            d["gate_seconds"] = gate_seconds
            if ev is not None:
                d["evalue"] = ev
                d["shadow_seconds"] = shadow_seconds
            # Which trigger, if any, defers this decision into a provisional root. `shadow`
            # computes everything and acts on nothing, which is what makes P0b a null check.
            provisional_here = False
            if ev is not None and not force:
                if cfg.provisional == "evalue":
                    provisional_here = (ev.get("state") == "undetermined")
                elif cfg.provisional == "se_proxy":
                    provisional_here = bool(ev.get("se_proxy_fires"))
                elif cfg.provisional == "always":
                    provisional_here = (gate_cache_n <= cfg.always_n_max)
            if provisional_here:
                # Defer: the ladder's chosen rung is replaced by a freshly grown ROOT, flagged.
                # Where the ladder already grows, the deferral is a no-op on the DAG and only the
                # flag is set, so a trigger never *removes* a root the gate wanted. The ladder's
                # own verdict is kept as `ladder_decision` so P3 can compare decided positions.
                d["ladder_decision"] = rec.decision
                rec.decision = "grow"
                d["decision"] = "grow"
            if cfg.provisional != "off":
                # Only when the feature is on, so `--provisional off` leaves the decision records
                # byte-identical to the published ones (P0a is a check on the JSON itself).
                d["provisional"] = bool(provisional_here)
            decisions.append(d)

            if rec.decision == "grow":
                # A raw-probe grow certified a ROOT's view (raw encoder features), so mint a root —
                # composition over existing concepts is reuse/search's job, not the grown node's.
                grow_as_root = force or use_raw_probe
                grow_parents = None if grow_as_root else parents
                node = DAGNode(task_id=t, concept_dim=cfg.concept_dim, cnn_out_dim=cfg.cnn_out_dim,
                               n_mlp_layers=cfg.n_mlp_layers, parent_models=grow_parents,
                               soft_pca_k=cfg.soft_pca_k, use_cnn=use_cnn, feature_dim=cfg.feature_dim,
                               root_family=cfg.root_family)
                # Loop 1 provisional roots are ordinary roots plus bookkeeping: no fast output
                # map, no writes, no schedule (those are [[provisional-root-plasticity]]). With
                # nothing plastic on the node, a provisional root cannot perturb a reader, which
                # is what keeps P3's null check a null check rather than a confound.
                node.provisional = bool(provisional_here)
                node.minted_at = t
                if provisional_here:
                    provisional_log[t] = {
                        "minted_at": t, "resolution": None, "resolved_at": None,
                        "trigger": cfg.provisional,
                        "e_state": (ev or {}).get("state"),
                        "log2_e_plus": (ev or {}).get("log2_e_plus"),
                        "log2_e_minus": (ev or {}).get("log2_e_minus"),
                        "n_score": (ev or {}).get("n"),
                        "n_needed_plus": (ev or {}).get("n_needed_plus"),
                        "n_needed_minus": (ev or {}).get("n_needed_minus"),
                        "ladder_decision": d.get("ladder_decision", rec.decision),
                        "params_at_mint": None, "_node": node,
                    }
                node.evalue_state = dict(ev) if (provisional_here and ev) else None
                train_node(node, head, task["train"], cfg.child_epochs, cfg.lr, device, cfg.log_every,
                           name=f"t{t}-{'prov' if provisional_here else 'grow'}"
                                f"{'-root' if grow_as_root else ''}", orth_weight=cfg.orth_weight)
                node.compute_concept_subspace(task["train"], device, top_k=cfg.subspace_k,
                                              max_batches=cfg.routing_batches)
                # A provisional root is frozen like any other in Loop 1 (nothing is plastic yet);
                # the flag only tells consolidation it may be resolved.
                node.freeze()
                nodes.append(node)
                if t in provisional_log:
                    provisional_log[t]["params_at_mint"] = sum(p.numel() for p in node.parameters())
                predictors.append(TaskPredictor("grow", head, node=node))
                if dump_gate_tensors and node.is_root:
                    # Provisional or not: a provisional root is an ordinary root plus a flag, and
                    # it is precisely the one consolidation is allowed to drop.
                    roots_at_mint.append(_root_mint_snapshot(node, predictors[-1], t))
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
                d["oracle_accs"], d["oracle_val_bits"] = _oracle_rungs(
                    rec, X, y, parents, task, t, cfg, spec, use_cnn, device, acc, predictors[t])

            # --- Update rung (§4): refinement placement, cannot mask a grow decision. ---
            if cfg.enable_update and not use_cnn and cfg.feature_dim is not None:
                root_parent_idxs = [i for i, p in enumerate(parents) if p.is_root]
                if root_parent_idxs:
                    # The probe (randperm in update_probe's training loop, deepcopy init, etc.)
                    # consumes RNG state. Forked + deterministically reseeded so it neither leaks
                    # into nor depends on the main run's RNG stream — otherwise every later task's
                    # init/batch order would differ from a control run with enable_update=False,
                    # and borderline decisions on later tasks would flip for an RNG reason rather
                    # than because of the update rung.
                    #
                    # The fork must cover the DEVICE generator, not just the CPU one: this is the
                    # P0b bug (commit 8d65718) at a second call site. `torch.manual_seed` below
                    # re-seeds EVERY device generator (it fans out to `cuda.manual_seed_all` and
                    # `mps.manual_seed`), while `torch.random.fork_rng(devices=[])` names no
                    # device and puts back only the CPU half — so the probe used to leave the
                    # device generator wherever its own training loop ended, and the dropout mask
                    # stream of every later `train_node` in the run moved with it. Invisible on
                    # CPU, which is why the suite said nothing; it makes `--enable_update` fail to
                    # be a clean comparison against its own control on any GPU/MPS run.
                    with fork_rng_all_devices():
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
                        drift = float((_flat_theta(module_copy) - _flat_theta(orig_module)).norm().item())
                        best_p.concept_module.load_state_dict(module_copy.state_dict())
                        best_p.compute_concept_subspace(tasks[best_p.task_id]["train"], device,
                                                        top_k=cfg.subspace_k,
                                                        max_batches=cfg.routing_batches)
                        best_p.freeze()
                        reader_audit.append({
                            "at_task": t, "kind": "update_commit", "node": best_p.task_id,
                            "readers": sorted(affected), "drift": drift, "rel_error": None,
                            "reader_deltas_val": backward_deltas_val,
                            "reader_deltas_test": backward_deltas_test,
                            "tolerance": cfg.update_tolerance, "verdict": "kept",
                        })
                        X2, Xraw2, y2 = _cache_parent_stack(parents, task["train"], device,
                                                            max_batches=gate_batches)
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
        # `param_curve` keeps its published meaning (CONCEPT-MODULE parameters only, as in every
        # earlier run, where a SmallCNN root's backbone was excluded too). `param_curve_total`
        # is the whole node: in token mode that adds each root's AttentionPool, which is the
        # honest parameter cost of the attention family. The two are equal in CLS feature mode.
        param_curve_total.append(sum(p.numel() for n in nodes for p in n.parameters()))
        print(f"  task {t:2d}: decision={decisions[t]['decision']:5s}  acc={acc:.4f}  "
              f"nodes={len(nodes)}  params={param_curve[-1]}")

        if cfg.consolidate_every and (t + 1) % cfg.consolidate_every == 0 and len(nodes) > 2:
            accept = make_accuracy_accept_fn(nodes, predictors, tasks, device, cfg.merge_tolerance)
            summ = consolidate_nodes(nodes, predictors, tasks, device, accept_fn=accept,
                                     similarity_threshold=cfg.similarity_threshold,
                                     reader_tolerance=cfg.merge_tolerance,
                                     subspace_k=cfg.subspace_k, distill=cfg.distill,
                                     distill_epochs=cfg.distill_epochs, merge_tolerance=cfg.merge_tolerance,
                                     functional_threshold=(cfg.functional_threshold
                                                           if cfg.functional_redundancy else None),
                                     merge_trigger=cfg.merge_trigger,
                                     merge_cka_threshold=cfg.merge_cka_threshold)
            print(f"  [consolidate @ task {t}] saved {summ['params_saved']} params, {summ['n_ops']} ops")
            summ["at_task"] = t
            summ["final"] = False
            for e in summ["reader_audit"]:
                e["at_task"] = t
            reader_audit.extend(summ["reader_audit"])
            consolidation_passes.append(summ)
            _resolve_provisional(t, pass_result=summ)
        elif cfg.provisional != "off":
            # Crystallisation is a property of the STREAM's clock — "a provisional root that is
            # still flagged `crystallise_after` tasks later has run out of time" — not of the
            # consolidation schedule. Running it only inside the `consolidate_every` branch meant
            # that at the default `consolidate_every=0` NO root could time out until the final
            # pass, so every run's P6 read "resolved at the last task" and the timeout clock was
            # never actually exercised (PR #8 review, should-fix 7). A merge still resolves a root
            # only in a consolidation pass, which is why no `pass_result` is passed here: with no
            # pass, no node can have vanished, and the merge bookkeeping has nothing to do.
            _resolve_provisional(t)
        _flush(device)

    # Final consolidation.
    accept = make_accuracy_accept_fn(nodes, predictors, tasks, device, cfg.merge_tolerance)
    consolidation = consolidate_nodes(nodes, predictors, tasks, device, accept_fn=accept,
                                      similarity_threshold=cfg.similarity_threshold,
                                      reader_tolerance=cfg.merge_tolerance,
                                      subspace_k=cfg.subspace_k, distill=cfg.distill,
                                      distill_epochs=cfg.distill_epochs, merge_tolerance=cfg.merge_tolerance,
                                      functional_threshold=(cfg.functional_threshold
                                                            if cfg.functional_redundancy else None),
                                      merge_trigger=cfg.merge_trigger,
                                      merge_cka_threshold=cfg.merge_cka_threshold)
    consolidation["at_task"] = len(tasks) - 1
    consolidation["final"] = True
    for e in consolidation["reader_audit"]:
        e["at_task"] = len(tasks) - 1
    reader_audit.extend(consolidation["reader_audit"])
    consolidation_passes.append(consolidation)

    _resolve_provisional(len(tasks) - 1, end_of_stream=True, pass_result=consolidation)

    # D: `average_accuracy`/`test_accs` price the DAG BEFORE its last consolidation pass (each
    # `test_accs[t]` is appended inside the task loop, ahead of that iteration's
    # `consolidate_every` pass) — a merge's or truncation's cost, bounded by `merge_tolerance` /
    # `reader_tolerance` per reader by construction, never reaches them. `test_accs_final` /
    # `average_accuracy_final` re-evaluate every task once more, after the run's last
    # consolidation, so the DAG is priced as it actually ends up.
    test_accs_final = [predictors[tt].accuracy(tasks[tt]["test"], device) for tt in range(len(tasks))]

    n_grow = sum(1 for d in decisions if d["decision"] == "grow")
    n_search = sum(1 for d in decisions if d["decision"] == "search")
    n_reuse = sum(1 for d in decisions if d["decision"] == "reuse")
    n_update = sum(1 for d in decisions if d["decision"] == "update")
    results = {
        "average_accuracy": float(np.mean(test_accs)),
        "test_accs": test_accs,
        "average_accuracy_final": float(np.mean(test_accs_final)),
        "test_accs_final": test_accs_final,
        "reader_audit": reader_audit,
        "n_grow": n_grow,
        "n_search": n_search,
        "n_reuse": n_reuse,
        "n_update": n_update,
        "reuse_rate": (len(tasks) - n_grow) / len(tasks),   # non-grow fraction (reuse+search+update)
        "param_curve": param_curve,
        "param_curve_total": param_curve_total,
        "params_final_pre_consolidation": param_curve[-1] if param_curve else 0,
        "params_total_pre_consolidation": param_curve_total[-1] if param_curve_total else 0,
        "params_per_root": [sum(p.numel() for p in n.parameters()) for n in nodes if n.is_root],
        "consolidation": consolidation,
        "consolidation_passes": consolidation_passes,
        "provisional": cfg.provisional,
        "provisional_roots": [{k: v for k, v in r.items() if k != "_node"}
                              for r in provisional_log.values()],
        "root_family": cfg.root_family,
        "n_tokens": cfg.n_tokens,
        "feature_dim": cfg.feature_dim,
        "gate_cache_max": cfg.gate_cache_max,
        "decisions": decisions,
    }
    out_path = os.path.join(cfg.results_dir, "exp3a_kan_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAA={results['average_accuracy']:.4f}  grow={n_grow}/{len(tasks)}  "
          f"reuse_rate={results['reuse_rate']:.2f}  "
          f"params {results['params_final_pre_consolidation']}→{consolidation['params_after']}")
    print(f"Results saved to {out_path}")

    if dump_gate_tensors:
        # Private, non-JSON-serialized: the exact `y` each gated task's live decision saw, for
        # tests to check `_build_gate_dump`'s "train_y" against without needing to re-derive
        # `_cache_parent_stack`'s output post hoc. Added after the JSON write above (tensors
        # aren't JSON-serializable) and only under this flag, so it never changes on-disk output.
        results["_gate_cache_y"] = {t: yv for t, (_, yv) in gate_cache.items()}

    if dump_gate_tensors and not use_cnn:
        dump = _build_gate_dump(cfg, tasks, nodes, predictors, decisions, gate_cache, roots_at_mint)
        dump_path = os.path.join(cfg.results_dir, "gate_dump.pt")
        # Estimated BEFORE writing: a dump that only reveals its size once it is on disk has
        # already cost the disk. Refusing is not fatal — the run's results are written above.
        est_bytes = _dump_nbytes(dump)
        print(f"[kan_exp] gate_dump.pt estimated tensor payload: {est_bytes / 1e9:.3f} GB "
              f"(limit --dump_max_bytes {cfg.dump_max_bytes / 1e9:.3f} GB)")
        if est_bytes > cfg.dump_max_bytes:
            print(f"[kan_exp] REFUSING to write {dump_path}: estimated {est_bytes / 1e9:.3f} GB "
                  f"exceeds --dump_max_bytes {cfg.dump_max_bytes / 1e9:.3f} GB. Lower "
                  f"--dump_max_per_split (currently {cfg.dump_max_per_split}) or raise the limit. "
                  f"The run itself is complete and its results JSON is already written.")
        else:
            torch.save(dump, dump_path)
            print(f"Gate tensor dump saved to {dump_path}")

    return results
