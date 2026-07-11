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
    TaskSpec, classification_task, decide_reuse_vs_grow, ReuseComposer, _fit_full,
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
        assert kind in ("grow", "reuse")
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
                        max_batches: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    stacks, ys = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device)
        outs = [forward_dag_memoized(p, x) for p in parents]      # each (B, D)
        stacks.append(torch.stack(outs, dim=1).cpu())             # (B, P, D)
        ys.append(y.cpu())
    return torch.cat(stacks, 0), torch.cat(ys, 0)


# ---------------------------------------------------------------------------
# Consolidation over the DAGNode list (reduction / "sleep" pass)
# ---------------------------------------------------------------------------


def _node_children(nodes: List[DAGNode], node: DAGNode) -> List[DAGNode]:
    return [c for c in nodes if node in c.parent_models]


def distill_merge(keep: DAGNode, drop: DAGNode, loader, device: str,
                  epochs: int = 30, lr: float = 1e-3) -> Dict[str, nn.Module]:
    """
    Functional merge for DAGNodes with *different but overlapping* subspaces.

    Trains ``keep`` to be a **sufficient statistic of the pair {keep, drop}**: a vector from which
    both concepts' outputs are linearly recoverable. Returns per-concept linear *recovery adapters*
    ``{"keep": W_keep, "drop": W_drop}`` — the transport maps a re-pointed child inserts before its
    aggregator so it still sees (approximately) the activations it was trained on. When the two
    subspaces are near-identical, W_keep ≈ W_drop ≈ I and this degrades to a structural merge; when
    their joint rank exceeds concept_dim the reconstruction error stays high and the caller's gate
    rejects the merge (they are not actually redundant). See module/keep docs.
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
                sim = _subspace_similarity(a, b, subspace_k)
                if sim < similarity_threshold:
                    continue
                affected = affected_task_ids(a) | affected_task_ids(b)

                if not distill:
                    # Structural merge (safe only for near-identical subspaces); gate decides.
                    if not accept_fn(a, b, affected):
                        continue
                    _merge_nodes(nodes, predictors, keep=a, drop=b)
                    ops.append({"op": "merge", "keep": a.task_id, "drop": b.task_id,
                                "similarity": sim, "distilled": False})
                    merged = True
                    break

                # Distilled merge: snapshot → distill keep → tentatively apply adapted merge →
                # forgetting check → commit or roll back (topology + keep weights).
                base = {t: predictors[t].accuracy(tasks[t]["test"], device)
                        for t in affected if t < len(predictors)}
                snap = _snapshot_topology(nodes, predictors, keep=a)
                loader = _combined_loader(tasks, a.task_id, b.task_id)
                W = distill_merge(a, b, loader, device, epochs=distill_epochs)
                _merge_nodes(nodes, predictors, keep=a, drop=b, W_keep=W["keep"], W_drop=W["drop"])
                ok = all(predictors[t].accuracy(tasks[t]["test"], device) >= base.get(t, 0.0) - merge_tolerance
                         for t in affected if t < len(predictors))
                if ok:
                    ops.append({"op": "merge", "keep": a.task_id, "drop": b.task_id,
                                "similarity": sim, "distilled": True, "recon_loss": W["recon_loss"]})
                    merged = True
                    break
                else:
                    _restore_topology(nodes, predictors, snap)
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
            # --- Kan gate on cached parent embeddings ---
            X, y = _cache_parent_stack(parents, task["train"], device, max_batches=cfg.routing_batches)
            rec = decide_reuse_vs_grow(
                X, y, new_module_factory(parents), spec,
                concept_dim=cfg.concept_dim, n_parents=len(parents), device=device,
                n_epochs=cfg.gate_epochs, lr=cfg.gate_lr, eps_rel=cfg.eps_rel,
            )
            decisions.append({"task": t, "decision": rec.decision, "parents": sel_idx,
                              **{k: getattr(rec, k) for k in ("rel_improvement", "L_reuse_bits", "L_grow_bits")}})

            if rec.decision == "grow":
                node = DAGNode(task_id=t, concept_dim=cfg.concept_dim, cnn_out_dim=cfg.cnn_out_dim,
                               n_mlp_layers=cfg.n_mlp_layers, parent_models=parents,
                               soft_pca_k=cfg.soft_pca_k, use_cnn=use_cnn, feature_dim=cfg.feature_dim)
                train_node(node, head, task["train"], cfg.child_epochs, cfg.lr, device, cfg.log_every,
                           name=f"t{t}-grow", orth_weight=cfg.orth_weight)
                node.compute_concept_subspace(task["train"], device, top_k=cfg.subspace_k,
                                              max_batches=cfg.routing_batches)
                node.freeze()
                nodes.append(node)
                predictors.append(TaskPredictor("grow", head, node=node))
            else:
                # Reuse: train a composer over frozen parents; add NO node.
                composer = ReuseComposer(parent_dim=cfg.concept_dim, n_parents=len(parents), head=head)
                _fit_full(composer, lambda m, xb: m(xb), spec, X, y, cfg.child_epochs, cfg.lr, device)
                predictors.append(TaskPredictor("reuse", head, parents=parents, composer=composer))

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
                                     distill_epochs=cfg.distill_epochs, merge_tolerance=cfg.merge_tolerance)
            print(f"  [consolidate @ task {t}] saved {summ['params_saved']} params, {summ['n_ops']} ops")
        _flush(device)

    # Final consolidation.
    accept = make_accuracy_accept_fn(nodes, predictors, tasks, device, cfg.merge_tolerance)
    consolidation = consolidate_nodes(nodes, predictors, tasks, device, accept_fn=accept,
                                      similarity_threshold=cfg.similarity_threshold,
                                      subspace_k=cfg.subspace_k, distill=cfg.distill,
                                      distill_epochs=cfg.distill_epochs, merge_tolerance=cfg.merge_tolerance)

    n_grow = sum(1 for d in decisions if d["decision"] == "grow")
    results = {
        "average_accuracy": float(np.mean(test_accs)),
        "test_accs": test_accs,
        "n_grow": n_grow,
        "n_reuse": len(tasks) - n_grow,
        "reuse_rate": (len(tasks) - n_grow) / len(tasks),
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
    return results
