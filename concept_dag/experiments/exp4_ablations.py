"""
Experiment 4 — Ablation Studies for the Concept DAG.

Five system variants, each toggling one design choice at a time, compared against
the full system (Exp 3a reference):

  Variant        Routing               Aggregation   Parents frozen?
  ─────────────────────────────────────────────────────────────────
  full           principal_angle       soft_pca      yes   ← reference (= Exp 3a)
  no_routing     random                soft_pca      yes   ← ablates Stage 1
  no_crystal     principal_angle       concat        yes   ← ablates crystallization
  no_freeze      principal_angle       soft_pca      no    ← ablates freeze protocol
  sequential     n/a (no parents)      n/a           n/a   ← ablates DAG structure

Causal ablation (exp 4f – "forced_hub"):
  Forces task 0 to be a parent of every subsequent task, regardless of its routing
  score, elevating its out_degree from 4 → n_tasks-1.  After building the forced DAG,
  exp 3b-style perturbation tests are run.  If task 0's drift drops significantly
  compared to the natural run, the structural effect is causal; if not, selection bias
  (some nodes are intrinsically stable AND attract children) is the likely explanation.

Run:
    python run_experiment.py --exp 4         --device cuda --data_root ./data
    python run_experiment.py --exp 4f        --device cuda --data_root ./data
    python run_experiment.py --exp 4 --n_tasks 10 --epochs 15   # fast ablation
"""

import gc
import os
import json
import math
import random
import itertools
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..modules.concept_module import ConceptModule
from ..models.baselines import SmallCNN, LinearHead
from ..utils.metrics import accuracy, evaluate
from ..data.loaders import make_split_cifar100
from .exp3_growing_dag import (
    Exp3Config, DAGNode, eval_node,
    route_for_task, _flush, forward_dag_memoized,
)


# ──────────────────────────────────────────────────────────────────────────────
# Ablation config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Exp4Config:
    """
    Extends Exp3Config with per-variant flags.
    Most numerical hyperparameters are intentionally kept equal across variants
    so comparisons are fair.
    """
    data_root:       str   = "./data"
    results_dir:     str   = "results/exp4"
    # Model
    cnn_out_dim:     int   = 256
    concept_dim:     int   = 128
    n_mlp_layers:    int   = 2
    soft_pca_k:      int   = 8
    # DAG growth
    n_tasks:         int   = 20
    n_parents:       int   = 2
    subspace_k:      int   = 8
    routing_batches: int   = 20
    # Training
    root_epochs:     int   = 25
    child_epochs:    int   = 25
    lr:              float = 1e-3
    batch_size:      int   = 128
    orth_weight:     float = 0.01
    # Perturbation (for forced_hub variant)
    perturb_steps:   int   = 60
    perturb_lr:      float = 5e-4
    # Ablation switches
    routing_mode:    str   = "principal_angle"  # "principal_angle" | "random"
    aggregation:     str   = "soft_pca"         # "soft_pca" | "concat" | "mean" | "cross_attention"
    freeze_parents:  bool  = True               # freeze parent nodes after training?
    is_sequential:   bool  = False              # no parents at all (baseline)
    # Backbone ("smallcnn" | "dinov2_vits14" | "clip_vitb16" | "resnet50")
    backbone:        str   = "smallcnn"
    feature_dim:     int   = None               # auto-set from encoder if backbone != smallcnn
    cache_dir:       str   = None               # where to store cached features
    # Misc
    seed:            int   = 42
    device:          str   = "cpu"
    log_every:       int   = 5
    variant_name:    str   = "full"


def exp4_cfg_from_exp3(cfg3: Exp3Config, **overrides) -> Exp4Config:
    """Convert an Exp3Config into an Exp4Config, then apply overrides."""
    base = Exp4Config(
        data_root      = cfg3.data_root,
        results_dir    = cfg3.results_dir.replace("exp3", "exp4"),
        cnn_out_dim    = cfg3.cnn_out_dim,
        concept_dim    = cfg3.concept_dim,
        n_mlp_layers   = cfg3.n_mlp_layers,
        soft_pca_k     = cfg3.soft_pca_k,
        n_tasks        = cfg3.n_tasks,
        n_parents      = cfg3.n_parents,
        subspace_k     = cfg3.subspace_k,
        routing_batches= cfg3.routing_batches,
        root_epochs    = cfg3.root_epochs,
        child_epochs   = cfg3.child_epochs,
        lr             = cfg3.lr,
        batch_size     = cfg3.batch_size,
        orth_weight    = cfg3.orth_weight,
        seed           = cfg3.seed,
        device         = cfg3.device,
        log_every      = cfg3.log_every,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# ──────────────────────────────────────────────────────────────────────────────
# Ablation-aware DAGNode builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_node(
    task_id:       int,
    cfg:           Exp4Config,
    parent_models: Optional[List[DAGNode]] = None,
) -> DAGNode:
    """
    Build a DAGNode that respects the ablation flags.
    For no_crystal: override aggregation to 'concat'.
    For sequential / no-parent cases: always root.
    Respects cfg.backbone / cfg.feature_dim for SSL encoder mode.
    """
    use_cnn     = (cfg.backbone == "smallcnn")
    feature_dim = cfg.feature_dim  # None for smallcnn (unused), set for SSL encoders

    if cfg.is_sequential or not parent_models:
        # Root node
        node = DAGNode(
            task_id      = task_id,
            concept_dim  = cfg.concept_dim,
            cnn_out_dim  = cfg.cnn_out_dim,
            n_mlp_layers = cfg.n_mlp_layers,
            parent_models= None,
            soft_pca_k   = cfg.soft_pca_k,
            use_cnn      = use_cnn,
            feature_dim  = feature_dim,
        )
    else:
        # Child node: build manually to override aggregation
        node = _ChildNodeWithAggregation(
            task_id      = task_id,
            cfg          = cfg,
            parent_models= parent_models,
        )
    return node


class _ChildNodeWithAggregation(DAGNode):
    """
    DAGNode subclass that replaces the default soft_pca aggregation with
    whatever cfg.aggregation specifies (used for no_crystal ablation).
    """
    def __init__(self, task_id: int, cfg: Exp4Config, parent_models: List[DAGNode]):
        # Call nn.Module.__init__ directly — we'll build our own internals
        nn.Module.__init__(self)
        self.task_id  = task_id
        self._is_root = False
        self.use_cnn  = False    # children never have a CNN
        self.cnn      = None
        self.parent_models: List[DAGNode] = list(parent_models)

        n_par = len(parent_models)
        agg = cfg.aggregation  # "soft_pca" | "concat" | "mean" | "cross_attention"
        agg_kwargs: dict = {}
        if agg == "soft_pca":
            agg_kwargs = {"top_k": min(cfg.soft_pca_k, cfg.concept_dim)}
        elif agg == "cross_attention":
            # 4 heads is a reasonable default; requires concept_dim divisible by n_heads
            agg_kwargs = {"n_heads": 4, "orth_weight": cfg.orth_weight, "entropy_weight": 0.0}

        self.concept_module = ConceptModule(
            module_id   = f"node_{task_id}",
            in_dim      = cfg.concept_dim,
            hidden_dim  = cfg.concept_dim,
            out_dim     = cfg.concept_dim,
            n_layers    = cfg.n_mlp_layers,
            n_parents   = n_par,
            aggregation = agg,
            agg_kwargs  = agg_kwargs,
        )


# ──────────────────────────────────────────────────────────────────────────────
# No-freeze-aware training helpers
# ──────────────────────────────────────────────────────────────────────────────

def _collect_ancestors(node: DAGNode) -> Tuple[List[DAGNode], List[DAGNode]]:
    """
    BFS over the DAG through `parent_models` starting at `node`. Returns
    (non_root_ancestors, root_ancestors).  Both exclude `node` itself.
    """
    non_root: List[DAGNode] = []
    roots:    List[DAGNode] = []
    seen_ids = set()
    frontier: List[DAGNode] = list(node.parent_models)
    while frontier:
        p = frontier.pop(0)
        pid = id(p)
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        if p.is_root:
            roots.append(p)
        else:
            non_root.append(p)
        # Walk up further in both cases (roots have no parents, so this is a no-op)
        frontier.extend(p.parent_models)
    return non_root, roots


def _make_memoized_no_freeze_forward(target: DAGNode):
    """
    Replacement for `target.forward` used in the Exp 4 `no_freeze` ablation.

    Delegates to `forward_dag_memoized`, which evaluates every ancestor
    exactly once per batch via topological order and returns the target's
    output. This eliminates the exponential recomputation of shared
    ancestors that otherwise makes `no_freeze` runtime-pathological in
    diamond-heavy DAGs (see WRITEUP_NOTES.md).

    `cnn_no_grad=True` keeps root-ancestor CNN backbones frozen — concept
    modules still receive gradients from the target's loss, but the CNN
    isn't retrained and its conv activations aren't retained for backprop.
    """
    def _fwd(x: torch.Tensor) -> torch.Tensor:
        return forward_dag_memoized(target, x, cnn_no_grad=True)
    return _fwd


def _train_node_with_freeze_mode(
    node:           DAGNode,
    head:           LinearHead,
    loader,
    n_epochs:       int,
    cfg:            Exp4Config,
    name:           str,
    existing_nodes: List[DAGNode],
) -> Dict:
    """
    Train a node.  If cfg.freeze_parents is False AND node has parents,
    include every reachable ancestor's **concept_module** parameters in the
    optimizer so gradients flow back through the full ancestor chain.

    Crucially, root-ancestor CNN backbones stay frozen even in no_freeze
    mode:
      - The ablation claim is about concept-module updatability, not CNN
        retraining.
      - Retaining conv activations for every CNN in the ancestor chain
        multiplies peak memory by an order of magnitude and will OOM on
        a single GPU past ~task 8. See WRITEUP_NOTES.md for detail.

    Because `DAGNode.forward` internally wraps its parent calls in
    `torch.no_grad()`, we replace the *training node's* forward with a
    memoized evaluator that walks the full ancestor DAG in topological
    order, evaluating every reachable ancestor exactly once per batch.
    Only the training node's forward is patched — the memoized evaluator
    calls each ancestor's local modules directly, so ancestor `forward`
    methods (and their no_grad wrappers) are bypassed entirely. This also
    fixes the exponential recomputation that otherwise occurs in
    diamond-shaped DAGs (see WRITEUP_NOTES.md §3). The patch is restored
    in a `finally` block.
    """
    device = cfg.device

    # Move node's own parameters to device
    if node.is_root:
        node.to(device)
    else:
        node.concept_module.to(device)

    # Collect full ancestor chain split by root/non-root
    non_root_ancestors: List[DAGNode] = []
    root_ancestors:     List[DAGNode] = []
    if not cfg.freeze_parents and not node.is_root:
        non_root_ancestors, root_ancestors = _collect_ancestors(node)

        # Non-root ancestors: fully unfreeze concept_module and move to device
        for a in non_root_ancestors:
            a.unfreeze()
            a.concept_module.to(device)

        # Root ancestors: unfreeze ONLY the concept_module, keep CNN frozen.
        # This is the memory-safety fix.
        for r in root_ancestors:
            # Ensure everything is on device (already should be)
            r.to(device)
            # Freeze the CNN explicitly, unfreeze just the concept_module
            if r.cnn is not None:
                for p in r.cnn.parameters():
                    p.requires_grad_(False)
                r.cnn.eval()
            for p in r.concept_module.parameters():
                p.requires_grad_(True)
            r.concept_module._frozen = False

    head.to(device)

    # Build parameter groups: current node + head + (if no_freeze) every
    # ancestor's concept_module params (NEVER root CNN params).
    trainable_params: List[nn.Parameter] = (
        node.trainable_parameters() + list(head.parameters())
    )
    if not cfg.freeze_parents and not node.is_root:
        seen_param_ids = {id(p) for p in trainable_params}
        for a in non_root_ancestors:
            for ap in a.concept_module.parameters():
                if id(ap) not in seen_param_ids:
                    trainable_params.append(ap)
                    seen_param_ids.add(id(ap))
        for r in root_ancestors:
            for ap in r.concept_module.parameters():
                if id(ap) not in seen_param_ids:
                    trainable_params.append(ap)
                    seen_param_ids.add(id(ap))

    opt   = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    # Monkey-patch forward on the training node only. The memoized evaluator
    # walks the full ancestor DAG internally (topologically by task_id),
    # evaluates each ancestor's local modules exactly once per batch, and
    # caches results. Ancestor `forward` methods — and their inner no_grad
    # wrappers — are bypassed entirely, so gradients flow back through every
    # ancestor concept_module that was added to the optimizer. Root CNN
    # backbones still run under no_grad (cnn_no_grad=True inside the evaluator).
    patches: List[Tuple[DAGNode, object]] = []  # (node, original_forward)
    if not cfg.freeze_parents and not node.is_root:
        patches.append((node, node.forward))
        node.forward = _make_memoized_no_freeze_forward(node)  # type: ignore[assignment]

    history = {"loss": [], "accuracy": []}
    try:
        for epoch in range(1, n_epochs + 1):
            node.concept_module.train()
            if node.is_root and node.cnn is not None:
                node.cnn.train()
            if not cfg.freeze_parents and not node.is_root:
                for a in non_root_ancestors:
                    a.concept_module.train()
                for r in root_ancestors:
                    r.concept_module.train()
                    # CNN stays in eval mode — it's frozen and we don't want
                    # batchnorm / dropout stats updating from other tasks' data.
                    if r.cnn is not None:
                        r.cnn.eval()
            head.train()

            ep_loss, ep_acc, n = 0.0, 0.0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                emb    = node(x)
                logits = head(emb)
                loss   = nn.functional.cross_entropy(logits, y)
                loss   = loss + cfg.orth_weight * node.concept_module.orth_loss()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                ep_loss += loss.item()
                ep_acc  += accuracy(logits, y)
                n       += 1

            sched.step()
            history["loss"].append(ep_loss / max(n, 1))
            history["accuracy"].append(ep_acc / max(n, 1))
            if epoch % cfg.log_every == 0 or epoch == 1:
                print(f"    [{name} | ep {epoch:3d}/{n_epochs}] "
                      f"loss={history['loss'][-1]:.4f}  acc={history['accuracy'][-1]:.3f}")
    finally:
        # Always restore every patched forward, even if training threw.
        for patched_node, orig_fwd in patches:
            patched_node.forward = orig_fwd  # type: ignore[assignment]

    node.concept_module.eval()
    if node.is_root and node.cnn is not None:
        node.cnn.eval()
    head.eval()
    node.concept_module.clear_activation_buffer()

    # In no_freeze we intentionally leave ancestors unfrozen; the ablation's
    # whole point is that later tasks can still modify earlier representations.
    del opt, sched
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Routing helpers (ablation-aware)
# ──────────────────────────────────────────────────────────────────────────────

def _route(
    existing_nodes: List[DAGNode],
    task_loader,
    cfg:            Exp4Config,
) -> Tuple[List[int], Dict[int, float]]:
    """
    Select parent indices according to cfg.routing_mode.

    principal_angle : standard principal-angle similarity (Stage 1 routing)
    random          : uniformly sample n_parents parents at random
    """
    n_available = len(existing_nodes)
    actual_k    = min(cfg.n_parents, n_available)

    if cfg.routing_mode == "random":
        selected = random.sample(range(n_available), k=actual_k)
        scores   = {i: 0.0 for i in range(n_available)}
        for i in selected:
            scores[i] = 1.0
        return selected, scores

    # Default: principal angle similarity
    return route_for_task(
        existing_nodes   = existing_nodes,
        new_task_loader  = task_loader,
        n_parents        = cfg.n_parents,
        subspace_k       = cfg.subspace_k,
        device           = cfg.device,
        routing_batches  = cfg.routing_batches,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Core ablation runner (single variant)
# ──────────────────────────────────────────────────────────────────────────────

def run_ablation_variant(
    cfg:   Exp4Config,
    tasks: List[Dict],
) -> Dict:
    """
    Grow the DAG for one ablation variant and return per-task metrics.
    `tasks` is the list of task dicts from make_split_cifar100.
    """
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device

    nodes:      List[DAGNode]    = []
    heads:      List[LinearHead] = []
    parent_map: Dict[int, List[int]] = {}
    test_accs:  List[float]      = []

    print(f"\n{'─'*70}")
    print(f"  Variant: {cfg.variant_name.upper()}")
    print(f"  routing={cfg.routing_mode}  agg={cfg.aggregation}  "
          f"freeze={cfg.freeze_parents}  sequential={cfg.is_sequential}")
    print(f"{'─'*70}")

    for t, task in enumerate(tasks):
        # ---- routing ----
        n_par = 0 if cfg.is_sequential else min(cfg.n_parents, t)
        if n_par == 0:
            selected_parents_idx, pa_scores = [], {}
        else:
            selected_parents_idx, pa_scores = _route(nodes, task["train"], cfg)

        parent_map[t] = selected_parents_idx
        selected_parents = [nodes[i] for i in selected_parents_idx]

        label = ("root" if n_par == 0
                 else f"parents={selected_parents_idx}")
        print(f"\n── Task {t:2d} | {label}")

        # ---- build node ----
        node = _build_node(t, cfg, selected_parents or None)
        head = LinearHead(cfg.concept_dim, task["n_classes"])

        # ---- train ----
        n_epochs = cfg.root_epochs if n_par == 0 else cfg.child_epochs
        _train_node_with_freeze_mode(
            node, head, task["train"], n_epochs, cfg,
            name=f"t{t}", existing_nodes=nodes,
        )
        test_acc = eval_node(node, head, task["test"], device)
        print(f"  Test acc: {test_acc:.4f}")
        test_accs.append(test_acc)

        # ---- compute subspace ----
        node.compute_concept_subspace(
            task["train"], device,
            top_k=cfg.subspace_k, max_batches=cfg.routing_batches,
        )

        # ---- freeze if requested ----
        if cfg.freeze_parents:
            node.freeze()
        # In no_freeze variant, nodes stay unfrozen — later tasks may update them

        nodes.append(node)
        heads.append(head)
        _flush(device)

    # ---- average accuracy ----
    aa = float(np.mean(test_accs))

    # ---- backward transfer (re-evaluate all previous tasks) ----
    print(f"\n  Re-evaluating all tasks for backward transfer...")
    final_accs = []
    for t, (node, head, task) in enumerate(zip(nodes, heads, tasks)):
        acc = eval_node(node, head, task["test"], device)
        final_accs.append(acc)
    bt = float(np.mean([final_accs[t] - test_accs[t] for t in range(len(tasks))]))

    # ---- out-degree ----
    out_degree = {t: 0 for t in range(len(tasks))}
    for t, parents in parent_map.items():
        for p in parents:
            out_degree[p] += 1

    print(f"\n  Average Accuracy     : {aa:.4f}")
    print(f"  Backward Transfer    : {bt:+.4f}  (negative = forgetting)")
    print(f"  Final-state accs mean: {np.mean(final_accs):.4f}")

    return {
        "variant":           cfg.variant_name,
        "routing_mode":      cfg.routing_mode,
        "aggregation":       cfg.aggregation,
        "freeze_parents":    cfg.freeze_parents,
        "is_sequential":     cfg.is_sequential,
        "test_accs":         test_accs,
        "final_accs":        final_accs,
        "average_accuracy":  aa,
        "backward_transfer": bt,
        "parent_map":        parent_map,
        "out_degree":        out_degree,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Causal ablation — forced hub
# ──────────────────────────────────────────────────────────────────────────────

def _route_forced_hub(
    existing_nodes: List[DAGNode],
    task_loader,
    cfg:            Exp4Config,
    forced_hub_idx: int = 0,
) -> Tuple[List[int], Dict[int, float]]:
    """
    Always include `forced_hub_idx` as one of the selected parents.
    Fill remaining parent slots from principal-angle routing on non-forced nodes.
    """
    n_available = len(existing_nodes)
    actual_k    = min(cfg.n_parents, n_available)

    # Score all nodes normally
    _, scores = route_for_task(
        existing_nodes   = existing_nodes,
        new_task_loader  = task_loader,
        n_parents        = actual_k,
        subspace_k       = cfg.subspace_k,
        device           = cfg.device,
        routing_batches  = cfg.routing_batches,
    )

    # Build selected list: forced hub first, then highest-scoring non-forced
    remaining_slots = actual_k - 1
    non_forced = sorted(
        [i for i in scores if i != forced_hub_idx],
        key=lambda i: scores[i],
        reverse=True,
    )
    selected = [forced_hub_idx] + non_forced[:remaining_slots]
    return selected, scores


def run_forced_hub_causal(
    cfg:   Exp4Config,
    tasks: List[Dict],
) -> Dict:
    """
    Causal ablation: force task 0 to be a parent for every subsequent task,
    regardless of routing score.  After building the DAG, run Exp 3b-style
    perturbation on ALL nodes and report drift vs. out_degree.

    Compare:
      - Forced hub (task 0, out_degree ~ n_tasks-1) drift in this run
      - vs. Natural hub (task 2, high out_degree) drift from Exp 3b results
      - vs. Task 0 drift from Exp 3b results (when it had out_degree 4)

    If forced_hub drift < natural_task0 drift → structural effect is causal.
    If forced_hub drift ≈ natural_task0 drift → selection bias explanation.
    """
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device
    forced_hub_idx = 0   # we force task 0 to be the hub

    print(f"\n{'='*70}")
    print("Experiment 4f (Causal Ablation): Forced Hub")
    print(f"  Task {forced_hub_idx} will be forced as parent of every subsequent task.")
    print(f"{'='*70}")

    nodes:      List[DAGNode]    = []
    heads:      List[LinearHead] = []
    parent_map: Dict[int, List[int]] = {}
    test_accs:  List[float]      = []

    for t, task in enumerate(tasks):
        n_par = min(cfg.n_parents, t)
        if n_par == 0 or cfg.is_sequential:
            selected_parents_idx = []
        elif t == 1 and n_par == 1:
            # Only one existing node (task 0) — forced hub trivially selected
            selected_parents_idx = [0]
        else:
            selected_parents_idx, _ = _route_forced_hub(
                existing_nodes  = nodes,
                task_loader     = task["train"],
                cfg             = cfg,
                forced_hub_idx  = forced_hub_idx,
            )

        parent_map[t] = selected_parents_idx
        selected_parents = [nodes[i] for i in selected_parents_idx]

        label = "root" if not selected_parents_idx else f"parents={selected_parents_idx}"
        print(f"\n── Task {t:2d} | {label}")

        node = _build_node(t, cfg, selected_parents or None)
        head = LinearHead(cfg.concept_dim, task["n_classes"])

        n_epochs = cfg.root_epochs if not selected_parents_idx else cfg.child_epochs
        _train_node_with_freeze_mode(
            node, head, task["train"], n_epochs, cfg,
            name=f"t{t}", existing_nodes=nodes,
        )
        test_acc = eval_node(node, head, task["test"], device)
        print(f"  Test acc: {test_acc:.4f}")
        test_accs.append(test_acc)

        node.compute_concept_subspace(
            task["train"], device,
            top_k=cfg.subspace_k, max_batches=cfg.routing_batches,
        )
        node.freeze()
        nodes.append(node)
        heads.append(head)
        _flush(device)

    # ── Out-degree
    out_degree = {t: 0 for t in range(len(tasks))}
    for t, parents in parent_map.items():
        for p in parents:
            out_degree[p] += 1

    aa = float(np.mean(test_accs))
    print(f"\n  Average Accuracy: {aa:.4f}")
    print(f"  Out-degrees: {dict(out_degree)}")
    print(f"  Forced hub (task {forced_hub_idx}) out_degree = {out_degree[forced_hub_idx]}")

    # ── Perturbation test (same as Exp 3b)
    print(f"\n── Perturbation test (Exp 3b protocol)...")
    n = len(nodes)
    results_pert = []
    for t in range(n):
        node = nodes[t]
        head = heads[t]
        d    = out_degree[t]
        base_acc = eval_node(node, head, tasks[t]["test"], device)

        snapshot = {name: p.detach().cpu().clone()
                    for name, p in node.named_parameters()}
        node.unfreeze()
        node.concept_module.train()
        if node.is_root and node.cnn is not None:
            node.cnn.train()

        wrong_task_idx  = (t + 1) % n
        wrong_loader    = tasks[wrong_task_idx]["train"]
        wrong_head = LinearHead(cfg.concept_dim, tasks[wrong_task_idx]["n_classes"]).to(device)
        opt = torch.optim.SGD(
            list(node.trainable_parameters()) + list(wrong_head.parameters()),
            lr=cfg.perturb_lr, momentum=0.9,
        )
        step, data_iter = 0, iter(wrong_loader)
        while step < cfg.perturb_steps:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(wrong_loader)
                x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            logits = wrong_head(node(x))
            loss   = nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(node.trainable_parameters(), 1.0)
            opt.step()
            step += 1
        del opt, wrong_head
        _flush(device)

        named = dict(node.named_parameters())
        drift = math.sqrt(sum(
            (named[nm].detach().cpu() - old_p).pow(2).sum().item()
            for nm, old_p in snapshot.items()
        ))

        node.eval()
        node.concept_module.eval()
        post_acc = eval_node(node, head, tasks[t]["test"], device)
        acc_drop = base_acc - post_acc

        with torch.no_grad():
            for nm, live_p in named.items():
                live_p.copy_(snapshot[nm].to(device))
        node.freeze()

        results_pert.append({
            "task": t, "out_degree": d,
            "base_acc": base_acc, "post_acc": post_acc,
            "acc_drop": acc_drop, "weight_drift": drift,
        })
        print(f"  Task {t:2d} | deg={d} | drift={drift:.4f} | "
              f"acc {base_acc:.4f}→{post_acc:.4f} (Δ={-acc_drop:+.4f})")
        _flush(device)

    degrees = [r["out_degree"]   for r in results_pert]
    drifts  = [r["weight_drift"] for r in results_pert]
    corr    = float(np.corrcoef(degrees, drifts)[0, 1]) if len(set(degrees)) > 1 else float("nan")

    # Spotlight: forced hub vs. natural results
    hub_result = next(r for r in results_pert if r["task"] == forced_hub_idx)
    print(f"\n  Forced hub (task {forced_hub_idx}, deg={hub_result['out_degree']}): "
          f"drift={hub_result['weight_drift']:.4f}")
    print(f"  Pearson r(out_degree, drift) = {corr:+.4f}  "
          f"(natural run was −0.586; same sign = structural effect holds)")

    output = {
        "variant":           "forced_hub",
        "forced_hub_task":   forced_hub_idx,
        "test_accs":         test_accs,
        "average_accuracy":  aa,
        "parent_map":        parent_map,
        "out_degree":        out_degree,
        "perturbation":      results_pert,
        "corr_out_degree_drift": corr,
    }
    return output


# ──────────────────────────────────────────────────────────────────────────────
# Master runner — all five variants
# ──────────────────────────────────────────────────────────────────────────────

VARIANTS = [
    # (variant_name,       routing_mode,      aggregation,       freeze_parents, is_sequential)
    ("full",                "principal_angle", "soft_pca",        True,           False),
    ("no_routing",          "random",          "soft_pca",        True,           False),
    ("no_crystal",          "principal_angle", "concat",          True,           False),
    ("no_freeze",           "principal_angle", "soft_pca",        False,          False),
    ("sequential",          "principal_angle", "soft_pca",        True,           True),
    # Post-Exp-5 extension: cross-attention aggregator replaces SoftPCA.
    # Tests whether input-conditioned routing over parents beats the
    # subspace-bottlenecked linear aggregation.
    ("cross_attention",     "principal_angle", "cross_attention", True,           False),
]


def run_all_ablations(cfg: Exp4Config, tasks=None) -> Dict:
    """
    Load the data once, then run all five variants sequentially.
    Saves per-variant JSON and a combined summary JSON.

    Args:
        tasks: Pre-loaded task list (e.g. with cached SSL features). If None,
               loads raw Split-CIFAR-100 images as before.
    """
    print("\n" + "=" * 70)
    print("Experiment 4: Ablation Studies")
    if cfg.backbone != "smallcnn":
        print(f"  Backbone: {cfg.backbone}  feature_dim={cfg.feature_dim}")
    print("=" * 70)
    print(f"  {len(VARIANTS)} variants × {cfg.n_tasks} tasks × up to {cfg.child_epochs} epochs")

    os.makedirs(cfg.results_dir, exist_ok=True)

    if tasks is None:
        print("\n[Step 0] Loading Split-CIFAR-100...")
        tasks = make_split_cifar100(
            data_root  = cfg.data_root,
            n_tasks    = cfg.n_tasks,
            batch_size = cfg.batch_size,
            seed       = cfg.seed,
        )
    else:
        print(f"\n[Step 0] Using pre-loaded tasks ({len(tasks)} tasks).")
    print(f"  {cfg.n_tasks} tasks, 5 classes each.")

    all_results = {}
    for (vname, routing, agg, freeze, seq) in VARIANTS:
        vcfg = Exp4Config(
            **{k: v for k, v in vars(cfg).items()
               if k not in ("routing_mode","aggregation","freeze_parents",
                            "is_sequential","variant_name")},
            routing_mode   = routing,
            aggregation    = agg,
            freeze_parents = freeze,
            is_sequential  = seq,
            variant_name   = vname,
        )
        result = run_ablation_variant(vcfg, tasks)
        all_results[vname] = result
        _save_variant(result, cfg.results_dir, vname)

    # ── Summary table
    print("\n" + "=" * 70)
    print("SUMMARY — Experiment 4: Ablation")
    print("=" * 70)
    print(f"{'Variant':<15}  {'AA':>8}  {'BT':>8}  {'Final-AA':>10}")
    print("-" * 45)
    for vname, res in all_results.items():
        aa     = res["average_accuracy"]
        bt     = res["backward_transfer"]
        fin_aa = float(np.mean(res["final_accs"]))
        print(f"  {vname:<13}  {aa:8.4f}  {bt:+8.4f}  {fin_aa:10.4f}")

    combined_path = os.path.join(cfg.results_dir, "exp4_all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {combined_path}")

    return all_results


def _save_variant(result: Dict, results_dir: str, vname: str):
    path = os.path.join(results_dir, f"exp4_{vname}_results.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → Saved {path}")
