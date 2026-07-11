"""
Experiment 3a — Growing Concept DAG on Split-CIFAR-100 (20 tasks × 5 classes).
Experiment 3b — Out-degree vs forgetting resistance.

3a:
  - Task  0 : root node — SmallCNN backbone + ConceptModule, trained from scratch.
  - Tasks 1–19 : child nodes — principal-angle routing over all existing nodes selects
                 the top-k parents; child trains with SoftPCA aggregation, parents frozen.
  - After DAG is built: evaluate test accuracy per task, average accuracy (AA),
    log parent assignments and per-node concept subspace orthogonality.

3b (built on top of the 3a DAG):
  - For every node t, record its out-degree d(t) = number of direct children.
  - Snapshot weights, temporarily unfreeze node t, fine-tune on a "wrong" task
    for N gradient steps, measure weight drift ‖Δθ‖₂ and accuracy drop on task t.
  - Restore snapshot. Repeat for all nodes.
  - Hypothesis: higher out-degree → concept subspace used by more children →
    stronger implicit constraint → less drift.

Run:
    python run_experiment.py --exp 3a --device mps --data_root ./data
    python run_experiment.py --exp 3b --device mps --data_root ./data
"""

import gc
import os
import json
import math
import itertools
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..modules.concept_module import ConceptModule
from ..models.baselines import SmallCNN, LinearHead
from ..utils.metrics import accuracy, evaluate, safe_cross_entropy
from ..data.loaders import make_split_cifar100


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Exp3Config:
    data_root:       str   = "./data"
    results_dir:     str   = "results/exp3"
    # Model
    cnn_out_dim:     int   = 256
    concept_dim:     int   = 128
    n_mlp_layers:   int   = 2
    soft_pca_k:      int   = 8
    # Backbone ("smallcnn" | "dinov2_vits14" | "clip_vitb16" | "resnet50")
    backbone:        str   = "smallcnn"
    feature_dim:     Optional[int] = None    # auto-set from encoder if backbone != smallcnn
    cache_dir:       Optional[str] = None    # where to store cached features
    # DAG growth
    n_tasks:         int   = 20
    n_parents:       int   = 2
    subspace_k:      int   = 8
    routing_batches: int   = 20   # batches used for principal-angle scoring
    # Training
    root_epochs:     int   = 25
    child_epochs:    int   = 25
    lr:              float = 1e-3
    batch_size:      int   = 128
    orth_weight:     float = 0.01
    # Exp 3b perturbation
    perturb_steps:   int   = 60
    perturb_lr:      float = 5e-4
    # Misc
    seed:            int   = 42
    device:          str   = "cpu"
    log_every:       int   = 5


# ---------------------------------------------------------------------------
# DAGNode — unified root / child
# ---------------------------------------------------------------------------


class DAGNode(nn.Module):
    """
    Single node in the growing Concept DAG.

    Root nodes (no parents) own a SmallCNN backbone that processes raw images,
    OR — when use_cnn=False — receive pre-extracted feature vectors directly
    from a frozen external encoder (e.g. DINOv2) via feature caching.

    Child nodes receive parent concept embeddings, aggregate them, and map
    to a concept embedding through a small MLP.

    Parent models are stored as a plain Python list — NOT an nn.ModuleList —
    so that MPS/CUDA can free child memory without holding references to parents.

    Args:
        use_cnn:      If True (default), root nodes build a SmallCNN.
                      If False, root nodes expect pre-extracted features of
                      size `feature_dim` — use with feature_cache.py.
        feature_dim:  Input feature dimension when use_cnn=False.
                      Ignored when use_cnn=True (cnn_out_dim is used instead).
    """

    def __init__(
        self,
        task_id:       int,
        concept_dim:   int,
        cnn_out_dim:   int   = 256,
        n_mlp_layers:  int   = 2,
        parent_models: Optional[List["DAGNode"]] = None,
        soft_pca_k:    int   = 8,
        use_cnn:       bool  = True,
        feature_dim:   Optional[int] = None,
    ):
        super().__init__()
        self.task_id     = task_id
        self._is_root    = not parent_models
        self.use_cnn     = use_cnn

        # Resolve input dimension for this node
        if self._is_root:
            in_dim = cnn_out_dim if use_cnn else (feature_dim or cnn_out_dim)
        else:
            in_dim = concept_dim

        if self._is_root:
            self.cnn = SmallCNN(in_channels=3, out_dim=cnn_out_dim) if use_cnn else None
            self.concept_module = ConceptModule(
                module_id  = f"node_{task_id}",
                in_dim     = in_dim,
                hidden_dim = concept_dim,
                out_dim    = concept_dim,
                n_layers   = n_mlp_layers,
                n_parents  = 0,
            )
            self.parent_models: List["DAGNode"] = []
        else:
            n_par = len(parent_models)
            self.cnn = None
            self.parent_models = list(parent_models)   # plain list
            self.concept_module = ConceptModule(
                module_id   = f"node_{task_id}",
                in_dim      = in_dim,
                hidden_dim  = concept_dim,
                out_dim     = concept_dim,
                n_layers    = n_mlp_layers,
                n_parents   = n_par,
                aggregation = "soft_pca",
                agg_kwargs  = {"top_k": min(soft_pca_k, concept_dim)},
            )

    @property
    def is_root(self) -> bool:
        return self._is_root

    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_root:
            feats = self.cnn(x) if self.use_cnn else x
            return self.concept_module(feats)
        # Child: collect frozen parent embeddings, then aggregate
        with torch.no_grad():
            parent_outs = [p(x) for p in self.parent_models]
        parent_outs = self._apply_parent_adapters(parent_outs)
        return self.concept_module(x, parent_outputs=parent_outs)

    def _apply_parent_adapters(self, parent_outs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply per-edge recovery adapters (installed by a consolidation merge). Each adapter is the
        left-Kan transport map that lets this child read a merged parent as if it were the concept it
        was originally trained against. None (the default) = identity edge.
        """
        adapters = getattr(self, "parent_adapters", None)
        if not adapters:
            return parent_outs
        return [po if ad is None else ad(po) for po, ad in zip(parent_outs, adapters)]

    # -----------------------------------------------------------------------

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Only own parameters — never parents'."""
        if self.is_root:
            return list(self.parameters())        # cnn (if present) + concept_module
        return list(self.concept_module.parameters())

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.concept_module._frozen = True

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)
        self.concept_module._frozen = False

    # -----------------------------------------------------------------------
    # Single-node "local" forward steps — consumed by the memoized evaluator
    # below. `DAGNode.forward` does recursive expansion, which re-evaluates
    # shared ancestors once per incoming edge (exponential in diamond DAGs).
    # These helpers do exactly one node's work, assuming the caller has
    # already evaluated its parents and passes their outputs in explicitly.
    # -----------------------------------------------------------------------

    def _local_forward_root(
        self, x: torch.Tensor, cnn_no_grad: bool = False
    ) -> torch.Tensor:
        """
        Root node's one-shot forward.
        - SmallCNN mode: cnn(x) → concept_module. cnn_no_grad prevents
          conv activations from being retained for backprop (Exp 4 no_freeze).
        - Feature mode (use_cnn=False): x is already (B, feature_dim);
          pass directly to concept_module. cnn_no_grad is ignored.
        """
        if not self.use_cnn:
            return self.concept_module(x)
        if cnn_no_grad:
            with torch.no_grad():
                feats = self.cnn(x)
        else:
            feats = self.cnn(x)
        return self.concept_module(feats)

    def _local_forward_child(
        self, x: torch.Tensor, parent_outs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Child node's one-shot forward. Assumes `parent_outs[i]` is already
        the output of `self.parent_models[i]` for this batch.
        x is passed through for cross-attention query (ignored by linear aggs).
        """
        parent_outs = self._apply_parent_adapters(parent_outs)
        return self.concept_module(x, parent_outputs=parent_outs)

    def compute_concept_subspace(
        self, loader, device: str, top_k: int = 8, max_batches: int = 0
    ) -> torch.Tensor:
        """Collect activations and compute concept subspace via SVD."""
        self.eval()
        cm = self.concept_module
        cm.clear_activation_buffer()
        cm._collecting_subspace = True
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                if max_batches and i >= max_batches:
                    break
                self(x.to(device))
        cm._collecting_subspace = False
        return cm.compute_concept_subspace(top_k=top_k)


# ---------------------------------------------------------------------------
# Memoized DAG forward
# ---------------------------------------------------------------------------


def forward_dag_memoized(
    target:      DAGNode,
    x:           torch.Tensor,
    cnn_no_grad: bool = False,
) -> torch.Tensor:
    """
    Evaluate the DAG rooted at `target` with each node computed **exactly
    once** per batch. Drop-in replacement for `target.forward(x)` that
    eliminates the exponential recomputation of shared ancestors in
    diamond-heavy DAGs.

    Why this matters:
      The naive `DAGNode.forward` expands the ancestor subgraph recursively
      on every call — in a DAG where task 19's ancestors share task 0/1/2
      along many paths, task 0's CNN can be called ~9× per batch and task 2's
      concept_module ~4×. Under `torch.no_grad()` that's wasted compute; in
      the Exp 4 `no_freeze` variant it's catastrophic because every redundant
      call builds a separate autograd subgraph.

    Algorithm:
      1. BFS over `parent_models` to collect every ancestor reachable from
         `target` (plus `target` itself).
      2. Sort by `task_id`. The DAG growth protocol guarantees that a child's
         parents all have strictly smaller task_ids, so `task_id` order is a
         valid topological order.
      3. Walk the sorted list; for each node, look up its parents' cached
         outputs and invoke its local forward step. Cache the result keyed
         by `id(node)`.
      4. Return `cache[id(target)]`.

    Result: one `concept_module` call per ancestor per batch, one `cnn` call
    per root ancestor per batch. Linear in |DAG| instead of exponential.

    `cnn_no_grad=True` keeps root-ancestor CNN backbones out of the gradient
    graph — needed for the `no_freeze` ablation so conv activations aren't
    retained (see WRITEUP_NOTES.md for rationale).
    """
    # 1. Collect all reachable nodes via BFS
    all_nodes: List[DAGNode] = []
    seen_ids: set = set()
    frontier: List[DAGNode] = [target]
    while frontier:
        n = frontier.pop(0)
        nid = id(n)
        if nid in seen_ids:
            continue
        seen_ids.add(nid)
        all_nodes.append(n)
        frontier.extend(n.parent_models)

    # 2. Topological order = ascending task_id (guaranteed by the growth protocol:
    #    every parent task was created before its children).
    all_nodes.sort(key=lambda n: n.task_id)

    # 3. Evaluate each node exactly once, caching its output tensor by id(node).
    cache: Dict[int, torch.Tensor] = {}
    for n in all_nodes:
        if n.is_root:
            cache[id(n)] = n._local_forward_root(x, cnn_no_grad=cnn_no_grad)
        else:
            parent_outs = [cache[id(p)] for p in n.parent_models]
            cache[id(n)] = n._local_forward_child(x, parent_outs=parent_outs)

    return cache[id(target)]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _flush(device: str):
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def train_node(
    node:      DAGNode,
    head:      LinearHead,
    loader,
    n_epochs:  int,
    lr:        float,
    device:    str,
    log_every: int,
    name:      str,
    orth_weight: float = 0.01,
) -> Dict:
    """Train a DAGNode (root or child) + its classification head."""
    # Move only this node's own parameters to device.
    # For roots: cnn + concept_module. For children: concept_module only.
    # Parents are a plain list and are already on device from their own training.
    if node.is_root:
        node.to(device)
    else:
        node.concept_module.to(device)
    head.to(device)

    params = node.trainable_parameters() + list(head.parameters())
    opt    = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    history = {"loss": [], "accuracy": []}

    for epoch in range(1, n_epochs + 1):
        node.concept_module.train()
        if node.is_root and node.cnn is not None:
            node.cnn.train()
        head.train()

        ep_loss, ep_acc, n = 0.0, 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            emb    = node(x)
            logits = head(emb)
            loss   = safe_cross_entropy(logits, y)
            loss   = loss + orth_weight * node.concept_module.orth_loss()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += loss.item()
            ep_acc  += accuracy(logits, y)
            n       += 1

        sched.step()
        history["loss"].append(ep_loss / max(n, 1))
        history["accuracy"].append(ep_acc  / max(n, 1))
        if epoch % log_every == 0 or epoch == 1:
            print(
                f"    [{name} | ep {epoch:3d}/{n_epochs}] "
                f"loss={history['loss'][-1]:.4f}  acc={history['accuracy'][-1]:.3f}"
            )

    node.concept_module.eval()
    if node.is_root and node.cnn is not None:
        node.cnn.eval()
    head.eval()
    node.concept_module.clear_activation_buffer()
    del opt, sched
    return history


@torch.no_grad()
def eval_node(node: DAGNode, head: LinearHead, loader, device: str) -> float:
    correct, total = 0, 0
    node.eval()
    head.eval()
    for x, y in loader:
        x, y  = x.to(device), y.to(device)
        preds = head(node(x)).argmax(-1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Stage 1 routing
# ---------------------------------------------------------------------------


def route_for_task(
    existing_nodes: List[DAGNode],
    new_task_loader,
    n_parents:      int,
    subspace_k:     int,
    device:         str,
    routing_batches: int = 20,
) -> Tuple[List[int], Dict[int, float]]:
    """
    For a new task, score every existing node using principal-angle similarity
    between the node's stored concept subspace and the new task's embedding
    distribution when passed through that node.

    Returns (selected_parent_indices, full_score_dict).
    """
    n_available  = len(existing_nodes)
    actual_k     = min(n_parents, n_available)
    scores: Dict[int, float] = {}

    for idx, node in enumerate(existing_nodes):
        cs = node.concept_module.get_concept_subspace()
        if cs is None:
            scores[idx] = 0.0
            continue

        # Collect embeddings of the new task through this node
        embs = []
        node.eval()
        with torch.no_grad():
            for i, (x, _) in enumerate(new_task_loader):
                if i >= routing_batches:
                    break
                embs.append(node(x.to(device)).cpu())
        E = torch.cat(embs, dim=0)
        E = E - E.mean(0, keepdim=True)
        _, _, Vh = torch.linalg.svd(E, full_matrices=False)
        query_sub = Vh[:subspace_k, :].t()   # (concept_dim, subspace_k)

        # Principal angle similarity: σ_i(Qa^T Qb)
        Qa, _ = torch.linalg.qr(query_sub)
        Qb, _ = torch.linalg.qr(cs.cpu())
        k     = min(Qa.shape[1], Qb.shape[1])
        svals = torch.linalg.svdvals(Qa[:, :k].t() @ Qb[:, :k]).clamp(0, 1)
        scores[idx] = float(svals.sum().item())

    selected = sorted(scores, key=lambda k: scores[k], reverse=True)[:actual_k]
    return selected, scores


# ---------------------------------------------------------------------------
# Exp 3a — growing DAG
# ---------------------------------------------------------------------------


def run_exp3a(cfg: Exp3Config, tasks: Optional[List[Dict]] = None) -> Dict:
    """
    Args:
        cfg:   Experiment config.
        tasks: Pre-loaded task list (e.g. with cached SSL features). If None,
               loads raw Split-CIFAR-100 images as before.
    """
    print("\n" + "=" * 70)
    print("Experiment 3a: Growing Concept DAG on Split-CIFAR-100 (20 tasks)")
    if cfg.backbone != "smallcnn":
        print(f"  Backbone: {cfg.backbone}  feature_dim={cfg.feature_dim}")
    print("=" * 70)

    torch.manual_seed(cfg.seed)
    device = cfg.device
    os.makedirs(cfg.results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if tasks is None:
        print("\n[Step 0] Loading Split-CIFAR-100 (20 × 5-class tasks)...")
        tasks = make_split_cifar100(
            data_root  = cfg.data_root,
            n_tasks    = cfg.n_tasks,
            batch_size = cfg.batch_size,
            seed       = cfg.seed,
        )
    else:
        print(f"\n[Step 0] Using pre-loaded tasks ({len(tasks)} tasks).")
    print(f"  {cfg.n_tasks} tasks, {tasks[0]['n_classes']} classes each.")

    # ------------------------------------------------------------------
    # Grow the DAG
    # ------------------------------------------------------------------
    print("\n[Step 1] Growing DAG task-by-task...")
    nodes:       List[DAGNode]   = []
    heads:       List[LinearHead] = []
    parent_map:  Dict[int, List[int]] = {}   # task_id → list of parent task_ids
    histories    = []
    test_accs    = []

    for t, task in enumerate(tasks):
        n_par = min(cfg.n_parents, t)   # task 0 has no parents
        print(f"\n── Task {t:2d} | classes {task['class_ids']} "
              f"({'root' if n_par == 0 else f'{n_par}-parent child'}) ──")

        # ---- routing ----
        if n_par == 0:
            selected_parents = []
            parent_map[t] = []
        else:
            selected_parents_idx, pa_scores = route_for_task(
                existing_nodes  = nodes,
                new_task_loader = task["train"],
                n_parents       = n_par,
                subspace_k      = cfg.subspace_k,
                device          = device,
                routing_batches = cfg.routing_batches,
            )
            parent_map[t] = selected_parents_idx
            score_str = "  ".join(
                f"t{i}:{pa_scores[i]:.3f}" for i in sorted(pa_scores)
            )
            print(f"  Scores : {score_str}")
            print(f"  Parents: {selected_parents_idx}")
            selected_parents = [nodes[i] for i in selected_parents_idx]

        # ---- build node ----
        node = DAGNode(
            task_id       = t,
            concept_dim   = cfg.concept_dim,
            cnn_out_dim   = cfg.cnn_out_dim,
            n_mlp_layers  = cfg.n_mlp_layers,
            parent_models = selected_parents if selected_parents else None,
            soft_pca_k    = cfg.soft_pca_k,
            use_cnn       = (cfg.backbone == "smallcnn"),
            feature_dim   = cfg.feature_dim,
        )
        head = LinearHead(cfg.concept_dim, task["n_classes"])

        # ---- train ----
        n_epochs = cfg.root_epochs if n_par == 0 else cfg.child_epochs
        hist = train_node(
            node, head, task["train"],
            n_epochs, cfg.lr, device, cfg.log_every,
            name=f"t{t}", orth_weight=cfg.orth_weight,
        )
        test_acc = eval_node(node, head, task["test"], device)
        print(f"  Test acc: {test_acc:.4f}")

        histories.append({"task": t, "history": hist, "test_acc": test_acc})
        test_accs.append(test_acc)

        # ---- compute subspace, freeze ----
        node.compute_concept_subspace(
            task["train"], device,
            top_k=cfg.subspace_k, max_batches=cfg.routing_batches,
        )
        node.freeze()
        nodes.append(node)
        heads.append(head)

        # flush device memory between tasks
        _flush(device)

    # ------------------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------------------
    aa = float(np.mean(test_accs))
    print("\n" + "=" * 70)
    print("SUMMARY — Exp 3a: Growing Concept DAG")
    print("=" * 70)
    print(f"{'Task':>5}  {'Parents':>20}  {'Test Acc':>10}")
    print("-" * 70)
    for t in range(cfg.n_tasks):
        pstr = str(parent_map[t]) if parent_map[t] else "root"
        print(f"  {t:3d}  {pstr:>20}  {test_accs[t]:>10.4f}")
    print(f"\n  Average Accuracy (AA): {aa:.4f}")

    # Out-degree map
    out_degree = {t: 0 for t in range(cfg.n_tasks)}
    for t, parents in parent_map.items():
        for p in parents:
            out_degree[p] += 1

    print("\n  Out-degrees:", {t: out_degree[t] for t in range(cfg.n_tasks)})

    # Concept subspace orthogonality per node
    orth_scores = {}
    for t, node in enumerate(nodes):
        cs = node.concept_module.get_concept_subspace()
        if cs is not None:
            gram = cs.t() @ cs  # (k, k) — should be ~I if subspace is orthonormal
            off  = (gram - torch.eye(gram.shape[0])).abs().mean().item()
            orth_scores[t] = float(off)

    results = {
        "test_accs":    test_accs,
        "average_accuracy": aa,
        "parent_map":   parent_map,
        "out_degree":   out_degree,
        "orth_scores":  orth_scores,
        "histories":    [{"task": h["task"], "test_acc": h["test_acc"]} for h in histories],
    }

    out_path = os.path.join(cfg.results_dir, "exp3a_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results, nodes, heads, parent_map, tasks


# ---------------------------------------------------------------------------
# Exp 3b — out-degree vs forgetting resistance
# ---------------------------------------------------------------------------


def run_exp3b(
    cfg:        Exp3Config,
    nodes:      List[DAGNode],
    heads:      List[LinearHead],
    parent_map: Dict[int, List[int]],
    tasks:      List[Dict],
) -> Dict:
    """
    For each node t:
      1. Record out-degree d(t).
      2. Snapshot weights.
      3. Unfreeze t, fine-tune for cfg.perturb_steps on a *different* task.
      4. Measure weight drift ‖Δθ‖₂ and accuracy drop on task t.
      5. Restore snapshot.

    Correlates d(t) with drift — tests the "crystallisation = forgetting resistance"
    hypothesis.
    """
    print("\n" + "=" * 70)
    print("Experiment 3b: Out-degree vs forgetting resistance")
    print("=" * 70)

    device = cfg.device
    n      = len(nodes)

    # Out-degree
    out_degree = {t: 0 for t in range(n)}
    for t, parents in parent_map.items():
        for p in parents:
            out_degree[p] += 1

    results_3b = []

    for t in range(n):
        node = nodes[t]
        head = heads[t]
        d    = out_degree[t]

        # ---- baseline accuracy ----
        base_acc = eval_node(node, head, tasks[t]["test"], device)

        # ---- snapshot — always store on CPU so drift calc is device-agnostic ----
        snapshot = {name: p.detach().cpu().clone()
                    for name, p in node.named_parameters()}

        # ---- temporarily unfreeze ----
        node.unfreeze()
        node.concept_module.train()
        if node.is_root and node.cnn is not None:
            node.cnn.train()

        # Use the "next" task's data as the wrong task (wraps around)
        wrong_task_idx  = (t + 1) % n
        wrong_loader    = tasks[wrong_task_idx]["train"]
        wrong_n_classes = tasks[wrong_task_idx]["n_classes"]

        # Temporary wrong-task head (same dim, different output size)
        wrong_head = LinearHead(cfg.concept_dim, wrong_n_classes).to(device)

        opt = torch.optim.SGD(
            list(node.trainable_parameters()) + list(wrong_head.parameters()),
            lr=cfg.perturb_lr, momentum=0.9,
        )

        step = 0
        data_iter = iter(wrong_loader)
        while step < cfg.perturb_steps:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(wrong_loader)
                x, y = next(data_iter)
            x, y   = x.to(device), y.to(device)
            logits = wrong_head(node(x))
            loss   = safe_cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(node.trainable_parameters(), 1.0)
            opt.step()
            step += 1

        del opt, wrong_head
        _flush(device)

        # ---- measure drift — both sides on CPU ----
        named = dict(node.named_parameters())
        drift = 0.0
        for name, old_p in snapshot.items():   # old_p is always CPU
            drift += (named[name].detach().cpu() - old_p).pow(2).sum().item()
        drift = math.sqrt(drift)

        # ---- measure accuracy drop ----
        node.eval()
        if node.is_root and node.cnn is not None:
            node.cnn.eval()
        node.concept_module.eval()
        post_acc = eval_node(node, head, tasks[t]["test"], device)
        acc_drop = base_acc - post_acc

        # ---- restore — copy CPU snapshot back onto the live (device) parameters ----
        with torch.no_grad():
            for name, live_p in named.items():   # 'named' already built above
                live_p.copy_(snapshot[name].to(device))
        node.freeze()

        results_3b.append({
            "task":       t,
            "out_degree": d,
            "base_acc":   base_acc,
            "post_acc":   post_acc,
            "acc_drop":   acc_drop,
            "weight_drift": drift,
        })
        print(
            f"  Task {t:2d} | out_degree={d} | "
            f"drift={drift:.4f} | "
            f"acc {base_acc:.4f} → {post_acc:.4f} (Δ={-acc_drop:+.4f})"
        )

        _flush(device)

    # ---- correlation summary ----
    degrees = [r["out_degree"]   for r in results_3b]
    drifts  = [r["weight_drift"] for r in results_3b]
    drops   = [r["acc_drop"]     for r in results_3b]

    if len(set(degrees)) > 1:
        corr_drift = float(np.corrcoef(degrees, drifts)[0, 1])
        corr_drop  = float(np.corrcoef(degrees, drops)[0, 1])
    else:
        corr_drift = corr_drop = float("nan")

    print(f"\n  Pearson r(out_degree, weight_drift) = {corr_drift:+.4f}")
    print(f"  Pearson r(out_degree, acc_drop)     = {corr_drop:+.4f}")
    print("  (Negative correlation supports the forgetting-resistance hypothesis.)")

    output = {
        "per_node": results_3b,
        "corr_out_degree_drift": corr_drift,
        "corr_out_degree_drop":  corr_drop,
    }
    out_path = os.path.join(cfg.results_dir, "exp3b_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {out_path}")
    return output
