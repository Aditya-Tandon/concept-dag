"""
Experiment 2a — Two-stage routing comparison on Split-CIFAR-10.

Tests whether Stage 1 principal-angle routing finds better parents than
random selection, and how that parent quality affects child accuracy.

Setup:
  - Split CIFAR-10 into 5 tasks (2 classes each):
      Task 0: airplane, automobile   ← vehicles/transport
      Task 1: bird, cat              ← animals
      Task 2: deer, dog              ← animals
      Task 3: frog, horse            ← animals
      Task 4: ship, truck            ← vehicles/transport  ← NEW TASK (routing target)

  - Train a RootConceptModel (SmallCNN + ConceptModule) on each of tasks 0–3.
  - For task 4, compare three routing strategies:
      (a) principal_angle  — Stage 1: cosine similarity between concept subspaces
      (b) random           — random parent pair (averaged over 5 seeds)
      (c) probe_head       — Stage 1 with a temporary probe head (cold-start mitigation)
      (d) oracle           — exhaustive search over all C(4,2)=6 parent pairs

  - For each strategy, train a child module (SoftPCA aggregation) on task 4.
  - Measure: child test accuracy, parent selection, backward transfer on tasks 0-3.

Ground-truth expectation: task 0 (airplane/automobile) should be the best parent
for task 4 (ship/truck), since both are vehicle tasks. The routing should discover this.

Run:
    python run_experiment.py --exp 2a --data_root ./data
"""

import os
import json
import itertools
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..modules.concept_module import ConceptModule
from ..modules.dag import ConceptDAG
from ..models.baselines import SmallCNN, LinearHead
from ..utils.metrics import CLMetricsTracker, evaluate, accuracy
from ..data.loaders import make_split_cifar10
from ..training.two_stage import compute_query_subspace, stage1_routing


CIFAR10_TASK_NAMES = {
    0: "airplane+automobile",
    1: "bird+cat",
    2: "deer+dog",
    3: "frog+horse",
    4: "ship+truck",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Exp2Config:
    data_root: str = "./data"
    results_dir: str = "results/exp2"
    # Model dims
    cnn_out_dim: int = 256
    concept_dim: int = 128
    n_mlp_layers: int = 2
    soft_pca_k: int = 8
    # Training
    root_epochs: int = 30
    child_epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 128
    # Routing
    n_parents: int = 2
    subspace_k: int = 8  # top-k singular vectors for subspace estimation
    probe_epochs: int = 5  # probe head epochs for cold-start mitigation
    n_random_seeds: int = 5  # how many random parent selections to average
    # Misc
    seed: int = 42
    device: str = "cpu"
    log_every: int = 10


# ---------------------------------------------------------------------------
# Root concept model: CNN backbone + ConceptModule
# ---------------------------------------------------------------------------


class RootConceptModel(nn.Module):
    """
    A task-specific root concept model: SmallCNN feature extractor + ConceptModule.
    The CNN learns task-specific visual features; ConceptModule maps them
    into the shared concept embedding space.
    """

    def __init__(
        self, task_id: int, concept_dim: int, cnn_out_dim: int, n_mlp_layers: int
    ):
        super().__init__()
        self.task_id = task_id
        self.cnn = SmallCNN(in_channels=3, out_dim=cnn_out_dim)
        self.concept_module = ConceptModule(
            module_id=f"root_{task_id}",
            in_dim=cnn_out_dim,
            hidden_dim=concept_dim,
            out_dim=concept_dim,
            n_layers=n_mlp_layers,
            n_parents=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(x)  # (B, cnn_out_dim)
        return self.concept_module(feats)  # (B, concept_dim)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.concept_module._frozen = True

    def compute_concept_subspace(self, loader, device, top_k: int = 8):
        """Collect activations and compute concept subspace for Stage 1 routing."""
        self.eval()
        self.concept_module.clear_activation_buffer()
        self.concept_module._collecting_subspace = True
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                self(x)
        self.concept_module._collecting_subspace = False
        return self.concept_module.compute_concept_subspace(top_k=top_k)


# ---------------------------------------------------------------------------
# Child concept model: aggregates parent outputs via SoftPCA
# ---------------------------------------------------------------------------


class ChildConceptModel(nn.Module):
    """
    Child module for task 4 that fuses selected parent concept embeddings
    via SoftPCA and maps to a concept embedding for classification.
    """

    def __init__(
        self,
        parent_models: List[RootConceptModel],
        concept_dim: int,
        n_mlp_layers: int,
        soft_pca_k: int,
    ):
        super().__init__()
        # Plain list — parents are frozen and already live in root_models.
        # Using nn.ModuleList would register them as owned submodules of the child,
        # preventing MPS from reclaiming their memory between child training runs.
        self.parent_models: List[RootConceptModel] = parent_models
        n_parents = len(parent_models)
        self.concept_module = ConceptModule(
            module_id="child_task4",
            in_dim=concept_dim,
            hidden_dim=concept_dim,
            out_dim=concept_dim,
            n_layers=n_mlp_layers,
            n_parents=n_parents,
            aggregation="soft_pca",
            agg_kwargs={"top_k": soft_pca_k},
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get parent embeddings (parents are frozen — no grad)
        with torch.no_grad():
            parent_outs = [p(x) for p in self.parent_models]
        return self.concept_module(x, parent_outputs=parent_outs)  # (B, concept_dim)

    def trainable_params(self):
        return list(self.concept_module.parameters())


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_root(
    model: RootConceptModel,
    head: LinearHead,
    loader,
    n_epochs: int,
    lr: float,
    device: str,
    log_every: int,
    name: str,
) -> Dict:
    model.to(device)
    head.to(device)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    history = {"loss": [], "accuracy": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        head.train()
        ep_loss, ep_acc, n = 0.0, 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = head(model(x))
            loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += loss.item()
            ep_acc += accuracy(logits, y)
            n += 1
        sched.step()
        history["loss"].append(ep_loss / max(n, 1))
        history["accuracy"].append(ep_acc / max(n, 1))
        if epoch % log_every == 0 or epoch == 1:
            print(
                f"    [{name} | ep {epoch:3d}/{n_epochs}] "
                f"loss={history['loss'][-1]:.4f}  acc={history['accuracy'][-1]:.3f}"
            )

    model.eval()
    head.eval()
    # Safety: clear any activation buffer that accumulated (shouldn't happen
    # now that training-time buffering is removed, but keeps state clean).
    model.concept_module.clear_activation_buffer()
    return history


def train_child(
    child: ChildConceptModel,
    head: LinearHead,
    loader,
    n_epochs: int,
    lr: float,
    device: str,
    log_every: int,
    orth_weight: float = 0.01,
) -> Dict:
    child.concept_module.to(device)
    head.to(device)
    params = child.trainable_params() + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    history = {"loss": [], "accuracy": []}

    for epoch in range(1, n_epochs + 1):
        child.concept_module.train()
        head.train()
        ep_loss, ep_acc, n = 0.0, 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            emb = child(x)
            logits = head(emb)
            loss = nn.functional.cross_entropy(logits, y)
            loss = loss + orth_weight * child.concept_module.orth_loss()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += loss.item()
            ep_acc += accuracy(logits, y)
            n += 1
        sched.step()
        history["loss"].append(ep_loss / max(n, 1))
        history["accuracy"].append(ep_acc / max(n, 1))
        if epoch % log_every == 0 or epoch == 1:
            print(
                f"      [child | ep {epoch:3d}/{n_epochs}] "
                f"loss={history['loss'][-1]:.4f}  acc={history['accuracy'][-1]:.3f}"
            )

    child.concept_module.eval()
    head.eval()
    del opt, sched  # free AdamW moment buffers immediately
    return history


@torch.no_grad()
def eval_model(model_fn, loader, device) -> float:
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model_fn(x).argmax(-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Routing strategies
# ---------------------------------------------------------------------------


def route_principal_angle(
    root_models: List[RootConceptModel],
    task4_train_loader,
    concept_dim: int,
    n_parents: int,
    subspace_k: int,
    device: str,
) -> Tuple[List[int], Dict[int, float]]:
    """
    Compute query subspace for task 4 by collecting activations from a shared
    linear probe on the raw CNN features, then compare against root concept subspaces
    using principal angle similarity.

    Returns selected parent indices and the full score dict.
    """
    # Collect task-4 features using a simple average pooling of the image
    # We use root_models[0]'s CNN as a feature extractor for the query
    # (a frozen backbone is the canonical approach; here we use root 0's CNN
    #  since all CNNs are in the same image space and the routing is in concept space)
    query_encoder = root_models[0].cnn
    query_encoder.eval()

    acts = []
    with torch.no_grad():
        for i, (x, _) in enumerate(task4_train_loader):
            if i >= 20:
                break
            acts.append(query_encoder(x.to(device)).cpu())
    A = torch.cat(acts, dim=0)
    A = A - A.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(A, full_matrices=False)
    query_subspace = Vh[:subspace_k, :].t()  # (cnn_out_dim, subspace_k)

    # Each root has a concept subspace in concept_dim space
    # We need to project query into concept space for fair comparison.
    # Use each root's concept_module linear weights as proxy projection.
    # More principled: collect activations of each root on task-4 data and compare.
    scores = {}
    for idx, root in enumerate(root_models):
        root.eval()
        # Collect root's concept embeddings when shown task-4 images
        embs = []
        with torch.no_grad():
            for i, (x, _) in enumerate(task4_train_loader):
                if i >= 20:
                    break
                embs.append(root(x.to(device)).cpu())
        E = torch.cat(embs, dim=0)
        E = E - E.mean(0, keepdim=True)
        _, S, _ = torch.linalg.svd(E, full_matrices=False)
        # Score = total variance captured (how much structure does this parent
        # still have when processing the new task's images?)
        scores[idx] = float(S[:subspace_k].sum().item())

    # Also compute principal angle similarity between stored concept subspaces
    # and a task-4 concept subspace derived from root 0's embeddings
    # This is the primary routing criterion
    angle_scores = {}
    task4_embs = []
    with torch.no_grad():
        for i, (x, _) in enumerate(task4_train_loader):
            if i >= 20:
                break
            task4_embs.append(root_models[0](x.to(device)).cpu())
    T4 = torch.cat(task4_embs, dim=0)
    T4 = T4 - T4.mean(0, keepdim=True)
    _, _, Vh4 = torch.linalg.svd(T4, full_matrices=False)
    query_concept_subspace = Vh4[:subspace_k, :].t()  # (concept_dim, subspace_k)

    for idx, root in enumerate(root_models):
        cs = root.concept_module.get_concept_subspace()
        if cs is None:
            angle_scores[idx] = 0.0
            continue
        Qa, _ = torch.linalg.qr(query_concept_subspace)
        Qb, _ = torch.linalg.qr(cs)
        cross = Qa.t() @ Qb
        svals = torch.linalg.svdvals(cross).clamp(0, 1)
        angle_scores[idx] = float(svals.sum().item())

    # Select top n_parents by principal angle score
    sorted_by_angle = sorted(angle_scores, key=lambda k: angle_scores[k], reverse=True)
    selected = sorted_by_angle[:n_parents]

    print(
        f"    Principal angle scores: { {CIFAR10_TASK_NAMES[i]: round(angle_scores[i], 4) for i in angle_scores} }"
    )
    print(f"    Selected parents: {[CIFAR10_TASK_NAMES[i] for i in selected]}")
    return selected, angle_scores


def route_random(n_roots: int, n_parents: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    return list(rng.choice(n_roots, size=n_parents, replace=False))


def route_oracle(
    root_models: List[RootConceptModel],
    heads: List[LinearHead],
    task4_task: dict,
    cfg: Exp2Config,
    device: str,
) -> Tuple[List[int], Dict[str, float]]:
    """
    Exhaustive search: train a child for every parent pair, return the best.
    C(4, 2) = 6 combinations.
    """
    all_pairs = list(itertools.combinations(range(len(root_models)), cfg.n_parents))
    pair_scores = {}

    print(f"    Oracle: evaluating {len(all_pairs)} parent combinations...")
    for pair in all_pairs:
        pair_name = "+".join(CIFAR10_TASK_NAMES[i] for i in pair)
        parents = [root_models[i] for i in pair]
        for p in parents:
            p.freeze()
        child = ChildConceptModel(
            parents, cfg.concept_dim, cfg.n_mlp_layers, cfg.soft_pca_k
        ).to(device)
        head = LinearHead(cfg.concept_dim, task4_task["n_classes"]).to(device)

        # Short training (half epochs) for efficiency
        train_child(
            child,
            head,
            task4_task["train"],
            cfg.child_epochs // 2,
            cfg.lr,
            device,
            log_every=999,
        )  # silent
        acc = eval_model(lambda x: head(child(x)), task4_task["val"], device)
        pair_scores[str(pair)] = acc
        print(f"      {pair_name:40s}: val_acc={acc:.4f}")
        del child, head
        _free_device_memory(device)

    best_pair_str = max(pair_scores, key=lambda k: pair_scores[k])
    best_pair = eval(best_pair_str)
    print(
        f"    Oracle best: {'+'.join(CIFAR10_TASK_NAMES[i] for i in best_pair)} "
        f"(val_acc={pair_scores[best_pair_str]:.4f})"
    )
    return list(best_pair), pair_scores


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_exp2a(cfg: Exp2Config) -> Dict:
    print("\n" + "=" * 70)
    print("Experiment 2a: Two-stage routing comparison on Split-CIFAR-10")
    print("=" * 70)

    torch.manual_seed(cfg.seed)
    device = cfg.device
    os.makedirs(cfg.results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n[Step 0] Loading Split-CIFAR-10...")
    tasks = make_split_cifar10(
        data_root=cfg.data_root,
        n_tasks=5,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )
    # Tasks 0-3 are "known concepts"; task 4 is the new concept to route into
    root_tasks = tasks[:4]
    new_task = tasks[4]
    print(f"  Root tasks : {[CIFAR10_TASK_NAMES[t['task_id']] for t in root_tasks]}")
    print(f"  New task   : {CIFAR10_TASK_NAMES[new_task['task_id']]}")

    # ------------------------------------------------------------------
    # Train root concept models (one per task 0-3)
    # ------------------------------------------------------------------
    print("\n[Step 1] Training root concept models...")
    root_models: List[RootConceptModel] = []
    root_heads: List[LinearHead] = []
    root_train_histories = []

    for task in root_tasks:
        tid = task["task_id"]
        print(f"\n  Task {tid}: {CIFAR10_TASK_NAMES[tid]}")
        model = RootConceptModel(
            tid, cfg.concept_dim, cfg.cnn_out_dim, cfg.n_mlp_layers
        )
        head = LinearHead(cfg.concept_dim, task["n_classes"])
        hist = train_root(
            model,
            head,
            task["train"],
            cfg.root_epochs,
            cfg.lr,
            device,
            cfg.log_every,
            CIFAR10_TASK_NAMES[tid],
        )
        test_acc = eval_model(lambda x, m=model, h=head: h(m(x)), task["test"], device)
        print(f"  Test acc: {test_acc:.4f}")
        root_train_histories.append(
            {"task": tid, "history": hist, "test_acc": test_acc}
        )

        # Compute and cache concept subspace
        model.compute_concept_subspace(task["train"], device, top_k=cfg.subspace_k)
        # Freeze for use as parents
        model.freeze()
        root_models.append(model)
        root_heads.append(head)

    # ------------------------------------------------------------------
    # Routing strategies + child training
    # ------------------------------------------------------------------
    print("\n[Step 2] Routing strategies for new task...")
    results = {
        "root_train_histories": root_train_histories,
        "routing_results": {},
    }

    # --- (a) Principal angle routing ---
    print("\n  (a) Principal angle routing")
    pa_parents, pa_scores = route_principal_angle(
        root_models,
        new_task["train"],
        cfg.concept_dim,
        cfg.n_parents,
        cfg.subspace_k,
        device,
    )
    pa_result = _train_and_eval_child(
        pa_parents, root_models, new_task, cfg, device, label="principal_angle"
    )
    pa_result["selected_parents"] = [CIFAR10_TASK_NAMES[i] for i in pa_parents]
    pa_result["angle_scores"] = {CIFAR10_TASK_NAMES[k]: v for k, v in pa_scores.items()}
    results["routing_results"]["principal_angle"] = pa_result

    # --- (b) Random routing (averaged over multiple seeds) ---
    print("\n  (b) Random parent selection")
    random_accs = []
    for rs in range(cfg.n_random_seeds):
        rand_parents = route_random(
            len(root_models), cfg.n_parents, seed=cfg.seed + rs + 100
        )
        rand_result = _train_and_eval_child(
            rand_parents,
            root_models,
            new_task,
            cfg,
            device,
            label=f"random_seed{rs}",
            silent=True,
        )
        random_accs.append(rand_result["test_accuracy"])
        print(
            f"    seed {rs}: parents={[CIFAR10_TASK_NAMES[i] for i in rand_parents]}  acc={rand_result['test_accuracy']:.4f}"
        )
    rand_mean = float(np.mean(random_accs))
    rand_std = float(np.std(random_accs))
    print(f"    Random mean±std: {rand_mean:.4f} ± {rand_std:.4f}")
    results["routing_results"]["random"] = {
        "test_accuracy_mean": rand_mean,
        "test_accuracy_std": rand_std,
        "per_seed_accs": random_accs,
    }

    # --- (c) Oracle ---
    print("\n  (c) Oracle (exhaustive parent search)")
    oracle_parents, oracle_pair_scores = route_oracle(
        root_models, root_heads, new_task, cfg, device
    )
    oracle_result = _train_and_eval_child(
        oracle_parents, root_models, new_task, cfg, device, label="oracle"
    )
    oracle_result["selected_parents"] = [CIFAR10_TASK_NAMES[i] for i in oracle_parents]
    oracle_result["all_pair_scores"] = oracle_pair_scores
    results["routing_results"]["oracle"] = oracle_result

    # --- (d) Monolithic baseline: fine-tune a fresh CNN on task 4 only ---
    print("\n  (d) Monolithic baseline (fresh CNN, no parents)")
    mono_result = _train_monolithic_baseline(new_task, cfg, device)
    results["routing_results"]["monolithic"] = mono_result

    # ------------------------------------------------------------------
    # Backward transfer: did frozen parents forget their tasks?
    # ------------------------------------------------------------------
    print(
        "\n[Step 3] Backward transfer check (frozen parents should show ~0 forgetting)..."
    )
    bt_results = {}
    for i, (root, head, task) in enumerate(zip(root_models, root_heads, root_tasks)):
        acc_now = eval_model(lambda x, m=root, h=head: h(m(x)), task["test"], device)
        orig = root_train_histories[i]["test_acc"]
        bt_results[CIFAR10_TASK_NAMES[task["task_id"]]] = {
            "original_acc": orig,
            "final_acc": acc_now,
            "forgetting": orig - acc_now,
        }
        print(
            f"  {CIFAR10_TASK_NAMES[task['task_id']]}: {orig:.4f} → {acc_now:.4f}  (Δ={orig-acc_now:+.4f})"
        )
    results["backward_transfer"] = bt_results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _print_exp2_summary(results)

    out_path = os.path.join(cfg.results_dir, "exp2a_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_safe)
    print(f"\nResults saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_device_memory(device: str):
    """Flush device memory cache. Critical on MPS — it doesn't release eagerly."""
    import gc
    gc.collect()
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _train_and_eval_child(
    parent_indices: List[int],
    root_models: List[RootConceptModel],
    task: dict,
    cfg: Exp2Config,
    device: str,
    label: str = "",
    silent: bool = False,
) -> Dict:
    parents = [root_models[i] for i in parent_indices]
    child = ChildConceptModel(
        parents, cfg.concept_dim, cfg.n_mlp_layers, cfg.soft_pca_k
    ).to(device)
    head = LinearHead(cfg.concept_dim, task["n_classes"]).to(device)
    log_every = 999 if silent else cfg.log_every
    hist = train_child(
        child,
        head,
        task["train"],
        cfg.child_epochs,
        cfg.lr,
        device,
        log_every=log_every,
    )
    test_acc = eval_model(lambda x: head(child(x)), task["test"], device)
    if not silent:
        print(f"    [{label}] test_acc={test_acc:.4f}")

    result = {
        "test_accuracy": test_acc,
        "train_history": hist,
        "parent_indices": parent_indices,
    }

    # Explicitly free child and head, then flush MPS/CUDA cache.
    # Without this, MPS stacks up tensors from sequential child runs (5 random
    # seeds + 6 oracle pairs) until the process runs out of memory.
    del child, head
    _free_device_memory(device)

    return result


def _train_monolithic_baseline(task: dict, cfg: Exp2Config, device: str) -> Dict:
    """Train a fresh SmallCNN + linear head directly on task 4. No parents, no DAG."""
    cnn = SmallCNN(in_channels=3, out_dim=cfg.cnn_out_dim).to(device)
    head = LinearHead(cfg.cnn_out_dim, task["n_classes"]).to(device)
    params = list(cnn.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.child_epochs)
    history = {"loss": [], "accuracy": []}

    for epoch in range(1, cfg.child_epochs + 1):
        cnn.train()
        head.train()
        ep_loss, ep_acc, n = 0.0, 0.0, 0
        for x, y in task["train"]:
            x, y = x.to(device), y.to(device)
            logits = head(cnn(x))
            loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += loss.item()
            ep_acc += accuracy(logits, y)
            n += 1
        sched.step()
        history["loss"].append(ep_loss / max(n, 1))
        history["accuracy"].append(ep_acc / max(n, 1))
        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"      [monolithic | ep {epoch:3d}/{cfg.child_epochs}] "
                f"loss={history['loss'][-1]:.4f}  acc={history['accuracy'][-1]:.3f}"
            )

    cnn.eval()
    head.eval()
    test_acc = eval_model(lambda x: head(cnn(x)), task["test"], device)
    print(f"    [monolithic] test_acc={test_acc:.4f}")
    return {"test_accuracy": test_acc, "train_history": history}


def _print_exp2_summary(results: Dict):
    rr = results["routing_results"]
    print("\n" + "=" * 70)
    print("SUMMARY — Routing strategy comparison (Task 4: ship+truck)")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Test Acc':>10} {'Parents selected'}")
    print("-" * 70)
    for name, r in rr.items():
        acc = r.get("test_accuracy") or r.get("test_accuracy_mean", 0)
        acc_str = f"{acc:.4f}"
        if "test_accuracy_std" in r:
            acc_str += f" ±{r['test_accuracy_std']:.4f}"
        parents = r.get("selected_parents", "N/A")
        if isinstance(parents, list):
            parents = " + ".join(parents)
        print(f"{name:<20} {acc_str:>14}  {parents}")

    bt = results.get("backward_transfer", {})
    if bt:
        print("\nBackward Transfer (forgetting on root tasks):")
        for task_name, v in bt.items():
            print(
                f"  {task_name:<25} Δ={v['forgetting']:+.4f}  "
                f"({v['original_acc']:.4f} → {v['final_acc']:.4f})"
            )
    print("=" * 70)


def _json_safe(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="results/exp2")
    args = parser.parse_args()

    cfg = Exp2Config(
        data_root=args.data_root,
        device=args.device,
        root_epochs=args.epochs,
        child_epochs=args.epochs,
        results_dir=args.out_dir,
    )
    run_exp2a(cfg)
