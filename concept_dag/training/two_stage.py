"""
Two-stage training protocol for growing the Concept DAG.

Stage 1 — Routing:
  Given a new task's data, compute the query subspace from a probe forward pass
  through an existing encoder. Use principal angle similarity to select the best
  n_parents parent modules from the existing DAG.

  Optional: "probe head" variant — train a temporary linear head for a few
  epochs before committing to the parent selection, to mitigate the cold-start problem.

Stage 2 — Concept training:
  Freeze all selected parent modules (and everything else in the DAG).
  Instantiate a new ConceptModule as a child of the selected parents.
  Train the new child module and its aggregation layer on the task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable
import copy

from ..modules.dag import ConceptDAG
from ..modules.concept_module import ConceptModule
from ..utils.metrics import accuracy


# ---------------------------------------------------------------------------
# Stage 1: Routing
# ---------------------------------------------------------------------------

def compute_query_subspace(
    encoder: nn.Module,
    loader: DataLoader,
    top_k: int = 8,
    device: str = "cpu",
    max_batches: int = 20,
) -> torch.Tensor:
    """
    Compute the query subspace of a new task by running a probe forward pass
    through the encoder (the raw input encoder, not the DAG) and taking the
    top-k right singular vectors of the resulting activation matrix.

    Args:
        encoder:     A model that maps raw inputs to a feature vector (B, D).
                     This is typically a shallow CNN or the raw input itself
                     for small experiments.
        loader:      DataLoader for the new task's training data.
        top_k:       Number of subspace directions to extract.
        device:      Compute device.
        max_batches: Cap how many batches to use (speed vs. quality trade-off).

    Returns:
        V: (D, top_k) — right singular vectors of the activation matrix.
    """
    encoder.eval()
    activations = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            z = encoder(x)  # (B, D)
            activations.append(z.cpu())

    A = torch.cat(activations, dim=0)   # (N, D)
    A = A - A.mean(dim=0, keepdim=True)  # centre
    _, _, Vh = torch.linalg.svd(A, full_matrices=False)  # Vh: (min(N,D), D)
    V = Vh[:top_k, :].t()               # (D, top_k)
    return V


def stage1_routing(
    dag: ConceptDAG,
    query_subspace: torch.Tensor,
    n_parents: int = 2,
    top_k: int = 8,
    exclude_ids: Optional[List[str]] = None,
) -> List[str]:
    """
    Stage 1: Select the best n_parents from the existing DAG using
    principal angle similarity to the query subspace.

    Calls dag.find_best_parents() internally.

    Returns:
        List of n_parents module IDs.
    """
    selected = dag.find_best_parents(
        query_subspace=query_subspace,
        n_parents=n_parents,
        top_k=top_k,
        exclude_ids=exclude_ids,
    )
    return selected


def stage1_routing_with_probe(
    dag: ConceptDAG,
    query_subspace: torch.Tensor,
    loader: DataLoader,
    probe_epochs: int = 3,
    n_parents: int = 2,
    device: str = "cpu",
    lr: float = 1e-3,
) -> List[str]:
    """
    Stage 1 with a "probe head" to mitigate the cold-start problem.

    Instead of routing based purely on the raw input subspace, we:
      1. Attach a temporary linear classification head to each candidate parent.
      2. Train the head for probe_epochs epochs (parents remain frozen).
      3. Select parents based on probe head accuracy (better accuracy = better parent fit).

    This is more expensive but produces better routing when the raw input subspace
    is not well-aligned with the learned concept subspaces.

    Returns:
        List of n_parents module IDs.
    """
    # First use subspace similarity for an initial ranking (to avoid evaluating all)
    try:
        candidates = dag.find_best_parents(query_subspace, n_parents=min(6, len(dag.all_module_ids())), top_k=8)
    except RuntimeError:
        candidates = dag.all_module_ids()

    scores = {}
    for cid in candidates:
        candidate_module = dag.get_module(cid)
        out_dim = candidate_module.out_dim
        # Determine n_classes from loader
        all_labels = [y for _, y in loader]
        n_classes = int(torch.cat(all_labels).max().item()) + 1

        probe = nn.Linear(out_dim, n_classes).to(device)
        opt = optim.Adam(probe.parameters(), lr=lr)
        candidate_module.eval()

        for _ in range(probe_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    # Get embedding from candidate module (root: needs raw input)
                    z = candidate_module(x)
                logits = probe(z)
                loss = nn.functional.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # Evaluate probe accuracy
        correct, total = 0, 0
        probe.eval()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                z = candidate_module(x)
                preds = probe(z).argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        scores[cid] = correct / max(total, 1)

    selected = sorted(scores, key=lambda k: scores[k], reverse=True)[:n_parents]
    return selected


# ---------------------------------------------------------------------------
# Stage 2: Concept training
# ---------------------------------------------------------------------------

def stage2_train(
    dag: ConceptDAG,
    new_module: ConceptModule,
    new_module_id: str,
    parent_ids: List[str],
    classifier_head: nn.Module,
    loader: DataLoader,
    n_epochs: int = 20,
    lr: float = 1e-3,
    orth_weight: float = 0.01,
    device: str = "cpu",
    log_every: int = 5,
    input_encoder: Optional[nn.Module] = None,
) -> Dict[str, List[float]]:
    """
    Stage 2: Train the new ConceptModule with all other modules frozen.

    The training loop:
      1. Freeze everything in the DAG except the new child and its aggregator.
      2. Run forward pass through parents (frozen) → new child (trainable) → classifier head.
      3. Optimise: task loss + orth_weight * orthogonality loss (if SoftPCA).

    Args:
        dag:             The ConceptDAG.
        new_module:      The newly initialised ConceptModule (not yet in DAG).
        new_module_id:   Its ID (will be added to DAG in this function).
        parent_ids:      Selected parent module IDs.
        classifier_head: A linear head (out_dim → n_classes) for the task.
        loader:          Training DataLoader.
        n_epochs:        Training epochs.
        lr:              Learning rate.
        orth_weight:     Weight on orthogonality regularisation (SoftPCA only).
        device:          Compute device.
        log_every:       Print logs every N epochs.
        input_encoder:   Optional encoder to project raw inputs before the DAG
                         (used when raw input dim != module in_dim).

    Returns:
        History dict with 'loss', 'accuracy', 'orth_loss' lists (per epoch).
    """
    # Add new module to DAG
    dag.add_module(new_module_id, new_module, parents=parent_ids)

    # Freeze all except the new child
    dag.freeze_except([new_module_id])

    # Move everything to device
    dag._modules_dict.to(device)
    classifier_head = classifier_head.to(device)

    # Optimise: new child parameters + classifier head
    params = list(new_module.parameters()) + list(classifier_head.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"loss": [], "accuracy": [], "orth_loss": []}

    for epoch in range(1, n_epochs + 1):
        new_module.train()
        classifier_head.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if input_encoder is not None:
                with torch.no_grad():
                    x = input_encoder(x)

            # Forward through the ancestor subgraph and the new child
            active_nodes = parent_ids + [new_module_id]
            _, embeddings = dag.forward(x, active_nodes=active_nodes, return_all_embeddings=True)
            child_emb = embeddings[new_module_id]  # (B, out_dim)

            logits = classifier_head(child_emb)
            task_loss = nn.functional.cross_entropy(logits, y)

            # Orthogonality regularisation (SoftPCA only)
            orth_loss_val = new_module.orth_loss()
            total_loss = task_loss + orth_weight * orth_loss_val

            optimizer.zero_grad()
            total_loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            acc = accuracy(logits, y)
            epoch_loss += task_loss.item()
            epoch_acc += acc
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_acc / max(n_batches, 1)
        orth_val = float(new_module.orth_loss().item()) if hasattr(new_module.orth_loss(), 'item') else 0.0

        history["loss"].append(avg_loss)
        history["accuracy"].append(avg_acc)
        history["orth_loss"].append(orth_val)

        if epoch % log_every == 0 or epoch == 1:
            print(
                f"  [Stage 2 | {new_module_id} | epoch {epoch:3d}/{n_epochs}] "
                f"loss={avg_loss:.4f}  acc={avg_acc:.3f}  orth_loss={orth_val:.4f}"
            )

    # After training, compute and cache this module's concept subspace
    # (used by future children in Stage 1 routing)
    new_module.eval()
    new_module.clear_activation_buffer()
    new_module._collecting_subspace = True  # bypass frozen/train guards
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            if input_encoder is not None:
                x = input_encoder(x)
            active_nodes = parent_ids + [new_module_id]
            _, embeddings = dag.forward(x, active_nodes=active_nodes, return_all_embeddings=True)
    new_module._collecting_subspace = False
    new_module.compute_concept_subspace(top_k=8)
    print(f"  [Stage 2 | {new_module_id}] Concept subspace computed. Module frozen.")

    # Freeze the newly trained module
    new_module.freeze()

    return history
