"""
Consolidation pass — the reduction (Bayesian-model-reduction) half of the loop.

The growth side mints concepts; this side reclaims parameters between growth phases, so the DAG
tracks *distinct crystallised concepts* rather than task count. Run it periodically (a "sleep" pass),
the direct analogue of offline replay / systems consolidation.

Two operations, gated so reduction never silently reintroduces forgetting:

  1. low_rank_factorize_final_layer  — re-crystallise a single frozen module to its effective rank
     ("keep the top orthogonal directions"). Function-preserving SVD factorisation of the final
     layer; reclaims parameters inside a node with no topology change.

  2. consolidate                     — find redundant concept pairs by principal-angle similarity and
     merge them (graph surgery in ConceptDAG.merge_modules), each merge gated by a caller-supplied
     ``accept_fn`` that MUST re-evaluate the affected tasks (backward-interference check).

The gate is where the rigour lives: a reduction is the inverse of a discovery, so it is accepted only
if it is evidence-preserving (ΔL ≤ 0) and disturbs no downstream task beyond ε. Those checks are the
caller's ``accept_fn`` — this module performs the surgery and the bookkeeping, not the policy.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from ..modules.dag import ConceptDAG
from ..modules.concept_module import ConceptModule


# ---------------------------------------------------------------------------
# Op 1 — low-rank re-crystallisation of a single module
# ---------------------------------------------------------------------------


def effective_rank(singular_values: torch.Tensor, energy: float = 0.99) -> int:
    """Smallest r whose top-r singular values capture `energy` fraction of the spectral energy."""
    s2 = singular_values.pow(2)
    csum = torch.cumsum(s2, dim=0) / s2.sum()
    r = int(torch.searchsorted(csum, torch.tensor(energy)).item()) + 1
    return max(1, min(r, singular_values.numel()))


def low_rank_factorize_final_layer(
    module: ConceptModule,
    energy: float = 0.99,
    max_rel_error: float = 0.05,
    rank: Optional[int] = None,
) -> Dict[str, float]:
    """
    Factorise the module's final Linear (hidden → out_dim) into (hidden → r) → (r → out_dim) using a
    rank-r SVD of its weight, when that reclaims parameters at acceptable reconstruction error.

    Returns a record dict: {applied, rank, out_dim, hidden, params_before, params_after, rel_error}.
    A no-op (applied=False) is returned when factorisation would not save params or the rank-r
    approximation exceeds ``max_rel_error`` (relative Frobenius error).
    """
    mlp = module.mlp
    final = mlp[-1]
    if not isinstance(final, nn.Linear):
        return {"applied": False, "reason": "final layer is not nn.Linear"}
    out_dim, hidden = final.weight.shape  # (out, hidden)
    dev = final.weight.device
    W = final.weight.detach().cpu()  # SVD on CPU: MPS lacks a reliable linalg.svd
    b = final.bias.detach().cpu() if final.bias is not None else None

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # W = U diag(S) Vh
    r = rank if rank is not None else effective_rank(S, energy)

    params_before = out_dim * hidden + (out_dim if b is not None else 0)
    params_after = hidden * r + out_dim * r + (out_dim if b is not None else 0)
    if params_after >= params_before:
        return {"applied": False, "reason": f"rank {r} saves nothing", "rank": r,
                "params_before": params_before, "params_after": params_after}

    # Rank-r reconstruction and its relative Frobenius error.
    W_r = (U[:, :r] * S[:r]) @ Vh[:r, :]
    rel_error = float((W - W_r).norm() / (W.norm() + 1e-12))
    if rel_error > max_rel_error:
        return {"applied": False, "reason": f"rel_error {rel_error:.3f} > {max_rel_error}",
                "rank": r, "rel_error": rel_error}

    # Build the factorised replacement: A: (r, hidden) = diag(S_r) Vh_r ; B: (out, r) = U_r.
    A = (S[:r].unsqueeze(1) * Vh[:r, :])   # (r, hidden)
    B = U[:, :r]                            # (out, r)
    lin1 = nn.Linear(hidden, r, bias=False)
    lin2 = nn.Linear(r, out_dim, bias=(b is not None))
    with torch.no_grad():
        lin1.weight.copy_(A)
        lin2.weight.copy_(B)
        if b is not None:
            lin2.bias.copy_(b)

    # Splice back in place of the old final layer (on the module's device); re-freeze.
    lin1, lin2 = lin1.to(dev), lin2.to(dev)
    new_layers = list(mlp[:-1]) + [lin1, lin2]
    module.mlp = nn.Sequential(*new_layers)
    if module.is_frozen:
        module.freeze()

    return {"applied": True, "rank": r, "out_dim": out_dim, "hidden": hidden,
            "params_before": params_before, "params_after": params_after,
            "params_saved": params_before - params_after, "rel_error": rel_error}


# ---------------------------------------------------------------------------
# Op 2 — subspace-redundancy merge over the frozen population
# ---------------------------------------------------------------------------


def find_redundant_pairs(
    dag: ConceptDAG,
    similarity_threshold: float,
    top_k: int = 8,
) -> List[Tuple[str, str, float]]:
    """
    All unordered concept pairs whose cached subspaces are near-identical (principal-angle similarity
    ≥ threshold), sorted most-similar first. Only nodes in an ancestor/descendant relationship are
    excluded (they can't be merged). Similarity is in [0, top_k]; threshold near top_k = near-identical.
    """
    ids = [mid for mid in dag.all_module_ids()
           if dag.get_module(mid).get_concept_subspace() is not None]
    pairs: List[Tuple[str, str, float]] = []
    for i, a in enumerate(ids):
        sub_a = dag.get_module(a).get_concept_subspace()
        for b in ids[i + 1:]:
            if a in dag.descendants([b]) or b in dag.descendants([a]):
                continue  # stacked, not parallel — never a merge candidate
            sim = dag.principal_angle_similarity(sub_a, b, top_k=top_k)
            if sim >= similarity_threshold:
                pairs.append((a, b, sim))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs


def consolidate(
    dag: ConceptDAG,
    *,
    accept_fn: Callable[[str, str, Set[str]], bool],
    similarity_threshold: float,
    top_k: int = 8,
    distill_fn: Optional[Callable[[str, str], None]] = None,
    truncate_energy: Optional[float] = 0.99,
    truncate_max_rel_error: float = 0.05,
) -> Dict[str, object]:
    """
    One consolidation pass.

    Order: (1) low-rank re-crystallise every module (cheap, topology-safe), then (2) merge redundant
    pairs, each gated by ``accept_fn``.

    Parameters
    ----------
    accept_fn(keep_id, drop_id, affected_task_ids) -> bool
        The backward-interference + ΔL gate. **Must** re-evaluate every task in ``affected_task_ids``
        (obtained from ``dag.affected_tasks(keep_id)`` post-merge) and return False if any regresses
        beyond tolerance or the merge does not reduce total code length. This module supplies the
        candidate and the affected set; the caller owns the accept/reject policy.
    distill_fn(keep_id, drop_id) -> None
        Optional: make ``keep_id`` reproduce both concepts' outputs (functional merge) BEFORE the
        graph surgery, so re-pointed children see approximately unchanged parent activations. If None,
        the merge is structural only (children are re-pointed as-is) — appropriate only when the two
        subspaces are essentially identical.

    Returns a summary dict with params before/after and the list of applied ops.
    """
    params_before = dag.parameter_count()
    ops: List[dict] = []

    # --- (1) Low-rank re-crystallisation. ---
    if truncate_energy is not None:
        for mid in list(dag.all_module_ids()):
            rec = low_rank_factorize_final_layer(
                dag.get_module(mid), energy=truncate_energy, max_rel_error=truncate_max_rel_error,
            )
            rec["op"] = "truncate"
            rec["module"] = mid
            if rec.get("applied"):
                ops.append(rec)

    # --- (2) Subspace-redundancy merges (re-scan after each accepted merge; ids change). ---
    merged = True
    while merged:
        merged = False
        for keep_id, drop_id, sim in find_redundant_pairs(dag, similarity_threshold, top_k):
            if distill_fn is not None:
                distill_fn(keep_id, drop_id)
            # After (hypothetical) merge, the tasks that route through keep_id are the ones at risk.
            affected = dag.affected_tasks(keep_id) | dag.affected_tasks(drop_id)
            if not accept_fn(keep_id, drop_id, affected):
                continue  # gate rejected: keep both concepts
            dag.merge_modules(keep_id, drop_id)
            ops.append({"op": "merge", "keep": keep_id, "drop": drop_id, "similarity": sim})
            merged = True
            break  # restart the scan on the modified graph

    params_after = dag.parameter_count()
    return {
        "params_before": params_before,
        "params_after": params_after,
        "params_saved": params_before - params_after,
        "n_ops": len(ops),
        "ops": ops,
    }
