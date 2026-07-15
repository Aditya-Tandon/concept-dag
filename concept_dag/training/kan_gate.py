"""
Kan-gated growth — the reuse-vs-grow decision for the Concept DAG.

Motivation
----------
The current protocol (`two_stage.stage2_train`) *always* mints a new ConceptModule per task, so
parameters grow linearly with the number of tasks. The Kan gate replaces that unconditional growth
with an information test:

    Grow a new concept  iff  the task cannot be solved by *recomposing existing concepts* —
    i.e. a new concept beats pure recombination by more than its own description cost.

This is the categorical "Kan obstruction" (no morphism from existing concept types to the needed
sufficient statistic) made operational as a **paired minimum-description-length (MDL)** test, plus an
optional geometric check on concept subspaces.

The arbitrary-task problem
--------------------------
A task is arbitrary: classification, regression, density estimation, ranking… each has its own
natural loss, on its own scale. If the gate hardcoded "accuracy" it would break on non-classification
tasks, and an absolute loss threshold is meaningless across tasks (a 100-class task floors near
log2(100) bits/sample; a binary task near 1). We handle this with three moves:

  1. **Common currency = bits (negative log-likelihood).** Every task is cast as a probabilistic
     prediction and scored by its own code length in bits. The gate never sees "accuracy"; it sees
     ``nll_bits(prediction, target)`` supplied by the task. This is Wang & Buehler's per-regime
     description-length functional ``L_b`` and Finzi's two-part code, made literal.

  2. **Paired, scale-free decisions.** The decision variable is a *difference on the same held-out
     data* — ``L_reuse - L_grow`` — compared against the new module's description cost, and against a
     *relative* floor ``eps_rel * L_reuse``. Because both code lengths are measured on the same task,
     the task's intrinsic difficulty cancels; the threshold lives in the universal bit currency, so
     it transfers across tasks unchanged.

  3. **The description-length functional travels with the task, not the gate.** A task registers a
     :class:`TaskSpec` (a head factory + an ``nll_bits`` function). Adding a new *kind* of task means
     writing a TaskSpec, never editing this file. The gate logic is task-agnostic.

A fourth, metric-free signal is available for free: concept **subspace geometry**
(``ConceptDAG.principal_angle_similarity``) is representational and independent of any loss, so
"is the needed structure already present" has a geometric component that needs no per-task metric.
The gate can require *both* a geometric obstruction and a codelength obstruction (AND) for a robust,
two-independent-signals decision.
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..modules.dag import ConceptDAG
from ..modules.concept_module import ConceptModule
from ..utils.metrics import safe_cross_entropy

_LOG2 = math.log(2.0)


# ---------------------------------------------------------------------------
# TaskSpec — the per-task description-length functional (L_b)
# ---------------------------------------------------------------------------


@dataclass
class TaskSpec:
    """
    A task-agnostic description-length functional.

    Fields
    ------
    name:      Human-readable task name (for logging / provenance).
    make_head: ``make_head(in_dim) -> nn.Module`` builds a head mapping a concept embedding of width
               ``in_dim`` to the parameters of the task's output distribution.
    nll_bits:  ``nll_bits(head_out, target) -> Tensor`` returns the per-sample negative log-likelihood
               **in bits** (i.e. natural-log NLL divided by ln 2). This is the code length of the
               target under the model — the only task-specific quantity the gate consumes.
    """

    name: str
    make_head: Callable[[int], nn.Module]
    nll_bits: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def classification_task(n_classes: int, name: str = "classification") -> TaskSpec:
    """Categorical target → cross-entropy code length in bits."""

    def make_head(in_dim: int) -> nn.Module:
        return nn.Linear(in_dim, n_classes)

    def nll_bits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # cross_entropy returns nats; convert to bits. Keep per-sample (reduction='none').
        return safe_cross_entropy(logits, y, reduction="none") / _LOG2

    return TaskSpec(name=name, make_head=make_head, nll_bits=nll_bits)


def regression_task(dim: int = 1, name: str = "regression") -> TaskSpec:
    """
    Continuous target → Gaussian code length in bits, with a learned homoscedastic log-variance.
    The head emits the mean; the log-variance is a free parameter of the head so the code length is
    a proper (calibrated) NLL rather than a raw MSE.
    """

    class GaussHead(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.mean = nn.Linear(in_dim, dim)
            self.log_var = nn.Parameter(torch.zeros(dim))

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            mu = self.mean(z)
            log_var = self.log_var.expand_as(mu)
            return torch.stack([mu, log_var], dim=-1)  # (B, dim, 2)

    def make_head(in_dim: int) -> nn.Module:
        return GaussHead(in_dim)

    def nll_bits(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, log_var = out[..., 0], out[..., 1]
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        # 0.5 * [ log(2π) + log_var + (y-μ)²/σ² ], summed over target dims, in nats → bits.
        nats = 0.5 * (math.log(2 * math.pi) + log_var + (y - mu) ** 2 / log_var.exp())
        return nats.sum(dim=-1) / _LOG2

    return TaskSpec(name=name, make_head=make_head, nll_bits=nll_bits)


# ---------------------------------------------------------------------------
# Reuse composer — solve the task by recomposing EXISTING concepts only
# ---------------------------------------------------------------------------


class ReuseComposer(nn.Module):
    """
    The "search / recombination" model: a **linear recombination** of the (frozen) parent concept
    embeddings, read out with the task head. It may freely mix and re-weight every dimension of every
    existing concept (a full linear map over the concatenated parents) but has **no non-linearity and
    no hidden layer** — it cannot mint a *new* concept representation. That is the precise reuse-vs-grow
    contrast: if a linear readout of existing concepts already solves the task, there is no
    obstruction; only a task needing genuinely new (non-linear) structure justifies growing a concept.
    """

    def __init__(self, parent_dim: int, n_parents: int, head: nn.Module):
        super().__init__()
        # Linear recombination of all parent features → a concept-width readout → head. No non-linearity.
        self.proj = nn.Linear(max(n_parents, 1) * parent_dim, parent_dim)
        self.head = head

    def forward(self, parent_stack: torch.Tensor) -> torch.Tensor:
        # parent_stack: (B, n_parents, D) → concat → linear → (B, D)
        return self.head(self.proj(parent_stack.flatten(1)))


class SearchComposer(nn.Module):
    """
    The **Search-level** model: a *low-rank non-linear* recombination of a chosen subset of the frozen
    parent concepts. It is deliberately intermediate between :class:`ReuseComposer` (linear, no hidden
    layer — retrieval) and a new :class:`ConceptModule` (a full new representation — discovery):

      concat(selected parents) → Linear(·, r) → GELU → Linear(r, D) → head          (r ≪ D)

    The bottleneck rank ``r`` keeps its capacity far below a concept, so a task it solves is solved by
    *recombining* existing concepts with a little non-linear glue — spending test-time compute — not by
    minting new structure. Searching over the parent ``subset`` is the routing axis of that compute.
    """

    def __init__(self, parent_dim: int, n_parents: int, head: nn.Module,
                 rank: int = 16, subset: Optional[Tuple[int, ...]] = None):
        super().__init__()
        self.subset = tuple(range(n_parents)) if subset is None else tuple(subset)
        in_dim = max(len(self.subset), 1) * parent_dim
        self.enc = nn.Linear(in_dim, rank)
        self.act = nn.GELU()
        self.dec = nn.Linear(rank, parent_dim)
        self.head = head

    def forward(self, parent_stack: torch.Tensor) -> torch.Tensor:
        sel = parent_stack[:, self.subset, :] if parent_stack.dim() == 3 else parent_stack
        z = self.act(self.enc(sel.flatten(1)))
        return self.head(self.dec(z))


# ---------------------------------------------------------------------------
# The gate record (slots into the vault's gate_verdict convention)
# ---------------------------------------------------------------------------


@dataclass
class KanGateRecord:
    task_name: str
    decision: str                 # "grow" | "reuse"
    L_reuse_bits: float           # held-out code length per sample, reuse-only model
    L_grow_bits: float            # held-out code length per sample, grow model
    delta_bits_per_sample: float  # L_reuse - L_grow  (>0 ⇒ growing helps)
    model_bits: float             # description cost of the extra parameters a new concept adds
    n_samples: int
    net_codelength_delta: float   # delta_bits_per_sample * N - model_bits  (>0 ⇒ MDL favours grow)
    rel_improvement: float        # delta_bits_per_sample / L_reuse  (scale-free)
    best_subspace_similarity: Optional[float]  # geometric signal in [0, top_k]; None if no parents
    obstruction_geometric: Optional[bool]
    obstruction_codelength: bool
    reason: str
    # --- Search level (three-way gate); None on the binary reuse-vs-grow path. ---
    L_search_bits: Optional[float] = None       # held-out bits of the best bounded-search composition
    rel_search: Optional[float] = None          # fraction of reducible info Search adds beyond reuse
    rel_grow: Optional[float] = None            # fraction of reducible info grow adds beyond best search
    search_meta: Optional[dict] = None          # winning search config {subset, rank} to rebuild it
    search_trace: Optional[list] = None         # [(T, best L_search so far)] — the compute knee

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ---------------------------------------------------------------------------
# Internals: cache frozen parent embeddings; train a readout to convergence
# ---------------------------------------------------------------------------


@torch.no_grad()
def _cache_parent_embeddings(
    dag: ConceptDAG,
    parent_ids: List[str],
    loader: DataLoader,
    device: str,
    input_encoder: Optional[nn.Module],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen ancestor subgraph once; return (parent_stack (N, P, D), targets (N,...))."""
    stacks, ys = [], []
    for x, y in loader:
        x = x.to(device)
        if input_encoder is not None:
            x = input_encoder(x)
        _, emb = dag.forward(x, active_nodes=parent_ids, return_all_embeddings=True)
        stacks.append(torch.stack([emb[pid] for pid in parent_ids], dim=1).cpu())  # (B, P, D)
        ys.append(y.cpu())
    return torch.cat(stacks, dim=0), torch.cat(ys, dim=0)


def _held_out_codelength(
    model: nn.Module,
    forward: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    spec: TaskSpec,
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    n_epochs: int,
    lr: float,
    device: str,
    batch_size: int = 128,
) -> float:
    """
    Fit `model` on (train_X, train_y) minimising bit code length; return the **best** held-out
    bits/sample seen during training (early stopping).

    Early stopping is essential, not cosmetic: without it an over-parameterised grow-probe overfits
    and its final val code length can exceed the reuse model's, masking a real obstruction. The
    minimum held-out code length is the honest estimate of what each model class can achieve, so the
    reuse-vs-grow comparison is between best-achievable code lengths, not arbitrary end-of-run ones.
    """
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = train_X.shape[0]
    best_val = float("inf")
    for _ in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = train_X[idx].to(device)
            yb = train_y[idx].to(device)
            out = forward(model, xb)
            loss = spec.nll_bits(out, yb).mean()  # bits/sample
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vout = forward(model, val_X.to(device))
            val_bits = float(spec.nll_bits(vout, val_y.to(device)).mean().item())
        best_val = min(best_val, val_bits)
    return best_val


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def reuse_vs_grow(
    dag: ConceptDAG,
    parent_ids: List[str],
    new_module_factory: Callable[[], ConceptModule],
    spec: TaskSpec,
    loader: DataLoader,
    query_subspace: Optional[torch.Tensor] = None,
    *,
    concept_dim: int,
    device: str = "cpu",
    input_encoder: Optional[nn.Module] = None,
    n_epochs: int = 15,
    lr: float = 1e-3,
    val_fraction: float = 0.3,
    eps_rel: float = 0.05,
    sim_threshold: Optional[float] = None,
    require_geometric: bool = False,
    require_mdl: bool = False,
    bits_per_param_fn: Optional[Callable[[int, int], float]] = None,
) -> KanGateRecord:
    """
    Decide whether to GROW a new concept or REUSE existing ones for this task.

    The decision is paired and scale-free (see module docstring):
      * ``L_reuse`` = held-out code length (bits/sample) of the recombination-only model.
      * ``L_grow``  = held-out code length of the model that adds a new ConceptModule.
      * MDL obstruction: net code length falls, ``(L_reuse - L_grow) * N > model_bits``.
      * Relative obstruction: ``(L_reuse - L_grow) / L_reuse > eps_rel`` — this is the term that makes
        the threshold transfer across arbitrary tasks, since it is dimensionless.
      * Optional geometric obstruction: no existing concept subspace aligns with the task probe
        subspace above ``sim_threshold``.

    GROW iff the codelength obstruction holds (and, if ``require_geometric``, the geometric one too).
    With no parents (task 0, or nothing routable) growth is forced.

    Returns a :class:`KanGateRecord` — log it as the task's gate artifact.
    """
    # --- No parents ⇒ nothing to reuse ⇒ growth is forced. ---
    if not parent_ids:
        return KanGateRecord(
            task_name=spec.name, decision="grow",
            L_reuse_bits=float("inf"), L_grow_bits=float("nan"),
            delta_bits_per_sample=float("inf"), model_bits=0.0, n_samples=0,
            net_codelength_delta=float("inf"), rel_improvement=float("inf"),
            best_subspace_similarity=None, obstruction_geometric=None,
            obstruction_codelength=True,
            reason="no parents to recompose — growth forced (root concept).",
        )

    # --- Cache frozen parent embeddings; compute the geometric signal; delegate to the core. ---
    X, y = _cache_parent_embeddings(dag, parent_ids, loader, device, input_encoder)
    best_sim: Optional[float] = None
    if query_subspace is not None:
        sims = [
            dag.principal_angle_similarity(query_subspace, mid)
            for mid in dag.all_module_ids()
            if dag.get_module(mid).get_concept_subspace() is not None
        ]
        if sims:
            best_sim = max(sims)
    return decide_reuse_vs_grow(
        X, y, new_module_factory, spec,
        concept_dim=concept_dim, n_parents=len(parent_ids), device=device,
        n_epochs=n_epochs, lr=lr, val_fraction=val_fraction, eps_rel=eps_rel,
        best_subspace_similarity=best_sim, sim_threshold=sim_threshold,
        require_geometric=require_geometric, require_mdl=require_mdl,
        bits_per_param_fn=bits_per_param_fn,
    )


def decide_reuse_vs_grow(
    parent_stack: torch.Tensor,
    targets: torch.Tensor,
    new_module_factory: Callable[[], ConceptModule],
    spec: TaskSpec,
    *,
    concept_dim: int,
    n_parents: int,
    device: str = "cpu",
    n_epochs: int = 15,
    lr: float = 1e-3,
    val_fraction: float = 0.3,
    eps_rel: float = 0.05,
    best_subspace_similarity: Optional[float] = None,
    sim_threshold: Optional[float] = None,
    require_geometric: bool = False,
    require_mdl: bool = False,
    bits_per_param_fn: Optional[Callable[[int, int], float]] = None,
) -> KanGateRecord:
    """
    Backbone-agnostic reuse-vs-grow decision on a precomputed parent stack.

    ``parent_stack`` is (N, n_parents, concept_dim) — the frozen parent concept embeddings — and
    ``targets`` is (N, ...). This is the shared core used by both the ConceptDAG path
    (:func:`reuse_vs_grow`) and the DAGNode experiment path, so the decision logic lives in exactly
    one place. ``best_subspace_similarity`` is the (loss-free) geometric signal, if available.
    """
    X, y = parent_stack, targets
    n = X.shape[0]
    n_val = max(1, int(round(val_fraction * n)))
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, ytr, Xval, yval = X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

    # --- Reuse-only model: linear recombination of frozen parents + task head. ---
    reuse_model = ReuseComposer(parent_dim=concept_dim, n_parents=n_parents,
                                head=spec.make_head(concept_dim))
    L_reuse = _held_out_codelength(
        reuse_model, lambda m, xb: m(xb), spec, Xtr, ytr, Xval, yval,
        n_epochs=n_epochs, lr=lr, device=device,
    )

    # --- Grow model: a new ConceptModule over the same parents + task head. ---
    new_module = new_module_factory()
    grow_head = spec.make_head(new_module.out_dim)

    class _GrowModel(nn.Module):
        def __init__(self, module: ConceptModule, head: nn.Module):
            super().__init__()
            self.module = module
            self.head = head

        def forward(self, parent_stack: torch.Tensor) -> torch.Tensor:
            outs = [parent_stack[:, i, :] for i in range(parent_stack.shape[1])]
            return self.head(self.module(x=None, parent_outputs=outs))

    grow_model = _GrowModel(new_module, grow_head)
    L_grow = _held_out_codelength(
        grow_model, lambda m, xb: m(xb), spec, Xtr, ytr, Xval, yval,
        n_epochs=n_epochs, lr=lr, device=device,
    )

    # --- Null (marginal) code length: best input-independent predictor. Sets the "reducible" scale. ---
    class _NullModel(nn.Module):
        def __init__(self, head: nn.Module):
            super().__init__()
            self.head = head

        def forward(self, xb: torch.Tensor) -> torch.Tensor:
            return self.head(torch.zeros(xb.shape[0], concept_dim, device=xb.device))

    L_null = _held_out_codelength(
        _NullModel(spec.make_head(concept_dim)), lambda m, xb: m(xb), spec,
        Xtr, ytr, Xval, yval, n_epochs=max(n_epochs // 2, 10), lr=lr, device=device,
    )

    # --- Description cost of the EXTRA parameters growth introduces (grow − reuse). ---
    k_extra = max(sum(p.numel() for p in grow_model.parameters())
                  - sum(p.numel() for p in reuse_model.parameters()), 0)
    if bits_per_param_fn is not None:
        model_bits = bits_per_param_fn(k_extra, n)
    else:
        model_bits = 0.5 * k_extra * math.log2(max(n, 2))  # BIC-style two-part code cost

    delta = L_reuse - L_grow                         # bits/sample the new concept saves
    net = delta * n - model_bits                     # net code length change (bits)

    # Primary decision: the fraction of *reducible* information (relative to the marginal L_null) that
    # ONLY a new concept captures — grow's extra reduction over reuse, normalised by how much is
    # reducible at all. This is scale-free (a fraction, transfers across arbitrary tasks) AND robust
    # when reuse already nearly solves the task: there delta→0, so the fraction →0 and we reuse —
    # unlike delta/L_reuse, which explodes as L_reuse→0. `rel` (the record field) now holds this
    # residual fraction.
    reducible = max(L_null - L_grow, 1e-6)
    rel = delta / reducible
    obstruction_code = rel > eps_rel
    if require_mdl:
        obstruction_code = obstruction_code and (net > 0.0)

    obstruction_geom: Optional[bool] = None
    if best_subspace_similarity is not None and sim_threshold is not None:
        obstruction_geom = best_subspace_similarity < sim_threshold

    if require_geometric and obstruction_geom is not None:
        grow = obstruction_code and obstruction_geom
    else:
        grow = obstruction_code

    reason = (
        f"ΔL={delta:.4f} bits/sample (L_null={L_null:.3f} L_reuse={L_reuse:.3f} L_grow={L_grow:.3f}), "
        f"residual_frac={rel:.3f} vs eps_rel={eps_rel}"
        + (f", best_sim={best_subspace_similarity:.3f}" if best_subspace_similarity is not None else "")
    )
    return KanGateRecord(
        task_name=spec.name, decision=("grow" if grow else "reuse"),
        L_reuse_bits=L_reuse, L_grow_bits=L_grow,
        delta_bits_per_sample=delta, model_bits=model_bits, n_samples=n,
        net_codelength_delta=net, rel_improvement=rel,
        best_subspace_similarity=best_subspace_similarity, obstruction_geometric=obstruction_geom,
        obstruction_codelength=obstruction_code, reason=reason,
    )


# ---------------------------------------------------------------------------
# The Search level — bounded test-time-compute search over existing concepts
# ---------------------------------------------------------------------------


def _enumerate_subsets(n_parents: int) -> List[Tuple[int, ...]]:
    """Non-empty parent subsets, singletons first then the full set (routing search space)."""
    singles = [(i,) for i in range(n_parents)]
    full = [tuple(range(n_parents))] if n_parents > 1 else []
    return singles + full


def search_compose(
    Xtr: torch.Tensor, ytr: torch.Tensor, Xval: torch.Tensor, yval: torch.Tensor,
    spec: TaskSpec, *, concept_dim: int, n_parents: int, device: str,
    n_epochs: int, lr: float, budget: int = 6, rank: int = 16,
) -> Tuple[float, dict, list]:
    """Bounded search for the best composition of EXISTING concepts (the middle rung).

    Spends up to ``budget`` trained candidates over the (parent-subset × restart) space, each a
    :class:`SearchComposer`, and returns the best held-out code length, the winning config
    ``{subset, rank}``, and the ``(T, best-L-so-far)`` trace. ``T`` (candidate count) is the reported
    compute budget: L_search is monotone non-increasing in T, so its knee is the epiplexity signal for
    how much structure is *compute-extractable* from existing concepts before growth is warranted.
    """
    subsets = _enumerate_subsets(n_parents)
    # Interleave: every subset once (seed 0), then extra restarts of each — best-first coverage.
    candidates: List[Tuple[Tuple[int, ...], int]] = []
    seed = 0
    while len(candidates) < budget:
        for sub in subsets:
            candidates.append((sub, seed))
            if len(candidates) >= budget:
                break
        seed += 1

    best_L = float("inf")
    best_cfg = {"subset": subsets[-1], "rank": rank}
    trace: List[Tuple[int, float]] = []
    for t, (sub, sd) in enumerate(candidates, start=1):
        torch.manual_seed(sd)
        model = SearchComposer(parent_dim=concept_dim, n_parents=n_parents,
                               head=spec.make_head(concept_dim), rank=rank, subset=sub)
        L = _held_out_codelength(model, lambda m, xb: m(xb), spec, Xtr, ytr, Xval, yval,
                                 n_epochs=n_epochs, lr=lr, device=device)
        if L < best_L:
            best_L, best_cfg = L, {"subset": sub, "rank": rank}
        trace.append((t, best_L))
    return best_L, best_cfg, trace


def decide_reuse_search_grow(
    parent_stack: torch.Tensor,
    targets: torch.Tensor,
    new_module_factory: Callable[[], ConceptModule],
    spec: TaskSpec,
    *,
    concept_dim: int,
    n_parents: int,
    device: str = "cpu",
    n_epochs: int = 15,
    lr: float = 1e-3,
    val_fraction: float = 0.3,
    eps_grow: float = 0.05,
    eps_search: float = 0.05,
    search_budget: int = 6,
    search_rank: int = 16,
    bits_per_param_fn: Optional[Callable[[int, int], float]] = None,
) -> KanGateRecord:
    """
    Three-way escalation: **reuse → search → grow** on one MDL axis.

    All four probes (null, reuse, search, grow) are fit on the SAME held-out split so their code
    lengths are directly comparable. Residual fractions share the reducible denominator
    ``(L_null − L_grow)`` so they are additive slices of the total reducible information:

      * ``rel_search = (L_reuse  − L_search) / (L_null − L_grow)`` — what bounded search adds beyond reuse
      * ``rel_grow   = (L_search − L_grow)   / (L_null − L_grow)`` — what a NEW concept adds beyond search

    Decision (cheapest sufficient rung): grow if ``rel_grow > eps_grow`` (obstruction survives the
    search budget); else search if ``rel_search > eps_search`` (compute closed the gap, no new concept);
    else reuse. See [[test-time-compute-search-level]].
    """
    X, y = parent_stack, targets
    n = X.shape[0]
    n_val = max(1, int(round(val_fraction * n)))
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, ytr, Xval, yval = X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

    # Reuse (linear recombination).
    reuse_model = ReuseComposer(parent_dim=concept_dim, n_parents=n_parents, head=spec.make_head(concept_dim))
    L_reuse = _held_out_codelength(reuse_model, lambda m, xb: m(xb), spec, Xtr, ytr, Xval, yval,
                                   n_epochs=n_epochs, lr=lr, device=device)

    # Search (bounded test-time compute over existing concepts).
    L_search, search_cfg, trace = search_compose(
        Xtr, ytr, Xval, yval, spec, concept_dim=concept_dim, n_parents=n_parents,
        device=device, n_epochs=n_epochs, lr=lr, budget=search_budget, rank=search_rank)

    # Grow (a new concept over the same parents).
    new_module = new_module_factory()
    grow_head = spec.make_head(new_module.out_dim)

    class _GrowModel(nn.Module):
        def __init__(self, module: ConceptModule, head: nn.Module):
            super().__init__()
            self.module = module; self.head = head

        def forward(self, parent_stack: torch.Tensor) -> torch.Tensor:
            outs = [parent_stack[:, i, :] for i in range(parent_stack.shape[1])]
            return self.head(self.module(x=None, parent_outputs=outs))

    grow_model = _GrowModel(new_module, grow_head)
    L_grow = _held_out_codelength(grow_model, lambda m, xb: m(xb), spec, Xtr, ytr, Xval, yval,
                                  n_epochs=n_epochs, lr=lr, device=device)

    # Null (marginal) — sets the reducible scale.
    class _NullModel(nn.Module):
        def __init__(self, head: nn.Module):
            super().__init__(); self.head = head

        def forward(self, xb: torch.Tensor) -> torch.Tensor:
            return self.head(torch.zeros(xb.shape[0], concept_dim, device=xb.device))

    L_null = _held_out_codelength(_NullModel(spec.make_head(concept_dim)), lambda m, xb: m(xb), spec,
                                  Xtr, ytr, Xval, yval, n_epochs=max(n_epochs // 2, 10), lr=lr, device=device)

    reducible = max(L_null - L_grow, 1e-6)
    rel_search = (L_reuse - L_search) / reducible
    rel_grow = (L_search - L_grow) / reducible

    k_extra = max(sum(p.numel() for p in grow_model.parameters())
                  - sum(p.numel() for p in reuse_model.parameters()), 0)
    model_bits = (bits_per_param_fn(k_extra, n) if bits_per_param_fn is not None
                  else 0.5 * k_extra * math.log2(max(n, 2)))

    if rel_grow > eps_grow:
        decision = "grow"
    elif rel_search > eps_search:
        decision = "search"
    else:
        decision = "reuse"

    reason = (f"L_null={L_null:.3f} L_reuse={L_reuse:.3f} L_search={L_search:.3f} L_grow={L_grow:.3f} | "
              f"rel_search={rel_search:.3f} (eps {eps_search}) rel_grow={rel_grow:.3f} (eps {eps_grow}) "
              f"→ {decision}; search_subset={search_cfg['subset']}")
    return KanGateRecord(
        task_name=spec.name, decision=decision,
        L_reuse_bits=L_reuse, L_grow_bits=L_grow,
        delta_bits_per_sample=(L_reuse - L_grow), model_bits=model_bits, n_samples=n,
        net_codelength_delta=(L_reuse - L_grow) * n - model_bits,
        rel_improvement=(L_reuse - L_grow) / reducible,   # keep the binary field for back-compat/logging
        best_subspace_similarity=None, obstruction_geometric=None,
        obstruction_codelength=(rel_grow > eps_grow), reason=reason,
        L_search_bits=L_search, rel_search=rel_search, rel_grow=rel_grow,
        search_meta=search_cfg, search_trace=trace,
    )


# ---------------------------------------------------------------------------
# Orchestrator — one call to add a task under the Kan gate
# ---------------------------------------------------------------------------


def _fit_full(model, forward, spec, X, y, n_epochs, lr, device, batch_size=128):
    """Train `model` to convergence on all of (X, y) minimising bit code length (final fit)."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = X.shape[0]
    for _ in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            out = forward(model, X[idx].to(device))
            loss = spec.nll_bits(out, y[idx].to(device)).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
    return model


def kan_gated_grow(
    dag: ConceptDAG,
    new_module_id: str,
    parent_ids: List[str],
    new_module_factory: Callable[[], ConceptModule],
    spec: TaskSpec,
    loader: DataLoader,
    *,
    concept_dim: int,
    query_subspace: Optional[torch.Tensor] = None,
    device: str = "cpu",
    input_encoder: Optional[nn.Module] = None,
    subspace_top_k: int = 8,
    final_epochs: int = 60,
    lr: float = 1e-3,
    gate_kwargs: Optional[dict] = None,
) -> dict:
    """
    Add a task to the DAG under the Kan gate — the single entry point for the gated architecture.

    Runs :func:`reuse_vs_grow`, then:
      * **grow**  → train a new ConceptModule (task-agnostic code-length loss from ``spec``), add it,
        cache its concept subspace, freeze it, and train a task head. A new concept node is created.
      * **reuse** → train a :class:`ReuseComposer` (recombination of existing frozen concepts) + head.
        **No DAG node is added** — the task is solved by existing concepts. This is the branch that
        keeps parameter growth ∝ genuine novelty rather than ∝ tasks.

    Returns a dict: ``{decision, record, parent_ids, solver | module_id, head}``. The caller stores
    the returned solver/head as this task's predictor for evaluation. (Wiring the reuse-task predictor
    into the experiment's eval bookkeeping is the remaining integration step; the DAG-node path is
    already compatible with the existing evaluator.)
    """
    gate_kwargs = gate_kwargs or {}
    record = reuse_vs_grow(
        dag, parent_ids, new_module_factory, spec, loader,
        query_subspace=query_subspace, concept_dim=concept_dim,
        device=device, input_encoder=input_encoder, **gate_kwargs,
    )

    # Cache parent embeddings + targets once for the final fit (frozen ancestors).
    if parent_ids:
        X, y = _cache_parent_embeddings(dag, parent_ids, loader, device, input_encoder)

    if record.decision == "reuse":
        head = spec.make_head(concept_dim)
        solver = ReuseComposer(parent_dim=concept_dim, n_parents=len(parent_ids), head=head)
        _fit_full(solver, lambda m, xb: m(xb), spec, X, y, final_epochs, lr, device)
        return {"decision": "reuse", "record": record, "parent_ids": parent_ids,
                "solver": solver, "head": head, "module_id": None}

    # --- grow ---
    module = new_module_factory()
    head = spec.make_head(module.out_dim)

    class _GrowModel(nn.Module):
        def __init__(self, m, h):
            super().__init__()
            self.module, self.head = m, h

        def forward(self, parent_stack):
            outs = [parent_stack[:, i, :] for i in range(parent_stack.shape[1])]
            return self.head(self.module(x=None, parent_outputs=outs))

    if parent_ids:
        gm = _GrowModel(module, head)
        _fit_full(gm, lambda m, xb: m(xb), spec, X, y, final_epochs, lr, device)
    # else: a true root (no parents) — caller trains it against raw inputs via the existing
    # stage-2 path; here we still register the (untrained-on-parents) module for topology.

    dag.add_module(new_module_id, module, parents=parent_ids)
    dag.freeze_except([new_module_id])
    # Cache concept subspace for future routing, then freeze.
    if parent_ids:
        module.clear_activation_buffer()
        module._collecting_subspace = True
        with torch.no_grad():
            outs = [X[:, i, :].to(device) for i in range(X.shape[1])]
            module(x=None, parent_outputs=outs)
        module._collecting_subspace = False
        module.compute_concept_subspace(top_k=subspace_top_k)
    module.freeze()
    return {"decision": "grow", "record": record, "parent_ids": parent_ids,
            "module_id": new_module_id, "head": head, "solver": None}
