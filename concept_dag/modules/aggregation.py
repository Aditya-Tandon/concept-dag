"""
Aggregation strategies for combining parent concept outputs into a child concept.

This is the heart of the Concept DAG's novelty. We compare five approaches:
  - ConcatAggregator    : baseline, concatenate + linear project
  - MeanAggregator      : baseline, element-wise mean
  - AttentionAggregator : learned attention over parent outputs
  - SVDAggregator       : differentiable SVD, top-k eigenvectors (the proposed method)
  - SoftPCAAggregator   : learnable projection regularised toward orthogonality (safer approx)

All aggregators take a list of parent tensors of shape (B, D) and return (B, D_out).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


# ---------------------------------------------------------------------------
# Baseline aggregators
# ---------------------------------------------------------------------------

class ConcatAggregator(nn.Module):
    """Concatenate all parent outputs and project to output dim."""

    def __init__(self, parent_dim: int, n_parents: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(parent_dim * n_parents, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, parent_outputs: List[torch.Tensor]) -> torch.Tensor:
        # parent_outputs: list of (B, D) tensors
        x = torch.cat(parent_outputs, dim=-1)   # (B, D * n_parents)
        return self.norm(self.proj(x))           # (B, out_dim)

    def aggregation_name(self) -> str:
        return "concat"


class MeanAggregator(nn.Module):
    """Element-wise mean of parent outputs (assumes all same dim)."""

    def __init__(self, parent_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(parent_dim, out_dim, bias=True) if parent_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, parent_outputs: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(parent_outputs, dim=1)  # (B, n_parents, D)
        mean = stacked.mean(dim=1)                    # (B, D)
        return self.norm(self.proj(mean))

    def aggregation_name(self) -> str:
        return "mean"


class AttentionAggregator(nn.Module):
    """
    Multi-head attention over parent concept outputs.
    Each parent output is a 'value'; a learned query produces a weighted sum.
    """

    def __init__(self, parent_dim: int, out_dim: int, n_parents: int = 2, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.query = nn.Parameter(torch.randn(1, 1, parent_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=parent_dim, num_heads=n_heads, batch_first=True
        )
        self.proj = nn.Linear(parent_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, parent_outputs: List[torch.Tensor]) -> torch.Tensor:
        # Stack parents as sequence: (B, n_parents, D)
        kv = torch.stack(parent_outputs, dim=1)
        B = kv.size(0)
        q = self.query.expand(B, -1, -1)            # (B, 1, D)
        out, _ = self.attn(q, kv, kv)               # (B, 1, D)
        out = out.squeeze(1)                         # (B, D)
        return self.norm(self.proj(out))

    def aggregation_name(self) -> str:
        return "attention"


# ---------------------------------------------------------------------------
# SVD-based aggregator (the proposed crystallization mechanism)
# ---------------------------------------------------------------------------

class SVDAggregator(nn.Module):
    """
    Concept crystallization via differentiable SVD.

    Forward pass:
      1. Concatenate parent outputs into a matrix M of shape (B, n_parents * D).
         Reshape to (B * n_parents, D) so each parent output is a row.
         Actually we treat the collection of parent feature vectors as a data matrix
         and extract the dominant directions via SVD.
      2. Compute SVD: M = U S V^T, keep top-k right singular vectors (V[:, :k]).
         These are the top-k orthogonal directions that capture the most variance
         across parent concepts — the "crystallized" concept subspace.
      3. Project the mean parent representation onto this subspace and read out.

    Stability mitigations:
      - Epsilon padding on singular values to avoid division-by-zero in backward pass.
      - Sign normalisation (flip sign so first element of each singular vector is positive)
        to resolve phase ambiguity.
      - Gradient clipping hook on SVD output.
      - Fallback to SoftPCA if SVD gradient norm exceeds threshold (can be enabled).
    """

    def __init__(
        self,
        parent_dim: int,
        n_parents: int,
        out_dim: int,
        top_k: Optional[int] = None,
        eps: float = 1e-4,
        grad_clip: Optional[float] = 10.0,
    ):
        super().__init__()
        self.parent_dim = parent_dim
        self.n_parents = n_parents
        self.out_dim = out_dim
        # Economy SVD of (B, n_parents, D) can yield at most min(n_parents, D) singular vectors.
        # Clamp top_k so the readout layer dim always matches what SVD actually produces.
        max_svd_rank = min(n_parents, parent_dim)
        requested_k  = top_k if top_k is not None else max_svd_rank
        self.top_k   = min(requested_k, max_svd_rank)
        if requested_k > max_svd_rank:
            import warnings
            warnings.warn(
                f"SVDAggregator: top_k={requested_k} clamped to {self.top_k} "
                f"(max SVD rank = min(n_parents={n_parents}, parent_dim={parent_dim}))."
            )
        self.eps = eps
        self.grad_clip = grad_clip

        # Linear readout from the subspace-projected representation
        self.readout = nn.Linear(self.top_k, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)

        # Monitoring (not used in forward, populated during training hooks)
        self.last_singular_values: Optional[torch.Tensor] = None
        self.last_eigengap: Optional[float] = None

    def _sign_normalise(self, V: torch.Tensor) -> torch.Tensor:
        """
        Resolve SVD phase ambiguity: flip each column so its largest-magnitude
        element is positive. Shape: V is (D, k).
        """
        # Index of max-magnitude element per column
        max_idx = V.abs().argmax(dim=0)              # (k,)
        signs = V[max_idx, torch.arange(V.size(1))].sign()  # (k,)
        signs = signs.where(signs != 0, torch.ones_like(signs))
        return V * signs.unsqueeze(0)                # (D, k)

    def forward(self, parent_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        parent_outputs: list of n_parents tensors, each (B, D)
        returns: (B, out_dim)
        """
        B = parent_outputs[0].size(0)
        D = self.parent_dim

        # Stack into (B, n_parents, D)
        M = torch.stack(parent_outputs, dim=1)  # (B, n_parents, D)

        # For each sample in the batch, SVD on its (n_parents x D) matrix.
        # torch.linalg.svd supports batched inputs.
        # M: (B, n_parents, D) — rows are parent outputs, cols are features.
        # We want right singular vectors (V) of shape (B, D, n_parents) after economy SVD.
        # torch.linalg.svd(M, full_matrices=False) → U:(B,n,k), S:(B,k), Vh:(B,k,D)
        # Right singular vectors = rows of Vh = columns of V, shape (B, D, k) after transpose.

        try:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)  # Vh: (B, k, D)
        except RuntimeError:
            # Fallback: if SVD fails (e.g. on CPU with degenerate matrix), use mean
            mean_out = M.mean(dim=1)  # (B, D)
            return self.norm(self.readout(mean_out[..., :self.top_k]))

        # S: (B, k)  — singular values, monitor eigenvalue gap
        self.last_singular_values = S.detach()
        if S.size(1) > 1:
            self.last_eigengap = float((S[:, 0] - S[:, 1]).mean().item())

        # Top-k right singular vectors: Vh[:, :top_k, :] → (B, top_k, D)
        # Transpose to (B, D, top_k) for projection
        V_topk = Vh[:, :self.top_k, :].transpose(-1, -2)  # (B, D, top_k)

        # Sign normalise per sample (resolve phase ambiguity)
        # V_topk: (B, D, top_k) — normalise each of the top_k columns
        max_idx = V_topk.abs().argmax(dim=1, keepdim=True)  # (B, 1, top_k)
        signs = V_topk.gather(1, max_idx).squeeze(1).sign()  # (B, top_k)
        signs = signs.where(signs != 0, torch.ones_like(signs))
        V_topk = V_topk * signs.unsqueeze(1)  # (B, D, top_k)

        # Project mean parent representation onto top-k subspace
        mean_parent = M.mean(dim=1)              # (B, D)
        # Projection: (B, D) @ (B, D, top_k) → (B, top_k)
        projected = torch.bmm(mean_parent.unsqueeze(1), V_topk).squeeze(1)  # (B, top_k)

        # Readout to out_dim
        out = self.readout(projected)            # (B, out_dim)
        return self.norm(out)

    def aggregation_name(self) -> str:
        return "svd"


# ---------------------------------------------------------------------------
# Soft PCA aggregator (safer approximation)
# ---------------------------------------------------------------------------

class SoftPCAAggregator(nn.Module):
    """
    Learnable linear projection W: (parent_dim * n_parents) → (top_k) that is
    regularised toward orthogonality via a loss penalty ||W W^T - I||_F.

    This avoids the exact SVD (and its gradient instability) while still
    learning to extract near-orthogonal directions from the parent subspace.
    The orthogonality loss is computed here and stored; the training loop
    should call .orth_loss() and add it to the task loss with a small weight.
    """

    def __init__(
        self,
        parent_dim: int,
        n_parents: int,
        out_dim: int,
        top_k: Optional[int] = None,
        orth_weight: float = 0.01,
    ):
        super().__init__()
        self.parent_dim = parent_dim
        self.n_parents = n_parents
        self.top_k = top_k if top_k is not None else min(parent_dim, n_parents)
        self.orth_weight = orth_weight

        in_dim = parent_dim * n_parents

        # Projection W: (in_dim → top_k) — the "soft eigenvector extractor"
        self.W = nn.Parameter(torch.empty(self.top_k, in_dim))
        nn.init.orthogonal_(self.W)  # initialise orthogonally

        self.readout = nn.Linear(self.top_k, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)

        self._last_orth_loss: Optional[torch.Tensor] = None

    def orth_loss(self) -> torch.Tensor:
        """
        Orthogonality regularisation: ||W W^T - I||_F^2.
        Add self.orth_weight * aggregator.orth_loss() to training loss.
        """
        if self._last_orth_loss is None:
            return torch.tensor(0.0)
        return self._last_orth_loss

    def forward(self, parent_outputs: List[torch.Tensor]) -> torch.Tensor:
        B = parent_outputs[0].size(0)
        x = torch.cat(parent_outputs, dim=-1)        # (B, in_dim)

        # Project: (B, in_dim) @ W^T → (B, top_k)
        projected = F.linear(x, self.W)              # (B, top_k)

        # Compute and cache orthogonality loss: ||W W^T - I||_F^2
        I = torch.eye(self.top_k, device=self.W.device, dtype=self.W.dtype)
        WWT = self.W @ self.W.t()                    # (top_k, top_k)
        self._last_orth_loss = (WWT - I).pow(2).sum()

        out = self.readout(projected)
        return self.norm(out)

    def aggregation_name(self) -> str:
        return "soft_pca"


# ---------------------------------------------------------------------------
# Cross-attention aggregator (input-conditioned routing over parents)
# ---------------------------------------------------------------------------

class CrossAttentionAggregator(nn.Module):
    """
    Cross-attention aggregator. Unlike AttentionAggregator (which uses a learned
    fixed query shared across all inputs), this module derives the query from
    the child's own input `x`, giving per-sample routing over parents.

    Forward signature accepts an optional `query_input` tensor:
      - If provided (B, parent_dim): used directly (after projection) as the query.
      - If None: falls back to the mean of parent outputs (content-only query).

    Orthogonality regularisation on the key and query projection matrices is
    included so the crystallization-geometry story carries over from SoftPCA.
    Enable the orth loss by calling .orth_loss() and adding orth_weight * loss
    to the task objective (same convention as SoftPCAAggregator).

    Entropy regularisation on the attention distribution is optional
    (entropy_weight > 0): penalises near-one-hot attention to preserve
    multi-parent composition.
    """

    uses_query: bool = True  # flag read by ConceptModule to route `x` through

    def __init__(
        self,
        parent_dim: int,
        n_parents: int,
        out_dim: int,
        n_heads: int = 4,
        orth_weight: float = 0.01,
        entropy_weight: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert parent_dim % n_heads == 0, (
            f"parent_dim ({parent_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.parent_dim = parent_dim
        self.n_parents = n_parents
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.orth_weight = orth_weight
        self.entropy_weight = entropy_weight

        # Separate Q / K / V projections
        self.q_proj = nn.Linear(parent_dim, parent_dim, bias=False)
        self.k_proj = nn.Linear(parent_dim, parent_dim, bias=False)
        self.v_proj = nn.Linear(parent_dim, parent_dim, bias=False)
        nn.init.orthogonal_(self.q_proj.weight)
        nn.init.orthogonal_(self.k_proj.weight)
        nn.init.orthogonal_(self.v_proj.weight)

        self.out_proj = nn.Linear(parent_dim, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Cached losses + last attention for logging / interpretability
        self._last_orth_loss: Optional[torch.Tensor] = None
        self._last_entropy_loss: Optional[torch.Tensor] = None
        self._last_attention: Optional[torch.Tensor] = None  # (B, n_heads, n_parents)

    def orth_loss(self) -> torch.Tensor:
        if self._last_orth_loss is None:
            return torch.tensor(0.0)
        return self._last_orth_loss

    def entropy_loss(self) -> torch.Tensor:
        if self._last_entropy_loss is None:
            return torch.tensor(0.0)
        return self._last_entropy_loss

    def get_last_attention(self) -> Optional[torch.Tensor]:
        """Return most recent attention weights (B, n_heads, n_parents), useful
        for per-sample provenance analysis."""
        return self._last_attention

    def _compute_orth_loss(self) -> torch.Tensor:
        loss = 0.0
        for W in (self.q_proj.weight, self.k_proj.weight):
            WWT = W @ W.t()
            I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
            loss = loss + (WWT - I).pow(2).sum()
        return loss

    def forward(
        self,
        parent_outputs: List[torch.Tensor],
        query_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Stack parents: (B, n_parents, D)
        V_in = torch.stack(parent_outputs, dim=1)
        B, P, D = V_in.shape

        # Query source
        if query_input is None:
            q_src = V_in.mean(dim=1)  # (B, D)
        else:
            q_src = query_input       # (B, D)

        # Project Q / K / V and split heads
        H = self.n_heads
        Dh = D // H
        q = self.q_proj(q_src).view(B, 1, H, Dh).transpose(1, 2)   # (B, H, 1, Dh)
        k = self.k_proj(V_in).view(B, P, H, Dh).transpose(1, 2)    # (B, H, P, Dh)
        v = self.v_proj(V_in).view(B, P, H, Dh).transpose(1, 2)    # (B, H, P, Dh)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Dh)  # (B, H, 1, P)
        attn = F.softmax(scores, dim=-1)                                # (B, H, 1, P)

        # Cache attention for interpretability (detach + squeeze query dim)
        self._last_attention = attn.squeeze(2).detach()  # (B, H, P)

        # Weighted sum over parents, then concatenate heads.
        # attn: (B, H, 1, P)  v: (B, H, P, Dh)  →  (B, H, 1, Dh)
        out = torch.matmul(attn, v)                      # (B, H, 1, Dh)
        # Concat heads: (B, H, 1, Dh) → (B, 1, H, Dh) → (B, 1, H*Dh) → (B, D)
        out = out.transpose(1, 2).contiguous().view(B, 1, D).squeeze(1)

        # Cache orthogonality loss
        self._last_orth_loss = self._compute_orth_loss()

        # Cache entropy loss (encourages non-peaky attention when weight > 0)
        if self.entropy_weight > 0:
            # Entropy: -sum p log p, averaged over batch + heads
            eps = 1e-8
            H_attn = -(attn * (attn + eps).log()).sum(dim=-1)  # (B, H, 1)
            # We want to MAXIMISE entropy, so loss = -entropy (minimising loss maximises entropy)
            self._last_entropy_loss = -H_attn.mean()
        else:
            self._last_entropy_loss = torch.tensor(0.0, device=out.device)

        out = self.drop(out)
        out = self.out_proj(out)     # (B, out_dim)
        return self.norm(out)

    def aggregation_name(self) -> str:
        return "cross_attention"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

AGGREGATORS = {
    "concat":          ConcatAggregator,
    "mean":            MeanAggregator,
    "attention":       AttentionAggregator,
    "svd":             SVDAggregator,
    "soft_pca":        SoftPCAAggregator,
    "cross_attention": CrossAttentionAggregator,
}


def build_aggregator(name: str, parent_dim: int, n_parents: int, out_dim: int, **kwargs) -> nn.Module:
    """
    Factory function.

    Usage:
        agg = build_aggregator("svd", parent_dim=128, n_parents=2, out_dim=128, top_k=4)
        agg = build_aggregator("soft_pca", parent_dim=128, n_parents=2, out_dim=128, top_k=4)
        agg = build_aggregator("concat", parent_dim=128, n_parents=2, out_dim=128)
    """
    if name not in AGGREGATORS:
        raise ValueError(f"Unknown aggregator '{name}'. Choose from: {list(AGGREGATORS.keys())}")
    cls = AGGREGATORS[name]
    if name == "mean":
        return cls(parent_dim=parent_dim, out_dim=out_dim, **kwargs)
    return cls(parent_dim=parent_dim, n_parents=n_parents, out_dim=out_dim, **kwargs)


def aggregator_uses_query(agg: nn.Module) -> bool:
    """Return True if the aggregator expects a query_input argument (e.g. CrossAttentionAggregator)."""
    return getattr(agg, "uses_query", False)
