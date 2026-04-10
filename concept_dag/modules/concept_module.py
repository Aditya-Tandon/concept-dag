"""
ConceptModule — the fundamental node of the Concept DAG.

Each module:
  1. Receives aggregated parent outputs (or raw input if a root node).
  2. Passes them through a small MLP to learn a concept representation.
  3. Exposes its concept subspace (top-k singular vectors of its activation matrix)
     for use by Stage 1 routing in child modules.

Frozen state: after a concept is learned, its weights are frozen.
New child modules only train their own weights and their aggregation layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math

from .aggregation import build_aggregator


class ConceptModule(nn.Module):
    """
    A single concept node in the DAG.

    Architecture:
        [parent outputs] → Aggregator → MLP → concept embedding (D_out)

    Root modules (no parents) just run: [raw input] → MLP → concept embedding.

    Args:
        module_id:    Unique string ID for this concept node.
        in_dim:       Dimensionality of inputs (for root nodes) or parent outputs.
        hidden_dim:   Width of the MLP hidden layers.
        out_dim:      Dimensionality of the concept embedding.
        n_layers:     Depth of the MLP.
        n_parents:    Number of parent modules (0 = root node).
        aggregation:  Aggregation strategy ('concat', 'mean', 'attention', 'svd', 'soft_pca').
        agg_kwargs:   Extra kwargs forwarded to the aggregator (e.g. top_k for SVD).
        dropout:      Dropout rate in MLP.
    """

    def __init__(
        self,
        module_id: str,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        n_parents: int = 0,
        aggregation: str = "soft_pca",
        agg_kwargs: Optional[dict] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.module_id = module_id
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_parents = n_parents
        self.aggregation_name = aggregation
        self._frozen = False

        agg_kwargs = agg_kwargs or {}

        # Aggregation layer (None for root nodes)
        if n_parents > 0:
            self.aggregator = build_aggregator(
                aggregation,
                parent_dim=in_dim,
                n_parents=n_parents,
                out_dim=in_dim,  # aggregator outputs same dim as input, MLP maps further
                **agg_kwargs,
            )
        else:
            self.aggregator = None

        # MLP body
        layers = []
        current_dim = in_dim
        for i in range(n_layers):
            out = hidden_dim if i < n_layers - 1 else out_dim
            layers.append(nn.Linear(current_dim, out))
            if i < n_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            current_dim = out
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)

        # Activation buffer for subspace estimation (used by Stage 1 routing).
        # Filled by calling .update_subspace_buffer(x) during probe phase.
        self._activation_buffer: List[torch.Tensor] = []
        self._concept_subspace: Optional[torch.Tensor] = None  # (out_dim, top_k)
        self._collecting_subspace: bool = False  # explicit collection mode, bypasses frozen/train guards

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        parent_outputs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, in_dim) — raw input (used for root nodes or passed through
                            when no aggregator; for non-root nodes this is ignored if
                            parent_outputs are provided).
            parent_outputs: List of (B, in_dim) tensors from parent modules.
        Returns:
            (B, out_dim) concept embedding.
        """
        if self.aggregator is not None and parent_outputs is not None:
            h = self.aggregator(parent_outputs)  # (B, in_dim)
        else:
            h = x  # root node

        out = self.norm(self.mlp(h))  # (B, out_dim)

        if self._collecting_subspace:
            self._activation_buffer.append(out.detach().cpu())

        return out

    # -----------------------------------------------------------------------
    # Concept subspace (used by Stage 1 routing)
    # -----------------------------------------------------------------------

    def clear_activation_buffer(self):
        self._activation_buffer = []

    def compute_concept_subspace(self, top_k: int = 8) -> torch.Tensor:
        """
        Run SVD over buffered activations to compute the module's concept subspace.
        Call this after a probe forward pass over the dataset.

        Returns:
            V: (out_dim, top_k) matrix whose columns are the top-k right singular
               vectors of the activation matrix. This is the concept's "subspace basis".
        """
        if not self._activation_buffer:
            raise RuntimeError(
                f"Module {self.module_id}: activation buffer is empty. "
                "Run a probe forward pass first."
            )
        A = torch.cat(self._activation_buffer, dim=0)  # (N, out_dim)
        # Centre
        A = A - A.mean(dim=0, keepdim=True)
        # Economy SVD
        _, _, Vh = torch.linalg.svd(A, full_matrices=False)  # Vh: (min(N,D), D)
        V = Vh[:top_k, :].t()                                 # (D, top_k)
        self._concept_subspace = V
        self.clear_activation_buffer()
        return V

    def get_concept_subspace(self) -> Optional[torch.Tensor]:
        """Return cached concept subspace, or None if not yet computed."""
        return self._concept_subspace

    # -----------------------------------------------------------------------
    # Freezing / unfreezing
    # -----------------------------------------------------------------------

    def freeze(self):
        """Freeze all parameters. Called after a concept is learned."""
        for p in self.parameters():
            p.requires_grad_(False)
        self._frozen = True

    def unfreeze(self):
        """Unfreeze all parameters (use carefully — only for perturbation tests)."""
        for p in self.parameters():
            p.requires_grad_(True)
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    # -----------------------------------------------------------------------
    # Regularisation helpers
    # -----------------------------------------------------------------------

    def orth_loss(self) -> torch.Tensor:
        """
        Returns the orthogonality regularisation loss from the aggregator,
        if it is a SoftPCAAggregator. Otherwise returns 0.
        """
        if self.aggregator is not None and hasattr(self.aggregator, "orth_loss"):
            return self.aggregator.orth_loss()
        return torch.tensor(0.0)

    # -----------------------------------------------------------------------
    # Weight drift measurement (for Phase 3 / Test 3)
    # -----------------------------------------------------------------------

    def snapshot_weights(self) -> dict:
        """Return a snapshot of current parameter tensors (detached, cloned)."""
        return {name: p.detach().clone() for name, p in self.named_parameters()}

    @staticmethod
    def compute_weight_drift(snapshot: dict, module: "ConceptModule") -> float:
        """
        L2 parameter drift between a snapshot and current weights.
        Used in Test 3 to measure how much a module drifted after perturbation.
        """
        total = 0.0
        count = 0
        current = dict(module.named_parameters())
        for name, old_param in snapshot.items():
            if name in current:
                diff = (current[name].detach().cpu() - old_param.cpu()).norm().item()
                total += diff ** 2
                count += 1
        return math.sqrt(total / max(count, 1))

    def __repr__(self):
        frozen_str = " [FROZEN]" if self._frozen else ""
        parents_str = f" n_parents={self.n_parents} agg={self.aggregation_name}" if self.n_parents else " [ROOT]"
        return (
            f"ConceptModule(id={self.module_id}, "
            f"in={self.in_dim}, hidden={self.hidden_dim}, out={self.out_dim}"
            f"{parents_str}{frozen_str})"
        )
