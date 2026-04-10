"""
ConceptDAG — the graph structure connecting ConceptModules.

Responsibilities:
  - Maintain the adjacency structure (parent → children).
  - Enforce the DAG invariant (no cycles) when adding new nodes.
  - Perform a topologically-sorted forward pass.
  - Provide an interface for Stage 1 routing: given a probe distribution,
    find the closest parent modules via principal angle similarity.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Tuple
import math

from .concept_module import ConceptModule


class ConceptDAG(nn.Module):
    """
    A growing DAG of ConceptModules.

    Nodes are ConceptModules identified by string IDs.
    Edges are directed: parent_id → child_id (parent is computed before child).

    Usage:
        dag = ConceptDAG()
        dag.add_module("root_a", module_a)       # root node
        dag.add_module("root_b", module_b)
        dag.add_module("child_1", module_c, parents=["root_a", "root_b"])
    """

    def __init__(self):
        super().__init__()
        # nn.ModuleDict so PyTorch tracks parameters
        self._modules_dict: nn.ModuleDict = nn.ModuleDict()
        # Adjacency: node → list of parent IDs
        self._parents: Dict[str, List[str]] = {}
        # Adjacency: node → list of child IDs
        self._children: Dict[str, List[str]] = {}
        # Insertion order (for stable topological sort)
        self._insertion_order: List[str] = []

    # -----------------------------------------------------------------------
    # Graph construction
    # -----------------------------------------------------------------------

    def add_module(
        self,
        module_id: str,
        module: ConceptModule,
        parents: Optional[List[str]] = None,
    ):
        """
        Add a ConceptModule to the DAG.

        Args:
            module_id: Unique string ID for this node.
            module:    The ConceptModule instance.
            parents:   List of parent module IDs. If None or empty, this is a root node.
        """
        if module_id in self._modules_dict:
            raise ValueError(f"Module '{module_id}' already exists in the DAG.")

        parents = parents or []

        # Validate parents exist
        for pid in parents:
            if pid not in self._modules_dict:
                raise ValueError(f"Parent '{pid}' not found in DAG. Add it first.")

        # Check would-be cycle (reachability from module_id to any ancestor)
        # Since module_id is new, the only cycle risk is if any ancestor can
        # reach module_id — impossible since it doesn't exist yet. But we
        # guard against duplicate edges just in case.
        if len(parents) != len(set(parents)):
            raise ValueError("Duplicate parents specified.")

        self._modules_dict[module_id] = module
        self._parents[module_id] = list(parents)
        self._children[module_id] = []
        self._insertion_order.append(module_id)

        # Register this node as a child of each parent
        for pid in parents:
            self._children[pid].append(module_id)

    def get_module(self, module_id: str) -> ConceptModule:
        return self._modules_dict[module_id]

    def all_module_ids(self) -> List[str]:
        return list(self._insertion_order)

    def root_ids(self) -> List[str]:
        return [mid for mid in self._insertion_order if not self._parents[mid]]

    def leaf_ids(self) -> List[str]:
        return [mid for mid in self._insertion_order if not self._children[mid]]

    def out_degree(self, module_id: str) -> int:
        return len(self._children[module_id])

    def in_degree(self, module_id: str) -> int:
        return len(self._parents[module_id])

    # -----------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # -----------------------------------------------------------------------

    def topological_order(self) -> List[str]:
        """Return all module IDs in topological order (parents before children)."""
        in_deg = {mid: len(self._parents[mid]) for mid in self._insertion_order}
        queue = [mid for mid, d in in_deg.items() if d == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self._children[node]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
        if len(order) != len(self._insertion_order):
            raise RuntimeError("DAG contains a cycle — this should never happen.")
        return order

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        active_nodes: Optional[List[str]] = None,
        return_all_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run a forward pass through the DAG.

        Args:
            x:                   (B, in_dim) — raw input fed into root nodes.
            active_nodes:        If provided, only compute outputs for these nodes
                                 (and their ancestors). Useful during Stage 2 when
                                 only the new child and its parents matter.
            return_all_embeddings: If True, return embeddings for every node.

        Returns:
            (leaf_embedding, node_embeddings)
            leaf_embedding: (B, out_dim) from the last leaf in topological order.
            node_embeddings: dict mapping module_id → (B, out_dim).
        """
        topo_order = self.topological_order()

        # Filter to active subgraph if specified
        if active_nodes is not None:
            needed = self._ancestors(active_nodes) | set(active_nodes)
            topo_order = [mid for mid in topo_order if mid in needed]

        embeddings: Dict[str, torch.Tensor] = {}

        for mid in topo_order:
            module: ConceptModule = self._modules_dict[mid]
            parent_ids = self._parents[mid]
            if parent_ids:
                parent_outs = [embeddings[pid] for pid in parent_ids]
                emb = module(x, parent_outputs=parent_outs)
            else:
                emb = module(x)
            embeddings[mid] = emb

        # Return the last node in topological order as the primary output
        last_emb = embeddings[topo_order[-1]]
        if not return_all_embeddings:
            return last_emb, {topo_order[-1]: last_emb}
        return last_emb, embeddings

    # -----------------------------------------------------------------------
    # Stage 1: Parent routing
    # -----------------------------------------------------------------------

    def principal_angle_similarity(
        self,
        query_subspace: torch.Tensor,
        candidate_id: str,
        top_k: int = 8,
    ) -> float:
        """
        Compute the similarity between a query subspace and a candidate module's
        concept subspace, using the sum of cosines of principal angles.

        Principal angles θ_i between subspaces A and B are defined by:
            cos(θ_i) = σ_i(A^T B)
        where σ_i are the singular values of A^T B.

        Higher score = more similar subspaces = better parent candidate.

        Args:
            query_subspace:  (D, k_q) — subspace of the new task's representations.
            candidate_id:    Module ID whose stored subspace to compare against.
            top_k:           Number of singular vectors to use from stored subspace.

        Returns:
            Scalar similarity score in [0, top_k].
        """
        module = self._modules_dict[candidate_id]
        candidate_subspace = module.get_concept_subspace()  # (D, k_c)
        if candidate_subspace is None:
            return 0.0

        # Orthonormalise both subspaces (they should already be, but guard for safety)
        Q_q, _ = torch.linalg.qr(query_subspace)
        Q_c, _ = torch.linalg.qr(candidate_subspace)

        # Cross-Gram matrix
        cross = Q_q.t() @ Q_c  # (k_q, k_c)
        svals = torch.linalg.svdvals(cross)  # singular values = cos(principal angles)
        # Clamp to [0, 1] for numerical safety
        svals = svals.clamp(0.0, 1.0)
        # Sum of cosines: higher is more similar
        return float(svals.sum().item())

    def find_best_parents(
        self,
        query_subspace: torch.Tensor,
        n_parents: int = 2,
        top_k: int = 8,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Stage 1 routing: find the n_parents existing modules whose concept
        subspace is most similar to the query subspace.

        Args:
            query_subspace:  (D, k) subspace of new task's representations.
            n_parents:       How many parents to select.
            top_k:           Number of singular vectors used for comparison.
            exclude_ids:     Module IDs to exclude (e.g. already selected parents).

        Returns:
            List of n_parents module IDs (best parents first).
        """
        exclude = set(exclude_ids or [])
        scores = {}
        for mid in self._insertion_order:
            if mid in exclude:
                continue
            module = self._modules_dict[mid]
            if module.get_concept_subspace() is None:
                continue  # skip modules without a computed subspace
            scores[mid] = self.principal_angle_similarity(query_subspace, mid, top_k)

        if not scores:
            raise RuntimeError(
                "No modules with computed subspaces found. "
                "Run probe passes to compute concept subspaces first."
            )

        sorted_ids = sorted(scores, key=lambda k: scores[k], reverse=True)
        selected = sorted_ids[:n_parents]
        return selected

    # -----------------------------------------------------------------------
    # Freezing helpers
    # -----------------------------------------------------------------------

    def freeze_all(self):
        """Freeze every module in the DAG."""
        for module in self._modules_dict.values():
            module.freeze()

    def freeze_except(self, active_ids: List[str]):
        """Freeze all modules EXCEPT the ones in active_ids."""
        for mid, module in self._modules_dict.items():
            if mid not in active_ids:
                module.freeze()
            else:
                module.unfreeze()

    # -----------------------------------------------------------------------
    # Topology analysis (for Test 3)
    # -----------------------------------------------------------------------

    def out_degree_map(self) -> Dict[str, int]:
        """Return dict mapping module_id → out_degree."""
        return {mid: len(self._children[mid]) for mid in self._insertion_order}

    def snapshot_all_weights(self) -> Dict[str, dict]:
        """Snapshot weights of all modules."""
        return {mid: self._modules_dict[mid].snapshot_weights() for mid in self._insertion_order}

    def compute_all_drifts(self, snapshots: Dict[str, dict]) -> Dict[str, float]:
        """Compute weight drift for all modules relative to a snapshot."""
        return {
            mid: ConceptModule.compute_weight_drift(snapshots[mid], self._modules_dict[mid])
            for mid in self._insertion_order
        }

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _ancestors(self, module_ids: List[str]) -> Set[str]:
        """Return the set of all ancestors of the given module IDs."""
        visited = set()
        queue = list(module_ids)
        while queue:
            node = queue.pop()
            for pid in self._parents.get(node, []):
                if pid not in visited:
                    visited.add(pid)
                    queue.append(pid)
        return visited

    def summary(self) -> str:
        lines = ["ConceptDAG summary:"]
        for mid in self.topological_order():
            m = self._modules_dict[mid]
            parents_str = ", ".join(self._parents[mid]) if self._parents[mid] else "ROOT"
            children_str = ", ".join(self._children[mid]) if self._children[mid] else "LEAF"
            frozen_str = "❄" if m.is_frozen else "🔥"
            lines.append(
                f"  {frozen_str} {mid:20s} | parents: [{parents_str:30s}] "
                f"| children: [{children_str}] | out_deg={self.out_degree(mid)}"
            )
        return "\n".join(lines)
