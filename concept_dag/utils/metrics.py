"""
Continual learning metrics and general utilities.

Standard CL metrics (following Lopez-Paz & Ranzato, 2017):
  - Average Accuracy (AA): mean accuracy over all seen tasks after final task.
  - Backward Transfer (BT): how much performance on old tasks changes after training on new ones.
    BT < 0 means forgetting; BT > 0 means backward positive transfer (rare but possible).
  - Forward Transfer (FT): how much pre-training on old tasks helps on new tasks
    (compared to training from scratch on each new task).

Also includes:
  - Subspace orthogonality measurement (for Phase 1 / Test 2).
  - Gradient norm tracking (for SVD stability analysis).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Basic
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy for a batch."""
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def evaluate(
    model_fn,          # callable: (x: Tensor) → logits: Tensor
    loader,
    device: str = "cpu",
) -> float:
    """Evaluate a model function on a DataLoader, return accuracy."""
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model_fn(x)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Continual Learning metrics
# ---------------------------------------------------------------------------

class CLMetricsTracker:
    """
    Tracks performance across tasks for continual learning evaluation.

    Usage:
        tracker = CLMetricsTracker(n_tasks=5)

        # After training on task t, evaluate on all tasks 0..t:
        for task_id in range(t + 1):
            acc = evaluate(model, loaders[task_id], device)
            tracker.record(train_task=t, eval_task=task_id, accuracy=acc)

        # After all tasks:
        print(tracker.average_accuracy())
        print(tracker.backward_transfer())
        print(tracker.forward_transfer(baseline_accs))
    """

    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        # R[i][j] = accuracy on task j after training on task i
        self.R: Dict[Tuple[int, int], float] = {}

    def record(self, train_task: int, eval_task: int, accuracy: float):
        self.R[(train_task, eval_task)] = accuracy

    def average_accuracy(self) -> float:
        """
        AA = (1/T) * sum_{j=0}^{T-1} R[T-1][j]
        Accuracy on all tasks after training on the final task.
        """
        T = self.n_tasks
        accs = [self.R.get((T - 1, j), 0.0) for j in range(T)]
        return float(np.mean(accs))

    def backward_transfer(self) -> float:
        """
        BT = (1 / T(T-1)/2) * sum_{j < i} (R[T-1][j] - R[j][j])
        Measures forgetting: how much did performance on task j drop after training beyond it?
        Negative = forgetting. Zero = perfect retention.
        """
        T = self.n_tasks
        diffs = []
        for j in range(T - 1):
            r_final = self.R.get((T - 1, j), 0.0)
            r_when_trained = self.R.get((j, j), 0.0)
            diffs.append(r_final - r_when_trained)
        return float(np.mean(diffs)) if diffs else 0.0

    def forward_transfer(self, baseline_accs: Optional[List[float]] = None) -> float:
        """
        FT = (1 / T(T-1)/2) * sum_{i > j} (R[j-1][j] - b_j)
        Measures how much pre-training on prior tasks helps on new ones.
        b_j = accuracy on task j from a single-task baseline (train only on j from scratch).
        If baseline_accs is None, returns a relative measure (R[j-1][j] - first_task_acc).
        """
        T = self.n_tasks
        baseline = baseline_accs if baseline_accs is not None else [0.0] * T
        diffs = []
        for j in range(1, T):
            r_before_training = self.R.get((j - 1, j), 0.0)
            diffs.append(r_before_training - baseline[j])
        return float(np.mean(diffs)) if diffs else 0.0

    def forgetting_per_task(self) -> Dict[int, float]:
        """Return forgetting for each task individually."""
        T = self.n_tasks
        result = {}
        for j in range(T - 1):
            r_final = self.R.get((T - 1, j), 0.0)
            r_when_trained = self.R.get((j, j), 0.0)
            result[j] = r_final - r_when_trained
        return result

    def accuracy_matrix(self) -> np.ndarray:
        """Return the full T x T accuracy matrix for visualisation."""
        T = self.n_tasks
        mat = np.zeros((T, T))
        for i in range(T):
            for j in range(T):
                mat[i, j] = self.R.get((i, j), float("nan"))
        return mat

    def summary(self) -> str:
        lines = [
            f"CL Metrics ({self.n_tasks} tasks):",
            f"  Average Accuracy    : {self.average_accuracy():.4f}",
            f"  Backward Transfer   : {self.backward_transfer():.4f}",
            f"  Forgetting per task : {self.forgetting_per_task()}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subspace metrics (Phase 1 / Test 2)
# ---------------------------------------------------------------------------

def representation_orthogonality(
    activations: torch.Tensor,
    top_k: int = 8,
) -> float:
    """
    Measure how orthogonal the top-k principal directions of an activation matrix are.
    An ideal crystallised concept has k dominant directions that are perfectly orthogonal.

    Returns:
        Mean absolute off-diagonal element of V^T V where V = top-k right singular vectors.
        0.0 = perfectly orthogonal. Higher = more entangled.
    """
    A = activations - activations.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(A, full_matrices=False)
    V = Vh[:top_k, :]  # (top_k, D)
    G = V @ V.t()       # Gram matrix (top_k, top_k)
    I = torch.eye(top_k, device=G.device)
    off_diag = (G - I).abs()
    # Mask diagonal
    mask = ~torch.eye(top_k, dtype=torch.bool, device=G.device)
    return float(off_diag[mask].mean().item())


def variance_explained(
    activations: torch.Tensor,
    top_k: int = 8,
) -> float:
    """
    Fraction of total variance captured by the top-k principal components.
    Higher = more successfully crystallised (information compressed into few directions).
    """
    A = activations - activations.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(A, full_matrices=False)
    S2 = S.pow(2)
    return float((S2[:top_k].sum() / S2.sum()).item())


def principal_angles_between(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute principal angles between subspace A (D, k_a) and subspace B (D, k_b).

    Returns:
        angles: Tensor of principal angles in radians (length = min(k_a, k_b)).
    """
    Qa, _ = torch.linalg.qr(A)
    Qb, _ = torch.linalg.qr(B)
    M = Qa.t() @ Qb                       # (k_a, k_b)
    svals = torch.linalg.svdvals(M).clamp(0.0, 1.0)
    return torch.acos(svals)


# ---------------------------------------------------------------------------
# Gradient norm tracking (for SVD stability analysis)
# ---------------------------------------------------------------------------

class GradientNormTracker:
    """
    Register hooks on a module to track gradient norms during training.
    Used to detect SVD gradient explosions.
    """

    def __init__(self, module: nn.Module, name: str = ""):
        self.name = name
        self.grad_norms: List[float] = []
        self._hook = module.register_full_backward_hook(self._hook_fn)

    def _hook_fn(self, module, grad_input, grad_output):
        for g in grad_output:
            if g is not None:
                norm = g.norm().item()
                self.grad_norms.append(norm)

    def remove(self):
        self._hook.remove()

    def max_norm(self) -> float:
        return max(self.grad_norms) if self.grad_norms else 0.0

    def mean_norm(self) -> float:
        return float(np.mean(self.grad_norms)) if self.grad_norms else 0.0

    def had_explosion(self, threshold: float = 100.0) -> bool:
        return any(n > threshold for n in self.grad_norms)
