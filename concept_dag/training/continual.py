"""
Continual Learning harness.

Wraps the data pipeline, task sequence, and model evaluation into a clean
interface used by all three experiments. Supports:
  - Sequential task presentation (no replay, no task boundary info at test time for final eval)
  - Per-task train/val/test loaders
  - Integration with CLMetricsTracker
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from ..utils.metrics import CLMetricsTracker, evaluate


class TaskSequence:
    """
    Holds the sequence of tasks and their data loaders.
    Each task has: train_loader, val_loader, test_loader, n_classes.
    """

    def __init__(self):
        self._tasks: List[dict] = []

    def add_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        n_classes: int,
        name: str = "",
    ):
        self._tasks.append({
            "task_id": task_id,
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
            "n_classes": n_classes,
            "name": name or f"Task_{task_id}",
        })

    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, idx: int) -> dict:
        return self._tasks[idx]

    def n_tasks(self) -> int:
        return len(self._tasks)


class CLExperiment:
    """
    Orchestrates a continual learning experiment.

    The user provides:
      - task_sequence: TaskSequence
      - train_fn: callable(task_info, model, ...) → trains model on this task
      - eval_fn: callable(task_info, model) → float accuracy
      - device: str

    After running, access:
      - self.metrics: CLMetricsTracker
      - self.per_task_history: list of training history dicts
    """

    def __init__(
        self,
        task_sequence: TaskSequence,
        device: str = "cpu",
    ):
        self.task_sequence = task_sequence
        self.device = device
        self.metrics = CLMetricsTracker(n_tasks=task_sequence.n_tasks())
        self.per_task_history: List[dict] = []

    def run(
        self,
        train_fn: Callable,   # train_fn(task_info: dict, task_idx: int) → history dict
        model_fn_factory: Callable,  # model_fn_factory(task_idx: int) → callable(x) → logits
        evaluate_after_each_task: bool = True,
    ):
        """
        Run the full continual learning sequence.

        For each task t in order:
          1. Call train_fn(task_info, t) to train the model.
          2. Evaluate on all tasks 0..t and record in metrics.
        """
        n_tasks = self.task_sequence.n_tasks()

        for t in range(n_tasks):
            task_info = self.task_sequence[t]
            print(f"\n{'='*60}")
            print(f"Task {t}: {task_info['name']}")
            print(f"{'='*60}")

            # Train
            history = train_fn(task_info, t)
            self.per_task_history.append(history)

            if not evaluate_after_each_task:
                continue

            # Evaluate on all tasks seen so far
            for j in range(t + 1):
                eval_task = self.task_sequence[j]
                model_fn = model_fn_factory(j)
                acc = evaluate(model_fn, eval_task["test"], device=self.device)
                self.metrics.record(train_task=t, eval_task=j, accuracy=acc)
                print(f"  Eval task {j} ({eval_task['name']}): acc={acc:.4f}")

        print(f"\n{self.metrics.summary()}")

    def final_evaluation(self, model_fn_factory: Callable) -> Dict[int, float]:
        """Run final evaluation on all task test sets."""
        n_tasks = self.task_sequence.n_tasks()
        results = {}
        for j in range(n_tasks):
            eval_task = self.task_sequence[j]
            model_fn = model_fn_factory(j)
            acc = evaluate(model_fn, eval_task["test"], device=self.device)
            results[j] = acc
        return results


# ---------------------------------------------------------------------------
# Simple baseline trainer (monolithic model, no DAG)
# ---------------------------------------------------------------------------

def train_monolithic(
    model: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    n_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    log_every: int = 5,
) -> Dict[str, List[float]]:
    """
    Standard SGD training for monolithic baselines (ResNet, MLP, etc).
    All parameters are trainable — no freezing.
    """
    model.to(device)
    head.to(device)
    model.train()
    head.train()

    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"loss": [], "accuracy": []}

    for epoch in range(1, n_epochs + 1):
        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feats = model(x)
            logits = head(feats)
            loss = nn.functional.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            acc = (logits.argmax(-1) == y).float().mean().item()
            epoch_loss += loss.item()
            epoch_acc += acc
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_acc / max(n_batches, 1)
        history["loss"].append(avg_loss)
        history["accuracy"].append(avg_acc)

        if epoch % log_every == 0 or epoch == 1:
            print(f"  [Monolithic | epoch {epoch:3d}/{n_epochs}] loss={avg_loss:.4f}  acc={avg_acc:.3f}")

    return history
