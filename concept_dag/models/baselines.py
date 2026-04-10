"""
Baseline models for comparison against the Concept DAG.

  - SimpleMLP:            Small MLP as a lightweight encoder for toy experiments.
  - SmallCNN:             Shallow CNN for CIFAR-scale inputs.
  - ProgressiveNetwork:   Progressive Neural Network (PNN) — one column per task,
                          with lateral connections from all previous columns.
                          Prevents forgetting by design (frozen old columns).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Encoder backbones
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    """Flat MLP encoder. Used for toy experiments (Phase 1)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        dim = in_dim
        for i in range(n_layers):
            out = out_dim if i == n_layers - 1 else hidden_dim
            layers.append(nn.Linear(dim, out))
            if i < n_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            dim = out
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        return self.norm(self.net(x))


class SmallCNN(nn.Module):
    """
    3-block convolutional encoder for 32x32 CIFAR-scale inputs.
    Outputs a flat feature vector of size out_dim.
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32x32 → 16x16
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 16x16 → 8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: 8x8 → 4x4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # → (B, 256, 1, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class LinearHead(nn.Module):
    """Simple linear classification head. One per task."""

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Progressive Neural Network (PNN)
# ---------------------------------------------------------------------------


class PNNColumn(nn.Module):
    """
    A single column of the PNN — one column per task.
    Receives: its own previous layer activations + lateral inputs from all prior columns.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        n_lateral_inputs: int = 0,
        lateral_dim: int = 0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_lateral_inputs = n_lateral_inputs

        # Main layers
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        for i in range(n_layers):
            total_in = dims[i] + n_lateral_inputs * lateral_dim if i > 0 else dims[i]
            self.layers.append(nn.Linear(total_in, dims[i + 1]))
            self.layer_norms.append(nn.LayerNorm(dims[i + 1]))

        # Lateral adapters (one per prior column, one per layer)
        # lateral_adapters[layer_idx][col_idx] : (lateral_dim → lateral_dim)
        if n_lateral_inputs > 0:
            self.lateral_adapters = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            nn.Linear(lateral_dim, lateral_dim)
                            for _ in range(n_lateral_inputs)
                        ]
                    )
                    for _ in range(n_layers - 1)  # only hidden layers get laterals
                ]
            )
        else:
            self.lateral_adapters = nn.ModuleList()

        self.out_dim = out_dim

    def forward(
        self,
        x: torch.Tensor,
        lateral_activations: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x:                   (B, in_dim) input.
            lateral_activations: List[List[Tensor]] — outer list = layer index,
                                 inner list = activations from each prior column at that layer.
        Returns:
            (output, layer_activations)
        """
        h = x
        layer_outputs = []

        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            if (
                i > 0
                and lateral_activations
                and self.n_lateral_inputs > 0
                and len(self.lateral_adapters) > 0
            ):
                lat_layer_idx = i - 1
                if lat_layer_idx < len(self.lateral_adapters):
                    adapted = []
                    for col_idx, lat_act in enumerate(
                        lateral_activations[lat_layer_idx]
                    ):
                        adapter = self.lateral_adapters[lat_layer_idx][col_idx]
                        adapted.append(F.relu(adapter(lat_act)))
                    h = torch.cat([h] + adapted, dim=-1)

            h = F.gelu(norm(layer(h))) if i < self.n_layers - 1 else norm(layer(h))
            layer_outputs.append(h)

        return h, layer_outputs


class ProgressiveNeuralNetwork(nn.Module):
    """
    Progressive Neural Network (Rusu et al., 2016).
    One column per task, lateral connections from all previous columns at each layer.
    Previous columns are frozen when a new task starts.

    This is Baseline B in the experiments.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        self.columns: nn.ModuleList = nn.ModuleList()
        self.heads: nn.ModuleList = nn.ModuleList()
        self._n_tasks = 0

    def add_task(self, n_classes: int) -> int:
        """Add a new column for a new task. Returns the task index."""
        n_prev = len(self.columns)
        col = PNNColumn(
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            n_layers=self.n_layers,
            n_lateral_inputs=n_prev,
            lateral_dim=self.hidden_dim,
        )
        head = LinearHead(self.out_dim, n_classes)

        # Freeze all previous columns
        for prev_col in self.columns:
            for p in prev_col.parameters():
                p.requires_grad_(False)

        self.columns.append(col)
        self.heads.append(head)
        self._n_tasks += 1
        return self._n_tasks - 1

    def forward_task(
        self,
        x: torch.Tensor,
        task_idx: int,
    ) -> torch.Tensor:
        """Forward pass for task_idx, return logits."""
        if x.dim() > 2:
            x = x.flatten(1)

        # Collect activations from all previous columns
        all_layer_activations: List[List[torch.Tensor]] = [
            [] for _ in range(self.n_layers)
        ]

        for col_idx, col in enumerate(self.columns[:task_idx]):
            with torch.no_grad():
                _, layer_outs = col.forward(x, lateral_activations=None)
            for layer_idx, act in enumerate(layer_outs[:-1]):  # skip output layer
                all_layer_activations[layer_idx].append(act)

        col = self.columns[task_idx]
        out, _ = col.forward(x, lateral_activations=all_layer_activations)
        return self.heads[task_idx](out)

    def trainable_parameters_for_task(self, task_idx: int):
        """Return trainable parameters for the given task's column + head."""
        return list(self.columns[task_idx].parameters()) + list(
            self.heads[task_idx].parameters()
        )
