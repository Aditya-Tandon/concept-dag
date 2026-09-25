"""
Root module families for the H6 module-family ablation.

Every concept module in the DAG currently reads the DINOv2 **CLS token** and nothing
else; the 256 patch tokens the backbone computes are discarded. This module holds the
five root families that ablation compares, behind one interface so a single training
loop can drive all of them:

    family                 arm  input                      module
    ---------------------- ---- -------------------------- ---------------------------------
    "mlp_cls"              A    (B, 384) CLS               2-layer MLP (the current root)
    "mlp_cls_meanpool"     B    (B, 1+T, 384) tokens       CLS ⊕ mean patch tokens (768) → MLP
    "proj_uniform_pool"    C0   (B, 1+T, 384) tokens       Linear+LN, UNIFORM pooling → MLP
    "attn_pool"            C    (B, 1+T, 384) tokens       learned-query attention pool → MLP
    "self_attn"            D    (B, 1+T, 384) tokens       1 self-attn layer → pool → MLP
    "cnn"                  E    (B, 3, 32, 32) raw images  SmallCNN → same MLP (pre-DINO root)
    "mlp_meanpool"         --   (B, 1+T, 384) tokens       mean patch tokens only → MLP

B is the pre-registered information arm because it *contains* A (concatenation, not
replacement): B ⊇ A means B cannot lose to A for want of the CLS token, so M1 measures the
patch tokens' contribution and nothing else. C0 is the capacity-matched control for
attention — the same token projection, LayerNorm and MLP as C with the learned query
removed — so M2 (C − C0, D − C0) isolates attention rather than "tokens plus more
parameters". The patches-only "mlp_meanpool" is the superseded first-pass arm B; it is kept
runnable for continuity but is not part of the pre-registered set.

Every family returns a 128-d concept embedding, exposes ``out_dim`` and
``trainable_parameters()``, and — for the families whose body is a plain
:class:`ConceptModule` root — ``orth_loss()``, so the ablation's train loop is the
`train_node` recipe with no per-family branching beyond the input it is handed. (A root
ConceptModule has no aggregator, so that orth term is identically 0 for every family here;
it is kept so the recipe is literally the DAG's.)

A vs B changes only the input vector; C/D are compared to C0, which differs from them only
by the pooling rule; E is the substrate control, not a module comparison. No positional
encoding is added anywhere: DINO's tokens already carry position.

Nothing here is imported by the DAG/gate code path — this is a single-task ablation.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .concept_module import ConceptModule
from ..models.baselines import SmallCNN


#: The pre-registered arms, in the note's order (A, B, C0, C, D, E).
PREREGISTERED = ("mlp_cls", "mlp_cls_meanpool", "proj_uniform_pool", "attn_pool",
                 "self_attn", "cnn")

FAMILIES = ("mlp_cls", "mlp_cls_meanpool", "proj_uniform_pool", "attn_pool",
            "self_attn", "cnn", "mlp_meanpool")

#: Arm letters from the hypothesis note, for reporting.
ARM_OF = {"mlp_cls": "A", "mlp_cls_meanpool": "B", "proj_uniform_pool": "C0",
          "attn_pool": "C", "self_attn": "D", "cnn": "E", "mlp_meanpool": "B0"}

#: What each family expects as `x`. The ablation script uses this to pick a loader:
#: "cls" → (B, 384), "tokens" → (B, 1 + T, 384), "image" → (B, 3, 32, 32).
FAMILY_INPUT = {
    "mlp_cls":           "cls",
    "mlp_cls_meanpool":  "tokens",
    "proj_uniform_pool": "tokens",
    "attn_pool":         "tokens",
    "self_attn":         "tokens",
    "cnn":               "image",
    "mlp_meanpool":      "tokens",
}


def _concept_mlp(in_dim: int, concept_dim: int, module_id: str) -> ConceptModule:
    """The DAG's root body verbatim: 2-layer MLP + LayerNorm, no aggregator."""
    return ConceptModule(
        module_id  = module_id,
        in_dim     = in_dim,
        hidden_dim = concept_dim,
        out_dim    = concept_dim,
        n_layers   = 2,
        n_parents  = 0,
    )


class _RootBase(nn.Module):
    """Shared interface: `out_dim`, `trainable_parameters()`, `n_params`."""

    family: str = ""

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _check_tokens(x: torch.Tensor, feature_dim: int, family: str) -> None:
    if x.ndim != 3 or x.shape[-1] != feature_dim:
        raise ValueError(
            f"{family} expects token input (B, 1 + T, {feature_dim}); got {tuple(x.shape)}."
        )


# ---------------------------------------------------------------------------
# Arm A — CLS + MLP (the current root)
# ---------------------------------------------------------------------------

class MlpRoot(_RootBase):
    """(B, feature_dim) → 2-layer MLP → (B, concept_dim). The root the DAG uses today."""

    family = "mlp_cls"

    def __init__(self, feature_dim: int = 384, concept_dim: int = 128):
        super().__init__(concept_dim)
        self.feature_dim = feature_dim
        self.concept_module = _concept_mlp(feature_dim, concept_dim, "root_mlp_cls")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            # Tolerate a token tensor by taking the CLS slot, so a caller that only has
            # the token cache still gets exactly arm A's input.
            x = x[:, 0]
        if x.ndim != 2 or x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"mlp_cls expects (B, {self.feature_dim}); got {tuple(x.shape)}."
            )
        return self.concept_module(x)

    def orth_loss(self) -> torch.Tensor:
        return self.concept_module.orth_loss()


# ---------------------------------------------------------------------------
# Arm B — mean of the patch tokens + the same MLP
# ---------------------------------------------------------------------------

class MeanPoolMlpRoot(_RootBase):
    """(B, 1 + T, feature_dim) → mean over the T patch tokens (CLS excluded) → arm A's MLP.

    The information control for H6a: identical module, identical parameter count,
    only the input vector changes.
    """

    family = "mlp_meanpool"

    def __init__(self, feature_dim: int = 384, concept_dim: int = 128):
        super().__init__(concept_dim)
        self.feature_dim = feature_dim
        self.concept_module = _concept_mlp(feature_dim, concept_dim, "root_mlp_meanpool")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_tokens(x, self.feature_dim, "mlp_meanpool")
        if x.shape[1] < 2:
            raise ValueError("mlp_meanpool needs at least one patch token besides CLS.")
        return self.concept_module(x[:, 1:].mean(dim=1))

    def orth_loss(self) -> torch.Tensor:
        return self.concept_module.orth_loss()


class ClsMeanPoolMlpRoot(_RootBase):
    """Arm B — (B, 1 + T, feature_dim) → [CLS ⊕ mean of the T patch tokens] (2·feature_dim)
    → 2-layer MLP.

    The information arm. Concatenating rather than replacing makes B a strict superset of
    A's input, so M1 (B − A) cannot be negative for the trivial reason that the mean-pooled
    patch summary happens to be a worse single vector than the CLS token.
    """

    family = "mlp_cls_meanpool"

    def __init__(self, feature_dim: int = 384, concept_dim: int = 128):
        super().__init__(concept_dim)
        self.feature_dim = feature_dim
        self.concept_module = _concept_mlp(2 * feature_dim, concept_dim,
                                           "root_mlp_cls_meanpool")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_tokens(x, self.feature_dim, "mlp_cls_meanpool")
        if x.shape[1] < 2:
            raise ValueError("mlp_cls_meanpool needs at least one patch token besides CLS.")
        return self.concept_module(torch.cat([x[:, 0], x[:, 1:].mean(dim=1)], dim=-1))

    def orth_loss(self) -> torch.Tensor:
        return self.concept_module.orth_loss()


# ---------------------------------------------------------------------------
# Attention pooling — shared by arms C and D
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """Single-head learned-query attention pooling over a token sequence.

    K, V = Linear(in_dim → dim) over all 1 + T tokens; one learned query (1, dim);
    softmax(qKᵀ/√dim) V → LayerNorm. No positional encoding.
    """

    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.dim   = dim
        self.key   = nn.Linear(in_dim, dim)
        self.value = nn.Linear(in_dim, dim)
        self.query = nn.Parameter(torch.randn(1, dim) * dim ** -0.5)
        self.norm  = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        k = self.key(tokens)                                     # (B, S, dim)
        v = self.value(tokens)                                   # (B, S, dim)
        attn = (k @ self.query.t()).squeeze(-1) * self.dim ** -0.5   # (B, S)
        w = torch.softmax(attn, dim=-1).unsqueeze(1)             # (B, 1, S)
        pooled = (w @ v).squeeze(1)                              # (B, dim)
        return self.norm(pooled)


# ---------------------------------------------------------------------------
# Arm C — attention pooling over CLS + patch tokens
# ---------------------------------------------------------------------------

class AttnPoolRoot(_RootBase):
    """(B, 1 + T, feature_dim) → learned-query attention pooling → 2-layer MLP."""

    family = "attn_pool"

    def __init__(self, feature_dim: int = 384, concept_dim: int = 128):
        super().__init__(concept_dim)
        self.feature_dim = feature_dim
        self.pool = AttentionPool(feature_dim, concept_dim)
        self.concept_module = _concept_mlp(concept_dim, concept_dim, "root_attn_pool")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_tokens(x, self.feature_dim, "attn_pool")
        return self.concept_module(self.pool(x))


# ---------------------------------------------------------------------------
# Arm C0 — arm C with the attention removed (capacity-matched control)
# ---------------------------------------------------------------------------

class ProjUniformPoolRoot(_RootBase):
    """Arm C0 — (B, 1 + T, feature_dim) → Linear(feature_dim → concept_dim) per token →
    UNIFORM (mean) pooling over all 1 + T tokens → LayerNorm → the same 2-layer MLP.

    Arm C minus the learned query and key projection: same token projection, same
    LayerNorm, same MLP, same input. Whatever C or D buy over C0 is attention, not
    "tokens" and not "more parameters".
    """

    family = "proj_uniform_pool"

    def __init__(self, feature_dim: int = 384, concept_dim: int = 128):
        super().__init__(concept_dim)
        self.feature_dim = feature_dim
        self.value = nn.Linear(feature_dim, concept_dim)
        self.norm = nn.LayerNorm(concept_dim)
        self.concept_module = _concept_mlp(concept_dim, concept_dim, "root_proj_uniform")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_tokens(x, self.feature_dim, "proj_uniform_pool")
        pooled = self.norm(self.value(x).mean(dim=1))
        return self.concept_module(pooled)


# ---------------------------------------------------------------------------
# Arm D — one self-attention layer, then arm C's pooling
# ---------------------------------------------------------------------------

class SelfAttnRoot(_RootBase):
    """(B, 1 + T, feature_dim) → Linear(feature_dim → concept_dim) → one
    TransformerEncoderLayer (4 heads, FFN 256, pre-norm) → arm C's pooling → MLP."""

    family = "self_attn"

    def __init__(self, feature_dim: int = 384, concept_dim: int = 128,
                 n_heads: int = 4, ff_dim: int = 256):
        super().__init__(concept_dim)
        self.feature_dim = feature_dim
        self.proj = nn.Linear(feature_dim, concept_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=concept_dim, nhead=n_heads, dim_feedforward=ff_dim,
            batch_first=True, norm_first=True,
        )
        self.pool = AttentionPool(concept_dim, concept_dim)
        self.concept_module = _concept_mlp(concept_dim, concept_dim, "root_self_attn")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_tokens(x, self.feature_dim, "self_attn")
        h = self.encoder_layer(self.proj(x))     # (B, S, concept_dim)
        return self.concept_module(self.pool(h))


# ---------------------------------------------------------------------------
# Arm E — the pre-DINO SmallCNN substrate control
# ---------------------------------------------------------------------------

class CnnRoot(_RootBase):
    """(B, 3, 32, 32) → SmallCNN → the same 2-layer MLP.

    Composition-identical to a root `DAGNode(use_cnn=True)`: the same
    `SmallCNN(in_channels=3, out_dim=cnn_out_dim)` feeding the same
    `ConceptModule(in_dim=cnn_out_dim, hidden=out=concept_dim, n_layers=2, n_parents=0)`,
    in the same order. Reimplemented here rather than importing `DAGNode` so this
    ablation module stays independent of the experiment package (and of the DAG's
    routing/parent machinery, none of which a single-task arm uses);
    `tests/test_module_family.py` asserts parameter-count parity with a real `DAGNode`.
    """

    family = "cnn"

    def __init__(self, concept_dim: int = 128, cnn_out_dim: int = 256,
                 in_channels: int = 3):
        super().__init__(concept_dim)
        self.cnn = SmallCNN(in_channels=in_channels, out_dim=cnn_out_dim)
        self.concept_module = _concept_mlp(cnn_out_dim, concept_dim, "root_cnn")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"cnn root expects (B, C, H, W) images; got {tuple(x.shape)}.")
        return self.concept_module(self.cnn(x))

    def orth_loss(self) -> torch.Tensor:
        return self.concept_module.orth_loss()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_root(family: str, feature_dim: int = 384, concept_dim: int = 128) -> nn.Module:
    """Instantiate one root family by name. `feature_dim` is ignored by "cnn"."""
    family = family.lower()
    if family == "mlp_cls":
        return MlpRoot(feature_dim=feature_dim, concept_dim=concept_dim)
    if family == "mlp_meanpool":
        return MeanPoolMlpRoot(feature_dim=feature_dim, concept_dim=concept_dim)
    if family == "mlp_cls_meanpool":
        return ClsMeanPoolMlpRoot(feature_dim=feature_dim, concept_dim=concept_dim)
    if family == "proj_uniform_pool":
        return ProjUniformPoolRoot(feature_dim=feature_dim, concept_dim=concept_dim)
    if family == "attn_pool":
        return AttnPoolRoot(feature_dim=feature_dim, concept_dim=concept_dim)
    if family == "self_attn":
        return SelfAttnRoot(feature_dim=feature_dim, concept_dim=concept_dim)
    if family == "cnn":
        return CnnRoot(concept_dim=concept_dim)
    raise ValueError(f"Unknown root family '{family}'. Known: {list(FAMILIES)}")


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
