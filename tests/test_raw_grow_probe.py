"""
Smoke tests for the raw-input (pure-root) grow probe.

Runs on CPU with tiny synthetic data. The gate's legacy grow probe consumes only the frozen parent
stack, so it cannot certify an obstruction on structure absent from the parents (the 5-Datasets SVHN
failure: grow and reuse fail together, rel_improvement < 0, and the gate reuses at chance accuracy).
Raw-root mode gives the grow probe a real grown root's view — the raw encoder features. These tests
pin both sides of that fix (the legacy miss itself is pinned empirically by the headline run, not
synthetically — a synthetic parent either saturates or under-fits by optimisation accident):

  1. NOVEL DOMAIN (SVHN analogue): the parent embedding exposes only a coarse split of the label;
     the raw features carry the rest. The raw-root probe must GROW with positive rel_improvement.
  2. COVERED TASK (MNIST-dup analogue): the parent embedding already linearly encodes the labels.
     The raw-root probe must still REUSE — raw access must not make the gate grow-happy on tasks
     existing concepts solve (grow's extra reduction over reuse is a tiny fraction of the reducible
     information).
  3. The three-way gate honours raw-root mode the same way.
"""

import torch

from concept_dag.modules.concept_module import ConceptModule
from concept_dag.training.kan_gate import (
    classification_task, decide_reuse_vs_grow, decide_reuse_search_grow,
)

torch.manual_seed(0)
DIM = 16       # concept_dim
RAW_DIM = 24   # encoder feature dim (≠ concept_dim, so mixing them up fails loudly)
N = 900


def _child_probe_factory(n_parents):
    def factory():
        return ConceptModule(module_id="__probe__", in_dim=DIM, hidden_dim=64, out_dim=DIM,
                             n_layers=2, n_parents=n_parents, aggregation="mean", dropout=0.0)
    return factory


def _root_probe_factory():
    return ConceptModule(module_id="__root_probe__", in_dim=RAW_DIM, hidden_dim=DIM, out_dim=DIM,
                         n_layers=2, n_parents=0, dropout=0.0)


def _novel_domain_case():
    """Raw features decide the 4-way label; the frozen parent embedding only exposes a COARSE
    2-way split of it (the SVHN-vs-MNIST-parent analogue: partial information, hard ceiling).
    Reuse and a parents-only grow probe both plateau at the same ~1-bit residual, so the legacy
    gate sees no obstruction; only raw access reveals the remaining reducible structure."""
    raw = torch.randn(N, RAW_DIM)
    w = torch.randn(RAW_DIM, 4)
    y = (raw @ w).argmax(dim=1)
    lift = torch.randn(2, DIM)
    parent = torch.nn.functional.one_hot(y // 2, 2).float() @ lift + 0.1 * torch.randn(N, DIM)
    return parent.unsqueeze(1), raw, y


def _covered_task_case():
    """The parent embedding linearly encodes the label; raw features carry the same information."""
    raw = torch.randn(N, RAW_DIM)
    w = torch.randn(RAW_DIM, 4)
    y = (raw @ w).argmax(dim=1)
    # Parent = a frozen concept whose embedding exposes the label directionally.
    lift = torch.randn(4, DIM)
    parent = torch.nn.functional.one_hot(y, 4).float() @ lift + 0.1 * torch.randn(N, DIM)
    return parent.unsqueeze(1), raw, y


def test_raw_root_probe_grows_on_novel_domain():
    parent_stack, raw, y = _novel_domain_case()
    rec = decide_reuse_vs_grow(
        parent_stack, y, _child_probe_factory(1), classification_task(4),
        concept_dim=DIM, n_parents=1, n_epochs=25, eps_rel=0.05,
        raw_stack=raw, root_module_factory=_root_probe_factory,
    )
    assert rec.grow_probe_input == "raw-root"
    assert rec.decision == "grow"
    assert rec.rel_improvement > 0.05


def test_raw_root_probe_still_reuses_on_covered_task():
    parent_stack, raw, y = _covered_task_case()
    rec = decide_reuse_vs_grow(
        parent_stack, y, _child_probe_factory(1), classification_task(4),
        concept_dim=DIM, n_parents=1, n_epochs=8, eps_rel=0.05,
        raw_stack=raw, root_module_factory=_root_probe_factory,
    )
    assert rec.grow_probe_input == "raw-root"
    assert rec.decision == "reuse"          # no grow-heavy bias on already-covered structure


def test_three_way_gate_raw_root_mode():
    parent_stack, raw, y = _novel_domain_case()
    rec = decide_reuse_search_grow(
        parent_stack, y, _child_probe_factory(1), classification_task(4),
        concept_dim=DIM, n_parents=1, n_epochs=25, eps_grow=0.05, eps_search=0.05,
        search_budget=2, raw_stack=raw, root_module_factory=_root_probe_factory,
    )
    assert rec.grow_probe_input == "raw-root"
    # Search over an uninformative parent cannot close the gap; only the raw-root concept can.
    assert rec.decision == "grow"
    assert rec.rel_grow > 0.05


# ---------------------------------------------------------------------------
# Ladder monotonicity: L_search must never exceed L_reuse
# ---------------------------------------------------------------------------


def test_search_never_scores_worse_than_reuse():
    """rel_search >= 0 — the reuse -> search -> grow ladder must be monotone.

    The rank-16 SearchComposer bottleneck is NARROWER than ReuseComposer's full-rank linear map,
    so before the fix a linearly-solvable task made L_search > L_reuse and rel_search went negative
    (measured on NGT: rel_search in [-0.107, +0.017] across every 5-Datasets task, Search never
    fired). search_compose now seeds the search with the trivial composition (plain linear
    recombination), so reuse is a member of the search space rather than its competitor.
    """
    torch.manual_seed(0)
    X = torch.randn(N, 2, DIM)
    w = torch.randn(2 * DIM, 4)
    y = (X.flatten(1) @ w).argmax(dim=1)      # purely linear in the parents: reuse is optimal
    rec = decide_reuse_search_grow(
        X, y, _child_probe_factory(2), classification_task(4),
        concept_dim=DIM, n_parents=2, n_epochs=8, eps_grow=0.05, eps_search=0.05,
        search_budget=3,
    )
    assert rec.rel_search >= 0.0
    assert rec.search_meta.get("trivial") is True   # search found nothing better than reuse
    assert rec.L_search_bits <= rec.L_reuse_bits + 1e-9
    # (the grow/reuse verdict is incidental here — at this toy width a fresh concept can legitimately
    # beat linear reuse; what this test pins is that the SEARCH rung never scores worse than reuse.)


def test_search_does_not_reset_global_rng():
    """search_compose must not touch the global RNG: two runs under different global seeds must
    leave the generator in different states afterwards (before the fix, torch.manual_seed(cand)
    inside the loop made everything downstream of the first Search call seed-independent)."""
    from concept_dag.training.kan_gate import search_compose
    X = torch.randn(200, 2, DIM); y = torch.randint(0, 4, (200,))
    outs = []
    for g in (1, 2):
        torch.manual_seed(g)
        search_compose(X[:140], y[:140], X[140:], y[140:], classification_task(4),
                       concept_dim=DIM, n_parents=2, device="cpu", n_epochs=1, lr=1e-3, budget=2)
        outs.append(torch.rand(1).item())
    assert outs[0] != outs[1]
