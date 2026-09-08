"""
Unit tests for the gate-estimator / denominator / update-arm additions to the Kan gate core
(`concept_dag/training/kan_gate.py`): the crossfit estimator, the two recorded denominators
(``reducible_grow`` vs ``reducible_best``) and which one decides (``reducible_mode``), the
score-set path of ``_held_out_codelength``, and the ``update_probe`` refinement arm.

Runs on CPU with tiny synthetic data, mirroring the style and fixtures of
``tests/test_raw_grow_probe.py`` (same DIM/RAW_DIM, same novel-domain construction, same
``_child_probe_factory`` / raw-root ``_root_probe_factory``).
"""

import copy
import math

import pytest
import torch

from concept_dag.modules.concept_module import ConceptModule
from concept_dag.training.kan_gate import (
    ReuseComposer,
    _held_out_codelength,
    classification_task,
    decide_reuse_search_grow,
    decide_reuse_vs_grow,
    pack_update_input,
    update_probe,
)

torch.manual_seed(0)
DIM = 16       # concept_dim
RAW_DIM = 24   # encoder feature dim (!= concept_dim, so mixing them up fails loudly)
N = 900


def _child_probe_factory(n_parents):
    def factory():
        return ConceptModule(module_id="__probe__", in_dim=DIM, hidden_dim=64, out_dim=DIM,
                             n_layers=2, n_parents=n_parents, aggregation="mean", dropout=0.0)
    return factory


def _root_probe_factory():
    return ConceptModule(module_id="__root_probe__", in_dim=RAW_DIM, hidden_dim=DIM, out_dim=DIM,
                         n_layers=2, n_parents=0, dropout=0.0)


def _novel_domain_case(n=N):
    """Same construction as test_raw_grow_probe.py: raw features decide a 4-way label; the frozen
    parent embedding only exposes a coarse 2-way split of it, so only raw access reveals the
    remaining reducible structure and the gate must grow."""
    raw = torch.randn(n, RAW_DIM)
    w = torch.randn(RAW_DIM, 4)
    y = (raw @ w).argmax(dim=1)
    lift = torch.randn(2, DIM)
    parent = torch.nn.functional.one_hot(y // 2, 2).float() @ lift + 0.1 * torch.randn(n, DIM)
    return parent.unsqueeze(1), raw, y


def _linear_ladder_case(n=N):
    """Purely linear-in-the-parents 4-class task (as in test_search_never_scores_worse_than_reuse):
    all three probes should comfortably beat the marginal null once trained."""
    X = torch.randn(n, 2, DIM)
    w = torch.randn(2 * DIM, 4)
    y = (X.flatten(1) @ w).argmax(dim=1)
    return X, y


# ---------------------------------------------------------------------------
# 1. crossfit estimator
# ---------------------------------------------------------------------------


def test_crossfit_estimator_grows_on_novel_domain():
    # crossfit trains each rung on a smaller per-fold slice (K=3 folds, held-out fold scored,
    # 20% of the remainder held for selection) than the "single" estimator's 70% train split, so
    # it needs more N/epochs than the n_epochs=25 used for "single" elsewhere in this repo's
    # test_raw_grow_probe.py to reach the same clean grow verdict; n_epochs=25 at N=900 was flaky
    # here (empirically landed on "reuse"). N=1200, n_epochs=40 grows reliably (~1s runtime).
    torch.manual_seed(0)
    parent_stack, raw, y = _novel_domain_case(n=1200)
    rec = decide_reuse_search_grow(
        parent_stack, y, _child_probe_factory(1), classification_task(4),
        concept_dim=DIM, n_parents=1, n_epochs=40, eps_grow=0.05, eps_search=0.05,
        search_budget=2, raw_stack=raw, root_module_factory=_root_probe_factory,
        estimator="crossfit", n_splits=3, split_generator=torch.Generator().manual_seed(0),
    )
    assert math.isfinite(rec.L_grow_bits)
    assert math.isfinite(rec.L_reuse_bits)
    assert math.isfinite(rec.L_search_bits)

    assert rec.estimator_meta["estimator"] == "crossfit"
    assert len(rec.estimator_meta["folds"]) == 3
    for key in ("se_reuse_minus_search", "se_search_minus_grow", "se_reuse_minus_grow"):
        assert rec.estimator_meta[key] >= 0.0

    tr = rec.split_meta["tr_idx"]
    val = rec.split_meta["val_idx"]
    assert len(set(tr) & set(val)) == 0
    assert set(tr) | set(val) == set(range(len(y)))

    assert rec.decision == "grow"
    assert rec.rel_grow > 0.05


def test_crossfit_selection_sets_independent_across_folds():
    """Should-fix (code review): ``sel = rest[:n_sel]`` used to be a positional prefix of
    ``torch.cat([folds[j] for j != k])`` — since ``rest`` is a fixed-order concatenation of the
    surviving folds, that prefix was (for K=5) the identical 64-of-400 examples of fold 0 in 4 of
    the 5 folds, silently correlating the crossfit "independent" selection subsets used by the
    per-fold SE. Fixed by shuffling ``rest`` (with the caller's ``split_generator``) before
    slicing off the selection set. This asserts each fold's selection subset is disjoint from its
    own score (held-out) subset, and that no two folds land on the identical selection subset."""
    torch.manual_seed(0)
    X, y = _linear_ladder_case(n=500)
    rec = decide_reuse_search_grow(
        X, y, _child_probe_factory(2), classification_task(4),
        concept_dim=DIM, n_parents=2, n_epochs=2, eps_grow=0.05, eps_search=0.05,
        search_budget=1, estimator="crossfit", n_splits=5,
        split_generator=torch.Generator().manual_seed(0),
    )
    folds = rec.estimator_meta["folds"]
    assert len(folds) == 5

    sel_sets = [set(f["sel_idx"]) for f in folds]
    score_sets = [set(f["score_idx"]) for f in folds]

    for sel, sc in zip(sel_sets, score_sets):
        assert sel.isdisjoint(sc)

    for i in range(len(sel_sets)):
        for j in range(i + 1, len(sel_sets)):
            assert sel_sets[i] != sel_sets[j], f"folds {i} and {j} share an identical selection set"


# ---------------------------------------------------------------------------
# 2. split determinism
# ---------------------------------------------------------------------------


def test_split_generator_is_deterministic_and_seed_sensitive():
    parent_stack, raw, y = _novel_domain_case()

    def _run(seed):
        gen = torch.Generator().manual_seed(seed)
        rec = decide_reuse_search_grow(
            parent_stack, y, _child_probe_factory(1), classification_task(4),
            concept_dim=DIM, n_parents=1, n_epochs=2, eps_grow=0.05, eps_search=0.05,
            search_budget=1, raw_stack=raw, root_module_factory=_root_probe_factory,
            estimator="crossfit", n_splits=3, split_generator=gen,
        )
        return rec.split_meta

    m1 = _run(0)
    m2 = _run(0)
    assert m1 == m2

    m3 = _run(1)
    assert m3["tr_idx"] != m1["tr_idx"]


# ---------------------------------------------------------------------------
# 3. both denominators recorded
# ---------------------------------------------------------------------------


def test_both_denominators_recorded_grow_mode():
    torch.manual_seed(0)
    parent_stack, raw, y = _novel_domain_case()
    rec = decide_reuse_search_grow(
        parent_stack, y, _child_probe_factory(1), classification_task(4),
        concept_dim=DIM, n_parents=1, n_epochs=15, eps_grow=0.05, eps_search=0.05,
        search_budget=2, raw_stack=raw, root_module_factory=_root_probe_factory,
        reducible_mode="grow",
    )
    for name in ("reducible_grow", "reducible_best", "rel_search_best", "rel_grow_best",
                 "L_null_bits", "n_rungs_above_null"):
        assert getattr(rec, name) is not None

    assert rec.reducible_best >= rec.reducible_grow - 1e-9
    assert abs(rec.rel_search_best) <= abs(rec.rel_search)


def test_bounded_fractions_when_every_rung_beats_null():
    torch.manual_seed(1)
    X, y = _linear_ladder_case()
    rec = decide_reuse_search_grow(
        X, y, _child_probe_factory(2), classification_task(4),
        concept_dim=DIM, n_parents=2, n_epochs=20, eps_grow=0.05, eps_search=0.05,
        search_budget=2, reducible_mode="grow",
    )
    assert rec.n_rungs_above_null == 0, (
        f"test fixture did not reach the all-rungs-beat-null regime: "
        f"L_null={rec.L_null_bits}, L_reuse={rec.L_reuse_bits}, "
        f"L_search={rec.L_search_bits}, L_grow={rec.L_grow_bits}"
    )
    assert abs(rec.rel_search_best) <= 1.0
    assert abs(rec.rel_grow_best) <= 1.0


# ---------------------------------------------------------------------------
# 4. reducible_mode="best" decides with the best-rung fractions; "bogus" raises
# ---------------------------------------------------------------------------


def test_reducible_mode_best_matches_best_fractions_three_way():
    torch.manual_seed(0)
    parent_stack, raw, y = _novel_domain_case()
    rec = decide_reuse_search_grow(
        parent_stack, y, _child_probe_factory(1), classification_task(4),
        concept_dim=DIM, n_parents=1, n_epochs=15, eps_grow=0.05, eps_search=0.05,
        search_budget=2, raw_stack=raw, root_module_factory=_root_probe_factory,
        reducible_mode="best",
    )
    assert rec.rel_search == rec.rel_search_best
    assert rec.rel_grow == rec.rel_grow_best


def test_reducible_mode_bogus_raises_three_way():
    parent_stack, raw, y = _novel_domain_case(n=100)
    with pytest.raises(ValueError):
        decide_reuse_search_grow(
            parent_stack, y, _child_probe_factory(1), classification_task(4),
            concept_dim=DIM, n_parents=1, n_epochs=1, eps_grow=0.05, eps_search=0.05,
            search_budget=1, raw_stack=raw, root_module_factory=_root_probe_factory,
            reducible_mode="bogus",
        )


def test_reducible_mode_best_matches_best_fraction_binary():
    torch.manual_seed(0)
    parent_stack, raw, y = _novel_domain_case()
    rec = decide_reuse_vs_grow(
        parent_stack, y, _child_probe_factory(1), classification_task(4),
        concept_dim=DIM, n_parents=1, n_epochs=15, eps_rel=0.05,
        raw_stack=raw, root_module_factory=_root_probe_factory,
        reducible_mode="best",
    )
    assert rec.rel_improvement == rec.rel_improvement_best


def test_reducible_mode_bogus_raises_binary():
    parent_stack, raw, y = _novel_domain_case(n=100)
    with pytest.raises(ValueError):
        decide_reuse_vs_grow(
            parent_stack, y, _child_probe_factory(1), classification_task(4),
            concept_dim=DIM, n_parents=1, n_epochs=1, eps_rel=0.05,
            raw_stack=raw, root_module_factory=_root_probe_factory,
            reducible_mode="bogus",
        )


# ---------------------------------------------------------------------------
# 5. score-set scoring in _held_out_codelength
# ---------------------------------------------------------------------------


def test_held_out_codelength_scores_score_set_at_best_epoch():
    torch.manual_seed(3)
    n = 300
    X = torch.randn(n, DIM)
    y = torch.randint(0, 4, (n,))
    train_X, train_y = X[:150], y[:150]
    val_X, val_y = X[150:225], y[150:225]
    score_X, score_y = X[225:], y[225:]
    spec = classification_task(4)

    model = spec.make_head(DIM)  # plain nn.Linear(DIM, 4)
    result = _held_out_codelength(
        model, lambda m, xb: m(xb), spec, train_X, train_y, val_X, val_y,
        n_epochs=1, lr=1e-3, device="cpu", score_X=score_X, score_y=score_y,
    )
    # n_epochs=1 => the single epoch IS the best-selection epoch, so the model's post-call state
    # is exactly the state the returned bits were computed from: check independently.
    model.eval()
    with torch.no_grad():
        independent = float(spec.nll_bits(model(score_X), score_y).mean().item())
    assert math.isfinite(result)
    assert abs(result - independent) < 1e-5


def test_held_out_codelength_returns_val_bits_without_score_set():
    torch.manual_seed(3)
    n = 300
    X = torch.randn(n, DIM)
    y = torch.randint(0, 4, (n,))
    train_X, train_y = X[:150], y[:150]
    val_X, val_y = X[150:225], y[150:225]
    spec = classification_task(4)

    model = spec.make_head(DIM)
    result = _held_out_codelength(
        model, lambda m, xb: m(xb), spec, train_X, train_y, val_X, val_y,
        n_epochs=1, lr=1e-3, device="cpu",
    )
    model.eval()
    with torch.no_grad():
        independent = float(spec.nll_bits(model(val_X), val_y).mean().item())
    assert abs(result - independent) < 1e-5


# ---------------------------------------------------------------------------
# 6. update_probe
# ---------------------------------------------------------------------------


def test_update_probe_helps_or_ties_and_preserves_parent():
    torch.manual_seed(7)
    n = N
    raw = torch.randn(n, RAW_DIM)
    w = torch.randn(RAW_DIM, 4)
    y = (raw @ w).argmax(dim=1)  # linearly-solvable 4-class raw task

    parent = ConceptModule(module_id="root_parent", in_dim=RAW_DIM, hidden_dim=DIM, out_dim=DIM,
                           n_layers=2, n_parents=0, dropout=0.0)
    spec = classification_task(4)
    head = spec.make_head(DIM)
    opt = torch.optim.AdamW(list(parent.parameters()) + list(head.parameters()), lr=1e-3)
    for _ in range(3):  # deliberately under-trained: room left for the update arm to help
        logits = head(parent(raw))
        loss = spec.nll_bits(logits, y).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    parent_state_before = copy.deepcopy(parent.state_dict())
    emb = parent(raw).detach()
    parent_stack = emb.unsqueeze(1)  # (N, 1, D)

    perm = torch.randperm(n)
    n_val = int(round(0.3 * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    L_update, module_copy = update_probe(
        parent_stack, raw, y, parent, 0, spec,
        concept_dim=DIM, n_parents=1, tr_idx=tr_idx, val_idx=val_idx,
        n_epochs=15, update_lr=1e-3,
    )

    reuse_model = ReuseComposer(parent_dim=DIM, n_parents=1, head=spec.make_head(DIM))
    L_reuse = _held_out_codelength(
        reuse_model, lambda m, xb: m(xb), spec,
        parent_stack[tr_idx], y[tr_idx], parent_stack[val_idx], y[val_idx],
        n_epochs=15, lr=1e-3, device="cpu",
    )

    assert math.isfinite(L_update)
    assert L_update <= L_reuse + 0.05  # the update can only help or tie

    assert module_copy is not parent
    for k, v in parent.state_dict().items():
        assert torch.equal(v, parent_state_before[k]), f"parent parameter {k} was mutated"

    packed = pack_update_input(parent_stack, raw)
    assert packed.shape == (n, 1 * DIM + RAW_DIM)
