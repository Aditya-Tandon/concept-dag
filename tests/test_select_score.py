"""
Unit tests for the select-score estimator (INTERFACE_SPEC_H5.md §1-2): the score-set path of
``_held_out_codelength`` with ``return_both=True``, ``search_compose(..., select_on=...)``, the
``estimator="select-score"`` branch of ``decide_reuse_search_grow`` (three disjoint splits, the
paired standard errors, the search-selection-optimism counterfactual, and the tie rule with its
per-z counterfactual), and the CLI/`KanExpConfig` wiring for ``--gate_estimator select-score`` and
``--tie_rule/--tie_z/--tie_novelty``.

Runs on CPU with tiny synthetic data, mirroring the style and fixtures of
``tests/test_gate_estimators.py`` / ``tests/test_prequential.py`` (same DIM/RAW_DIM, same
novel-domain construction, same ``_child_probe_factory`` / raw-root ``_root_probe_factory``).
"""

import math

import pytest
import torch

from concept_dag.modules.concept_module import ConceptModule
from concept_dag.training.kan_gate import (
    classification_task,
    decide_reuse_search_grow,
    search_compose,
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
    """Same construction as test_gate_estimators.py: raw features decide a 4-way label; the frozen
    parent embedding only exposes a coarse 2-way split of it, so only raw access reveals the
    remaining reducible structure and the gate must grow."""
    raw = torch.randn(n, RAW_DIM)
    w = torch.randn(RAW_DIM, 4)
    y = (raw @ w).argmax(dim=1)
    lift = torch.randn(2, DIM)
    parent = torch.nn.functional.one_hot(y // 2, 2).float() @ lift + 0.1 * torch.randn(n, DIM)
    return parent.unsqueeze(1), raw, y


def _linear_ladder_case(n=N):
    """Purely linear-in-the-parents 4-class task: every rung comfortably beats the marginal null
    once trained."""
    X = torch.randn(n, 2, DIM)
    w = torch.randn(2 * DIM, 4)
    y = (X.flatten(1) @ w).argmax(dim=1)
    return X, y


# ---------------------------------------------------------------------------
# 1. search_compose(select_on="select") — the winner is chosen on SELECT, not SCORE
# ---------------------------------------------------------------------------


def test_search_compose_select_on_select_picks_select_winner(monkeypatch):
    """Stub _held_out_codelength so candidate A is best on SELECT but worst on SCORE, and
    candidate B is the reverse. Under select_on='select' A must win, and the reported bits must be
    A's SCORE bits — not B's (which select_on='score', the published default, would pick)."""
    import concept_dag.training.kan_gate as kg

    table = {
        (0,): (1.0, 5.0),   # candidate A: best on SELECT, worst on SCORE
        (1,): (2.0, 0.5),   # candidate B: worst on SELECT, best on SCORE
    }

    def fake_held_out(model, forward, spec, Xtr, ytr, Xval, yval, n_epochs, lr, device,
                      batch_size=128, score_X=None, score_y=None, return_both=False):
        sub = model.subset
        sel, sc = table[sub]
        if return_both:
            return sel, sc, torch.full((4,), sc)
        return sel if score_X is None else sc

    monkeypatch.setattr(kg, "_held_out_codelength", fake_held_out)

    spec = kg.classification_task(2)
    Xtr = torch.zeros(4, 1, DIM); ytr = torch.zeros(4, dtype=torch.long)
    Xsel = torch.zeros(4, 1, DIM); ysel = torch.zeros(4, dtype=torch.long)
    Xsc = torch.zeros(4, 1, DIM); ysc = torch.zeros(4, dtype=torch.long)

    out = kg.search_compose(
        Xtr, ytr, Xsel, ysel, spec, concept_dim=DIM, n_parents=2, device="cpu",
        n_epochs=1, lr=1e-3, budget=2, select_on="select",
        score_X=Xsc, score_y=ysc, baseline_L=10.0, baseline_select_L=10.0,
    )
    assert len(out) == 6
    L, cfg, trace, sel_bits, per_ex, score_argmin_L = out
    assert cfg["subset"] == (0,)      # A wins on SELECT despite being worse on SCORE
    assert L == 5.0                   # reported bits are A's SCORE bits, not B's
    assert sel_bits == 1.0
    assert torch.equal(per_ex, torch.full((4,), 5.0))
    # The score-argmin counterfactual, from the SAME two trained candidates, correctly finds B.
    assert score_argmin_L == 0.5


def test_search_compose_default_select_on_score_is_a_3_tuple_even_with_a_score_set():
    """Regression: the crossfit/single estimators call search_compose with a score set but the
    default select_on='score' — that path MUST stay the published 3-tuple (they unpack exactly 3
    values), even though a score_X is present."""
    torch.manual_seed(0)
    X, y = _linear_ladder_case(n=200)
    spec = classification_task(4)
    out = search_compose(X[:140], y[:140], X[140:170], y[140:170], spec, concept_dim=DIM,
                         n_parents=2, device="cpu", n_epochs=1, lr=1e-3, budget=2,
                         score_X=X[170:], score_y=y[170:])
    assert len(out) == 3


# ---------------------------------------------------------------------------
# 2. three disjoint splits covering N
# ---------------------------------------------------------------------------


def test_select_score_splits_cover_n_disjointly():
    torch.manual_seed(0)
    X, y = _linear_ladder_case(n=500)
    rec = decide_reuse_search_grow(
        X, y, _child_probe_factory(2), classification_task(4),
        concept_dim=DIM, n_parents=2, n_epochs=2, eps_grow=0.05, eps_search=0.05,
        search_budget=1, estimator="select-score",
        split_generator=torch.Generator().manual_seed(0),
    )
    meta = rec.estimator_meta
    assert meta["estimator"] == "select-score"
    assert meta["n_train"] + meta["n_select"] + meta["n_score"] == 500

    tr_and_sel = set(rec.split_meta["tr_idx"])
    score = set(rec.split_meta["val_idx"])
    assert tr_and_sel.isdisjoint(score)
    assert tr_and_sel | score == set(range(500))
    assert len(tr_and_sel) == meta["n_train"] + meta["n_select"]
    assert len(score) == meta["n_score"]


def test_select_score_ss_fracs_override_split_sizes():
    torch.manual_seed(0)
    X, y = _linear_ladder_case(n=1000)
    rec = decide_reuse_search_grow(
        X, y, _child_probe_factory(2), classification_task(4),
        concept_dim=DIM, n_parents=2, n_epochs=1, eps_grow=0.05, eps_search=0.05,
        search_budget=1, estimator="select-score", ss_fracs=(0.5, 0.25, 0.25),
        split_generator=torch.Generator().manual_seed(0),
    )
    meta = rec.estimator_meta
    assert meta["n_score"] == 250
    assert meta["n_select"] == 250
    assert meta["n_train"] == 500


# ---------------------------------------------------------------------------
# 3. standard errors, the score-selection-optimism counterfactual, tie_counterfactual
# ---------------------------------------------------------------------------


def test_se_and_selection_optimism_finite_and_signed():
    torch.manual_seed(0)
    parent_stack, raw, y = _novel_domain_case(n=900)
    rec = decide_reuse_search_grow(
        parent_stack, y, _child_probe_factory(1), classification_task(4),
        concept_dim=DIM, n_parents=1, n_epochs=15, eps_grow=0.05, eps_search=0.05,
        search_budget=2, raw_stack=raw, root_module_factory=_root_probe_factory,
        estimator="select-score", split_generator=torch.Generator().manual_seed(3),
    )
    meta = rec.estimator_meta
    for key in ("se_search_minus_grow", "se_reuse_minus_grow", "se_reuse_minus_search",
               "se_split_proxy"):
        assert math.isfinite(meta[key]), key
        assert meta[key] >= 0.0, key

    # score-argmin can only find bits at least as low as the select-argmin's, over the identical
    # candidate set, so the optimism it removes is <= 0 by construction.
    assert meta["search_selection_optimism"] <= 1e-9
    assert meta["search_selection_optimism_abs"] == pytest.approx(abs(meta["search_selection_optimism"]))
    assert meta["search_selection_optimism_abs"] >= 0.0
    assert meta["search_score_bits_selected_on_score"] <= meta["score_bits"]["search"] + 1e-9

    assert set(meta["tie_counterfactual"].keys()) == {"0.5", "1.0", "2.0"}
    assert meta["se_split_proxy"] == pytest.approx(
        meta["se_search_minus_grow"] * math.sqrt(meta["n_score"]))

    # "tie" is only recorded when tie_rule == "grow" (default "none" here).
    assert "tie" not in meta


# ---------------------------------------------------------------------------
# 4. the tie rule: fires on a constructed near-tie novel case, not otherwise
# ---------------------------------------------------------------------------


def _tie_fake_held_out(reuse_level, n_score=80):
    """Deterministic per-example SCORE bits, dispatched by model class name: grow and search are a
    near-tie (mean 3.0 each, paired differences with zero mean and a controlled non-zero SD, so the
    tie condition is exactly on the boundary regardless of stochastic training); reuse/null are
    controlled by `reuse_level` (novelty knob)."""
    half = n_score // 2
    noise = torch.tensor([0.05, -0.05] * half)
    pe_grow = torch.full((n_score,), 3.0)
    pe_search = pe_grow + noise                 # mean 3.0 too: margin == 0 exactly
    pe_reuse = torch.full((n_score,), reuse_level)
    pe_null = torch.full((n_score,), 5.0)
    profiles = {
        "reuse": (float(pe_reuse.mean()), pe_reuse),
        "grow": (float(pe_grow.mean()), pe_grow),
        "search": (float(pe_search.mean()), pe_search),
        "null": (float(pe_null.mean()), pe_null),
    }

    def fake(model, forward, spec, Xtr, ytr, Xval, yval, n_epochs, lr, device,
             batch_size=128, score_X=None, score_y=None, return_both=False):
        name = type(model).__name__
        key = ("search" if name == "SearchComposer" else
               "grow" if name in ("_GrowModel", "_RootGrowModel") else
               "null" if name == "_NullModel" else "reuse")
        sel, pe = profiles[key]
        sc = float(pe.mean())
        if return_both:
            return sel, sc, pe.clone()
        return sel if score_X is None else sc

    return fake


def _run_tie_case(monkeypatch, reuse_level, tie_rule, tie_novelty=0.1, tie_z=1.0, n=400):
    import concept_dag.training.kan_gate as kg
    monkeypatch.setattr(kg, "_held_out_codelength", _tie_fake_held_out(reuse_level))
    spec = kg.classification_task(2)
    parent_stack = torch.zeros(n, 1, DIM)
    y = torch.zeros(n, dtype=torch.long)
    return kg.decide_reuse_search_grow(
        parent_stack, y, _child_probe_factory(1), spec, concept_dim=DIM, n_parents=1, n_epochs=1,
        eps_grow=0.05, eps_search=0.05, search_budget=2, estimator="select-score",
        tie_rule=tie_rule, tie_z=tie_z, tie_novelty=tie_novelty,
        split_generator=torch.Generator().manual_seed(0),
    )


def test_tie_fires_on_near_tie_novel_case(monkeypatch):
    rec = _run_tie_case(monkeypatch, reuse_level=5.0, tie_rule="grow")   # novelty == 0.0 < 0.1
    assert rec.decision == "grow"   # the tie firing flips the pre-tie decision (search) to grow
    tie = rec.estimator_meta["tie"]
    assert tie["fired"] is True
    assert tie["margin"] == pytest.approx(0.0, abs=1e-6)
    assert tie["se"] > 0.0
    assert tie["novelty"] < 0.1
    # tie_counterfactual at z=1.0 must agree with the actual firing at tie_z=1.0.
    assert rec.estimator_meta["tie_counterfactual"]["1.0"] is True


def test_tie_does_not_fire_when_novelty_is_not_below_threshold(monkeypatch):
    rec = _run_tie_case(monkeypatch, reuse_level=4.0, tie_rule="grow")   # novelty == 0.2 >= 0.1
    assert rec.estimator_meta["novelty"] == pytest.approx(0.2)
    assert rec.estimator_meta["tie"]["fired"] is False
    assert rec.decision != "grow"
    assert rec.estimator_meta["tie_counterfactual"]["1.0"] is False


def test_tie_key_absent_when_tie_rule_is_none(monkeypatch):
    rec = _run_tie_case(monkeypatch, reuse_level=5.0, tie_rule="none")
    assert "tie" not in rec.estimator_meta
    # the counterfactual is still recorded, regardless of tie_rule.
    assert rec.estimator_meta["tie_counterfactual"]["1.0"] is True
    assert rec.decision != "grow"


def test_tie_rule_bogus_raises():
    with pytest.raises(ValueError):
        decide_reuse_search_grow(
            torch.zeros(100, 1, DIM), torch.zeros(100, dtype=torch.long),
            _child_probe_factory(1), classification_task(2), concept_dim=DIM, n_parents=1,
            n_epochs=1, search_budget=1, estimator="select-score", tie_rule="bogus",
        )


# ---------------------------------------------------------------------------
# 5. novel-domain raw-root case still grows under select-score
# ---------------------------------------------------------------------------


def test_select_score_grows_on_novel_domain_raw_root():
    torch.manual_seed(0)
    parent_stack, raw, y = _novel_domain_case(n=1200)
    rec = decide_reuse_search_grow(
        parent_stack, y, _child_probe_factory(1), classification_task(4),
        concept_dim=DIM, n_parents=1, n_epochs=40, eps_grow=0.05, eps_search=0.05,
        search_budget=2, raw_stack=raw, root_module_factory=_root_probe_factory,
        estimator="select-score", split_generator=torch.Generator().manual_seed(0),
    )
    assert math.isfinite(rec.L_grow_bits)
    assert rec.estimator_meta["estimator"] == "select-score"
    assert rec.decision == "grow"
    assert rec.rel_grow > 0.05


# ---------------------------------------------------------------------------
# 6. regression — the other estimators are untouched by the select-score branch
# ---------------------------------------------------------------------------


def test_single_estimator_unchanged_by_select_score_branch():
    def _run():
        torch.manual_seed(0)
        parent_stack, raw, y = _novel_domain_case(n=400)
        return decide_reuse_search_grow(
            parent_stack, y, _child_probe_factory(1), classification_task(4),
            concept_dim=DIM, n_parents=1, n_epochs=6, eps_grow=0.05, eps_search=0.05,
            search_budget=2, raw_stack=raw, root_module_factory=_root_probe_factory,
            estimator="single", split_generator=torch.Generator().manual_seed(7),
        ).as_dict()

    a, b = _run(), _run()
    assert a == b
    assert a["estimator_meta"]["estimator"] == "single"


def test_crossfit_estimator_unchanged_by_select_score_branch():
    def _run():
        torch.manual_seed(0)
        X, y = _linear_ladder_case(n=500)
        return decide_reuse_search_grow(
            X, y, _child_probe_factory(2), classification_task(4),
            concept_dim=DIM, n_parents=2, n_epochs=2, eps_grow=0.05, eps_search=0.05,
            search_budget=1, estimator="crossfit", n_splits=3,
            split_generator=torch.Generator().manual_seed(0),
        ).as_dict()

    a, b = _run(), _run()
    assert a == b
    assert a["estimator_meta"]["estimator"] == "crossfit"


def test_prequential_estimator_unchanged_by_select_score_branch():
    def _run():
        torch.manual_seed(0)
        X, y = _linear_ladder_case(n=200)
        return decide_reuse_search_grow(
            X, y, _child_probe_factory(2), classification_task(4), concept_dim=DIM, n_parents=2,
            n_epochs=2, search_budget=1, estimator="prequential",
        ).as_dict()

    a, b = _run(), _run()
    assert a == b
    assert a["estimator_meta"]["estimator"] == "prequential"


# ---------------------------------------------------------------------------
# 7. CLI / KanExpConfig wiring
# ---------------------------------------------------------------------------


def test_cli_gate_estimator_accepts_select_score():
    from run_experiment import build_parser

    parser = build_parser()
    args = parser.parse_args(["--gate_estimator", "select-score"])
    assert args.gate_estimator == "select-score"
    # pre-existing choices must still be accepted.
    for choice in ("single", "crossfit", "prequential"):
        assert parser.parse_args(["--gate_estimator", choice]).gate_estimator == choice


def test_cli_tie_flag_defaults():
    from run_experiment import build_parser

    args = build_parser().parse_args([])
    assert args.tie_rule == "none"
    assert args.tie_z == 1.0
    assert args.tie_novelty == 0.1


def test_cli_tie_flag_overrides():
    from run_experiment import build_parser

    args = build_parser().parse_args([
        "--gate_estimator", "select-score", "--tie_rule", "grow",
        "--tie_z", "2.0", "--tie_novelty", "0.2",
    ])
    assert args.tie_rule == "grow"
    assert args.tie_z == 2.0
    assert args.tie_novelty == 0.2


def test_cli_tie_rule_choices_reject_bad_value():
    from run_experiment import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--tie_rule", "bogus"])


def test_kan_exp_config_tie_defaults():
    from concept_dag.experiments.kan_exp import KanExpConfig

    assert KanExpConfig.__dataclass_fields__["tie_rule"].default == "none"
    assert KanExpConfig.__dataclass_fields__["tie_z"].default == 1.0
    assert KanExpConfig.__dataclass_fields__["tie_novelty"].default == 0.1


def test_run_exp3a_kan_wires_select_score_and_tie_rule():
    """End-to-end smoke test (INTERFACE_SPEC_H5.md §2): a tiny synthetic feature-mode run with
    gate_estimator='select-score' must produce gated decisions carrying the select-score
    estimator_meta, and with tie_rule='grow' a 'tie' key."""
    import torch as _torch
    from torch.utils.data import DataLoader, TensorDataset
    from concept_dag.experiments.kan_exp import KanExpConfig, run_exp3a_kan

    def _make_synth_tasks(n_tasks=3, feature_dim=20, n_per_class=80, seed=5):
        g = _torch.Generator().manual_seed(seed)
        tasks = []
        for t in range(n_tasks):
            base = t % 2
            mu = _torch.zeros(feature_dim)
            mu[base * 2] = 2.0
            mu[base * 2 + 1] = -2.0
            xs, ys = [], []
            for c in range(2):
                centre = mu if c == 0 else -mu
                xs.append(centre + _torch.randn(n_per_class, feature_dim, generator=g))
                ys.append(_torch.full((n_per_class,), c, dtype=_torch.long))
            X = _torch.cat(xs); Y = _torch.cat(ys)
            perm = _torch.randperm(X.shape[0], generator=g)
            X, Y = X[perm], Y[perm]
            n_tr = int(0.7 * X.shape[0])
            tr = DataLoader(TensorDataset(X[:n_tr], Y[:n_tr]), batch_size=32, shuffle=True)
            te = DataLoader(TensorDataset(X[n_tr:], Y[n_tr:]), batch_size=32)
            tasks.append({"train": tr, "test": te, "n_classes": 2})
        return tasks

    tasks = _make_synth_tasks()
    cfg = KanExpConfig(
        backbone="synthetic", feature_dim=20, concept_dim=16, cnn_out_dim=16, n_mlp_layers=2,
        n_tasks=3, n_parents=2, subspace_k=8, soft_pca_k=8, routing_batches=10,
        root_epochs=4, child_epochs=4, gate_epochs=8, gate_lr=3e-3, lr=3e-3,
        eps_rel=0.1, eps_search=0.05, enable_search=True, search_skip=True,
        gate_estimator="select-score", tie_rule="grow", tie_z=1.0, tie_novelty=0.1,
        results_dir="/tmp/test_select_score_smoke", device="cpu", log_every=1000,
    )
    res = run_exp3a_kan(cfg, tasks)
    gated = [d for d in res["decisions"] if d["task"] >= 1]
    assert gated, "expected at least one gated (non-root) task"
    for d in gated:
        em = d.get("estimator_meta")
        assert em is not None
        assert em["estimator"] == "select-score"
        assert "tie_counterfactual" in em
        assert "tie" in em   # tie_rule="grow" -> always recorded (fired True or False)
