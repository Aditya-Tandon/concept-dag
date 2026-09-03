"""
Unit tests for the prequential estimator in the Kan gate core (`concept_dag/training/kan_gate.py`):
the block schedule (``_blocks_from_order``), the tail extrapolation (``_fit_tail``), the online code
length itself (``_prequential_codelength``) and the ``estimator="prequential"`` branch of
``decide_reuse_search_grow``.

Why the rung is scored this way at all: every held-out estimator trains the grow probe on a fraction
of the task (70 % single split, 64 % cross-fit) and scores it there, while the grown root is deployed
after training on ALL of it — and grow is the rung whose code length falls fastest with n
([[gate-arms-multiseed-ctrl-result]]). Prequential coding
([[blier-ollivier-2018-description-length]], [[prequential-grow-probe]]) pays for every example once,
as a prediction before it is trained on, and yields two quantities: the strict ``total`` (which keeps
MDL's small-n penalty) and the ``tail`` extrapolation to the deployed n (which removes the
training-fraction bias). Both are recorded; ``preq_decide`` picks which one decides.

Runs on CPU with tiny synthetic data, mirroring the style and fixtures of
``tests/test_raw_grow_probe.py`` / ``tests/test_gate_estimators.py`` (same DIM/RAW_DIM, same
novel-domain construction, same ``_child_probe_factory`` / raw-root ``_root_probe_factory``).
"""

import math

import pytest
import torch

from concept_dag.modules.concept_module import ConceptModule
from concept_dag.training.kan_gate import (
    ReuseComposer,
    _blocks_from_order,
    _fit_tail,
    _prequential_codelength,
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
    """Same construction as test_raw_grow_probe.py: raw features decide a 4-way label; the frozen
    parent embedding only exposes a coarse 2-way split of it, so only raw access reveals the
    remaining reducible structure and the gate must grow."""
    raw = torch.randn(n, RAW_DIM)
    w = torch.randn(RAW_DIM, 4)
    y = (raw @ w).argmax(dim=1)
    lift = torch.randn(2, DIM)
    parent = torch.nn.functional.one_hot(y // 2, 2).float() @ lift + 0.1 * torch.randn(n, DIM)
    return parent.unsqueeze(1), raw, y


def _linear_case(n=N):
    """Purely linear-in-the-parents 4-class task: a ReuseComposer learns it, so the block curve
    actually falls and the tail has something to extrapolate."""
    X = torch.randn(n, 2, DIM)
    w = torch.randn(2 * DIM, 4)
    y = (X.flatten(1) @ w).argmax(dim=1)
    return X, y


# ---------------------------------------------------------------------------
# 1. the online code length itself
# ---------------------------------------------------------------------------


def test_prequential_codelength_curve_and_totals():
    torch.manual_seed(0)
    X, y = _linear_case(n=600)
    spec = classification_task(4)
    model = ReuseComposer(parent_dim=DIM, n_parents=2, head=spec.make_head(DIM))
    B = 5
    order = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(0))
    meta = _prequential_codelength(model, lambda m, xb: m(xb), spec, X, y, n_blocks=B,
                                   n_epochs=6, lr=1e-3, device="cpu", order=order,
                                   n_classes_or_uniform_bits=math.log2(4))

    assert meta["n_blocks"] == B
    assert meta["n_deploy"] == X.shape[0]
    assert len(meta["curve"]) == B - 1                      # block 0 is coded, never scored
    n_seen = [c[0] for c in meta["curve"]]
    assert n_seen == sorted(n_seen) and len(set(n_seen)) == len(n_seen)   # strictly increasing
    assert n_seen[-1] == X.shape[0] - len(_blocks_from_order(order, B)[-1])

    bits = [c[1] for c in meta["curve"]]
    assert all(math.isfinite(b) for b in bits)
    for key in ("total", "tail", "last_block"):
        assert math.isfinite(meta[key])
    # total is a weighted mean over {uniform block 0} ∪ {block bits}, so it cannot beat the best block
    assert meta["total"] >= min(bits)
    assert meta["last_block"] == bits[-1]
    # the tail extrapolates BEYOND the last block, and the clamp bounds how far below it can go
    assert meta["last_block"] >= meta["tail"] - 0.5
    assert set(meta["tail_by_exponent"]) == {"0.25", "0.5", "1.0"}
    assert meta["tail_by_exponent"]["0.5"] == meta["tail"]  # default exponent, same fit
    # A learnable task: the model must code the last block better than the uniform rate it started at.
    assert meta["last_block"] < meta["uniform_bits"]


# ---------------------------------------------------------------------------
# 2. the tail fit is exact on an exact power law
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exponent", [0.25, 0.5, 1.0])
def test_fit_tail_reproduces_exact_power_law(exponent):
    """A curve that follows a + c·n^(−e) exactly must extrapolate to a + c·N^(−e) exactly (the
    clamp is inactive here: c is small enough that the extrapolation stays within half a bit of the
    best observed block)."""
    a, c, n_deploy = 1.0, 3.0, 2000
    curve = [(n, a + c * n ** (-exponent)) for n in (100, 200, 400, 800)]
    expected = a + c * n_deploy ** (-exponent)
    assert abs(_fit_tail(curve, n_deploy, exponent) - expected) < 1e-6


def test_fit_tail_guards():
    # fewer than 2 points -> last block's bits (nothing to extrapolate from)
    assert _fit_tail([(100, 1.25)], 1000, 0.5) == 1.25
    # singular design: every n_seen identical -> the n^(-e) column has no variance
    assert _fit_tail([(100, 1.5), (100, 1.0), (100, 1.25)], 1000, 0.5) == 1.25
    # clamp: a curve dropping fast enough to extrapolate below zero is held at min(bits) - 0.5
    steep = [(100, 4.0), (200, 2.0), (400, 0.5)]
    assert _fit_tail(steep, 100000, 0.5) == pytest.approx(0.5 - 0.5)
    # ... and an INCREASING curve cannot extrapolate above the worst observed block
    rising = [(100, 0.5), (200, 2.0), (400, 4.0)]
    assert _fit_tail(rising, 100000, 0.5) <= 4.0


def test_fit_tail_floors_negative_extrapolation_at_zero():
    """On a nearly-solved task the clamp (`[min(bits) - 0.5, max(bits)]`) can still land below zero
    — bits are a code length and cannot be negative, so the tail must be floored at 0."""
    nearly_solved = [(100, 0.30), (200, 0.12), (300, 0.05)]
    n_deploy, exponent = 5000, 0.5
    # Reproduce the same closed-form fit (a + c*n^-e, OLS in {1, n^-e}) to confirm that WITHOUT the
    # floor, the clamped extrapolation would indeed be negative here.
    bits = [b for _, b in nearly_solved]
    u = torch.tensor([float(n) ** (-exponent) for n, _ in nearly_solved], dtype=torch.float64)
    b64 = torch.tensor(bits, dtype=torch.float64)
    u_mean, b_mean = u.mean(), b64.mean()
    du = u - u_mean
    c = float((du * (b64 - b_mean)).sum()) / float((du * du).sum())
    a = float(b_mean) - c * float(u_mean)
    pred = a + c * float(n_deploy) ** (-exponent)
    lo, hi = min(bits) - 0.5, max(bits)
    unfloored = min(max(pred, lo), hi)
    assert unfloored < 0.0  # confirms the floor is actually load-bearing for this curve
    assert _fit_tail(nearly_solved, n_deploy, exponent) == 0.0


# ---------------------------------------------------------------------------
# 3. pairing: one order -> one block schedule, whatever the rung
# ---------------------------------------------------------------------------


def test_blocks_from_order_pairs_rungs():
    order = torch.randperm(103, generator=torch.Generator().manual_seed(0))
    blocks = _blocks_from_order(order, 5)
    assert len(blocks) == 5
    sizes = [int(b.numel()) for b in blocks]
    assert sum(sizes) == 103
    assert max(sizes) - min(sizes) <= 1                       # near-equal
    assert torch.equal(torch.cat(blocks), order)              # contiguous chunks, in order
    assert set(torch.cat(blocks).tolist()) == set(range(103))  # a partition of the cache

    # Two rungs handed the SAME order see the identical block index sets — that is the pairing.
    other = _blocks_from_order(order, 5)
    assert all(torch.equal(a, b) for a, b in zip(blocks, other))
    # ... and a different order does not.
    shuffled = _blocks_from_order(torch.randperm(103, generator=torch.Generator().manual_seed(1)), 5)
    assert not all(torch.equal(a, b) for a, b in zip(blocks, shuffled))

    # Degenerate requests are clamped rather than producing an empty/absent prediction step.
    assert len(_blocks_from_order(torch.randperm(4), 1)) == 2
    assert len(_blocks_from_order(torch.randperm(3), 9)) == 3
    with pytest.raises(ValueError):
        _blocks_from_order(torch.arange(1), 2)


# ---------------------------------------------------------------------------
# 4. the gate branch
# ---------------------------------------------------------------------------


def test_prequential_gate_grows_on_novel_domain():
    """The novel-domain (SVHN analogue) case of test_raw_grow_probe.py, scored prequentially.

    N=1200 / n_epochs=15 / B=4: each rung is fit 3 times (on 1/4, 2/4, 3/4 of the cache, warm
    started), so this is ~1 s. Smaller N was flaky — the raw-root probe needs a few hundred
    examples per block before its curve separates from reuse's.
    """
    torch.manual_seed(0)
    parent_stack, raw, y = _novel_domain_case(n=1200)
    rec = decide_reuse_search_grow(
        parent_stack, y, _child_probe_factory(1), classification_task(4),
        concept_dim=DIM, n_parents=1, n_epochs=15, eps_grow=0.05, eps_search=0.05,
        search_budget=1, raw_stack=raw, root_module_factory=_root_probe_factory,
        estimator="prequential", preq_blocks=4,
        split_generator=torch.Generator().manual_seed(0),
    )
    for L in (rec.L_null_bits, rec.L_reuse_bits, rec.L_search_bits, rec.L_grow_bits):
        assert math.isfinite(L)

    em = rec.estimator_meta
    assert em["estimator"] == "prequential"
    assert em["decide"] == "tail" and em["n_blocks"] == 4 and em["exponent"] == 0.5
    assert em["n_train"] == 1200
    assert set(em["rungs"]) == {"null", "reuse", "search", "grow"}
    for name, meta in em["rungs"].items():
        assert len(meta["curve"]) == 3, name
        for key in ("total", "tail", "last_block"):
            assert math.isfinite(meta[key]), (name, key)
        assert set(meta["tail_by_exponent"]) == {"0.25", "0.5", "1.0"}, name
    # both rules recorded for every rung, so either decision can be recomputed offline
    assert set(em["alt"]) == {"tail", "total"}
    for rule in ("tail", "total"):
        assert set(em["alt"][rule]) == {"null", "reuse", "search", "grow"}
        assert em["alt"][rule]["grow"] == em["rungs"]["grow"][rule]
    # the deciding rule is the one the record's L_* came from
    assert rec.L_grow_bits == em["rungs"]["grow"]["tail"]

    # one order for the whole call: the split_meta is train-prefix vs final block
    tr, val = rec.split_meta["tr_idx"], rec.split_meta["val_idx"]
    assert len(set(tr) & set(val)) == 0
    assert set(tr) | set(val) == set(range(1200))
    assert len(val) == 300

    assert rec.grow_probe_input == "raw-root"
    assert rec.decision == "grow"
    assert rec.rel_grow > 0.05


def test_prequential_total_rule_decides_on_the_total():
    torch.manual_seed(0)
    parent_stack, raw, y = _novel_domain_case(n=1200)
    rec = decide_reuse_search_grow(
        parent_stack, y, _child_probe_factory(1), classification_task(4),
        concept_dim=DIM, n_parents=1, n_epochs=15, eps_grow=0.05, eps_search=0.05,
        search_budget=1, raw_stack=raw, root_module_factory=_root_probe_factory,
        estimator="prequential", preq_blocks=4, preq_decide="total",
        split_generator=torch.Generator().manual_seed(0),
    )
    em = rec.estimator_meta
    assert em["decide"] == "total"
    assert rec.L_grow_bits == em["rungs"]["grow"]["total"]
    assert rec.L_reuse_bits == em["rungs"]["reuse"]["total"]
    assert rec.L_null_bits == em["rungs"]["null"]["total"]
    # the tail is still recorded under the total rule (that is the point of "alt")
    assert em["alt"]["tail"]["grow"] == em["rungs"]["grow"]["tail"]
    assert em["alt"]["tail"]["grow"] != rec.L_grow_bits


def test_prequential_rejects_unknown_decide_rule():
    torch.manual_seed(0)
    X, y = _linear_case(n=200)
    with pytest.raises(ValueError):
        decide_reuse_search_grow(
            X, y, _child_probe_factory(2), classification_task(4), concept_dim=DIM, n_parents=2,
            n_epochs=1, search_budget=1, estimator="prequential", preq_decide="mean",
        )


# ---------------------------------------------------------------------------
# 5. the published estimators are untouched
# ---------------------------------------------------------------------------


def test_single_estimator_unchanged_by_the_prequential_branch():
    """The ``single`` path must be bit-for-bit what it was: same seed + same split_generator ⇒ the
    identical record. (The prequential branch only ADDS a branch; the rung models it shares with
    ``_fit_rungs`` are built in the same order, so the global-RNG stream is unchanged.)"""
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


def test_search_compose_still_returns_a_triple_without_estimator_fn():
    torch.manual_seed(0)
    X, y = _linear_case(n=200)
    spec = classification_task(4)
    out = search_compose(X[:140], y[:140], X[140:], y[140:], spec, concept_dim=DIM, n_parents=2,
                         device="cpu", n_epochs=1, lr=1e-3, budget=2)
    assert len(out) == 3
    L, cfg, trace = out
    assert math.isfinite(L) and "subset" in cfg and len(trace) == 2

    # With an estimator_fn the return grows a fourth element: the winner's meta.
    calls = []

    def _fake_estimator(model, forward):
        meta = {"total": 1.0 - 0.1 * len(calls), "tail": 0.5, "curve": [(10, 1.0)]}
        calls.append(meta)
        return meta["total"], meta

    out4 = search_compose(X[:140], y[:140], X[140:], y[140:], spec, concept_dim=DIM, n_parents=2,
                          device="cpu", n_epochs=1, lr=1e-3, budget=3, baseline_L=10.0,
                          estimator_fn=_fake_estimator)
    assert len(out4) == 4
    L4, cfg4, trace4, meta4 = out4
    assert len(calls) == 3                 # the estimator replaced _held_out_codelength entirely
    assert L4 == pytest.approx(min(m["total"] for m in calls))
    assert meta4 is calls[-1]              # monotone-decreasing fake ⇒ the last candidate wins
    assert [t[1] for t in trace4] == sorted([t[1] for t in trace4], reverse=True)

    # A baseline no candidate beats ⇒ the trivial composition wins and there is no winner meta.
    out_triv = search_compose(X[:140], y[:140], X[140:], y[140:], spec, concept_dim=DIM, n_parents=2,
                              device="cpu", n_epochs=1, lr=1e-3, budget=2, baseline_L=-1.0,
                              estimator_fn=_fake_estimator)
    assert out_triv[1].get("trivial") is True and out_triv[3] is None
