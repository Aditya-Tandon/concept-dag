"""R3 — `search_compose`'s candidate seeding must not leak the DEVICE generator, and must depend
on the run's seed.

The published block is

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(sd)                 # sd = the candidate INDEX
        model = SearchComposer(...)

and it has two defects that compound. `fork_rng(devices=[])` saves and restores the CPU generator
and nothing else, while `torch.manual_seed` inside it fans out to `torch.cuda.manual_seed_all` /
`torch.mps.manual_seed` — so on an accelerator the device generator is left at
`manual_seed(last candidate index)` plus that candidate's own training draws. And `sd` is the
candidate *index*, never a function of `--seed`. Together: after the first gated task's Search
rung, every later `train_node`'s dropout-mask stream is a function of `(budget, n_parents, n,
epochs)` alone. Point estimates and within-seed paired comparisons are unaffected; multi-seed
spreads understate seed variance (R3 of [[provisional-growth-v2-validity-rerun-result]],
should-fix 3 of [[provisional-growth-code-review]]).

Fixing it changes the numerics of every accelerator arm including `off`, so the fix is opt-in
(`candidate_seed_base`, `KanExpConfig.search_device_rng_fix`, `--search_device_rng_fix`) and the
default path must stay byte-identical until the pre-registered re-baseline replaces the
references ([[search-compose-device-reseed]]).

As in `tests/test_shadow_refit_null.py`, the device-RNG contract is exercised against a stand-in
`torch.Generator` wired into `concept_dag.utils.rng._device_rng_hooks`, so it runs on any machine:
`torch.mps` cannot certify a fork at all (its `get_rng_state` round-trip drops the philox offset)
and CUDA is not available on the laptop this is developed on. The CUDA variant of the same
assertion is gated and skipped off-GPU.
"""

from __future__ import annotations

import pytest
import torch

from concept_dag.training.kan_gate import (
    _candidate_init_seed,
    classification_task,
    search_compose,
)
from concept_dag.utils import rng as rng_utils

DIM = 16
N = 200
N_CLASSES = 4


def _spec():
    return classification_task(N_CLASSES)


def _data():
    torch.manual_seed(0)
    X = torch.randn(N, 2, DIM)
    y = torch.randint(0, N_CLASSES, (N,))
    return X, y


def _install_stand_in_device(monkeypatch) -> torch.Generator:
    """A `torch.Generator` posing as a device generator, wired into `_device_rng_hooks`.

    It is NOT reseeded by `torch.manual_seed` (only real device generators are), so a test that
    wants to emulate the fan-out has to do it explicitly — which is exactly what
    `test_default_path_still_leaks_the_device_generator` does.
    """
    gen = torch.Generator().manual_seed(1234)
    monkeypatch.setattr(
        rng_utils, "_device_rng_hooks",
        lambda: [("stand-in", gen.get_state, gen.set_state)],
    )
    return gen


def _run_search(X, y, *, candidate_seed_base, budget=3, device="cpu"):
    return search_compose(
        X[:140], y[:140], X[140:], y[140:], _spec(),
        concept_dim=DIM, n_parents=2, device=device, n_epochs=1, lr=1e-3, budget=budget,
        candidate_seed_base=candidate_seed_base)


# ---------------------------------------------------------------------------
# 1. The seed derivation
# ---------------------------------------------------------------------------


def test_candidate_init_seed_is_the_index_when_no_base_is_given():
    """The published behaviour, pinned: with no base the seed IS the candidate index."""
    assert [_candidate_init_seed(None, i) for i in range(6)] == list(range(6))


def test_candidate_init_seed_is_seed_dependent_and_collision_free():
    bases = [42 * 1000 + t for t in range(6)] + [43 * 1000 + t for t in range(6)]
    seeds = {(b, i): _candidate_init_seed(b, i) for b in bases for i in range(8)}
    # Distinct (base, index) pairs must not share a seed — in particular consecutive tasks, whose
    # bases differ by 1, must not alias through the index.
    assert len(set(seeds.values())) == len(seeds)
    # ... and every seed must be a legal argument to torch.manual_seed.
    for v in seeds.values():
        assert 0 <= v < 2 ** 31 - 1
        torch.manual_seed(v)


# ---------------------------------------------------------------------------
# 2. The device-RNG contract, against a stand-in device generator
# ---------------------------------------------------------------------------


def test_search_compose_restores_the_device_generator_when_the_seed_is_threaded(monkeypatch):
    """With `candidate_seed_base` given, a whole Search gate is a no-op on the device generator.

    `search_compose`'s own work draws on the CPU here (device="cpu"), so what this pins is the
    fork contract: the block that reseeds every generator puts the device half back. On CUDA the
    candidates' training draws happen OUTSIDE the block and legitimately advance the device
    stream — hence the `_draw` calls below stand in for that work, and the assertion is that the
    stream the run sees afterwards is the stream it would have seen with no reseeding at all.
    """
    gen = _install_stand_in_device(monkeypatch)
    X, y = _data()

    _draw = lambda k=4: torch.rand(k, generator=gen).tolist()   # noqa: E731

    before = _draw()                                  # the run's stream before the gate
    _run_search(X, y, candidate_seed_base=42_000)
    after = _draw()

    gen.manual_seed(1234)
    assert _draw() == before
    assert _draw() == after, "a Search gate must leave the device generator where it found it"


def test_default_path_still_leaks_the_device_generator(monkeypatch):
    """The negative control, and the reason the fix needs a re-baseline.

    `torch.manual_seed` reseeds every *real* device generator; the stand-in is not wired to it, so
    the fan-out is emulated by seeding the stand-in from the same value. What the test shows is
    that the default path's fork does not put that back, while the fixed path's does.
    """
    gen = _install_stand_in_device(monkeypatch)

    def _leaky_manual_seed(value):
        torch.random.manual_seed(value)
        gen.manual_seed(value)                        # the torch.cuda/torch.mps fan-out

    monkeypatch.setattr(torch, "manual_seed", _leaky_manual_seed)
    X, y = _data()

    gen.manual_seed(1234)
    _run_search(X, y, candidate_seed_base=None)
    leaked = torch.rand(4, generator=gen).tolist()

    gen.manual_seed(1234)
    _run_search(X, y, candidate_seed_base=42_000)
    restored = torch.rand(4, generator=gen).tolist()

    gen.manual_seed(1234)
    untouched = torch.rand(4, generator=gen).tolist()

    assert leaked != untouched, (
        "the default path is expected to leave the device generator shifted — if this ever fails, "
        "torch.random.fork_rng has grown device coverage and the opt-in flag can go")
    assert restored == untouched


def test_default_path_device_leak_is_seed_independent(monkeypatch):
    """The defect itself, stated as the archive feels it.

    Two runs differing only in `--seed` reach the first Search gate with DIFFERENT device-generator
    states (their earlier training drew different masks). On the default path the gate overwrites
    that state with `manual_seed(last candidate index)` — a value neither run's seed enters — so
    the two runs continue on the *same* device stream from there on: the dropout masks in every
    later `train_node` are shared, and a multi-seed spread measures less variance than it claims.
    With the seed threaded, each run's own state survives the gate and the streams stay distinct.
    """
    gen = _install_stand_in_device(monkeypatch)

    def _leaky_manual_seed(value):
        torch.random.manual_seed(value)
        gen.manual_seed(value)                        # the torch.cuda/torch.mps fan-out

    monkeypatch.setattr(torch, "manual_seed", _leaky_manual_seed)
    X, y = _data()

    def _post_gate_stream(base, device_state_seed):
        gen.manual_seed(device_state_seed)            # stands for "run seed 42" vs "run seed 43"
        _run_search(X, y, candidate_seed_base=base)
        return torch.rand(4, generator=gen).tolist()

    assert _post_gate_stream(None, 1234) == _post_gate_stream(None, 5678), (
        "the published path is expected to ERASE the run's device state — if this ever fails the "
        "leak is gone and the opt-in flag can go")
    assert _post_gate_stream(42_000, 1234) != _post_gate_stream(42_000, 5678), (
        "with the fix, two runs' device streams must stay distinct across the gate")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_search_compose_restores_the_cuda_generator_when_the_seed_is_threaded():
    """The same contract on the real thing — the only place the leak actually bites."""
    X, y = _data()
    torch.cuda.manual_seed_all(7)
    before = torch.cuda.get_rng_state().clone()
    _run_search(X, y, candidate_seed_base=42_000, device="cuda")
    assert torch.equal(torch.cuda.get_rng_state(), before)

    torch.cuda.manual_seed_all(7)
    _run_search(X, y, candidate_seed_base=None, device="cuda")
    assert not torch.equal(torch.cuda.get_rng_state(), before), (
        "the published path is expected to leave the CUDA generator at manual_seed(last candidate "
        "index) — this is the defect the flag exists to fix")


# ---------------------------------------------------------------------------
# 3. Reproducibility and seed-dependence of the candidate inits themselves
# ---------------------------------------------------------------------------


def _candidate_weights(base, *, global_seed=None):
    """The parameters of every SearchComposer `search_compose` builds, in order.

    The spy is installed and removed by hand rather than through `monkeypatch`, whose undo runs at
    the end of the test — nesting two installs inside one test would make each spy wrap the
    previous one and double-count.
    """
    import concept_dag.training.kan_gate as kg

    built = []
    real = kg.SearchComposer

    def _spy(*args, **kwargs):
        m = real(*args, **kwargs)
        built.append(torch.cat([p.detach().reshape(-1) for p in m.parameters()]).clone())
        return m

    X, y = _data()                                    # `_data` seeds the global RNG itself ...
    if global_seed is not None:
        torch.manual_seed(global_seed)                # ... so set the run's state after it
    kg.SearchComposer = _spy
    try:
        _run_search(X, y, candidate_seed_base=base)
    finally:
        kg.SearchComposer = real
    return built


def test_threaded_seed_is_reproducible_and_seed_dependent():
    a1 = _candidate_weights(42_000)
    a2 = _candidate_weights(42_000)
    b = _candidate_weights(43_000)

    assert len(a1) == len(a2) == len(b) == 3           # budget 3, n_parents 2 ⇒ 3 subsets
    assert all(torch.equal(x, y) for x, y in zip(a1, a2)), "same base ⇒ identical candidate inits"
    assert not any(torch.equal(x, y) for x, y in zip(a1, b)), (
        "different run seeds must give different candidate inits")


def test_default_path_candidate_inits_are_seed_independent():
    """The other half of the defect, at the CPU level: under the published seeding the candidate
    inits do not depend on the global RNG state at all, so two `--seed` values build the same
    candidates. (What still differs between seeds on the default path is the data subsampling and
    the split permutations — the understatement of seed variance is partial, not total.)"""
    a = _candidate_weights(None, global_seed=1)
    b = _candidate_weights(None, global_seed=2)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    # ... and with the fix the same two global states still give the same inits (the base decides,
    # not the ambient state) — which is what makes the fixed path reproducible.
    c = _candidate_weights(42_000, global_seed=1)
    d = _candidate_weights(42_000, global_seed=2)
    assert all(torch.equal(x, y) for x, y in zip(c, d))
    assert not any(torch.equal(x, y) for x, y in zip(a, c))


# ---------------------------------------------------------------------------
# 4. The default path is byte-identical
# ---------------------------------------------------------------------------


def test_default_path_bits_are_unchanged_by_the_new_kwarg():
    """`candidate_seed_base=None` must be indistinguishable from not passing it at all."""
    X, y = _data()
    spec = _spec()
    common = dict(concept_dim=DIM, n_parents=2, device="cpu", n_epochs=2, lr=1e-3, budget=3)

    torch.manual_seed(11)
    L_a, cfg_a, trace_a = search_compose(X[:140], y[:140], X[140:], y[140:], spec, **common)
    torch.manual_seed(11)
    L_b, cfg_b, trace_b = search_compose(X[:140], y[:140], X[140:], y[140:], spec,
                                         candidate_seed_base=None, **common)

    assert L_a == L_b
    assert cfg_a == cfg_b
    assert trace_a == trace_b


def test_default_path_still_does_not_reset_the_global_rng():
    """`tests/test_raw_grow_probe.py::test_search_does_not_reset_global_rng`, restated here so the
    CPU-fork property is asserted beside the device one on both paths."""
    X, y = _data()
    outs = {}
    for base in (None, 42_000):
        seen = []
        for g in (1, 2):
            torch.manual_seed(g)
            _run_search(X, y, candidate_seed_base=base, budget=2)
            seen.append(torch.rand(1).item())
        outs[base] = seen
    assert outs[None][0] != outs[None][1]
    assert outs[42_000][0] != outs[42_000][1]


# ---------------------------------------------------------------------------
# 5. End to end: the flag reaches the gate, and only when it is set
# ---------------------------------------------------------------------------
#
# The three tests above pin `search_compose`'s contract. These pin the wiring —
# `--search_device_rng_fix` → `KanExpConfig.search_device_rng_fix` → `candidate_seed_base` — so the
# flag cannot become dead code, and so the claim "the default path is byte-identical" is asserted
# at the level the archive was produced at, not only at the function's.


def _stream_cfg(results_dir, *, fix: bool, seed: int = 42):
    from concept_dag.experiments.kan_exp import KanExpConfig
    return KanExpConfig(
        backbone="synthetic", feature_dim=24, concept_dim=16, cnn_out_dim=16,
        n_parents=1, subspace_k=4, soft_pca_k=4, routing_batches=4,
        root_epochs=2, child_epochs=2, gate_epochs=3, gate_lr=3e-3, lr=3e-3,
        eps_rel=0.1, batch_size=16, device="cpu", results_dir=str(results_dir),
        log_every=100000, gate_cache_max=192, seed=seed,
        enable_search=True, search_skip=True, raw_grow_probe=True,
        search_device_rng_fix=fix,
    )


def _run_stream(tmp_path, name, *, fix: bool, seed: int = 42):
    import json

    from concept_dag.experiments.kan_exp import run_exp3a_kan
    from tests.fixtures.cls_identity_stream import make_cls_tasks

    res = run_exp3a_kan(
        _stream_cfg(tmp_path / name, fix=fix, seed=seed),
        make_cls_tasks(n_tasks=2, n_per_class=96, feature_dim=24, batch_size=16, seed=11))
    decisions = [{k: v for k, v in d.items() if k not in ("gate_seconds", "shadow_seconds")}
                 for d in res["decisions"]]
    return json.loads(json.dumps({"decisions": decisions, "test_accs": res["test_accs"],
                                  "average_accuracy": res["average_accuracy"]}))


def test_the_default_stream_is_reproducible_and_the_flag_changes_it(tmp_path):
    a = _run_stream(tmp_path, "off_a", fix=False)
    b = _run_stream(tmp_path, "off_b", fix=False)
    assert a == b, "the default path must be deterministic run-to-run at a fixed seed"

    fixed = _run_stream(tmp_path, "fixed", fix=True)
    assert fixed != a, (
        "--search_device_rng_fix must actually change the candidate seeds; if this passes the flag "
        "is dead code and the re-baseline would certify nothing")
    # Every gated decision still comes out of the same ladder — the change is the candidates' init,
    # so `L_search` is what moves, and the ladder must stay monotone (rel_search >= 0).
    gated = [d for d in fixed["decisions"] if "L_search_bits" in d or "L_search" in d]
    assert gated
    for d in gated:
        if d.get("rel_search") is not None:
            assert d["rel_search"] >= -1e-9


def test_the_fixed_path_is_reproducible_too(tmp_path):
    a = _run_stream(tmp_path, "fix_a", fix=True)
    b = _run_stream(tmp_path, "fix_b", fix=True)
    assert a == b, (
        "the fix must stay deterministic at a fixed seed — the re-baseline's bit-identity "
        "pre-flight is exactly this assertion on the GPU")
