"""P0b — the shadow refit must leave the run it observes exactly as it found it.

`--provisional shadow` computes the third gate state and acts on nothing, which is what makes
it the pre-registered null for [[provisional-growth-undetermined-gate]]. In the H8 sweep it
was not a null: the one 5-Datasets run moved (AA 0.92999 -> 0.92893, `L_reuse` at t2 1.0650 ->
1.0766) while all 20 CTrL runs stayed bit-identical.

The leak was the fork. `torch.random.fork_rng(devices=[])` saves and restores the CPU generator
and nothing else — its `devices` argument only ever names CUDA devices, and `[]` names none —
while `torch.manual_seed` inside the block re-seeds *every* device generator. Dropout masks are
drawn on the training device, so the shadow refit left the device generator wherever its own
refit ended and every later `train_node` in the run saw a different mask stream.

Whether that showed up was arithmetic. The live rung splits under `single` (val_fraction 0.3)
and the shadow refit under `select-score` (ss_fracs 0.65/0.15/0.20); the two give the same
number of minibatches per epoch at CTrL's gate-cache sizes (n = 200 -> 2 and 2, n = 2,000 -> 11
and 11) and different numbers at the 16,384-row 5-Datasets gate cache (90 and 84). Equal counts
consume equal device-RNG offsets, so the CTrL arms coincided by luck.

It is invisible on CPU — the CPU generator is the one `fork_rng` does protect — which is why a
green CPU suite said nothing about it. The tests below therefore come in two halves: the
device-RNG contract, exercised against a stand-in generator so it runs anywhere, and an
end-to-end identity check pinned to the split arithmetic that broke (n = 192: 2 minibatches
live, 1 in the shadow), which runs on CPU as a regression guard and on CUDA as the real proof.

`torch.mps` is deliberately not used as the stand-in: on torch 2.0 its `get_rng_state` /
`set_rng_state` round-trips the 36-byte state tensor without the philox offset, so an MPS run
cannot be forked at all and cannot certify anything.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from concept_dag.experiments.kan_exp import KanExpConfig, run_exp3a_kan
from concept_dag.utils import rng as rng_utils
from tests.fixtures.cls_identity_stream import make_cls_tasks

# The gate-cache size whose live/shadow minibatch counts differ — the 5-Datasets configuration,
# shrunk. `_assert_split_batch_counts_differ` re-derives it so the test cannot silently drift
# into the coincident (CTrL) case where the bug is invisible.
GATE_CACHE_N = 192
BATCH_SIZE = 16              # stream batch size; `gate_cache_max` is read in these units
PROBE_BATCH = 128            # `_held_out_codelength`'s own minibatch size

# Keys that are absent from an `off` run by construction (the H8 fields) or are wall-clock.
_ARM_ONLY_KEYS = {"provisional", "provisional_roots"}
_DECISION_ARM_ONLY_KEYS = {"provisional", "evalue", "shadow_seconds", "ladder_decision"}
_WALL_TIME_KEYS = {"gate_seconds", "shadow_seconds"}


# ---------------------------------------------------------------------------
# 1. The device-RNG contract, against a stand-in device generator
# ---------------------------------------------------------------------------


def _install_stand_in_device(monkeypatch) -> torch.Generator:
    """A `torch.Generator` posing as a device generator, wired into `_device_rng_hooks`."""
    gen = torch.Generator().manual_seed(1234)
    monkeypatch.setattr(
        rng_utils, "_device_rng_hooks",
        lambda: [("stand-in", gen.get_state, gen.set_state)],
    )
    return gen


def _draw(gen: torch.Generator, k: int = 4):
    return torch.rand(k, generator=gen).tolist()


def test_fork_rng_all_devices_restores_a_device_generator(monkeypatch):
    gen = _install_stand_in_device(monkeypatch)

    _draw(gen)                                    # the block is entered mid-stream, as in a run
    with rng_utils.fork_rng_all_devices():
        torch.manual_seed(999)                    # what the shadow refit does
        _draw(gen, 7)                             # ... and the work it does with it
    after_fork = _draw(gen)

    gen.manual_seed(1234)
    _draw(gen)
    no_block = _draw(gen)

    assert after_fork == no_block


def test_torch_fork_rng_devices_empty_leaves_a_device_generator_shifted(monkeypatch):
    """The negative control: the construction this replaced does NOT restore the device."""
    gen = _install_stand_in_device(monkeypatch)

    _draw(gen)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(999)
        _draw(gen, 7)
    after_fork = _draw(gen)

    gen.manual_seed(1234)
    _draw(gen)
    no_block = _draw(gen)

    assert after_fork != no_block, (
        "torch.random.fork_rng(devices=[]) is expected to leave a non-CPU generator shifted — "
        "if this ever passes, fork_rng has grown device coverage and the helper can go")


def test_fork_rng_all_devices_restores_the_cpu_generator_too(monkeypatch):
    _install_stand_in_device(monkeypatch)
    torch.manual_seed(5)
    before = torch.get_rng_state().clone()
    with rng_utils.fork_rng_all_devices():
        torch.manual_seed(999)
        torch.rand(16)
    assert torch.equal(torch.get_rng_state(), before)


def test_fork_rng_all_devices_restores_after_an_exception(monkeypatch):
    gen = _install_stand_in_device(monkeypatch)
    state = gen.get_state().clone()
    with pytest.raises(RuntimeError):
        with rng_utils.fork_rng_all_devices():
            _draw(gen, 3)
            raise RuntimeError("boom")
    assert torch.equal(gen.get_state(), state)


# ---------------------------------------------------------------------------
# 2. End to end: `off` and `shadow` must produce the same run
# ---------------------------------------------------------------------------


def _live_and_shadow_batch_counts(n: int):
    """Minibatches per probe epoch under the live (`single`) and shadow (`select-score`) splits."""
    n_train_single = n - int(0.3 * n)
    n_train_select_score = n - round(0.20 * n) - round(0.15 * n)
    return (math.ceil(n_train_single / PROBE_BATCH),
            math.ceil(n_train_select_score / PROBE_BATCH))


def _cfg(results_dir, provisional: str, device: str) -> KanExpConfig:
    return KanExpConfig(
        backbone="synthetic", feature_dim=24, concept_dim=16, cnn_out_dim=16,
        n_parents=1, subspace_k=4, soft_pca_k=4, routing_batches=4,
        root_epochs=3, child_epochs=3, gate_epochs=4, gate_lr=3e-3, lr=3e-3,
        eps_rel=0.1, batch_size=BATCH_SIZE, device=device, results_dir=str(results_dir),
        log_every=100000, gate_cache_max=GATE_CACHE_N,
        enable_search=True, search_skip=True, raw_grow_probe=True, oracle_rungs=True,
        provisional=provisional,
    )


def _comparable(results: dict) -> dict:
    out = {k: v for k, v in results.items() if k not in _ARM_ONLY_KEYS}
    out["decisions"] = [
        {k: v for k, v in d.items()
         if k not in _DECISION_ARM_ONLY_KEYS and k not in _WALL_TIME_KEYS}
        for d in out["decisions"]
    ]
    return json.loads(json.dumps(out))


def _run_both_arms(tmp_path, device: str):
    # 4 classes x 96 = 384 samples/task, 60 % train = 230 rows — ABOVE the 192-row gate cache,
    # so `_cache_parent_stack` caps exactly as it does on 5-Datasets (16,384 of 54,000).
    off = run_exp3a_kan(_cfg(tmp_path / "off", "off", device),
                        make_cls_tasks(n_tasks=3, n_per_class=96, feature_dim=24,
                                       batch_size=BATCH_SIZE, seed=11))
    shadow = run_exp3a_kan(_cfg(tmp_path / "shadow", "shadow", device),
                           make_cls_tasks(n_tasks=3, n_per_class=96, feature_dim=24,
                                          batch_size=BATCH_SIZE, seed=11))
    gated = [d for d in off["decisions"] if d["task"] > 0]
    assert gated and all(d["gate_cache_n"] == GATE_CACHE_N for d in gated), (
        "the gate cache must bind for this test to reproduce the 5-Datasets shape; got "
        f"{[d.get('gate_cache_n') for d in gated]}")
    assert [d for d in shadow["decisions"] if d["task"] > 0 and "evalue" in d], (
        "the shadow arm must have computed the third state on at least one gated decision")
    return off, shadow


def test_the_configuration_under_test_is_the_one_that_broke():
    live, shadow = _live_and_shadow_batch_counts(GATE_CACHE_N)
    assert live != shadow, (
        f"at n={GATE_CACHE_N} the live and shadow splits give the same minibatch count "
        f"({live}); pick an n where they differ or this test cannot see the P0b leak")
    # ... and the CTrL sizes, which coincided, are why the bug hid on 20 of 21 runs.
    assert _live_and_shadow_batch_counts(200)[0] == _live_and_shadow_batch_counts(200)[1]
    assert _live_and_shadow_batch_counts(2000)[0] == _live_and_shadow_batch_counts(2000)[1]
    assert _live_and_shadow_batch_counts(16384) == (90, 84)


def test_shadow_refit_is_a_null_on_a_capped_cache_stream_cpu(tmp_path):
    off, shadow = _run_both_arms(tmp_path, "cpu")
    assert _comparable(off) == _comparable(shadow)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device-RNG leak needs a device")
def test_shadow_refit_is_a_null_on_a_capped_cache_stream_cuda(tmp_path):
    off, shadow = _run_both_arms(tmp_path, "cuda")
    assert _comparable(off) == _comparable(shadow)
