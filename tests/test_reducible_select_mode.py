"""The e-process's indifference margin must use the normaliser the run CONFIGURED.

ref: PR #8 independent review, should-fix 8 (Track C item 8)

The three-way ladder normalises its margins by `reducible`, and `--reducible {grow,best}` picks
which: `L_null - L_grow` or `L_null - min(L_reuse, L_search, L_grow)`. The e-process's own
threshold is `eps_grow * reducible` measured on SELECT bits — the same quantity, same rungs — but
it was hard-coded to the "best" form, so a `--reducible grow` run tested a DIFFERENT margin from
the one its ladder decided with. Default is "best", so nothing archived moves; the bug was
waiting for the first run of the other arm.
"""

from __future__ import annotations

import math

import pytest
import torch

from concept_dag.experiments.kan_exp import KanExpConfig, run_exp3a_kan
from tests.fixtures.cls_identity_stream import make_cls_tasks


def _cfg(results_dir, reducible_mode: str) -> KanExpConfig:
    return KanExpConfig(
        results_dir=str(results_dir), device="cpu", backbone="dinov2_vits14",
        feature_dim=24, concept_dim=16, seed=42,
        root_epochs=3, child_epochs=3, gate_epochs=3,
        n_tasks=3, n_parents=2, routing_batches=4, gate_cache_max=256, batch_size=16,
        raw_grow_probe=True, enable_search=True, search_skip=True,
        provisional="shadow", reducible_mode=reducible_mode, log_every=100000,
    )


def _evalues(results):
    return [d["evalue"] for d in results["decisions"] if "evalue" in d]


@pytest.mark.parametrize("mode", ["best", "grow"])
def test_reducible_select_follows_the_configured_mode(tmp_path, mode):
    res = run_exp3a_kan(_cfg(tmp_path / mode, mode), make_cls_tasks(n_tasks=3, seed=11))
    evs = _evalues(res)
    assert evs, "the shadow arm must compute the third state on at least one gated decision"
    for ev in evs:
        assert ev["reducible_mode"] == mode
        expected = ev["reducible_select_grow"] if mode == "grow" else ev["reducible_select_best"]
        assert ev["reducible_select"] == expected
        # ... and the threshold the e-process actually tested against is eps_grow * that.
        assert ev["threshold_bits"] == pytest.approx(ev["eps_grow"] * expected)


def test_the_two_denominators_are_recorded_and_ordered(tmp_path):
    """`best` subtracts the BEST rung, which codes at least as well as grow, so it is >= it."""
    res = run_exp3a_kan(_cfg(tmp_path / "rec", "best"), make_cls_tasks(n_tasks=3, seed=11))
    evs = _evalues(res)
    assert evs
    for ev in evs:
        assert ev["reducible_select_best"] >= ev["reducible_select_grow"] - 1e-9
        assert ev["reducible_select_grow"] > 0 and ev["reducible_select_best"] > 0


def test_default_mode_is_best_and_leaves_the_published_denominator(tmp_path):
    """The published arm is `best`; switching the code must not have moved it."""
    assert KanExpConfig().reducible_mode == "best"
    res = run_exp3a_kan(_cfg(tmp_path / "def", "best"), make_cls_tasks(n_tasks=3, seed=11))
    for ev in _evalues(res):
        assert ev["reducible_select"] == ev["reducible_select_best"]
