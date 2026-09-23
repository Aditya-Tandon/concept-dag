"""P4 looked for the wrong provisional root, so it could not have passed in any world.

`s_interleave`'s ground truth marks position 3 — the revisit of position 1 — as the one that
`should` merge. Position 1 is the ORIGINAL: it is grown as an ordinary root at a position the
gate never defers, so no provisional root is ever minted there and `minted_at == 1` selects
nothing. The gate scored 0/10 on the archive for that reason, not because merges never fired:
8 of the 10 `evalue` seeds removed the position-3 provisional root mid-stream (`param_curve`
133,120 -> 99,840 between t3 and t4, one root's worth, while the recorded final pass saved 0).

The accepted op behind such a merge is only visible if the pass that ran it was recorded, and
on the legacy archive it never was (`--consolidate_every 2` fires after t3; only the final pass
reached the JSON). So the pair identity — ground-truth pair or the negative control — cannot be
recovered there, and the gate must say so rather than pick a side: hence the `undetermined`
verdict, which names no branch and raises the `P4-PAIR-UNVERIFIABLE` flag.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "eval_provisional",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "eval_provisional.py")
ep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ep)


GROUND_TRUTH = [
    {"revisit_of": None, "relation": None, "n_train": 4000, "should": None},
    {"revisit_of": None, "relation": None, "n_train": 400, "should": None},
    {"revisit_of": None, "relation": None, "n_train": 400, "should": "not_merge"},
    {"revisit_of": 1, "relation": "same", "n_train": 400, "should": "merge"},
    {"revisit_of": 0, "relation": "same", "n_train": 400, "should": "reuse"},
]


def _run(*, resolution="merge", merged_into=None, passes=None, minted_at=3):
    """One `s_interleave` evalue run, shaped like the archived JSONs."""
    run = {
        "ctrl_ground_truth": GROUND_TRUTH,
        "provisional_roots": [{"minted_at": minted_at, "resolution": resolution,
                               "resolved_at": 3, "trigger": "evalue"}],
        "decisions": [{"task": t, "decision": "grow"} for t in range(5)],
        "consolidation": {"ops": [{"op": "merge_rejected", "keep": 0, "drop": 2}],
                          "params_saved": 0},
    }
    if merged_into is not None:
        run["provisional_roots"][0]["merged_into"] = merged_into
    if passes is not None:
        run["consolidation_passes"] = passes
    return run


def test_target_pair_comes_from_the_runs_own_ground_truth():
    assert ep._target_pair(_run()) == (3, 1)


def test_target_pair_falls_back_when_ground_truth_is_absent():
    assert ep._target_pair({}) == (3, 1)


def test_the_old_selector_would_have_found_nothing():
    """Position 1 never carries a provisional root — the reason P4 read 0/10."""
    run = _run()
    assert not [p for p in run["provisional_roots"] if p["minted_at"] == 1]
    assert [p for p in run["provisional_roots"] if p["minted_at"] == 3]


def test_legacy_archive_gives_the_rate_but_not_the_pair():
    gate = ep.gate_p4({seed: _run() for seed in ep.INT_SEEDS})
    assert gate["value"]["n_merge"] == len(ep.INT_SEEDS)
    assert gate["value"]["frac_merge"] == 1.0
    assert gate["value"]["n_pair_unverified"] == len(ep.INT_SEEDS)
    assert gate["verdict"] == "undetermined"


def test_a_recorded_accepted_op_verifies_the_pair_and_passes():
    passes = [{"at_task": 3, "final": False,
               "ops": [{"op": "merge", "keep": 1, "drop": 3, "similarity": 0.95}]},
              {"at_task": 4, "final": True, "ops": []}]
    gate = ep.gate_p4({seed: _run(passes=passes) for seed in ep.INT_SEEDS})
    assert gate["value"]["n_pair_unverified"] == 0
    assert gate["verdict"] == "pass"


def test_merging_into_the_negative_control_fails():
    passes = [{"at_task": 3, "final": False,
               "ops": [{"op": "merge", "keep": 2, "drop": 3, "similarity": 0.95}]}]
    gate = ep.gate_p4({seed: _run(passes=passes) for seed in ep.INT_SEEDS})
    assert gate["value"]["n_wrong_pair"] == len(ep.INT_SEEDS)
    assert gate["value"]["any_wrong_merge"] is True     # pair {2,3} is the negative control
    assert gate["verdict"] == "fail"


def test_merged_into_on_the_resolution_is_enough_when_the_ops_are_gone():
    gate = ep.gate_p4({seed: _run(merged_into=1) for seed in ep.INT_SEEDS})
    assert gate["value"]["n_pair_unverified"] == 0
    assert gate["verdict"] == "pass"


def test_a_timeout_rate_below_threshold_still_fails():
    runs = {seed: _run(resolution="timeout") for seed in ep.INT_SEEDS}
    gate = ep.gate_p4(runs)
    assert gate["value"]["n_merge"] == 0
    assert gate["verdict"] == "fail"


@pytest.mark.parametrize("verdict,expect_flag",
                         [("undetermined", True), ("fail", False), ("pass", False)])
def test_the_unverifiable_verdict_raises_its_own_branch_flag(verdict, expect_flag):
    gates = {gid: {"verdict": "pass"} for gid in ("P0a", "P0b", "D2", "P1", "P2", "P5")}
    gates["P4"] = {"verdict": verdict}
    branch = ep.determine_branch(gates, p1_value=None, p2_value=None, p6_value=None,
                                 p4_value={"t3_reuse_fraction": 0.0})
    assert ("P4-PAIR-UNVERIFIABLE" in branch["flags"]) is expect_flag
