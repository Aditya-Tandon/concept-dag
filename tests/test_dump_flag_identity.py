"""`--dump_gate_tensors` must be an observer: turning it on may not change the run.

ref: Hypotheses/concept-dag/kan-gated-growth/provisional-growth-v2-validity-rerun.md,
     judge change 7 ("one in-run dump-flag identity check") — Track C item 10

The token-mode dump is new code inside the task loop: it retains the live gate cache, and it
snapshots every root and its reader at mint. All of that is tensor copying, which draws nothing —
but "draws nothing" is exactly the kind of claim that is true until someone adds a `randperm`, and
H9's whole V0 block rests on the dumped arms being comparable to the undumped baselines. So it is
pinned here rather than argued.

Fields compared are the note's "Bit-identity, operationally" list: the ordered decision list,
`average_accuracy`, every `L_*` key in every gate record, `test_accs`, `param_curve`,
`param_curve_total`, and — when present on both sides — `test_accs_final` and
`average_accuracy_final`. Exact equality; a field present on one side and absent on the other is
a failure, not a skip.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from concept_dag.experiments.kan_exp import KanExpConfig, run_exp3a_kan
from tests.fixtures.cls_identity_stream import make_cls_tasks
from tests.test_attn_root_adoption import _make_synthetic_token_tasks

#: The note's bit-identity field list, minus the per-decision keys handled separately.
RUN_FIELDS = ("average_accuracy", "test_accs", "param_curve", "param_curve_total")
OPTIONAL_RUN_FIELDS = ("test_accs_final", "average_accuracy_final")


def _results_json(results_dir) -> dict:
    """The run's results as they land ON DISK — the artifact every evaluator reads.

    The in-memory dict additionally carries `_gate_cache_y` (raw tensors, private, added after
    the JSON write and only under the dump flag), which is precisely not part of the comparison.
    """
    with open(os.path.join(str(results_dir), "exp3a_kan_results.json")) as f:
        return json.load(f)


def _assert_bit_identical(a: dict, b: dict, label_a: str, label_b: str) -> None:
    for f in RUN_FIELDS:
        assert (f in a) == (f in b), f"{f} present on one side only ({label_a} vs {label_b})"
        assert a.get(f) == b.get(f), f"{f} differs: {label_a}={a.get(f)!r} {label_b}={b.get(f)!r}"
    for f in OPTIONAL_RUN_FIELDS:
        if f in a and f in b:
            assert a[f] == b[f], f"{f} differs: {label_a}={a[f]!r} {label_b}={b[f]!r}"

    da, db = a["decisions"], b["decisions"]
    assert len(da) == len(db)
    for x, y in zip(da, db):
        assert x["task"] == y["task"]
        assert x["decision"] == y["decision"], f"task {x['task']}: decision differs"
        l_keys = {k for k in list(x) + list(y) if k.startswith("L_")}
        for k in sorted(l_keys):
            assert (k in x) == (k in y), f"task {x['task']}: {k} present on one side only"
            assert x.get(k) == y.get(k), (
                f"task {x['task']}: {k} differs: {label_a}={x.get(k)!r} {label_b}={y.get(k)!r}")


def _token_cfg(results_dir, dump: bool) -> KanExpConfig:
    return KanExpConfig(
        backbone="dinov2_vits14", feature_dim=24, n_tokens=7, token_pool=2,
        root_family="attn_pool", concept_dim=16, seed=42,
        root_epochs=3, child_epochs=3, gate_epochs=3,
        n_parents=2, routing_batches=4, gate_cache_max=256, batch_size=16,
        raw_grow_probe=True, enable_search=True, search_skip=True, oracle_rungs=True,
        dump_gate_tensors=dump, dump_max_per_split=8,
        log_every=100000, device="cpu", results_dir=str(results_dir))


def _token_tasks():
    return _make_synthetic_token_tasks(n_tasks=4, n_classes=4, feature_dim=24, n_tokens=7,
                                       n_per_class=48, batch_size=16, seed=0)


def test_token_mode_dump_flag_does_not_change_the_run(tmp_path):
    run_exp3a_kan(_token_cfg(tmp_path / "off", dump=False), _token_tasks())
    run_exp3a_kan(_token_cfg(tmp_path / "on", dump=True), _token_tasks())

    _assert_bit_identical(_results_json(tmp_path / "off"), _results_json(tmp_path / "on"),
                          "dump-off", "dump-on")
    # ... and the dump really was produced, so the comparison is against a run that paid for it.
    assert os.path.exists(os.path.join(str(tmp_path / "on"), "gate_dump.pt"))
    assert not os.path.exists(os.path.join(str(tmp_path / "off"), "gate_dump.pt"))


def test_token_mode_dump_flag_identity_holds_with_provisional_minting(tmp_path):
    """The snapshot hook sits next to the provisional mint — the arm that has roots to snapshot."""
    def cfg(d, dump):
        c = _token_cfg(d, dump)
        c.provisional = "always"
        c.always_n_max = 100000
        return c

    run_exp3a_kan(cfg(tmp_path / "off", False), _token_tasks())
    on = run_exp3a_kan(cfg(tmp_path / "on", True), _token_tasks())

    assert any(d.get("provisional") for d in on["decisions"]), "no provisional root was minted"
    _assert_bit_identical(_results_json(tmp_path / "off"), _results_json(tmp_path / "on"),
                          "dump-off", "dump-on")

    dump = torch.load(os.path.join(str(tmp_path / "on"), "gate_dump.pt"))
    minted = sum(1 for d in on["decisions"] if d["decision"] == "grow")
    assert len(dump["roots_at_mint"]) == minted


def test_cls_mode_dump_flag_does_not_change_the_run(tmp_path):
    """The same claim on the published CLS path, where the dump is the pre-existing one."""
    def cfg(d, dump):
        return KanExpConfig(
            results_dir=str(d), device="cpu", backbone="dinov2_vits14",
            feature_dim=24, concept_dim=16, seed=42,
            root_epochs=3, child_epochs=3, gate_epochs=3,
            n_tasks=4, n_parents=2, routing_batches=4, gate_cache_max=256, batch_size=16,
            raw_grow_probe=True, enable_search=True, search_skip=True, oracle_rungs=True,
            dump_gate_tensors=dump, log_every=100000)

    run_exp3a_kan(cfg(tmp_path / "off", False), make_cls_tasks())
    run_exp3a_kan(cfg(tmp_path / "on", True), make_cls_tasks())
    _assert_bit_identical(_results_json(tmp_path / "off"), _results_json(tmp_path / "on"),
                          "dump-off", "dump-on")
