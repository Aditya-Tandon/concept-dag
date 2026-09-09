"""The published `mlp_cls` root path must be BIT-IDENTICAL after the attention-root adoption.

ref: [[attn-root-adoption]], [[rebaseline-2026-09-08-result]]

The adoption adds a second ROOT family (learned-query attention pooling over DINOv2 patch tokens)
behind `--root_family`. Everything the re-baseline claims rests on the control arm being the
*same* code the published runs used: if `--root_family mlp_cls` drifted by so much as an RNG draw,
the arm comparison would be confounded and the null-validity check ("the mlp_cls arm reproduces
rebaseline-2026-09-08") would be measuring the wrong thing.

Two checks:

* **Determinism** — the reference stream, run twice in one process with the same seed, produces
  the identical results dict. This catches accidental global-RNG consumption added by the change.
* **Pre-change fixture** — the same stream, run on the pre-change commit `e258d96` (the branch
  point), produced `tests/fixtures/mlp_cls_baseline.json`. The current code must reproduce it
  exactly, float for float: decisions, code lengths, rel_* margins, accuracies, parameter curve
  and the consolidation record.
* **Pre-change fixture, consolidation ON** — `tests/fixtures/mlp_cls_consolidating_baseline.json`,
  from a stream with a deliberate duplicate task and `consolidate_every=2`, so the identity check
  also covers `consolidate_nodes`, `_functional_similarity`, `distill_merge` (which *trains*
  weights) and the backward-safety accept. The plain stream never gets a merge past the detector,
  so without this case the whole merge path was outside the identity guarantee (PR #7 review,
  should-fix 3). The recorded run accepts 2 merges and saves 1,408 parameters.

Regenerate the fixture (only when a deliberate change to the CLS path is being made, and then say
so in the commit message):

    git archive e258d96 | tar -x -C /tmp/prechange
    cp tests/fixtures/cls_identity_stream.py /tmp/prechange/
    cd /tmp/prechange && PYTHONPATH=. python3 cls_identity_stream.py \
        "$OLDPWD/tests/fixtures/mlp_cls_baseline.json"

Fields the change legitimately ADDS (`root_family`, `n_tokens`, `param_curve_total`, …) and the
wall-clock `gate_seconds` are excluded by `cls_identity_stream.comparable`; everything else is
compared with `==` on the JSON round-trip, i.e. exact float equality.
"""

from __future__ import annotations

import json
import os

from tests.fixtures.cls_identity_stream import (
    comparable, run_reference, run_reference_consolidating,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mlp_cls_baseline.json")
FIXTURE_CONSOL = os.path.join(os.path.dirname(__file__), "fixtures",
                              "mlp_cls_consolidating_baseline.json")


def _json_roundtrip(obj):
    """Compare through JSON so the fixture and a live run are the same kind of object."""
    return json.loads(json.dumps(obj, sort_keys=True))


def test_mlp_cls_stream_is_deterministic(tmp_path):
    a = comparable(run_reference(str(tmp_path / "a")))
    b = comparable(run_reference(str(tmp_path / "b")))
    assert _json_roundtrip(a) == _json_roundtrip(b), (
        "the CLS-mode gated run is no longer deterministic under a fixed seed"
    )


def test_mlp_cls_path_reproduces_the_pre_change_fixture(tmp_path):
    with open(FIXTURE) as f:
        expected = json.load(f)
    got = _json_roundtrip(comparable(run_reference(str(tmp_path / "live"))))
    # Report the first offending key rather than a 200-line dict diff.
    for key in sorted(set(expected) | set(got)):
        assert expected.get(key) == got.get(key), (
            f"mlp_cls path changed at results['{key}']:\n"
            f"  pre-change (e258d96): {expected.get(key)!r}\n"
            f"  current:              {got.get(key)!r}"
        )
    assert expected == got


def test_mlp_cls_consolidating_path_reproduces_the_pre_change_fixture(tmp_path):
    """As above, but on the stream that actually merges — so `distill_merge` is covered."""
    with open(FIXTURE_CONSOL) as f:
        expected = json.load(f)
    got = _json_roundtrip(comparable(run_reference_consolidating(str(tmp_path / "consol"))))
    # The point of this fixture: it must contain accepted merges, or it is not testing the path.
    assert expected["consolidation"]["params_saved"] > 0
    assert any(op["op"] == "merge" for op in expected["consolidation"]["ops"])
    for key in sorted(set(expected) | set(got)):
        assert expected.get(key) == got.get(key), (
            f"mlp_cls consolidation path changed at results['{key}']:\n"
            f"  pre-change (e258d96): {expected.get(key)!r}\n"
            f"  current:              {got.get(key)!r}"
        )
    assert expected == got
