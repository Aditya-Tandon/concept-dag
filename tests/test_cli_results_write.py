"""The CTrL CLI branch re-serialises the results dict after appending ground truth. A
--dump_gate_tensors run keeps private tensors under leading-underscore keys; serialising them
truncated every dumped CTrL results JSON mid-write (H9, 2026-09-24)."""
import json
import os

import torch

import run_experiment


def test_private_keys_are_stripped_and_file_is_valid_json(tmp_path):
    results = {"average_accuracy": 0.5, "decisions": [{"decision": "grow"}],
               "_gate_cache_y": {0: torch.zeros(3)}}
    out = str(tmp_path / "exp3a_kan_results.json")
    written = run_experiment.write_public_results_json(results, out)
    assert "_gate_cache_y" not in written
    on_disk = json.load(open(out))
    assert on_disk == {"average_accuracy": 0.5, "decisions": [{"decision": "grow"}]}
    assert not os.path.exists(out + ".tmp")


def test_public_content_is_unchanged_without_private_keys(tmp_path):
    results = {"average_accuracy": 0.5, "ctrl_ground_truth": [None, {"revisit_of": 0}]}
    out = str(tmp_path / "r.json")
    run_experiment.write_public_results_json(results, out)
    assert json.load(open(out)) == results


def test_failed_write_leaves_no_truncated_file(tmp_path):
    out = str(tmp_path / "r.json")
    json.dump({"old": True}, open(out, "w"))
    bad = {"average_accuracy": 0.5, "unserialisable": torch.zeros(2)}  # public key, still a tensor
    try:
        run_experiment.write_public_results_json(bad, out)
    except TypeError:
        pass
    else:
        raise AssertionError("expected the write to fail")
    assert json.load(open(out)) == {"old": True}
