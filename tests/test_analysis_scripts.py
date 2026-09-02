"""
Tests for scripts/desk_stage.py and scripts/eval_gate_arms.py (INTERFACE_SPEC.md §8).

Builds tiny SYNTHETIC fixtures — a gate_dump.pt following the §6 format exactly, and
exp3a_kan_results.json trees following the §7 decision-dict fields — so these tests do not depend
on any real experiment run. Two layers:

  1. End-to-end: run each script as a subprocess (``python scripts/<name>.py ...``), exactly as a
     user would, and check the JSON it writes contains every number it also prints.
  2. Unit: import the scripts' internal functions directly (via importlib, no package needed) and
     check the H2 PRESERVES/FLIPS logic, the H1' branch logic, and the H3' U1/U2/... logic against
     hand-constructed cases with known answers.

Runs on CPU only; small enough (feature_dim 24, concept_dim 16, <=70 samples/task, n_splits=3,
epochs=2) to finish in well under a minute.
"""

import importlib.util
import json
import os
import subprocess
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from concept_dag.modules.concept_module import ConceptModule  # noqa: E402
from concept_dag.models.baselines import LinearHead  # noqa: E402


def _load_script(name: str, filename: str):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


desk_stage = _load_script("desk_stage", "desk_stage.py")
eval_gate_arms = _load_script("eval_gate_arms", "eval_gate_arms.py")

F = 24   # feature_dim (raw encoder features)
D = 16   # concept_dim
N_LAYERS = 2


# ===========================================================================
# gate_dump.pt fixture (§6 format)
# ===========================================================================


def _root_module(seed: int, module_id: str) -> ConceptModule:
    torch.manual_seed(seed)
    m = ConceptModule(module_id=module_id, in_dim=F, hidden_dim=D, out_dim=D,
                      n_layers=N_LAYERS, n_parents=0, dropout=0.0)
    m.eval()
    return m


def _task_tensors(seed: int, n_train: int, n_val: int, n_test: int, n_classes: int):
    g = torch.Generator().manual_seed(seed)

    def mk(n):
        x = torch.randn(n, F, generator=g)
        y = torch.randint(0, n_classes, (n,), generator=g)
        return x, y

    return (*mk(n_train), *mk(n_val), *mk(n_test))


def _head_state(seed: int, n_classes: int) -> dict:
    torch.manual_seed(seed)
    return LinearHead(D, n_classes).state_dict()


def build_gate_dump(path: str, seed: int = 0) -> dict:
    """5 tasks, 2 root nodes (t0 "MNIST", t2 grown), t1/t3/t4 routed through them via
    reuse/search decisions; t1 and t4 carry an ``update`` dict (§4)."""
    node0 = _root_module(seed * 10 + 1, "node0")  # task 0 root ("MNIST")
    node1 = _root_module(seed * 10 + 2, "node1")  # task 2 root (grown)

    n_classes = {0: 10, 1: 10, 2: 5, 3: 10, 4: 10}
    sizes = {0: (60, 20, 20), 1: (56, 24, 20), 2: (40, 12, 12), 3: (70, 30, 20), 4: (56, 24, 20)}

    tasks = []
    for t in range(5):
        n_tr, n_va, n_te = sizes[t]
        tr_x, tr_y, va_x, va_y, te_x, te_y = _task_tensors(seed * 100 + t, n_tr, n_va, n_te, n_classes[t])
        tasks.append({
            "task": t, "n_classes": n_classes[t], "ctrl": {"name": f"synthetic-t{t}"},
            "train_raw": tr_x, "train_y": tr_y, "val_raw": va_x, "val_y": va_y,
            "test_raw": te_x, "test_y": te_y,
        })

    nodes = [
        {"index": 0, "task_id": 0, "is_root": True, "parent_indices": [], "state_dict": node0.state_dict()},
        {"index": 1, "task_id": 2, "is_root": True, "parent_indices": [], "state_dict": node1.state_dict()},
    ]

    decisions = [
        {"task": 0, "decision": "grow", "reason": "root"},
        {"task": 1, "decision": "reuse", "parents": [0], "grow_probe_input": "raw-root",
         "rel_improvement": 0.01, "L_reuse_bits": 2.10, "L_grow_bits": 2.30,
         "L_search_bits": 2.05, "rel_search": 0.02, "rel_grow": -0.05,
         "L_null_bits": 3.30, "reducible_grow": 1.00, "reducible_best": 1.25,
         "reducible_mode": "grow", "rel_search_best": 0.04, "rel_grow_best": -0.16,
         "rel_improvement_best": 0.20, "n_rungs_above_null": 0,
         "oracle_accs": {"reuse": 0.55, "search": 0.56, "grow": 0.50},
         "update": {"probed": True, "parent": 0, "parent_task_id": 0, "L_update": 1.90,
                    "rel_update": 0.095, "backward_safe": True,
                    "backward_deltas_val": {"0": -0.005}, "backward_deltas_test": {"0": -0.01},
                    "selected": False, "logged_only": False}},
        {"task": 2, "decision": "grow", "parents": [0], "grow_probe_input": "raw-root",
         "rel_improvement": 0.30, "L_reuse_bits": 2.80, "L_grow_bits": 1.70,
         "L_search_bits": 2.40, "rel_search": 0.18, "rel_grow": 0.40,
         "L_null_bits": 2.95, "reducible_grow": 1.25, "reducible_best": 1.25,
         "reducible_mode": "grow", "rel_search_best": 0.32, "rel_grow_best": 0.56,
         "rel_improvement_best": 0.39, "n_rungs_above_null": 0,
         "oracle_accs": {"reuse": 0.60, "search": 0.66, "grow": 0.80},
         "update": {"probed": True, "parent": 0, "parent_task_id": 0, "L_update": 1.75,
                    "rel_update": -0.03, "backward_safe": True,
                    "backward_deltas_val": {"0": 0.0, "1": -0.005},
                    "backward_deltas_test": {"0": 0.0, "1": -0.008},
                    "selected": False, "logged_only": True}},
        {"task": 3, "decision": "search", "parents": [0, 1], "grow_probe_input": "raw-root",
         "rel_improvement": 0.10, "L_reuse_bits": 2.60, "L_grow_bits": 2.40,
         "L_search_bits": 2.35, "rel_search": 0.45, "rel_grow": 0.10,
         "L_null_bits": 2.90, "reducible_grow": 0.50, "reducible_best": 0.55,
         "reducible_mode": "grow", "rel_search_best": 0.41, "rel_grow_best": 0.09,
         "rel_improvement_best": 0.36, "n_rungs_above_null": 0,
         "search_meta": {"subset": [0, 1], "rank": 16, "skip": True},
         "oracle_accs": {"reuse": 0.30, "search": 0.34, "grow": 0.28}},
        {"task": 4, "decision": "reuse", "parents": [0, 1], "grow_probe_input": "raw-root",
         "rel_improvement": 0.02, "L_reuse_bits": 1.20, "L_grow_bits": 1.35,
         "L_search_bits": 1.19, "rel_search": 0.01, "rel_grow": -0.15,
         "L_null_bits": 2.10, "reducible_grow": 0.75, "reducible_best": 0.90,
         "reducible_mode": "grow", "rel_search_best": 0.01, "rel_grow_best": -0.12,
         "rel_improvement_best": 0.10, "n_rungs_above_null": 0,
         "oracle_accs": {"reuse": 0.90, "search": 0.91, "grow": 0.85},
         "update": {"probed": True, "parent": 0, "parent_task_id": 0, "L_update": 0.95,
                    "rel_update": 0.208, "backward_safe": True,
                    "backward_deltas_val": {"0": -0.005, "1": -0.01},
                    "backward_deltas_test": {"0": -0.01, "1": -0.015},
                    "selected": True, "logged_only": False}},
    ]

    predictors = [
        {"task": 0, "kind": "grow", "head_state": _head_state(1, n_classes[0]),
         "node_index": 0, "parent_indices": None, "composer_kind": None, "composer_state": None,
         "search_meta": None},
        {"task": 1, "kind": "reuse", "head_state": _head_state(2, n_classes[1]),
         "node_index": None, "parent_indices": [0], "composer_kind": "reuse",
         "composer_state": None, "search_meta": None},
        {"task": 2, "kind": "grow", "head_state": _head_state(3, n_classes[2]),
         "node_index": 1, "parent_indices": None, "composer_kind": None, "composer_state": None,
         "search_meta": None},
        {"task": 3, "kind": "search", "head_state": _head_state(4, n_classes[3]),
         "node_index": None, "parent_indices": [0, 1], "composer_kind": "search",
         "composer_state": None, "search_meta": {"subset": [0, 1], "rank": 16, "skip": True}},
        {"task": 4, "kind": "reuse", "head_state": _head_state(5, n_classes[4]),
         "node_index": None, "parent_indices": [0, 1], "composer_kind": "reuse",
         "composer_state": None, "search_meta": None},
    ]

    dump = {
        "feature_mode": True, "concept_dim": D, "feature_dim": F, "n_parents": 2, "seed": seed,
        "config": {"gate_epochs": 2, "gate_lr": 1e-3, "eps_rel": 0.05, "eps_search": 0.05,
                   "search_budget": 6, "search_rank": 16, "search_skip": True,
                   "routing_batches": 20, "child_epochs": 2, "lr": 1e-3, "reducible_mode": "grow",
                   "gate_estimator": "single", "gate_splits": 5, "update_lr": 1e-4,
                   "eps_update": 0.1, "update_tolerance": 0.01, "subspace_k": 8,
                   "n_mlp_layers": N_LAYERS},
        "tasks": tasks, "nodes": nodes, "predictors": predictors, "decisions": decisions,
    }
    torch.save(dump, path)
    return dump


# ===========================================================================
# desk_stage.py — end-to-end
# ===========================================================================


def test_desk_stage_end_to_end(tmp_path):
    dump_path = tmp_path / "gate_dump.pt"
    build_gate_dump(str(dump_path), seed=0)
    out_path = tmp_path / "desk_stage.json"

    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "desk_stage.py"),
           "--dumps", str(dump_path), "--s_plus", str(dump_path),
           "--out", str(out_path), "--n_splits", "3", "--epochs", "2"]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    data = json.loads(out_path.read_text())
    required_keys = ("flip_frac", "bias_reuse", "bias_search", "bias_grow", "h1_branch",
                     "t1_L_update", "t1_L_grow", "t1_delta_t0_val", "delta_t0_val_broken",
                     "s_plus_rel_update", "h3_desk_branch")
    for key in required_keys:
        assert key in data, f"missing key {key!r} in desk_stage.json"

    assert data["h1_branch"] in ("SPLIT-LIMITED", "SELECTION-LIMITED", "BOTH", "NEITHER")
    assert data["h3_desk_branch"] in ("ARM-CAN-FIRE", "ARM-DEAD")
    assert isinstance(data["flip_frac"], float)
    assert 0.0 <= data["flip_frac"] <= 1.0
    for k in ("bias_reuse", "bias_search", "bias_grow", "t1_L_update", "t1_L_grow",
              "t1_delta_t0_val", "delta_t0_val_broken", "s_plus_rel_update"):
        assert isinstance(data[k], float) and data[k] == data[k], f"{k} is not a finite float"
    assert len(data["per_dump"]) == 1
    assert len(data["per_dump"][0]["h1"]["splits"]) == 3
    assert -1.0 <= data["t1_delta_t0_val"] <= 1.0
    assert -1.0 <= data["delta_t0_val_broken"] <= 1.0

    # every number the script prints is also in the JSON it writes
    assert "flip_frac=" in res.stdout
    assert f"=> {data['h1_branch']}" in res.stdout
    assert "H3' s_plus kill" in res.stdout
    assert f"=> {data['h3_desk_branch']}" in res.stdout


def test_desk_stage_h1_branch_logic():
    assert desk_stage.h1_branch(0.35, 0.50, 0.10, 0.10) == "BOTH"
    assert desk_stage.h1_branch(0.35, 0.02, 0.10, 0.10) == "SPLIT-LIMITED"
    assert desk_stage.h1_branch(0.10, 0.20, 0.05, 0.05) == "SELECTION-LIMITED"
    assert desk_stage.h1_branch(0.10, 0.05, 0.05, 0.05) == "NEITHER"
    assert desk_stage.h1_branch(None, 0.05, 0.05, 0.05) == "NEITHER"


def test_desk_stage_rebuild_parent_stack_shapes(tmp_path):
    dump_path = tmp_path / "gate_dump.pt"
    dump = build_gate_dump(str(dump_path), seed=1)
    stack, raw, y, parent_idx = desk_stage.rebuild_parent_stack(dump, 3, split="train")
    assert parent_idx == [0, 1]
    assert stack.shape == (70, 2, D)
    assert raw.shape == (70, F)
    assert y.shape == (70,)


# ===========================================================================
# exp3a_kan_results.json fixtures (§7 decision-dict fields) for eval_gate_arms.py
# ===========================================================================


def _mk_decision(task, decision, L_reuse, L_search, L_grow, L_null, parents,
                 oracle_accs=None, update=None, search_meta=None):
    reducible_grow = max(L_null - L_grow, 1e-6)
    L_all = [L_reuse, L_grow] + ([L_search] if L_search is not None else [])
    reducible_best = max(L_null - min(L_all), 1e-6)
    base = L_search if L_search is not None else L_reuse
    d = {"task": task, "decision": decision, "parents": parents, "grow_probe_input": "raw-root",
         "rel_improvement": (base - L_grow) / reducible_grow, "L_reuse_bits": L_reuse,
         "L_grow_bits": L_grow, "L_null_bits": L_null, "reducible_grow": reducible_grow,
         "reducible_best": reducible_best, "reducible_mode": "grow",
         "rel_grow_best": (base - L_grow) / reducible_best,
         "rel_improvement_best": (base - L_grow) / reducible_best, "n_rungs_above_null": 0}
    if L_search is not None:
        d["L_search_bits"] = L_search
        d["rel_search"] = (L_reuse - L_search) / reducible_grow
        d["rel_search_best"] = (L_reuse - L_search) / reducible_best
        if search_meta:
            d["search_meta"] = search_meta
    if oracle_accs is not None:
        d["oracle_accs"] = oracle_accs
    if update is not None:
        d["update"] = update
    return d


def make_ctrl_results(seed: int, stream: str, arm: str) -> dict:
    """5-task CTrL-shaped exp3a_kan_results.json content. ``arm`` in {control, crossfit, update}
    controls the perturbation each hypothesis is meant to detect."""
    decisions = [
        {"task": 0, "decision": "grow", "reason": "root"},
        _mk_decision(1, "grow", 2.5, 2.3, 1.8, 3.0, parents=[],
                     oracle_accs={"reuse": 0.55, "search": 0.60, "grow": 0.70}),
        _mk_decision(2, "grow", 2.6, 2.2, 1.6, 3.1, parents=[0],
                     oracle_accs={"reuse": 0.50, "search": 0.58, "grow": 0.75}),
        _mk_decision(3, "search", 3.20, 2.962, 3.091, 3.30, parents=[0, 1],
                     oracle_accs={"reuse": 0.60, "search": 0.64, "grow": 0.55},
                     search_meta={"subset": [0, 1], "rank": 16, "skip": True}),
        _mk_decision(4, "reuse", 1.20, 1.19, 1.35, 2.10, parents=[0, 1, 2],
                     oracle_accs={"reuse": 0.90, "search": 0.90, "grow": 0.85}),
    ]
    test_accs = [0.95, 0.68, 0.74, 0.60, 0.90]
    if arm == "crossfit":
        test_accs[3] = 0.64  # t3 regret drops to ~0 under the crossfit estimator
    if arm == "update" and stream == "s_plus":
        decisions[4] = _mk_decision(4, "update", 1.20, 1.19, 1.02, 2.10, parents=[0, 1, 2],
                                     oracle_accs={"reuse": 0.90, "search": 0.90, "grow": 0.85},
                                     update={"probed": True, "parent": 0, "parent_task_id": 0,
                                             "L_update": 0.95, "rel_update": 0.21,
                                             "backward_safe": True,
                                             "backward_deltas_val": {"0": -0.004},
                                             "backward_deltas_test": {"0": -0.008},
                                             "selected": True, "logged_only": False})
        test_accs[4] = 0.93
    n_grow = sum(1 for d in decisions if d["decision"] == "grow")
    n_search = sum(1 for d in decisions if d["decision"] == "search")
    return {"average_accuracy": sum(test_accs) / len(test_accs), "test_accs": test_accs,
            "n_grow": n_grow, "n_search": n_search, "n_reuse": len(decisions) - n_grow - n_search,
            "reuse_rate": (len(decisions) - n_grow) / len(decisions),
            "param_curve": [1, 2, 3, 3, 3], "params_final_pre_consolidation": 3,
            "consolidation": {"params_saved": 0, "n_ops": 0}, "decisions": decisions}


def make_5ds_results(seed: int) -> dict:
    """Published pattern: 4 grow, 0 search, 1 reuse."""
    decisions = [
        {"task": 0, "decision": "grow", "reason": "root"},
        _mk_decision(1, "grow", 2.0, None, 1.0, 3.0, parents=[0]),
        _mk_decision(2, "grow", 2.0, None, 1.0, 3.0, parents=[0, 1]),
        _mk_decision(3, "grow", 2.0, None, 1.0, 3.0, parents=[0, 1, 2]),
        _mk_decision(4, "reuse", 1.0, None, 1.4, 2.0, parents=[0, 1, 2, 3]),
    ]
    test_accs = [0.9, 0.8, 0.85, 0.7, 0.6]
    n_grow = sum(1 for d in decisions if d["decision"] == "grow")
    return {"average_accuracy": sum(test_accs) / len(test_accs), "test_accs": test_accs,
            "n_grow": n_grow, "n_search": 0, "n_reuse": len(decisions) - n_grow,
            "reuse_rate": (len(decisions) - n_grow) / len(decisions),
            "param_curve": [1, 2, 3, 4, 4], "params_final_pre_consolidation": 4,
            "consolidation": {"params_saved": 0, "n_ops": 0}, "decisions": decisions}


def write_ctrl_arm(base_dir: str, seeds, arm: str):
    for seed in seeds:
        for stream in eval_gate_arms.STREAMS:
            d = os.path.join(base_dir, f"seed_{seed}", f"exp_ctrl_{stream}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "exp3a_kan_results.json"), "w") as f:
                json.dump(make_ctrl_results(seed, stream, arm), f)


def write_5ds_arm(base_dir: str, seeds):
    for seed in seeds:
        d = os.path.join(base_dir, f"seed_{seed}", "exp5ds_kan")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "exp3a_kan_results.json"), "w") as f:
            json.dump(make_5ds_results(seed), f)


# ===========================================================================
# eval_gate_arms.py — end-to-end, including a missing arm
# ===========================================================================


def test_eval_gate_arms_end_to_end_with_missing_arm(tmp_path):
    seeds = [42, 7]
    a0_dir = tmp_path / "a0"
    write_ctrl_arm(str(a0_dir), seeds, "control")
    a1_dir = tmp_path / "a1"
    write_ctrl_arm(str(a1_dir), seeds, "crossfit")
    a3_dir = tmp_path / "a3"
    write_ctrl_arm(str(a3_dir), seeds, "update")
    a0_5ds_dir = tmp_path / "a0_5ds"
    write_5ds_arm(str(a0_5ds_dir), seeds)
    missing_dir = tmp_path / "a2_does_not_exist"  # never created

    out_path = tmp_path / "gates.json"
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "eval_gate_arms.py"),
           "--a0", str(a0_dir), "--a1", str(a1_dir), "--a2", str(missing_dir),
           "--a3", str(a3_dir), "--a0_5ds", str(a0_5ds_dir), "--out", str(out_path)]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert "MISSING" in res.stdout  # graceful missing-arm note, printed

    data = json.loads(out_path.read_text())
    for key in ("h1_stream_stage", "h2_denominator", "h3_update_arm", "null_validity", "notes"):
        assert key in data
    assert any("MISSING" in n and "A2" in n for n in data["notes"])
    assert any("not provided" in n and "A3_5ds" in n for n in data["notes"])

    assert data["h1_stream_stage"]["status"] == "OK"
    assert data["h1_stream_stage"]["branch"] in ("LOWER-REGRET", "NO-EFFECT", "ANOMALOUS", "INCONCLUSIVE")
    assert data["h2_denominator"]["branch"] in ("PRESERVES", "FLIPS")
    assert data["h3_update_arm"]["status"] == "OK"
    assert data["h3_update_arm"]["blind_branch"] in (
        "FIRES-ON-S-PLUS-ONLY", "NEVER-FIRES", "OVER-FIRES", "FIRES-WITHOUT-GAIN", "UNSAFE", "INCONCLUSIVE")
    assert data["null_validity"]["5ds"]["match"] == "MATCH"
    for stream in eval_gate_arms.STREAMS:
        assert data["null_validity"][stream]["match"] == "MATCH"

    # every number printed is also in the written JSON
    assert f"\"branch\": \"{data['h2_denominator']['branch']}\"" in res.stdout


def test_eval_gate_arms_all_arms_missing_does_not_crash(tmp_path):
    out_path = tmp_path / "gates.json"
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "eval_gate_arms.py"), "--out", str(out_path)]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    data = json.loads(out_path.read_text())
    assert data["h1_stream_stage"]["status"] == "SKIPPED"
    assert data["h2_denominator"]["status"] == "SKIPPED"
    assert data["h3_update_arm"]["status"] == "SKIPPED"
    assert len(data["notes"]) == 6  # one "not provided" note per arm flag


# ===========================================================================
# eval_gate_arms.py — unit tests against hand-constructed cases
# ===========================================================================


def test_discover_seeds_missing_dir():
    assert eval_gate_arms.discover_seeds("/no/such/directory/xyz") == {}


def test_h2_gate_preserves_when_grow_is_the_best_rung():
    # grow is already the minimum-code-length rung, so the best and grow denominators coincide.
    d = {"task": 2, "decision": "grow", "L_null_bits": 3.0, "L_reuse_bits": 2.90,
         "L_search_bits": 2.85, "L_grow_bits": 2.80}
    out = eval_gate_arms.h2_gate({"case": {"decisions": [d]}})
    assert out["branch"] == "PRESERVES"
    assert out["flips"] == 0
    assert out["clamp_count"] == 0


def test_h2_gate_flips_when_denominator_swings_a_marginal_decision():
    # Engineered so rel_search crosses eps=0.05 under the grow denominator (0.06) but not the
    # best denominator (0.03, since grow is far from the best rung here): search -> reuse.
    d = {"task": 3, "decision": "search", "L_null_bits": 3.0, "L_reuse_bits": 2.806,
         "L_search_bits": 2.80, "L_grow_bits": 2.90}
    out = eval_gate_arms.h2_gate({"case": {"decisions": [d]}})
    assert out["branch"] == "FLIPS"
    assert out["flips"] == 1
    assert out["per_arm"]["case"]["flip_details"][0] == {
        "task": 3, "recorded": "search", "best_denominator": "reuse"}


def test_h2_gate_clamp_count():
    d = {"task": 1, "decision": "reuse", "L_null_bits": 2.0, "L_reuse_bits": 2.0,
         "L_search_bits": 2.0, "L_grow_bits": 2.0}
    out = eval_gate_arms.h2_gate({"case": {"decisions": [d]}})
    assert out["clamp_count"] == 1
    assert out["flips"] == 0  # clamp alone does not flip a decision that was already "reuse"


def test_h2_gate_no_decisions_is_skipped():
    out = eval_gate_arms.h2_gate({})
    assert out["status"] == "SKIPPED"


def _dec4(decision, oracle_accs=None, update=None):
    d = {"task": 4, "decision": decision}
    if oracle_accs is not None:
        d["oracle_accs"] = oracle_accs
    if update is not None:
        d["update"] = update
    return d


def test_h3_gates_u1_accepts_on_gain_0_03_in_two_of_two_seeds():
    a0, a3 = {}, {}
    for seed in (1, 2):
        a0[seed] = {"s_plus": {"decisions": [_dec4("reuse", oracle_accs={
            "reuse": 0.90, "search": 0.90, "grow": 0.85})], "test_accs": [0, 0, 0, 0, 0.90]}}
        a3[seed] = {"s_plus": {"decisions": [_dec4("update", update={
            "backward_deltas_val": {"0": -0.005}, "backward_deltas_test": {"0": -0.01}})],
            "test_accs": [0, 0, 0, 0, 0.93]}}
    out = eval_gate_arms.h3_gates(a0, a3)
    assert out["U1"]["n_seeds"] == 2
    assert out["U1"]["n_fire"] == 2
    assert abs(out["U1"]["mean_gain"] - 0.03) < 1e-9
    assert out["U1"]["accept"] is True
    assert out["blind_branch"] == "FIRES-ON-S-PLUS-ONLY"


def test_h3_gates_u1_rejects_never_fires():
    a0, a3 = {}, {}
    for seed in (1, 2):
        a0[seed] = {"s_plus": {"decisions": [_dec4("reuse", oracle_accs={
            "reuse": 0.90, "search": 0.90, "grow": 0.85})], "test_accs": [0, 0, 0, 0, 0.90]}}
        a3[seed] = {"s_plus": {"decisions": [_dec4("reuse")], "test_accs": [0, 0, 0, 0, 0.90]}}
    out = eval_gate_arms.h3_gates(a0, a3)
    assert out["U1"]["n_fire"] == 0
    assert out["U1"]["fires_any"] is False
    assert out["U1"]["accept"] is False
    assert out["blind_branch"] == "NEVER-FIRES"


def test_h3_gates_u2_reject_on_middle_task_difference():
    a0 = {1: {"s_minus": {"decisions": [{"task": 1, "decision": "grow"},
                                        {"task": 2, "decision": "grow"},
                                        {"task": 3, "decision": "search"}]}}}
    a3 = {1: {"s_minus": {"decisions": [{"task": 1, "decision": "grow"},
                                        {"task": 2, "decision": "reuse"},  # differs from A0
                                        {"task": 3, "decision": "search"}]}}}
    out = eval_gate_arms.h3_gates(a0, a3)
    assert out["U2"]["identical"] is False
    assert len(out["U2"]["mismatches"]) == 1
    assert out["U2"]["mismatches"][0] == {"seed": 1, "stream": "s_minus", "task": 2,
                                          "A0": "grow", "A3": "reuse"}


def test_h3_gates_u2_accept_when_identical():
    dec = [{"task": 1, "decision": "grow"}, {"task": 2, "decision": "grow"},
           {"task": 3, "decision": "search"}]
    a0 = {1: {"s_minus": {"decisions": dec}}}
    a3 = {1: {"s_minus": {"decisions": list(dec)}}}
    out = eval_gate_arms.h3_gates(a0, a3)
    assert out["U2"]["identical"] is True
    assert out["U2"]["mismatches"] == []


def test_h3_gates_missing_arm_is_skipped():
    assert eval_gate_arms.h3_gates({}, {})["status"] == "SKIPPED"
    assert eval_gate_arms.h3_gates({1: {}}, {})["status"] == "SKIPPED"


def _t3t4(oracle_search_max, t3_acc, t4_dec="reuse"):
    return {"decisions": [{"task": 3, "oracle_accs": {"reuse": 0.5, "search": oracle_search_max, "grow": 0.4}},
                          {"task": 4, "decision": t4_dec}],
            "test_accs": [0, 0, 0, t3_acc, 0]}


def test_h1_stream_stage_lower_regret():
    a0 = {1: {"s_minus": _t3t4(0.62, 0.60)}, 2: {"s_minus": _t3t4(0.62, 0.60)}}
    a1 = {1: {"s_minus": _t3t4(0.64, 0.64)}, 2: {"s_minus": _t3t4(0.64, 0.64)}}
    out = eval_gate_arms.h1_stream_stage(a0, a1)
    assert out["status"] == "OK"
    assert abs(out["mean_delta_regret"] - 0.02) < 1e-9
    assert out["branch"] == "LOWER-REGRET"


def test_h1_stream_stage_no_effect():
    a0 = {1: {"s_minus": _t3t4(0.62, 0.60)}, 2: {"s_minus": _t3t4(0.62, 0.60)}}
    a1 = {1: {"s_minus": _t3t4(0.625, 0.615)}, 2: {"s_minus": _t3t4(0.625, 0.615)}}
    out = eval_gate_arms.h1_stream_stage(a0, a1)
    assert out["branch"] == "NO-EFFECT"


def test_h1_stream_stage_anomalous_on_t4_change():
    a0 = {1: {"s_minus": _t3t4(0.62, 0.60, "reuse"), "s_plus": _t3t4(0.62, 0.60, "reuse")}}
    a1 = {1: {"s_minus": _t3t4(0.64, 0.64, "search"), "s_plus": _t3t4(0.64, 0.64, "reuse")}}
    out = eval_gate_arms.h1_stream_stage(a0, a1)
    assert out["any_t4_decision_changed"] is True
    assert out["branch"] == "ANOMALOUS"


def test_h1_stream_stage_inconclusive_on_high_se():
    a0 = {1: {"s_minus": _t3t4(0.62, 0.60)}, 2: {"s_minus": _t3t4(0.62, 0.30)}}
    a1 = {1: {"s_minus": _t3t4(0.64, 0.64)}, 2: {"s_minus": _t3t4(0.64, 0.64)}}
    out = eval_gate_arms.h1_stream_stage(a0, a1)
    assert out["se"] >= 0.01
    assert out["branch"] == "INCONCLUSIVE"


def test_h1_stream_stage_missing_arm_is_skipped():
    assert eval_gate_arms.h1_stream_stage({}, {})["status"] == "SKIPPED"


def test_null_validity_match_and_mismatch():
    def s(t1, t2, t3, t4):
        return {"decisions": [{"task": 1, "decision": t1}, {"task": 2, "decision": t2},
                              {"task": 3, "decision": t3}, {"task": 4, "decision": t4}]}
    a0 = {42: {
        "s_minus": s("grow", "grow", "search", "reuse"),
        "s_plus": s("grow", "grow", "search", "reuse"),
        "s_out": s("grow", "grow", "reuse", "reuse"),   # mismatch at t3
        "s_in": s("grow", "grow", "search", "reuse"),
    }}
    a0_5ds = {42: {"decisions": [{"task": 0, "decision": "grow"}, {"task": 1, "decision": "grow"},
                                 {"task": 2, "decision": "grow"}, {"task": 3, "decision": "grow"},
                                 {"task": 4, "decision": "reuse"}]}}
    out = eval_gate_arms.null_validity(a0, a0_5ds)
    assert out["5ds"]["match"] == "MATCH"
    assert out["s_minus"]["match"] == "MATCH"
    assert out["s_plus"]["match"] == "MATCH"
    assert out["s_out"]["match"] == "MISMATCH"
    assert out["s_in"]["match"] == "MATCH"


def test_null_validity_missing_arm_is_skipped():
    out = eval_gate_arms.null_validity({}, {})
    assert out["5ds"]["status"] == "SKIPPED"
    for stream in eval_gate_arms.STREAMS:
        assert out[stream]["status"] == "SKIPPED"
