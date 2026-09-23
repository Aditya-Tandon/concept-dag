"""
Tests for the UNDETERMINED gate state and provisional (deferred) growth
([[provisional-growth-undetermined-gate]]), and for the `s_interleave` CTrL stream that carries
its merge/reuse/negative-control ground truth (G4).

CPU-only, seconds, synthetic tensors — no network, no downloaded datasets. `s_interleave` is
exercised the same way `tests/test_ctrl_loader_flags.py` exercises the other CTrL streams:
`loaders._load_raw_dataset` (the single torchvision hook) is monkeypatched with an in-memory
fake dataset, so `make_ctrl_stream` never touches disk.

Covers:
  1. `--provisional off` is inert: no `provisional`/`evalue`/`shadow_seconds`/`ladder_decision`
     key appears on any decision, and the explicit-off and flag-absent runs are identical.
  2. `provisional="shadow"` computes the third state on every gated decision but never acts
     (P0b): `evalue` is always present, `provisional` is always False, and decisions are
     otherwise identical to the "off" run.
  3. `provisional="always"` mints at least one provisional root, and every entry in
     `provisional_roots` ends resolved (merge or timeout) with a non-null `resolved_at` — i.e.
     no root is left flagged at stream end (P9).
  4. `s_interleave`'s shape and ground truth, and the provable index-disjointness between t1 and
     t3's training sets.
  5. CLI defaults (`build_parser`) and the `--provisional != off` requires `--enable_search`
     guard in `run_experiment.main()`.
"""

from __future__ import annotations

import json
import sys

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from concept_dag.experiments.kan_exp import KanExpConfig, run_exp3a_kan


# ---------------------------------------------------------------------------
# Synthetic feature-mode task stream (no encoder, no network — matches the schema
# `tests/test_gate_cache_max.py` / `tests/test_attn_root_adoption.py` synthesize by hand).
# ---------------------------------------------------------------------------


def _make_feature_tasks(n_tasks=4, n_classes=2, feature_dim=16, n_per_class=40,
                        batch_size=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    tasks = []
    for t in range(n_tasks):
        feats, labels = [], []
        for c in range(n_classes):
            center = torch.randn(feature_dim, generator=g) * 3.0
            x = center + torch.randn(n_per_class, feature_dim, generator=g)
            feats.append(x)
            labels.append(torch.full((n_per_class,), c, dtype=torch.long))
        X = torch.cat(feats)
        Y = torch.cat(labels)
        perm = torch.randperm(len(X), generator=g)
        X, Y = X[perm], Y[perm]

        n = len(X)
        n_tr, n_val = int(0.6 * n), int(0.2 * n)

        def _loader(lo, hi, shuffle):
            return DataLoader(TensorDataset(X[lo:hi], Y[lo:hi]), batch_size=batch_size,
                              shuffle=shuffle)

        tasks.append({
            "task_id": t,
            "train": _loader(0, n_tr, True),
            "val": _loader(n_tr, n_tr + n_val, False),
            "test": _loader(n_tr + n_val, n, False),
            "n_classes": n_classes,
            "name": f"synthetic-provisional-task-{t}",
            "feature_dim": feature_dim,
        })
    return tasks


def _base_cfg(results_dir, **overrides):
    defaults = dict(
        backbone="synthetic", feature_dim=16, concept_dim=16, cnn_out_dim=16,
        n_parents=1, subspace_k=4, soft_pca_k=4, routing_batches=4,
        root_epochs=3, child_epochs=3, gate_epochs=5, gate_lr=3e-3, lr=3e-3,
        eps_rel=0.1, batch_size=16, device="cpu", results_dir=str(results_dir),
        log_every=1000, gate_cache_max=0,   # unlimited: gate cache == the (small) full train set
    )
    defaults.update(overrides)
    return KanExpConfig(**defaults)


_DECISION_ONLY_KEYS = {"provisional", "evalue", "shadow_seconds", "ladder_decision"}
# Wall-clock, never claimed deterministic — excluded from every equality check below the same
# way `tests/fixtures/cls_identity_stream.py`'s `comparable()` excludes `gate_seconds`.
_WALL_TIME_KEYS = {"gate_seconds", "shadow_seconds"}


def _strip(decisions, keys=_DECISION_ONLY_KEYS):
    drop = set(keys) | _WALL_TIME_KEYS
    return [{k: v for k, v in d.items() if k not in drop} for d in decisions]


def _as_json(obj):
    return json.loads(json.dumps(obj))


# ===========================================================================
# 1. --provisional off is inert
# ===========================================================================


def test_provisional_off_is_inert(tmp_path):
    tasks_explicit = _make_feature_tasks(n_tasks=4, seed=1)
    cfg_explicit = _base_cfg(tmp_path / "explicit", provisional="off")
    res_explicit = run_exp3a_kan(cfg_explicit, tasks_explicit)

    tasks_default = _make_feature_tasks(n_tasks=4, seed=1)
    cfg_default = _base_cfg(tmp_path / "default")   # provisional left at its dataclass default
    assert cfg_default.provisional == "off"
    res_default = run_exp3a_kan(cfg_default, tasks_default)

    for res in (res_explicit, res_default):
        assert len(res["decisions"]) == 4
        for d in res["decisions"]:
            assert "provisional" not in d
            assert "evalue" not in d
            assert "shadow_seconds" not in d
            assert "ladder_decision" not in d

    assert _as_json(_strip(res_explicit["decisions"])) == _as_json(_strip(res_default["decisions"]))


# ===========================================================================
# 2. provisional="shadow" computes but never acts (P0b)
# ===========================================================================


_EVALUE_KEYS = {"state", "log2_e_plus", "log2_e_minus", "n", "n_needed_plus", "n_needed_minus",
                "alt_rung", "optimism", "clipped_hi", "clipped_lo", "se_proxy_fires"}


def test_provisional_shadow_computes_but_never_acts(tmp_path):
    tasks_off = _make_feature_tasks(n_tasks=4, seed=2)
    cfg_off = _base_cfg(tmp_path / "off", provisional="off",
                        enable_search=True, search_skip=True)
    res_off = run_exp3a_kan(cfg_off, tasks_off)

    tasks_shadow = _make_feature_tasks(n_tasks=4, seed=2)
    cfg_shadow = _base_cfg(tmp_path / "shadow", provisional="shadow",
                           enable_search=True, search_skip=True)
    res_shadow = run_exp3a_kan(cfg_shadow, tasks_shadow)

    gated = [d for d in res_shadow["decisions"] if d["task"] > 0]
    assert gated, "expected at least one gated (non-root) decision"
    for d in gated:
        assert "evalue" in d
        ev = d["evalue"]
        assert _EVALUE_KEYS <= set(ev.keys()), f"missing evalue keys: {_EVALUE_KEYS - set(ev)}"
        assert ev["state"] in {"grow", "alt", "undetermined"}
        assert d["provisional"] is False

    assert _as_json(_strip(res_off["decisions"])) == _as_json(_strip(res_shadow["decisions"]))


# ===========================================================================
# 3. provisional="always" mints, and every provisional root ends resolved (P6/P9)
# ===========================================================================


def test_provisional_always_mints_and_resolves(tmp_path):
    tasks = _make_feature_tasks(n_tasks=4, seed=3)
    cfg = _base_cfg(tmp_path, provisional="always", enable_search=True, search_skip=True,
                    always_n_max=1000)
    res = run_exp3a_kan(cfg, tasks)

    minted = [d for d in res["decisions"] if d.get("provisional") is True]
    assert minted, "expected at least one minted provisional root under the 'always' arm"

    roots = res["provisional_roots"]
    assert roots, "expected a non-empty provisional_roots list"
    for rec in roots:
        assert rec["resolution"] in {"merge", "timeout"}
        assert rec["resolved_at"] is not None
    # P9: every provisional root resolved by stream end — asserted via the returned records
    # above (a leak would show up as `resolution is None`), per the task's own instruction not
    # to reach into node internals for this check.


# ===========================================================================
# 4. s_interleave: shape, ground truth, and provable t1/t3 index-disjointness
# ===========================================================================


class _FakeImageDataset(Dataset):
    """Minimal stand-in for a torchvision dataset (mirrors test_ctrl_loader_flags.py)."""

    def __init__(self, n, n_classes=10, img_size=4):
        self.n = n
        self.images = torch.randn(n, 3, img_size, img_size)
        self.labels = torch.randint(0, n_classes, (n,))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.images[idx], int(self.labels[idx])


def _install_fake_loader(monkeypatch, train_len, test_len):
    from concept_dag.data import loaders as loaders_mod

    def _fake_load_raw_dataset(name, data_root, train, transform, download):
        return _FakeImageDataset(train_len if train else test_len)

    monkeypatch.setattr(loaders_mod, "_load_raw_dataset", _fake_load_raw_dataset)


def test_s_interleave_shape_and_ground_truth(monkeypatch):
    from concept_dag.data.loaders import make_ctrl_stream

    _install_fake_loader(monkeypatch, train_len=6000, test_len=2000)

    tasks = make_ctrl_stream(
        "s_interleave", data_root="unused", first_dataset="mnist",
        middle_datasets=["fashion", "kmnist", "svhn"],
        n_large=4000, n_small=400, batch_size=32, download=False, seed=0,
    )

    assert len(tasks) == 5
    assert [t["dataset"] for t in tasks] == ["mnist", "fashion", "svhn", "fashion", "mnist"]

    ctrl = [t["ctrl"] for t in tasks]
    assert ctrl[0] == {"revisit_of": None, "relation": None, "n_train": 4000, "should": None}
    assert ctrl[1] == {"revisit_of": None, "relation": None, "n_train": 400, "should": None}
    assert ctrl[2] == {"revisit_of": None, "relation": None, "n_train": 400, "should": "not_merge"}
    assert ctrl[3] == {"revisit_of": 1, "relation": "same", "n_train": 400, "should": "merge"}
    assert ctrl[4] == {"revisit_of": 0, "relation": "same", "n_train": 400, "should": "reuse"}

    # Index disjointness between t1 and t3's TRAINING sets must be provable, not a shuffle
    # coincidence: assert directly on the stored Subset indices.
    t1_train_idx = tasks[1]["train"].dataset.indices
    t3_train_idx = tasks[3]["train"].dataset.indices
    assert len(t1_train_idx) == 400 and len(t3_train_idx) == 400
    assert set(t1_train_idx).isdisjoint(set(t3_train_idx))


def test_s_interleave_default_middles(monkeypatch):
    """Default `middle_datasets` (the other four of the five datasets) still works: t2 falls
    back to middles[1] only when fewer than 3 middles are given; with the full default set of
    4, t2 is middles[2]."""
    from concept_dag.data.loaders import make_ctrl_stream

    _install_fake_loader(monkeypatch, train_len=6000, test_len=2000)

    tasks = make_ctrl_stream("s_interleave", data_root="unused", first_dataset="mnist",
                             n_large=4000, n_small=400, batch_size=32, download=False, seed=0)
    assert len(tasks) == 5
    # default middles (excluding "mnist"): fashion, kmnist, svhn, cifar10 -> middles[2] = svhn
    assert tasks[1]["dataset"] == "fashion"
    assert tasks[2]["dataset"] == "svhn"
    assert tasks[3]["dataset"] == "fashion"

    tasks2 = make_ctrl_stream("s_interleave", data_root="unused", first_dataset="mnist",
                              middle_datasets=["fashion", "kmnist"],
                              n_large=4000, n_small=400, batch_size=32, download=False, seed=0)
    # only 2 middles given -> falls back to middles[1] = "kmnist"
    assert tasks2[2]["dataset"] == "kmnist"


def test_s_interleave_unknown_stream_still_rejected(monkeypatch):
    from concept_dag.data.loaders import make_ctrl_stream

    with pytest.raises(ValueError):
        make_ctrl_stream("s_bogus", data_root="unused")


# ===========================================================================
# 5. CLI defaults and the --provisional/--enable_search guard
# ===========================================================================


def test_cli_provisional_flag_defaults():
    from run_experiment import build_parser

    args = build_parser().parse_args([])
    assert args.provisional == "off"
    assert args.provisional_alpha == 0.05
    assert args.provisional_z == 1.0
    assert args.always_n_max == 1000
    assert args.crystallise_after == 3


def test_cli_provisional_flag_overrides_and_ctrl_stream_choice():
    from run_experiment import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--provisional", "se_proxy", "--provisional_alpha", "0.1", "--provisional_z", "2.0",
        "--always_n_max", "500", "--crystallise_after", "5",
        "--enable_search", "--ctrl_stream", "s_interleave",
    ])
    assert args.provisional == "se_proxy"
    assert args.provisional_alpha == 0.1
    assert args.provisional_z == 2.0
    assert args.always_n_max == 500
    assert args.crystallise_after == 5
    assert args.ctrl_stream == "s_interleave"


def test_provisional_evalue_without_enable_search_exits(monkeypatch):
    import run_experiment

    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--provisional", "evalue"])
    with pytest.raises(SystemExit):
        run_experiment.main()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
