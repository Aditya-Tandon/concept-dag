"""
Tests for the gate-estimator / denominator / update-arm CLI flags and the
`make_ctrl_stream(val_frac=, n_test=)` loader plumbing (INTERFACE_SPEC.md §2).

No dataset downloads: `_load_raw_dataset` (the loader's single torchvision hook) is
monkeypatched with a tiny in-memory Dataset so `make_ctrl_stream` runs entirely on
synthetic tensors.

`KanExpConfig` (concept_dag/experiments/kan_exp.py) is being extended concurrently on
this branch with the new fields (`reducible_mode`, `gate_estimator`, ... — see
INTERFACE_SPEC.md §1). This file does not import those fields; it only checks the CLI
parser's flags/defaults and the `make_ctrl_stream` loader behaviour, which are this
assignment's actual surface.
"""

import torch
from torch.utils.data import Dataset

from concept_dag.data import loaders as loaders_mod
from concept_dag.data.loaders import make_ctrl_stream


class _FakeImageDataset(Dataset):
    """Minimal stand-in for a torchvision dataset: fixed length, random tiny tensors."""

    def __init__(self, n, n_classes=10, img_size=4):
        self.n = n
        self.images = torch.randn(n, 3, img_size, img_size)
        self.labels = torch.randint(0, n_classes, (n,))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.images[idx], int(self.labels[idx])


def _install_fake_loader(monkeypatch, train_len, test_len):
    """Replace loaders._load_raw_dataset so make_ctrl_stream never touches torchvision."""

    def _fake_load_raw_dataset(name, data_root, train, transform, download):
        return _FakeImageDataset(train_len if train else test_len)

    monkeypatch.setattr(loaders_mod, "_load_raw_dataset", _fake_load_raw_dataset)


# ─────────────────────────────────────────────────────────────────────────────
# (1) n_test=None keeps max(500, n_train // 4); val is drawn IN ADDITION to n_train.
# ─────────────────────────────────────────────────────────────────────────────

def test_n_test_none_keeps_default_and_val_is_additive(monkeypatch):
    _install_fake_loader(monkeypatch, train_len=3000, test_len=3000)
    tasks = make_ctrl_stream(
        "s_minus", data_root="unused", first_dataset="mnist",
        middle_datasets=["fashion"], n_large=50, n_small=400,
        batch_size=32, val_frac=0.1, download=False, seed=0, n_test=None,
    )
    # tasks: [0]=mnist (n_large=50), [1]=fashion (n_small=400), [2]=mnist revisit (n_small=400)
    middle = tasks[1]
    assert middle["ctrl"]["n_train"] == 400
    assert len(middle["train"].dataset) == 400          # training set exactly n_train
    assert len(middle["val"].dataset) == 40              # val_frac=0.1 of 400
    assert len(middle["test"].dataset) == 500             # max(500, 400 // 4)


# ─────────────────────────────────────────────────────────────────────────────
# (2) n_test as an int caps at min(n_test, len(test_full)); val_frac scales val only,
#     training count is unaffected.
# ─────────────────────────────────────────────────────────────────────────────

def test_n_test_int_capped_by_available_and_val_frac_does_not_shrink_train(monkeypatch):
    _install_fake_loader(monkeypatch, train_len=2000, test_len=1200)
    tasks = make_ctrl_stream(
        "s_minus", data_root="unused", first_dataset="mnist",
        middle_datasets=["fashion"], n_large=50, n_small=400,
        batch_size=32, val_frac=0.5, download=False, seed=0, n_test=5000,
    )
    middle = tasks[1]
    assert middle["ctrl"]["n_train"] == 400
    assert len(middle["train"].dataset) == 400            # training set unaffected by val_frac
    assert len(middle["val"].dataset) == 200               # val_frac=0.5 of 400
    assert len(middle["test"].dataset) == 1200             # min(5000, len(test_full)=1200)


# ─────────────────────────────────────────────────────────────────────────────
# (3) argparse defaults for the new flags.
# ─────────────────────────────────────────────────────────────────────────────

def test_cli_flag_defaults():
    from run_experiment import build_parser

    args = build_parser().parse_args([])
    assert args.reducible == "grow"
    assert args.gate_estimator == "single"
    assert args.gate_splits == 5
    assert args.routing_batches == 20
    assert args.ctrl_val_frac == 0.1
    assert args.ctrl_n_test is None
    assert args.update_lr == 1e-4
    assert args.eps_update == 0.1
    assert args.update_tolerance == 0.01
    assert args.oracle_rungs is False
    assert args.dump_gate_tensors is False
    assert args.enable_update is False


def test_cli_flag_choices_and_overrides():
    from run_experiment import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--reducible", "best", "--gate_estimator", "crossfit", "--gate_splits", "3",
        "--oracle_rungs", "--dump_gate_tensors", "--enable_update",
        "--update_lr", "5e-3", "--eps_update", "0.2", "--update_tolerance", "0.05",
        "--routing_batches", "10", "--ctrl_val_frac", "0.2", "--ctrl_n_test", "1000",
    ])
    assert args.reducible == "best"
    assert args.gate_estimator == "crossfit"
    assert args.gate_splits == 3
    assert args.oracle_rungs is True
    assert args.dump_gate_tensors is True
    assert args.enable_update is True
    assert args.update_lr == 5e-3
    assert args.eps_update == 0.2
    assert args.update_tolerance == 0.05
    assert args.routing_batches == 10
    assert args.ctrl_val_frac == 0.2
    assert args.ctrl_n_test == 1000
