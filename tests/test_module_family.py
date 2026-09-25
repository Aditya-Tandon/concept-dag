"""
Tests for the H6 module-family ablation (INTERFACE_SPEC_H6.md §1-§6):
`concept_dag/modules/token_roots.py`, the token path of `concept_dag/data/feature_cache.py`,
`DINOv2Encoder(return_tokens=True)`, `scripts/module_family_ablation.py` and
`scripts/eval_module_family.py`.

Everything here runs on CPU in seconds and touches no network: the DINOv2 hub model is
never loaded — its token reshape/pool path is exercised against a fake `forward_features`
dict, which is the only way this laptop (py3.9, no hub access) can guard the code the pod
actually runs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from concept_dag.modules.token_roots import (  # noqa: E402
    ARM_OF, FAMILIES, FAMILY_INPUT, PREREGISTERED, build_root, count_params,
)


def _load_script(name: str, filename: str):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ablation = _load_script("module_family_ablation", "module_family_ablation.py")
gate = _load_script("eval_module_family", "eval_module_family.py")


# ===========================================================================
# §3 — root module families
# ===========================================================================

def _stub_tokens(batch: int = 8, n_tokens: int = 65, dim: int = 384) -> torch.Tensor:
    """Stub encoder output: random (B, 1 + T, 384) token sequences."""
    g = torch.Generator().manual_seed(0)
    return torch.randn(batch, n_tokens, dim, generator=g)


def _root(family: str):
    """Build a family in eval mode — the MLP body has dropout, so train mode is stochastic."""
    return build_root(family, feature_dim=384, concept_dim=128).eval()


def test_every_family_builds_and_returns_128d():
    tokens = _stub_tokens()
    images = torch.randn(8, 3, 32, 32)
    counts = {}
    for family in FAMILIES:
        root = _root(family)
        assert root.out_dim == 128
        assert len(root.trainable_parameters()) > 0

        if FAMILY_INPUT[family] == "image":
            x = images
        elif FAMILY_INPUT[family] == "cls":
            x = tokens[:, 0]
        else:
            x = tokens
        out = root(x)
        assert out.shape == (8, 128), f"{family}: {tuple(out.shape)}"
        counts[family] = count_params(root)

    print("\n[h6] root family parameter counts (65 tokens, concept_dim 128):")
    for family, n in counts.items():
        print(f"    {ARM_OF[family]:<3}{family:<20}{n:>10,}")

    # Ranges from the note's arm table.
    assert 60_000 <= counts["mlp_cls"] <= 70_000               # A ≈ 66k
    assert counts["mlp_meanpool"] == counts["mlp_cls"]         # superseded B is A's body
    assert counts["mlp_cls_meanpool"] < 130_000                # B: 768-wide first layer
    assert counts["mlp_cls_meanpool"] > counts["mlp_cls"]
    assert counts["proj_uniform_pool"] < counts["attn_pool"]   # C0 = C minus the attention
    assert counts["attn_pool"] < 150_000                       # C
    assert counts["self_attn"] < 400_000                       # D
    assert counts["cnn"] > counts["self_attn"]                 # E: a whole conv stack


def test_preregistered_set_is_the_notes_six_arms():
    assert PREREGISTERED == ("mlp_cls", "mlp_cls_meanpool", "proj_uniform_pool",
                             "attn_pool", "self_attn", "cnn")
    assert [ARM_OF[f] for f in PREREGISTERED] == ["A", "B", "C0", "C", "D", "E"]


def test_mlp_cls_accepts_the_shared_token_cache():
    """Arm A reads index 0 of the same cache arms B-D use."""
    tokens = _stub_tokens()
    root = _root("mlp_cls")
    assert torch.allclose(root(tokens), root(tokens[:, 0]))


def test_arm_b_contains_arm_a():
    """B = CLS ⊕ mean patches: both halves must actually reach the MLP."""
    root = _root("mlp_cls_meanpool")
    tokens = _stub_tokens(batch=4)
    assert root.concept_module.in_dim == 768

    cls_hit = tokens.clone()
    cls_hit[:, 0] += 5.0
    patch_hit = tokens.clone()
    patch_hit[:, 1:] += 5.0
    base = root(tokens)
    assert not torch.allclose(base, root(cls_hit))      # CLS is read
    assert not torch.allclose(base, root(patch_hit))    # patches are read


def test_meanpool_excludes_cls():
    root = _root("mlp_meanpool")
    tokens = _stub_tokens(batch=4)
    poisoned = tokens.clone()
    poisoned[:, 0] = 1e3           # wreck the CLS slot only
    assert torch.allclose(root(tokens), root(poisoned))


def test_c0_is_c_with_uniform_pooling():
    """C0 must be capacity-comparable to C and permutation-invariant over the tokens."""
    c0, c = _root("proj_uniform_pool"), _root("attn_pool")
    tokens = _stub_tokens(batch=3)
    # Uniform pooling ignores token order; the learned query is applied to all tokens too,
    # so both are permutation invariant — but C0 must have NO query parameter at all.
    assert not any("query" in n for n, _ in c0.named_parameters())
    assert any("query" in n for n, _ in c.named_parameters())
    perm = torch.randperm(tokens.shape[1])
    assert torch.allclose(c0(tokens), c0(tokens[:, perm]), atol=1e-5)
    # C0's pooled vector is the mean of the projected tokens.
    pooled = c0.norm(c0.value(tokens).mean(dim=1))
    assert torch.allclose(c0(tokens), c0.concept_module(pooled))


def test_token_families_reject_wrong_input_shape():
    for family in ("mlp_meanpool", "mlp_cls_meanpool", "proj_uniform_pool",
                   "attn_pool", "self_attn"):
        with pytest.raises(ValueError):
            build_root(family)(torch.randn(4, 384))
    with pytest.raises(ValueError):
        build_root("cnn")(torch.randn(4, 384))
    with pytest.raises(ValueError):
        build_root("no_such_family")


def test_cnn_root_matches_the_dagnode_root_it_reuses():
    """Arm E must be the existing `DAGNode(use_cnn=True)` root, not a lookalike."""
    from concept_dag.experiments.exp3_growing_dag import DAGNode

    node = DAGNode(task_id=0, concept_dim=128, cnn_out_dim=256, n_mlp_layers=2,
                   use_cnn=True)
    root = build_root("cnn", concept_dim=128)
    assert count_params(root) == count_params(node)
    node_shapes = sorted(tuple(p.shape) for p in node.parameters())
    root_shapes = sorted(tuple(p.shape) for p in root.parameters())
    assert node_shapes == root_shapes


def test_orth_loss_is_a_no_op_where_it_exists():
    """Root ConceptModules have no aggregator, so the DAG's orth term is identically 0."""
    for family in ("mlp_cls", "mlp_meanpool", "mlp_cls_meanpool", "cnn"):
        assert float(build_root(family).orth_loss()) == 0.0
    for family in ("proj_uniform_pool", "attn_pool", "self_attn"):
        assert not hasattr(build_root(family), "orth_loss")


# ===========================================================================
# §1 — DINOv2Encoder(return_tokens=True), exercised without torch.hub
# ===========================================================================

class _FakeDino(nn.Module):
    """Stands in for the hub model: CLS = -1, patch (r, c) = r * side + c."""

    def __init__(self, side: int = 16, dim: int = 384, keys=("x_norm_clstoken",
                                                             "x_norm_patchtokens")):
        super().__init__()
        self.side, self.dim, self.keys = side, dim, keys
        self.called_plain = 0

    def forward(self, x):
        self.called_plain += 1
        return torch.zeros(x.shape[0], self.dim)

    def forward_features(self, x):
        B = x.shape[0]
        vals = torch.arange(self.side * self.side, dtype=torch.float32)
        patches = vals.view(1, -1, 1).expand(B, -1, self.dim).clone()
        out = {"x_norm_clstoken": torch.full((B, self.dim), -1.0),
               "x_norm_patchtokens": patches}
        return {k: v for k, v in out.items() if k in self.keys}


def _fake_encoder(return_tokens=True, token_pool=2, keys=("x_norm_clstoken",
                                                          "x_norm_patchtokens")):
    from concept_dag.models.root_encoder import DINOv2Encoder
    enc = DINOv2Encoder.__new__(DINOv2Encoder)      # bypass torch.hub.load
    nn.Module.__init__(enc)
    enc.return_tokens, enc.token_pool, enc._grid = return_tokens, token_pool, 16
    enc.model = _FakeDino(keys=keys)
    return enc


def test_n_tokens_property():
    assert _fake_encoder(return_tokens=False).n_tokens == 1
    assert _fake_encoder(token_pool=1).n_tokens == 257
    assert _fake_encoder(token_pool=2).n_tokens == 65
    assert _fake_encoder(token_pool=4).n_tokens == 17


def test_cls_only_path_is_untouched():
    """return_tokens=False must still be `self.model(x)` and nothing else."""
    enc = _fake_encoder(return_tokens=False)
    out = enc(torch.randn(3, 3, 224, 224))
    assert out.shape == (3, 384)
    assert enc.model.called_plain == 1


def test_token_forward_layout_and_pooling():
    enc = _fake_encoder(token_pool=2)
    out = enc(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 65, 384), tuple(out.shape)
    assert torch.allclose(out[:, 0], torch.full((2, 384), -1.0))   # CLS at index 0

    # 16x16 grid pooled 2x2: token 0 averages patches {0, 1, 16, 17} = 8.5,
    # token 1 averages {2, 3, 18, 19} = 10.5, token 8 (next pooled row) = 40.5.
    assert pytest.approx(float(out[0, 1, 0]), abs=1e-4) == 8.5
    assert pytest.approx(float(out[0, 2, 0]), abs=1e-4) == 10.5
    assert pytest.approx(float(out[0, 9, 0]), abs=1e-4) == 40.5

    unpooled = _fake_encoder(token_pool=1)(torch.randn(1, 3, 224, 224))
    assert unpooled.shape == (1, 257, 384)
    assert pytest.approx(float(unpooled[0, 1, 0])) == 0.0
    assert pytest.approx(float(unpooled[0, 17, 0])) == 16.0    # row 1, col 0


def test_token_forward_fails_loudly_on_a_changed_hub_api():
    enc = _fake_encoder(keys=("x_norm_clstoken",))
    with pytest.raises(RuntimeError, match="x_norm_patchtokens"):
        enc(torch.randn(1, 3, 224, 224))


def test_token_pool_must_divide_the_grid():
    from concept_dag.models.root_encoder import DINOv2Encoder
    enc = DINOv2Encoder.__new__(DINOv2Encoder)
    nn.Module.__init__(enc)
    enc.return_tokens, enc.token_pool, enc._grid = True, 3, 16
    enc.model = _FakeDino()
    with pytest.raises(RuntimeError, match="token_pool"):
        enc(torch.randn(1, 3, 224, 224))


# ===========================================================================
# §2 — token feature cache
# ===========================================================================

def _stub_encoder(token_pool: int = 4):
    return ablation.StubTokenEncoder(token_pool=token_pool)


def test_token_cache_roundtrips_float16_to_float32():
    from concept_dag.data.feature_cache import cache_split_features

    enc = _stub_encoder(token_pool=4)       # 2x2 grid + CLS = 5 tokens
    images = torch.randn(12, 3, 32, 32)
    labels = torch.randint(0, 10, (12,))
    ds = TensorDataset(images, labels)

    with tempfile.TemporaryDirectory() as tmp:
        feats, ys = cache_split_features(enc, ds, tmp, "svhn_train", tokens=True,
                                         batch_size=5)
        assert feats.shape == (12, enc.n_tokens, 384)
        assert feats.dtype == torch.float32
        assert torch.equal(ys, labels)

        on_disk = torch.load(os.path.join(tmp, "svhn_train_features.pt"),
                             map_location="cpu", weights_only=True)
        assert on_disk.dtype == torch.float16
        assert on_disk.shape == feats.shape

        # A cache hit must return exactly what the cache-miss call returned.
        again, _ = cache_split_features(enc, ds, tmp, "svhn_train", tokens=True)
        assert torch.equal(again, feats)
        assert torch.equal(feats, on_disk.float())


def test_cache_features_token_mode_suffix_meta_and_dataset():
    from concept_dag.data.feature_cache import cache_features
    from torch.utils.data import DataLoader

    enc = _stub_encoder(token_pool=4)
    images, labels = torch.randn(10, 3, 32, 32), torch.randint(0, 5, (10,))

    def _task(t):
        def dl(shuffle):
            return DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=shuffle)
        return {"task_id": t, "train": dl(True), "val": dl(False), "test": dl(False),
                "n_classes": 5, "class_ids": list(range(5)), "name": f"t{t}"}

    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "cache")
        tasks = cache_features(enc, [_task(0)], cache_dir=base, device="cpu",
                               tokens=True, token_pool=4)
        tok_dir = f"{base}_tok{enc.n_tokens - 1}"
        assert os.path.isdir(tok_dir), os.listdir(tmp)
        meta = json.load(open(os.path.join(tok_dir, "meta.json")))
        assert meta["tokens"] is True and meta["n_tokens"] == enc.n_tokens
        assert meta["token_pool"] == 4 and meta["feature_dim"] == 384
        assert tasks[0]["n_tokens"] == enc.n_tokens

        # FeatureTensorDataset must hand the token tensor back as `x`.
        x, y = next(iter(tasks[0]["test"]))
        assert x.shape[1:] == (enc.n_tokens, 384) and x.dtype == torch.float32
        assert y.shape[0] == x.shape[0]


def test_cls_only_cache_path_unchanged_by_token_support():
    """A pre-token CLS cache (meta without the token keys) still hits, never recomputes."""
    from concept_dag.data.feature_cache import cache_features
    from torch.utils.data import DataLoader

    class _CLSEncoder(nn.Module):
        feature_dim = 8

        def forward(self, x):
            return x.flatten(1)[:, : self.feature_dim]

    images, labels = torch.randn(8, 3, 4, 4), torch.randint(0, 5, (8,))

    def _task():
        def dl(shuffle):
            return DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=shuffle)
        return [{"task_id": 0, "train": dl(True), "val": dl(False), "test": dl(False),
                 "n_classes": 5, "class_ids": list(range(5)), "name": "t0"}]

    with tempfile.TemporaryDirectory() as tmp:
        cache_features(_CLSEncoder(), _task(), cache_dir=tmp, device="cpu", seed=0)
        meta_path = os.path.join(tmp, "meta.json")
        legacy = json.load(open(meta_path))
        legacy.pop("tokens", None)
        legacy.pop("n_tokens", None)
        json.dump(legacy, open(meta_path, "w"))       # simulate a cache written pre-H6

        feats = torch.load(os.path.join(tmp, "task_0_train_features.pt"),
                           map_location="cpu", weights_only=True)
        assert feats.dtype == torch.float32 and feats.shape == (8, 8)
        marker = feats.clone()
        marker[0, 0] = 12345.0                        # if it recomputes, this vanishes
        torch.save(marker, os.path.join(tmp, "task_0_train_features.pt"))

        tasks = cache_features(_CLSEncoder(), _task(), cache_dir=tmp, device="cpu", seed=0)
        x, _ = next(iter(tasks[0]["train"]))
        assert x.shape[1] == 8
        reread = torch.load(os.path.join(tmp, "task_0_train_features.pt"),
                            map_location="cpu", weights_only=True)
        assert float(reread[0, 0]) == 12345.0, "CLS-only cache was recomputed"


# ===========================================================================
# §4 — the ablation script, end to end on CPU with --backbone stub
# ===========================================================================

def test_build_splits_val_is_additional_and_test_regime_scales():
    full = ablation.build_splits(1000, 300, "full", seed=42, n_val=100, n_test_small=50)
    assert len(full["val"]) == 100
    assert len(full["train"]) == 900              # val comes off the pool, on top of n_train
    assert len(full["test"]) == 300               # full regime = the whole test split
    assert not set(full["train"]) & set(full["val"])

    small = ablation.build_splits(1000, 300, "400", seed=42, n_val=100, n_test_small=50)
    assert len(small["train"]) == 400 and len(small["val"]) == 100
    assert len(small["test"]) == 50
    assert not set(small["train"]) & set(small["val"])
    # Same seed ⇒ same val split in both regimes ⇒ every family sees the same images.
    assert list(small["val"]) == list(full["val"])

    other = ablation.build_splits(1000, 300, "400", seed=43, n_val=100, n_test_small=50)
    assert list(other["val"]) != list(small["val"])
    assert ablation.split_hash(small) != ablation.split_hash(other)
    assert ablation.split_hash(small) == ablation.split_hash(
        ablation.build_splits(1000, 300, "400", seed=42, n_val=100, n_test_small=50))

    with pytest.raises(ValueError):
        ablation.build_splits(1000, 300, "9999", seed=42, n_val=100, n_test_small=50)


def _run_ablation(out, cache, extra=()):
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "module_family_ablation.py"),
           "--dataset", "synthetic", "--backbone", "stub", "--n_train", "60",
           "--epochs", "1", "--n_val", "20", "--n_test", "40",
           "--synthetic_n", "200", "--seeds", "42", "--batch_size", "20",
           "--out", out, "--cache_dir", cache, "--device", "cpu", "--log_every", "0",
           *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)


def test_ablation_script_runs_every_preregistered_family(tmp_path):
    out, cache = str(tmp_path / "results"), str(tmp_path / "cache")
    proc = _run_ablation(out, cache, ["--token_pool", "4"])
    assert proc.returncode == 0, proc.stdout + proc.stderr

    regime = os.path.join(out, "synthetic_60_pool4")
    hashes = set()
    for family in PREREGISTERED:
        path = os.path.join(regime, f"family_{family}_seed_42.json")
        assert os.path.exists(path), os.listdir(regime)
        rec = json.load(open(path))
        for key in ("test_acc", "val_acc", "val_bits", "test_bits", "params", "wall_s",
                    "val_curve", "n_train_images", "n_val_images", "n_test_images",
                    "family", "seed", "arm", "split_hash", "token_pool"):
            assert key in rec, key
        assert rec["n_train_images"] == 60
        assert rec["n_val_images"] == 20
        assert rec["n_test_images"] == 40
        assert len(rec["val_curve"]) == 1
        assert 0.0 <= rec["test_acc"] <= 1.0
        assert rec["val_bits"] > 0 and rec["test_bits"] > 0
        assert rec["params"] == count_params(build_root(family))
        # Arm E reads no tokens, so it carries no pool value.
        assert rec["token_pool"] == (None if FAMILY_INPUT[family] == "image" else 4)
        hashes.add(rec["split_hash"])

    # Paired design: one index draw per seed, shared by every family.
    assert len(hashes) == 1, hashes

    summary = json.load(open(os.path.join(out, "summary.json")))
    assert set(summary["synthetic_60_pool4"]) == {f for f in PREREGISTERED
                                                  if FAMILY_INPUT[f] != "image"}
    assert set(summary["synthetic_60"]) == {"cnn"}    # E is pool-free
    assert os.path.exists(os.path.join(regime, "summary.json"))
    cache_dirs = os.listdir(cache)
    assert len(cache_dirs) == 1 and cache_dirs[0].endswith("_tok4"), cache_dirs


def test_ablation_token_pool_sweep_writes_one_dir_and_one_cache_per_pool(tmp_path):
    out, cache = str(tmp_path / "results"), str(tmp_path / "cache")
    proc = _run_ablation(out, cache, ["--token_pool", "4", "8",
                                      "--families", "mlp_cls", "attn_pool", "cnn"])
    assert proc.returncode == 0, proc.stdout + proc.stderr

    assert sorted(d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d))) == \
        ["synthetic_60_pool4", "synthetic_60_pool8"]
    assert sorted(os.listdir(cache)) == ["synthetic_stub_p4_tok4",
                                         "synthetic_stub_p8_tok1"]

    # Arm E is trained once, under the first pool value only.
    assert os.path.exists(os.path.join(out, "synthetic_60_pool4", "family_cnn_seed_42.json"))
    assert not os.path.exists(os.path.join(out, "synthetic_60_pool8",
                                           "family_cnn_seed_42.json"))
    # Same seed ⇒ same images at every pool value.
    h = {p: json.load(open(os.path.join(out, f"synthetic_60_pool{p}",
                                        "family_attn_pool_seed_42.json")))["split_hash"]
         for p in (4, 8)}
    assert h[4] == h[8]


def test_ablation_is_deterministic_under_a_fixed_seed(tmp_path):
    """Two runs at the same seed must produce identical numbers."""
    def _run(tag):
        out = str(tmp_path / tag)
        argv = ["--dataset", "synthetic", "--backbone", "stub", "--n_train", "40",
                "--epochs", "1", "--n_val", "20", "--n_test", "20", "--token_pool", "4",
                "--synthetic_n", "150", "--seeds", "42", "--batch_size", "20",
                "--families", "attn_pool", "--out", out,
                "--cache_dir", str(tmp_path / "cache"), "--log_every", "0"]
        ablation.main(argv)
        return json.load(open(os.path.join(out, "synthetic_40_pool4",
                                           "family_attn_pool_seed_42.json")))

    a, b = _run("run_a"), _run("run_b")
    assert a["test_acc"] == b["test_acc"]
    assert a["val_bits"] == b["val_bits"]
    assert a["val_curve"] == b["val_curve"]
    assert a["split_hash"] == b["split_hash"]


# ===========================================================================
# §M5 — the gate-probe noise script
# ===========================================================================

def test_probe_noise_script_end_to_end(tmp_path):
    out, cache = str(tmp_path / "probe"), str(tmp_path / "cache")
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "module_family_probe_noise.py"),
           "--dataset", "synthetic", "--backbone", "stub", "--n_train", "60",
           "--n_splits", "3", "--gate_epochs", "2", "--holdout", "20",
           "--token_pool", "4", "--synthetic_n", "200", "--n_val", "20", "--n_test", "40",
           "--seeds", "42", "--batch_size", "20", "--families", "mlp_cls", "attn_pool",
           "--out", out, "--cache_dir", cache]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    path = os.path.join(out, "synthetic_60_pool4", "probe_mlp_cls_seed_42.json")
    rec = json.load(open(path))
    assert rec["n_gate_samples"] == 60 and rec["holdout"] == 20
    assert len(rec["splits"]) == 3
    for sp in rec["splits"]:
        # min-over-epochs selection: the kept value can never exceed the final epoch's.
        assert sp["min_held_out_bits"] <= sp["final_held_out_bits"] + 1e-9
        assert sp["optimism"] >= -1e-9
        assert 1 <= sp["best_epoch"] <= 2
        assert sp["test_bits_at_best"] is not None
    assert rec["sd_min_held_out_bits"] >= 0.0
    assert rec["mean_optimism"] == pytest.approx(
        sum(s["optimism"] for s in rec["splits"]) / 3)

    summary = json.load(open(os.path.join(out, "probe_summary.json")))
    assert set(summary["synthetic_60_pool4"]) == {"mlp_cls", "attn_pool"}


# ===========================================================================
# §5 — the gate script
# ===========================================================================

def _write_runs(root: str, dataset: str, n_train: str, accs: dict, bits: dict = None,
                seeds=(42, 43, 44), pool=2):
    """Hand-build the per-run JSONs the gate reads. `accs[family]` may be a float
    (same for every seed) or a per-seed list."""
    suffix = "" if pool is None else f"_pool{pool}"
    d = os.path.join(root, f"{dataset}_{n_train}{suffix}")
    os.makedirs(d, exist_ok=True)
    for family, acc in accs.items():
        for i, seed in enumerate(seeds):
            a = acc[i] if isinstance(acc, (list, tuple)) else acc
            b = (bits or {}).get(family, 1.0)
            b = b[i] if isinstance(b, (list, tuple)) else b
            rec = {"dataset": dataset, "n_train": n_train, "family": family,
                   "arm": None, "seed": seed, "test_acc": a, "val_acc": a,
                   "val_bits": b, "test_bits": b, "params": 66048, "wall_s": 1.0,
                   "token_pool": None if family == "cnn" else pool,
                   "split_hash": f"hash{seed}"}
            with open(os.path.join(d, f"family_{family}_seed_{seed}.json"), "w") as f:
                json.dump(rec, f)
    return d


def _svhn(root, a, b, c0, c, d_, e, n_train="full", **kw):
    return _write_runs(root, "svhn", n_train,
                       {"mlp_cls": a, "mlp_cls_meanpool": b, "proj_uniform_pool": c0,
                        "attn_pool": c, "self_attn": d_, "cnn": e}, **kw)


def _branch(root, probe=None):
    summary = gate.load_results(root)
    m1 = gate.m1_information(summary)
    m2 = gate.m2_attention(summary)
    m3b = gate.m3b_substrate_small_n(summary)
    m4 = gate.m4_control(summary)
    power = gate.power_check(summary)
    m5 = gate.m5_gate_noise(gate.load_probe(probe) if probe else [])
    return (gate.blind_branch(m1, m2, m3b, power), gate.collateral_flag(m4),
            gate.gate_noisier_flag(m5), summary)


def test_gate_branch_attention_wins(tmp_path):
    root = str(tmp_path)
    #        A     B     C0    C     D     E
    _svhn(root, 0.60, 0.65, 0.64, 0.70, 0.65, 0.60, "full")
    _svhn(root, 0.25, 0.30, 0.29, 0.35, 0.30, 0.20, "400")
    branch, collateral, noisier, summary = _branch(root)
    assert branch == "ATTENTION-WINS + SUBSTRATE-OK", branch
    assert collateral is False and noisier is False
    m2 = gate.m2_attention(summary)
    assert m2["accept_attn_pool"] is True       # C - C0 = 0.06 in both regimes
    assert m2["accept_self_attn"] is False      # D - C0 = 0.01
    assert m2["accept"] is True                 # "at least one, in both regimes"


def test_gate_branch_information_wins_plus_substrate_at_small_n(tmp_path):
    root = str(tmp_path)
    _svhn(root, 0.60, 0.65, 0.66, 0.665, 0.66, 0.90, "full")
    _svhn(root, 0.25, 0.30, 0.30, 0.305, 0.30, 0.35, "400")   # E - A = 0.10 @400
    branch, collateral, _, summary = _branch(root)
    assert branch == "INFORMATION-WINS + SUBSTRATE-AT-SMALL-N", branch
    assert collateral is False
    m3 = gate.m3_substrate_full(summary)
    assert m3["descriptive"] is True and "accept" not in m3
    assert m3["delta"] == pytest.approx(0.90 - 0.665)


def test_gate_branch_no_effect(tmp_path):
    root = str(tmp_path)
    _svhn(root, 0.60, 0.605, 0.61, 0.61, 0.60, 0.62, "full")
    _svhn(root, 0.25, 0.255, 0.26, 0.26, 0.25, 0.20, "400")
    branch, _, _, _ = _branch(root)
    assert branch == "NO-EFFECT + SUBSTRATE-OK", branch


def test_gate_branch_attention_only_is_named(tmp_path):
    """M1 rejects, M2 accepts — the cell the pre-registration left unnamed."""
    root = str(tmp_path)
    _svhn(root, 0.60, 0.605, 0.60, 0.66, 0.60, 0.61, "full")
    _svhn(root, 0.25, 0.255, 0.25, 0.31, 0.25, 0.20, "400")
    branch, _, _, _ = _branch(root)
    assert branch == "ATTENTION-ONLY + SUBSTRATE-OK", branch


def test_gate_requires_both_regimes_for_m1(tmp_path):
    """A gain @full alone is not M1: the note requires ≥ 0.02 in BOTH regimes."""
    root = str(tmp_path)
    _svhn(root, 0.60, 0.65, 0.61, 0.61, 0.60, 0.62, "full")
    _svhn(root, 0.25, 0.255, 0.26, 0.26, 0.25, 0.20, "400")     # no gain @400
    summary = gate.load_results(root)
    m1 = gate.m1_information(summary)
    assert m1["status"] == "OK" and m1["accept"] is False
    assert [r["accept"] for r in m1["rows"]] == [True, False]
    branch, _, _, _ = _branch(root)
    assert branch.startswith("NO-EFFECT")


def test_gate_uses_the_best_pool_per_family(tmp_path):
    """token_pool is a treatment: each family is read at its best pool."""
    root = str(tmp_path)
    _svhn(root, 0.60, 0.605, 0.60, 0.61, 0.60, 0.61, "full", pool=2)
    _svhn(root, 0.25, 0.255, 0.25, 0.26, 0.25, 0.20, "400", pool=2)
    # A better pool for C only, at both regimes.
    _write_runs(root, "svhn", "full", {"attn_pool": 0.66}, pool=1)
    _write_runs(root, "svhn", "400", {"attn_pool": 0.31}, pool=1)

    summary = gate.load_results(root)
    entry = summary["svhn_full"]["attn_pool"]
    assert set(entry["pools"]) == {"1", "2"} and entry["best_pool"] == "1"
    assert entry["acc_mean"] == pytest.approx(0.66)
    m2 = gate.m2_attention(summary)
    assert m2["accept_attn_pool"] is True
    assert m2["attn_pool_vs_uniform"]["rows"][0]["treat_pool"] == "1"


def test_gate_control_is_kmnist_and_flags_collateral(tmp_path):
    root = str(tmp_path)
    _svhn(root, 0.60, 0.605, 0.61, 0.61, 0.60, 0.62, "full")
    _svhn(root, 0.25, 0.255, 0.26, 0.26, 0.25, 0.20, "400")
    _write_runs(root, "kmnist", "400",
                {"mlp_cls": 0.592, "mlp_cls_meanpool": 0.60, "proj_uniform_pool": 0.59,
                 "attn_pool": 0.50, "self_attn": 0.60, "cnn": 0.59})
    branch, collateral, _, summary = _branch(root)
    assert collateral is True
    m4 = gate.m4_control(summary)
    assert m4["regime"] == "kmnist_400" and m4["accept"] is False
    assert [v["family"] for v in m4["violations"]] == ["attn_pool"]
    assert branch == "NO-EFFECT + SUBSTRATE-OK"      # collateral never enters the branch


def test_gate_control_accepts_when_every_arm_is_within_tolerance(tmp_path):
    root = str(tmp_path)
    _write_runs(root, "kmnist", "400",
                {"mlp_cls": 0.592, "mlp_cls_meanpool": 0.60, "proj_uniform_pool": 0.59,
                 "attn_pool": 0.585, "self_attn": 0.60, "cnn": 0.58})
    m4 = gate.m4_control(gate.load_results(root))
    assert m4["accept"] is True and m4["violations"] == []


def test_gate_inconclusive_on_wide_seed_spread(tmp_path):
    root = str(tmp_path)
    _svhn(root, 0.60, 0.65, 0.60, 0.70, 0.60, 0.62, "full")
    _write_runs(root, "svhn", "400",
                {"mlp_cls": [0.20, 0.30, 0.40], "mlp_cls_meanpool": 0.30,
                 "proj_uniform_pool": 0.29, "attn_pool": 0.35, "self_attn": 0.30,
                 "cnn": 0.20})
    summary = gate.load_results(root)
    power = gate.power_check(summary)
    assert power["inconclusive"] is True and power["over_arms"] == ["mlp_cls"]
    branch, _, _, _ = _branch(root)
    assert branch == "INCONCLUSIVE"


def test_gate_null_validity_reads_arm_a_against_the_archive(tmp_path):
    root = str(tmp_path)
    _svhn(root, 0.594, 0.60, 0.59, 0.60, 0.59, 0.61, "full")
    _svhn(root, 0.303, 0.31, 0.30, 0.31, 0.30, 0.20, "400")
    null = gate.null_validity(gate.load_results(root))
    assert null["matches"] is True
    _write_runs(root, "svhn", "400", {"mlp_cls": 0.10})   # overwrite arm A with a miss
    null = gate.null_validity(gate.load_results(root))
    assert null["matches"] is False
    assert [r["regime"] for r in null["rows"] if not r["matches_within_3sd"]] == ["svhn_400"]


# --- M5, from the probe-noise JSONs ---------------------------------------

def _write_probe(root: str, family: str, seed: int, mins, opts, dataset="svhn",
                 n_train="400", pool=2):
    d = os.path.join(root, f"{dataset}_{n_train}_pool{pool}")
    os.makedirs(d, exist_ok=True)
    splits = [{"split_seed": i, "min_held_out_bits": m, "optimism": o,
               "final_held_out_bits": m + o, "best_epoch": 3, "test_bits_at_best": m}
              for i, (m, o) in enumerate(zip(mins, opts))]
    rec = {"dataset": dataset, "n_train": n_train, "family": family, "seed": seed,
           "token_pool": pool, "splits": splits}
    with open(os.path.join(d, f"probe_{family}_seed_{seed}.json"), "w") as f:
        json.dump(rec, f)


def test_gate_m5_flags_families_noisier_than_arm_a(tmp_path):
    root = str(tmp_path)
    _write_probe(root, "mlp_cls", 42, [1.00, 1.10, 1.20], [0.10, 0.10, 0.10])
    _write_probe(root, "self_attn", 42, [1.00, 1.50, 2.00], [0.10, 0.10, 0.10])  # SD x5
    _write_probe(root, "attn_pool", 42, [1.00, 1.10, 1.20], [0.50, 0.50, 0.50])  # opt x5
    _write_probe(root, "proj_uniform_pool", 42, [1.00, 1.10, 1.20], [0.11, 0.10, 0.09])

    m5 = gate.m5_gate_noise(gate.load_probe(root))
    assert m5["status"] == "OK"
    rows = {r["family"]: r for r in m5["rows"]}
    assert rows["self_attn"]["sd_ratio_vs_A"] == pytest.approx(5.0, rel=1e-3)
    assert rows["attn_pool"]["optimism_ratio_vs_A"] == pytest.approx(5.0, rel=1e-3)
    assert sorted(m5["gate_noisier_arms"]) == ["attn_pool", "self_attn"]
    assert gate.gate_noisier_flag(m5) is True
    assert rows["proj_uniform_pool"]["gate_noisier"] is False


def test_gate_m5_skips_without_probe_data():
    m5 = gate.m5_gate_noise([])
    assert m5["status"] == "SKIPPED"
    assert gate.gate_noisier_flag(m5) is False


def test_gate_cli_end_to_end(tmp_path):
    root = str(tmp_path / "results")
    probe = str(tmp_path / "probe")
    os.makedirs(root)
    _svhn(root, 0.594, 0.65, 0.64, 0.70, 0.65, 0.62, "full")
    _svhn(root, 0.303, 0.35, 0.34, 0.40, 0.35, 0.40, "400")
    _write_runs(root, "kmnist", "400",
                {"mlp_cls": 0.592, "mlp_cls_meanpool": 0.60, "proj_uniform_pool": 0.59,
                 "attn_pool": 0.585, "self_attn": 0.60, "cnn": 0.58})
    _write_probe(probe, "mlp_cls", 42, [1.0, 1.1, 1.2], [0.1, 0.1, 0.1])
    _write_probe(probe, "self_attn", 42, [1.0, 1.5, 2.0], [0.1, 0.1, 0.1])

    out = str(tmp_path / "gates.json")
    proc = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, "eval_module_family.py"),
         "--results", root, "--probe", probe, "--out", out],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "svhn_full" in proc.stdout and "mlp_cls_meanpool" in proc.stdout

    result = json.load(open(out))
    assert result["blind_branch"] == "ATTENTION-WINS + SUBSTRATE-AT-SMALL-N"
    assert result["collateral_flag"] is False
    assert result["gate_noisier_flag"] is True
    assert result["gate_noisier_arms"] == ["self_attn"]
    assert result["null_validity"]["matches"] is True
    assert result["pairing"]["paired"] is True
    assert result["m3_substrate_full"]["best_dino_family"] == "attn_pool"
    assert result["summary"]["svhn_400"]["attn_pool"]["n_seeds"] == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
