"""
Smoke tests for Kan-gated growth + consolidation (reduction).

Runs on CPU with tiny synthetic data — no CIFAR, no encoder. Exercises:
  1. reuse_vs_grow decides REUSE when the task is linearly decodable from an existing concept.
  2. reuse_vs_grow decides GROW when the task is a non-linear (XOR) function of concept features
     that linear recombination cannot capture, and forces growth for a root (no parents).
  3. ConceptDAG.merge_modules re-points children and drops the redundant node (cycle-safe).
  4. low_rank_factorize_final_layer preserves the module's function and reclaims parameters.
"""

import math
import torch

from concept_dag.modules.dag import ConceptDAG
from concept_dag.modules.concept_module import ConceptModule
from concept_dag.training.kan_gate import (
    reuse_vs_grow, decide_reuse_search_grow, classification_task, kan_gated_grow,
)
from concept_dag.training.consolidate import low_rank_factorize_final_layer, effective_rank


torch.manual_seed(0)
DIM = 16


def _make_frozen_root(mid: str, in_dim=DIM, out_dim=DIM) -> ConceptModule:
    m = ConceptModule(module_id=mid, in_dim=in_dim, hidden_dim=32, out_dim=out_dim,
                      n_layers=2, n_parents=0, dropout=0.0)
    m.eval()
    m.freeze()
    return m


def _loader(x, y, bs=64):
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)


def _new_module_factory(n_parents):
    def factory():
        return ConceptModule(module_id="__probe_grow__", in_dim=DIM, hidden_dim=64, out_dim=DIM,
                             n_layers=2, n_parents=n_parents, aggregation="mean", dropout=0.0)
    return factory


def _build_dag_with_parent():
    dag = ConceptDAG()
    c0 = _make_frozen_root("c0")
    dag.add_module("c0", c0)
    # Parent embeddings for a batch of inputs (used to synthesise labels consistent with the DAG).
    x = torch.randn(600, DIM)
    with torch.no_grad():
        _, emb = dag.forward(x, active_nodes=["c0"], return_all_embeddings=True)
    e0 = emb["c0"]  # (N, DIM)
    return dag, x, e0


def test_gate_reuse_when_linearly_decodable():
    dag, x, e0 = _build_dag_with_parent()
    # Bayes-optimal predictor is LINEAR in the existing concept, with genuine label noise
    # (logistic). Reuse (linear head) already hits the noise floor; a new MLP concept can only
    # overfit, so on held-out data grow ties reuse → no obstruction → reuse.
    p = torch.sigmoid(2.5 * e0[:, 0])
    y = torch.bernoulli(p).long()
    rec = reuse_vs_grow(
        dag, parent_ids=["c0"], new_module_factory=_new_module_factory(1),
        spec=classification_task(2), loader=_loader(x, y),
        concept_dim=DIM, n_epochs=200, lr=3e-3, eps_rel=0.10,
    )
    print("REUSE-case record:", rec.reason, "->", rec.decision)
    assert rec.decision == "reuse", rec.as_dict()


def test_gate_grow_when_nonlinear():
    dag, x, e0 = _build_dag_with_parent()
    # XOR of two thresholds → not linearly separable → a new MLP concept is required.
    a = e0[:, 0] > e0[:, 0].median()
    b = e0[:, 1] > e0[:, 1].median()
    y = (a ^ b).long()
    rec = reuse_vs_grow(
        dag, parent_ids=["c0"], new_module_factory=_new_module_factory(1),
        spec=classification_task(2), loader=_loader(x, y),
        concept_dim=DIM, n_epochs=200, lr=3e-3, eps_rel=0.10,
    )
    print("GROW-case record:", rec.reason, "->", rec.decision)
    assert rec.decision == "grow", rec.as_dict()


def test_search_level_reclassifies_nonlinear_from_grow_to_search():
    """The XOR-of-a-concept task the BINARY gate calls 'grow' is caught by the three-way gate as
    'search': a bounded non-linear recombination of the EXISTING concept solves it, so a new concept
    adds little beyond the best search (rel_grow small) — test-time compute substitutes for growth."""
    dag, x, e0 = _build_dag_with_parent()
    a = e0[:, 0] > e0[:, 0].median()
    b = e0[:, 1] > e0[:, 1].median()
    y = (a ^ b).long()
    stack = e0.unsqueeze(1)                       # (N, 1, DIM) parent stack
    rec = decide_reuse_search_grow(
        stack, y, _new_module_factory(1), classification_task(2),
        concept_dim=DIM, n_parents=1, device="cpu", n_epochs=200, lr=3e-3,
        eps_grow=0.10, eps_search=0.05, search_budget=6, search_rank=16,
    )
    print("SEARCH-case:", rec.reason, "->", rec.decision)
    assert rec.decision == "search", rec.as_dict()
    assert rec.L_search_bits < rec.L_reuse_bits - 1e-3        # search beat linear reuse
    assert rec.rel_grow <= 0.10                               # a new concept adds little beyond search
    assert rec.search_trace[-1][1] <= rec.search_trace[0][1]  # (T, L_search) monotone — the knee


def test_gate_forces_growth_for_root():
    dag = ConceptDAG()
    rec = reuse_vs_grow(
        dag, parent_ids=[], new_module_factory=_new_module_factory(0),
        spec=classification_task(2), loader=_loader(torch.randn(10, DIM), torch.zeros(10).long()),
        concept_dim=DIM,
    )
    assert rec.decision == "grow" and "forced" in rec.reason


def test_merge_modules_surgery():
    dag = ConceptDAG()
    dag.add_module("a", _make_frozen_root("a"))
    dag.add_module("b", _make_frozen_root("b"))
    child = ConceptModule("child", in_dim=DIM, hidden_dim=32, out_dim=DIM, n_parents=1,
                          aggregation="mean", dropout=0.0)
    dag.add_module("child", child, parents=["b"])
    assert dag.out_degree("b") == 1 and dag.out_degree("a") == 0
    dag.merge_modules(keep_id="a", drop_id="b")   # b's child re-points to a; b removed
    assert "b" not in dag.all_module_ids()
    assert dag._parents["child"] == ["a"]
    assert dag.out_degree("a") == 1


def test_low_rank_factorize_preserves_function_and_saves_params():
    # Wide out_dim so a low-rank final layer genuinely saves params.
    m = ConceptModule("m", in_dim=DIM, hidden_dim=64, out_dim=64, n_layers=2, n_parents=0, dropout=0.0)
    m.eval(); m.freeze()
    # Make the final layer genuinely rank-4 (a crystallised concept lives in few directions), so the
    # rank-4 factorisation is (near-)lossless — the realistic case the pass targets.
    final = m.mlp[-1]
    with torch.no_grad():
        U = torch.randn(64, 4)
        V = torch.randn(4, 64)
        final.weight.copy_((U @ V) * 0.1)
    m.freeze()
    x = torch.randn(128, DIM)
    with torch.no_grad():
        before = m(x)
    p_before = sum(p.numel() for p in m.parameters())
    rec = low_rank_factorize_final_layer(m, rank=4, max_rel_error=1e-3)  # exact for a rank-4 weight
    assert rec["applied"], rec
    p_after = sum(p.numel() for p in m.parameters())
    with torch.no_grad():
        after = m(x)
    rel = (after - before).norm() / (before.norm() + 1e-9)
    print(f"truncate: params {p_before} -> {p_after}, output rel-error {rel:.5f}, "
          f"weight rel-error {rec['rel_error']:.5f}")
    assert p_after < p_before
    assert rel < 1e-4  # function preserved (exact low-rank factorisation)


def test_effective_rank_monotone():
    s = torch.tensor([10.0, 1.0, 0.1, 0.01])
    assert effective_rank(s, 0.99) <= effective_rank(s, 0.999)


def test_orchestrator_grows_node_on_obstruction():
    dag, x, e0 = _build_dag_with_parent()
    a = e0[:, 0] > e0[:, 0].median()
    b = e0[:, 1] > e0[:, 1].median()
    y = (a ^ b).long()
    n_before = len(dag.all_module_ids())
    out = kan_gated_grow(
        dag, new_module_id="t1", parent_ids=["c0"], new_module_factory=_new_module_factory(1),
        spec=classification_task(2), loader=_loader(x, y), concept_dim=DIM,
        final_epochs=80, lr=3e-3,
        gate_kwargs=dict(n_epochs=200, lr=3e-3, eps_rel=0.10),
    )
    assert out["decision"] == "grow"
    assert len(dag.all_module_ids()) == n_before + 1        # a concept node was added
    assert dag.get_module("t1").get_concept_subspace() is not None  # routable for future tasks
    assert dag.get_module("t1").is_frozen


def test_orchestrator_reuses_without_growing():
    dag, x, e0 = _build_dag_with_parent()
    p = torch.sigmoid(2.5 * e0[:, 0])
    y = torch.bernoulli(p).long()
    n_before = len(dag.all_module_ids())
    out = kan_gated_grow(
        dag, new_module_id="t1", parent_ids=["c0"], new_module_factory=_new_module_factory(1),
        spec=classification_task(2), loader=_loader(x, y), concept_dim=DIM,
        final_epochs=60, lr=3e-3,
        gate_kwargs=dict(n_epochs=200, lr=3e-3, eps_rel=0.10),
    )
    assert out["decision"] == "reuse"
    assert len(dag.all_module_ids()) == n_before           # NO node added — solved by reuse
    assert out["solver"] is not None


if __name__ == "__main__":
    test_gate_forces_growth_for_root()
    test_merge_modules_surgery()
    test_low_rank_factorize_preserves_function_and_saves_params()
    test_effective_rank_monotone()
    test_gate_reuse_when_linearly_decodable()
    test_gate_grow_when_nonlinear()
    test_orchestrator_grows_node_on_obstruction()
    test_orchestrator_reuses_without_growing()
    print("\nAll Kan-gate smoke tests passed.")
