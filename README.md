# concept-dag

A DAG-structured modular architecture for continual learning. Each task in a sequential stream grows a new node in a directed acyclic graph, where parent nodes are selected by principal-angle similarity between concept subspaces. Once trained, a node's parameters are frozen; later tasks compose over it rather than overwriting it.

The architecture is designed around a specific question: does structural connectivity protect learned representations from drift? The main empirical finding is that it does — highly-connected hub nodes (high out-degree) drift less under adversarial perturbation than leaf nodes, with Pearson ρ(out-degree, drift) = −0.586 on Split-CIFAR-100. A forced-hub intervention confirms the effect is partly structural (ρ = −0.337), not purely selection bias. We call this the crystallization effect.

---

## Architecture

Each node consists of a concept module (MLP + aggregation layer) and, for root nodes, a small CNN backbone. Five aggregation strategies are implemented: concatenation, element-wise mean, multi-head attention with a fixed learned query, differentiable top-k SVD, a learnable orthogonally-regularised projection (SoftPCA), and multi-head cross-attention where the query derives from the child's own input. The routing step that selects parents for a new task uses principal-angle similarity between concept subspaces.

---

## Results (Split-CIFAR-100, 20 tasks × 5 classes)

| Variant | Avg Acc | Backward Transfer | Notes |
|---|---:|---:|---|
| full | 0.5365 | +0.0000 | Principal-angle routing + SoftPCA + freeze |
| no_routing | 0.5444 | +0.0000 | Random parent selection |
| no_crystal | 0.5601 | +0.0000 | Concat aggregation |
| no_freeze | 0.4981 | −0.2687 | Catastrophic forgetting without freezing |
| cross_attention | 0.5671 | +0.0000 | Cross-attention aggregator |
| sequential | 0.7871 | +0.0000 | Parameter-isolation upper bound (20× params) |

Freezing is the critical ingredient — removing it produces severe backward transfer (−0.27). The routing mechanism has little effect at default settings, but a sensitivity sweep over `subspace_k` shows that `k = 16` (rather than the default 8) raises the full variant to 0.5621, matching the no_crystal baseline. Cross-attention achieves the best accuracy among shared-parameter variants at 0.5671.

---

## Repository structure

```
concept_dag/
  modules/
    aggregation.py       ConcatAgg, MeanAgg, AttentionAgg, SVDAgg,
                         SoftPCAAgg, CrossAttentionAgg
    concept_module.py    Core node: aggregator + MLP
    dag.py               DAG data structure and routing utilities
  models/baselines.py    SmallCNN feature extractor, LinearHead
  data/loaders.py        Split-CIFAR-100 task-stream loader
  training/
    two_stage.py         Route-then-train protocol
    continual.py         Continual-learning loop utilities
  experiments/           Experiments 1–6 and plotting
results/                 Per-experiment JSON outputs
run_experiment.py        CLI dispatcher
```

---

## Usage

```bash
# Main 20-task growing-DAG experiment
python run_experiment.py --exp 3 --device cuda --data_root ./data

# Ablation variants (full, no_routing, no_crystal, no_freeze, sequential, cross_attention)
python run_experiment.py --exp 4 --device cuda

# Causal forced-hub intervention
python run_experiment.py --exp 4f --device cuda

# Sensitivity sweeps (n_parents and subspace_k)
python run_experiment.py --exp 5 --device cuda

# Best-config confirmation (k=16, n_parents=4)
python run_experiment.py --exp 6 --device cuda --best_subspace_k 16 --best_n_parents 4
```

---

## Direction being explored

The sequential baseline achieves the highest accuracy (0.7871) but at 20× the parameter count — one full model per task. The shared-parameter variants close part of that gap while keeping backward transfer at zero, but the remaining accuracy shortfall suggests that fixed-size nodes under-represent later tasks as the DAG grows. An open question is whether the DAG structure can be extended with a routing or node-sharing strategy that grows total parameters sub-linearly in the number of tasks while preserving the crystallization guarantee. This is an idea being played around with and is not yet implemented.
