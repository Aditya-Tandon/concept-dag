# Concept DAG

A DAG-based modular neural architecture for continual learning with explicit provenance, crystallized non-interference, and compositional skill construction. Submitted as part of an Imperial College thesis; follow-up work scoped in `PLAN.md`.

## What it is

Each task in a continual stream grows a new **concept node** in a directed acyclic graph. Nodes consist of a concept module (MLP + aggregation layer) and — for root nodes — a small CNN backbone. A routing step selects parent nodes for each new task using principal-angle similarity between concept subspaces. Once trained, a node's parameters are **frozen**; later tasks compose over it rather than overwriting it.

The core empirical finding: highly-connected "hub" nodes (high out-degree) drift less under adversarial perturbation than leaf nodes, with Pearson ρ(out_degree, drift) = −0.586 on Split-CIFAR-100. We call this the **crystallization effect** and test its causal nature via a forced-hub intervention (ρ = −0.337 — same sign, reduced magnitude, suggesting structural effect plus partial selection bias).

## Repo layout

```
Memory_DAG/
├── concept_dag/
│   ├── modules/
│   │   ├── aggregation.py      # ConcatAgg, MeanAgg, AttentionAgg, SVDAgg,
│   │   │                       # SoftPCAAgg, CrossAttentionAgg
│   │   ├── concept_module.py   # Core node: aggregator + MLP
│   │   └── dag.py              # DAG data structure + routing utilities
│   ├── models/baselines.py     # SmallCNN feature extractor, LinearHead
│   ├── data/loaders.py         # Split-CIFAR-100 task-stream loader
│   ├── training/
│   │   ├── two_stage.py        # Route-then-train protocol
│   │   └── continual.py        # Continual-learning loop utilities
│   ├── experiments/
│   │   ├── exp1_crystallization.py  # Initial crystallization test
│   │   ├── exp2_routing.py          # Principal-angle router validation
│   │   ├── exp3_growing_dag.py      # Main 20-task growth experiment
│   │   ├── exp4_ablations.py        # 5-variant ablation + causal forced-hub
│   │   ├── exp5_sensitivity.py      # n_parents & subspace_k sweeps
│   │   ├── exp6_confirmation.py     # Best-config confirmation
│   │   └── plot_results.py          # Figure generation
│   └── utils/metrics.py        # Accuracy, BT, AA computation
├── results/                    # Per-experiment JSON outputs
├── run_experiment.py           # CLI dispatcher
├── PLAN.md                     # Post-thesis work plan
├── WRITEUP_NOTES.md            # Caveats / fixes documented for the paper
└── README.md                   # This file
```

## Quick start

```bash
# Main growing-DAG experiment (Exp 3, 20 tasks × 25 epochs)
python run_experiment.py --exp 3 --device cuda --data_root ./data

# All 6 ablation variants (including new cross_attention)
python run_experiment.py --exp 4 --device cuda

# Single ablation variant
python run_experiment.py --exp 4 --variants cross_attention --device cuda

# Causal forced-hub ablation
python run_experiment.py --exp 4f --device cuda

# Sensitivity sweeps (n_parents and subspace_k)
python run_experiment.py --exp 5 --device cuda

# Best-config confirmation (k=16, n_parents=4)
python run_experiment.py --exp 6 --device cuda --best_subspace_k 16 --best_n_parents 4
```

## Key results (Split-CIFAR-100, 20 tasks × 5 classes)

| Variant | Avg Acc | Backward Transfer | Notes |
|---|---:|---:|---|
| full | 0.5365 | +0.0000 | Reference (principal-angle routing + SoftPCA + freeze) |
| no_routing | 0.5444 | +0.0000 | Random parent selection — routing barely affects AA |
| no_crystal | 0.5601 | +0.0000 | Concat aggregation beats SoftPCA-k=8 at this width |
| no_freeze | 0.4981 | **−0.2687** | Catastrophic forgetting when ancestors can be modified |
| sequential | 0.7871 | +0.0000 | Parameter-isolation upper bound (no sharing, 20× params) |

From the Exp 5 sweep, `subspace_k=16` (not the default 8) gives `full`-variant AA of 0.5621, matching `no_crystal`. See `WRITEUP_NOTES.md` for the full discussion and `PLAN.md` for follow-up experiments.

## Aggregators

Five aggregation strategies are implemented in `concept_dag/modules/aggregation.py`:

- `concat` — concatenate parent outputs and project linearly (baseline).
- `mean` — element-wise mean of parent outputs (baseline).
- `attention` — multi-head attention with a *fixed learned query* (same for all inputs).
- `svd` — differentiable top-k SVD of parent stack with sign-normalisation + gradient clipping.
- `soft_pca` — learnable orthogonally-regularised linear projection (the method used throughout Exps 3–6).
- `cross_attention` — **(new)** multi-head cross-attention where the query is derived from the child's own input. Per-sample routing over parents; optional entropy regulariser to avoid collapse onto a single parent. Registered in `VARIANTS` as the 6th ablation.

## Citing

This is research code associated with an MSc thesis. Not yet published. If you use this code, please cite the thesis once it's deposited. Contact: Aditya (at3722@ic.ac.uk).
