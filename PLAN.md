# Concept DAG — Post-Thesis Work Plan

Scoped milestones for each open todo. Effort estimates are calendar-time for a single researcher working part-time around a thesis. Dependency order is top-to-bottom within each tier.

## Tier 1 — In-session deliverables (completed 2026-04-14)

### Cross-attention aggregator (critique #6) — DONE
- **Status:** implemented in `concept_dag/modules/aggregation.py` as `CrossAttentionAggregator`; plumbed through `ConceptModule.forward` via `aggregator_uses_query` flag; registered in Exp 4 config and VARIANTS list as `"cross_attention"`.
- **Next step for you:** run `python run_experiment.py --exp 4 --variants cross_attention --device cuda`. Expected runtime ~40 min at `(k=16, n_parents=2)`. Compare AA / BT / Final-AA against `no_crystal` (0.5601) and `full` (0.5365).
- **Key knobs to sweep if time allows:** `n_heads ∈ {2, 4, 8}`, `entropy_weight ∈ {0, 0.01, 0.1}`, `orth_weight ∈ {0, 0.01, 0.1}`.

### PLAN.md + README.md — DONE

## Tier 2 — I can prep, you run (1–2 weeks)

### Exp 6 at best-from-Exp-5 config
- **Config:** `subspace_k=16, n_parents=4, variant=full`. Single-seed first; add 3-seed re-run if time.
- **Effort:** ~1 hour to run, plus plotting.
- **Blocker:** none. Ready to launch.

### Multi-seed Exp 5 re-sweep at best config
- **Scope:** re-run Exp 5a at `n_parents=4` and Exp 5b at `k=16` with 3 seeds each, to kill the single-seed variance concern (critique #13).
- **Effort:** ~4 hours compute, trivial code change (wrap sweep in `for seed in seeds:`).

### Cross-attention ablation in Exp 4
- **Goal:** full 20-task comparison of cross-attention vs soft_pca vs concat on identical conditions.
- **Effort:** ~40 min compute. Already enabled by VARIANTS list above.
- **Status (2026-04-15):** DONE. Cross-attention AA = 0.5671 (beats all other Exp 4 variants except sequential).

### Forced-hub aggregator sweep (task-0 backbone confound)
- **Goal:** isolate aggregator expressivity from backbone-access bias. Cross-attention forced-hub run hit AA=0.5972, vs 0.5671 under natural routing — about 3 pp of the cross-attention advantage is routing/backbone-access, not aggregator power. To get a clean "aggregator-only" comparison, rerun exp 4f with `--aggregation soft_pca` and `--aggregation concat` and compare AAs at held-constant routing.
- **Command (cross_attention done):** `python run_experiment.py --exp 4f --aggregation {soft_pca,concat} --device cuda --data_root ./data`.
- **Effort:** ~45 min compute per aggregator. Small code change already applied to `run_experiment.py`.
- **Why it matters:** cleanly decomposes the cross-attention gain for the writeup. Also the cheapest confound kill before the DINO swap. See `.auto-memory/project_dag_task0_backbone_effect.md` for full decomposition.

## Tier 3 — Research weeks (post-thesis paper prep)

### DINO / SSL backbone swap (critique #2)
- **Design:** replace `SmallCNN` in `concept_dag/models/baselines.py` with a frozen encoder returning a fixed-dim feature vector. Factor out the "root encoder" interface so DINO, CLIP, MAE, or a vanilla pretrained ResNet-50 can be plugged in identically. The root node then becomes `encoder (frozen) + concept_module (trainable)` instead of the task-0-trained `SmallCNN + concept_module`.
- **Data implications:** input resolution likely changes (DINO expects 224×224); update `data/loaders.py` transforms.
- **Expected effect:** uniform AA gain across all variants (all roots get better features); should tighten the gap between `full` and `sequential` because sequential no longer benefits from training its own CNN per task.
- **Effort:** 1 week code + 2 days training time.

### Non-CIFAR benchmark port (critique #10)
- **Choice:** Split-MiniImageNet (vision) OR Meta-World ML45 (robotics-positioning). Recommend MiniImageNet first — smaller scope shift, validates the crystallization story at larger scale.
- **Effort:** 1–2 weeks (data loader, task splits, re-tuned hyperparameters).

### Cross-attention router (overlaps with SBI)
- **Decision gate:** only do this if you DON'T pursue the SBI extension. Otherwise SBI posterior is your router.

### Node merging protocol (critique #1)
- **Scope:** three-stage protocol (candidate selection via subspace similarity → permutation-aligned task-vector averaging → held-out validation gate).
- **Prerequisite:** small exemplar buffer per task (~50 examples) for the validation gate.
- **Effort:** 2–3 weeks implementation + validation experiments.

## Tier 4 — Post-thesis paper-scale work (months)

### SBI-gated grow-or-reuse extension
- See `.auto-memory/project_dag_sbi_extension.md` for the full design.
- **Effort:** 3–6 months.
- **Pairs naturally with:** node merging protocol (SBI grows, merging prunes).

### Physics-domain port
- **Target:** rMD17 + MARTINI CG trajectories (see `project_dag_applications.md`).
- **Blockers to unblock first:** DINO-style root backbone (here, equivariant GNN like NequIP/MACE), cross-attention aggregator (non-linear composition needed for MD → CG).
- **Effort:** 3 months as a self-contained paper.

### Thesis writeup framing
- Crystallization as empirical finding, not mechanism.
- Sequential positioned as parameter-isolation upper bound.
- SBI extension as explicit future work.
- Cross-attention aggregator as post-thesis addendum if results are ready in time.
