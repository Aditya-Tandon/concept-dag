# DINO backbone swap — implementation plan

Goal: replace the task-0-trained `SmallCNN` with a frozen self-supervised encoder so that every node starts from identical, task-agnostic features. This disentangles *aggregator expressivity* from *privileged access to the one trained CNN* (see `project_dag_task0_backbone_effect.md`).

## Motivation

Currently the DAG has a single structural anomaly: task 0's root module is the **only** node whose gradients flowed through the SmallCNN backbone. Every other node's "features" are some chain of frozen ConceptModule outputs rooted at task 0. That means:

1. Task 0 is implicitly the feature extractor for the entire DAG.
2. Descendants close to task 0 (direct children, or one hop via task 10) benefit from shorter, less-lossy pathways.
3. Cross-attention's observed +5.2 pp win on task-0-parent tasks is likely exploiting this — it's the only aggregator with enough expressivity to selectively bypass to the good path.

With a frozen DINO encoder:
- Every root sees the same pretrained features.
- No node is architecturally privileged.
- Comparisons between aggregators become clean.
- The DAG's role narrows to "compositional skill construction on top of a fixed representation" — which is the thesis's actual claim.

## Target encoder

**Primary:** DINOv2-ViT-S/14 (384-dim features, ~22M frozen params). Strong out-of-the-box representation for natural images; ImageNet-pretraining-free, so no label leakage into CIFAR-100.

**Fallbacks to keep on the table:**
- DINOv1-ViT-S/16 (384-dim) — slightly weaker, but 16-patch is friendlier at small input resolution.
- CLIP ViT-B/16 (512-dim) — stronger zero-shot transfer but trained with language supervision; useful as a "what if the encoder already knows CIFAR classes" upper-bound probe.
- MAE ViT-B/16 (768-dim) — reconstruction-based SSL, good contrast with contrastive SSL.
- Vanilla ImageNet-pretrained ResNet-50 (2048-dim) — sanity-check baseline; if DINO doesn't outperform it on this task, something is wrong.

All four should be pluggable via a single `RootEncoder` interface.

## Architectural changes

### 1. Introduce a `RootEncoder` abstraction

New file: `concept_dag/models/root_encoder.py`.

```python
class RootEncoder(nn.Module):
    """Frozen feature extractor. Same interface regardless of which SSL model."""
    def __init__(self, name: str, image_size: int): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) → (B, feature_dim)
        ...
    @property
    def feature_dim(self) -> int: ...
```

Implementations:
- `DINOv2Encoder` — load via `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`; take the CLS token.
- `CLIPEncoder` — `open_clip.create_model_and_transforms('ViT-B-16', 'openai')`; take the image projection output.
- `MAEEncoder` — `timm.create_model('vit_base_patch16_224.mae')` with `num_classes=0`.
- `ResNet50Encoder` — `torchvision.models.resnet50(weights=IMAGENET1K_V2)` with the final fc stripped.

All return `(B, feature_dim)` and are called in `torch.no_grad()` mode (encoder frozen, no optimizer attached to its params).

### 2. Replace `SmallCNN` inside `DAGNode` / `_build_node`

Currently `_build_node` constructs `SmallCNN + ConceptModule` for root nodes and `ConceptModule` only for children. After the swap:
- Roots: `RootEncoder (frozen, shared singleton) + ConceptModule`.
- Children: `RootEncoder (frozen, shared singleton) + Aggregator(parent outputs) + ConceptModule`.

**Key simplification:** with a frozen shared encoder, the encoder is literally the same nn.Module instance across all nodes — no per-task CNN at all. This removes ~90 % of the parameter count and means the sequential (parameter-isolation) baseline can no longer "cheat" by training a fresh CNN per task. Expect its AA to drop noticeably; this is intentional and makes sequential a fairer upper bound.

### 3. Update data loaders

DINOv2-ViT-S/14 expects 224×224 inputs normalized with ImageNet stats. Current `data/loaders.py` outputs 32×32 CIFAR tensors with CIFAR-100 normalization.

Two options:
- **Resize at load time** (simpler): add a 32→224 bilinear upsample + ImageNet normalisation transform in the loader. 7× upsample is noisy but standard practice.
- **Cache features once per task** (faster training, recommended): run the frozen encoder over each task split once at startup, save `(features, labels)` tensors to `./data/dino_features/task_{t}.pt`, and have the loaders return pre-computed features instead of images. This decouples encoder cost from training epochs and saves ~95 % of GPU-hours.

Recommend the caching path. Build a `feature_cache.py` utility.

### 4. Update concept dim defaults

DINOv2-ViT-S/14 output is 384-D. Current `concept_dim=128` may be a bottleneck relative to the richer input. Sweep `concept_dim ∈ {128, 256, 384}` in a quick ablation.

## Experiments to run after the swap

Goal: re-run the 6-variant Exp 4 and confirm or refute the task-0-backbone confound.

1. **Exp 4 replication with DINO.** All 6 variants (full, no_routing, no_crystal, no_freeze, sequential, cross_attention), 20 tasks × 25 epochs. The headline numbers we want:
   - Does cross_attention still beat full / no_crystal by the same margin?
   - Does the task-0-parent vs non-task-0-parent gap *disappear*? (Prediction: yes, drops from +5.2 pp to < 1 pp under DINO.)
   - Does sequential still win by a huge margin? (Prediction: no — it loses the per-task CNN advantage.)

2. **Exp 3b crystallization with DINO.** Does the ρ(out_degree, drift) = −0.586 correlation survive when roots are identical? If crystallization is real and structural, yes. If it was partly a "task 0 is special" artefact, the correlation will weaken.

3. **Sensitivity sweep.** Exp 5-style sweep of `n_parents ∈ {1,2,3,4,5}` and `subspace_k ∈ {4,8,16,32,64}`, since optimal hyperparameters likely shift with the richer input.

## Effort estimate

- `RootEncoder` + DINO integration: 1 day.
- Feature-caching pipeline: 1 day.
- Data loader + resolution transform updates: 0.5 day.
- Concept dim / head-size sweep (small, CPU): 0.5 day.
- Full Exp 4 re-run: ~2–3 hours compute per variant × 6 = ~15 GPU-hours.
- Exp 3b + sensitivity re-run: another ~20 GPU-hours.
- Analysis + writeup updates: 2 days.

**Total: ~1 week code + 1.5 days compute + 2 days writeup.** Good fit for one week post-thesis-submission.

## Pre-flight checks (can do before coding)

1. Verify DINOv2 loads on target hardware: `python -c "import torch; m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14'); print(m(torch.randn(1,3,224,224)).shape)"`. Should print `torch.Size([1, 384])`.
2. Check disk budget for cached features: 20 tasks × 2500 train + 500 test images × 384 floats × 4 bytes ≈ 90 MB total. Trivial.
3. Decide on random augmentation strategy. DINO was trained with heavy augmentation; adding RandomCrop/RandomFlip at 224 to CIFAR inputs is fine, but if caching features, augment *before* caching or cache multiple augmented versions per image.

## Decision gates

- **If cross_attention's advantage survives the swap at full margin:** aggregator expressivity is doing real compositional work. Headline result for paper.
- **If cross_attention collapses to par with soft_pca:** the Exp 4 win was backbone-access, and the thesis framing needs a retraction / reframe. Still a publishable finding (negative-result ablation), but requires rewrites.
- **If crystallization survives at ρ ≤ −0.4:** effect is real and structural; defend as-is.
- **If crystallization drops to |ρ| < 0.2:** the correlation was task-0-driven; demote from main finding to observation, rework thesis introduction.

These gates should be set *before* running, to avoid HARKing after the fact.

## Dependencies / blockers

- GPU with ≥ 8 GB VRAM (DINOv2-ViT-S easily fits).
- Network access for `torch.hub.load` on first run (or pre-download weights to a local cache).
- Updates to `PLAN.md` Tier 3 DINO section — this doc supersedes that section with more detail.
