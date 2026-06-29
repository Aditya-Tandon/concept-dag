# Writeup Notes — Concept DAG

Notes to remember while drafting the paper / thesis chapter.

## Exp 4 — `no_freeze` ablation: semantic caveat

The `no_freeze` variant is *not* a full "unfreeze the entire ancestor chain"
intervention. It is a **one-level (direct-parent) unfreeze**.

### Why

`DAGNode.forward` (in `concept_dag/experiments/exp3_growing_dag.py`) internally
wraps parent calls in `with torch.no_grad():`:

```python
def forward(self, x):
    if self.is_root:
        return self.concept_module(self.cnn(x))
    with torch.no_grad():
        parent_outs = [p(x) for p in self.parent_models]
    return self.concept_module(x, parent_outputs=parent_outs)
```

So even if we override `node.forward` for the *currently-training* node to
skip that `no_grad` block, every ancestor's own `forward` still cuts the
gradient at its own parent boundary.

Net effect of the original patch:
- Gradients flow into the currently-training child's weights ✓
- Gradients flow into the **direct parents'** `concept_module` weights ✓
- Gradients do **not** reach grandparents / higher ancestors ✗

### Fix applied (2026-04-11)

`_train_node_with_freeze_mode` in `exp4_ablations.py` now walks the entire
ancestor DAG via BFS from `node.parent_models` and patches every non-root
ancestor's `forward` to a gradient-transparent version for the duration of
the training call. All patches are restored in a `try/finally`.

With the fix, `no_freeze` now matches the semantic claim: any task can update
any ancestor's weights via backprop.

### How to describe this in the writeup

- Before the fix: the "no_freeze" label was technically wrong — describe it as
  "direct-parent unfreeze" if you use pre-fix results.
- After the fix: `no_freeze` is a genuine full-chain unfreeze. Report which
  version of the code produced the results you cite. All results collected
  from commit date ≥ 2026-04-11 use the full-chain version.

### What this means for the crystallization hypothesis

Crystallization predicts that *frozen* parents retain their concept subspaces.
The `no_freeze` ablation removes this protection entirely (after the fix). If
the full-chain version shows a catastrophic AA drop and a large positive
shift in ρ(out_degree, drift), that's stronger evidence for the structural
effect than the one-level-deep version would give. The one-level version
conflates "this layer's weights moved" with "subspace constraints were lost",
so the full-chain version is the cleaner experiment.

## Exp 4 — `no_freeze` memory caveat: root CNN backbones stay frozen

First attempt at the full-chain fix caused a GPU OOM at task 8 of the
`no_freeze` run. Root cause: task 0 owns a `SmallCNN` backbone, and once the
full ancestor chain was made grad-transparent, every forward pass retained
conv-layer activations for backprop. Peak memory scaled as
`(#CNN ancestors) × (conv activation size)` which exceeded VRAM by task 8.

### Second fix (2026-04-11, same day)

Root CNN backbones are now kept frozen even in `no_freeze` mode:

- Root ancestors get a specialised forward (`_make_frozen_cnn_root_forward`)
  that runs `self.cnn(x)` inside a `torch.no_grad()` block, then feeds the
  detached features to `self.concept_module`. Concept-module parameters still
  carry gradients.
- Root CNN parameters are excluded from the optimizer's trainable set.
- Root CNN stays in `eval()` mode throughout training so batchnorm/dropout
  stats don't drift.

Non-root ancestors and direct non-root parents remain fully grad-transparent
(their `concept_module` parameters are updated by every subsequent task that
uses them as ancestors).

### How to describe this in the writeup

The `no_freeze` ablation tests whether **concept-module crystallization**
matters. It deliberately does **not** also test whether the feature extractor
should be retrained, because:

1. The CNN backbone is an orthogonal architectural choice (any frozen
   pretrained encoder could substitute it). The crystallization claim is
   about the concept-module layer specifically.
2. Every other variant (`full`, `no_routing`, `no_crystal`, `sequential`)
   also treats the CNN as a fixed feature extractor trained only on task 0.
   Including CNN retraining in `no_freeze` would confound the ablation with
   a separate axis of variation.
3. Practical: training all ancestor CNNs jointly on the current task's data
   blows up peak GPU memory by an order of magnitude.

So the correct description in the paper is something like:

> "The `no_freeze` variant removes the parent-freezing constraint on every
> concept module in the ancestor chain. All previously learned concept
> modules remain updateable by later tasks via backprop through the DAG.
> Root feature-extractor backbones (`SmallCNN`) remain frozen across all
> variants and are never updated beyond their initial task-0 training; this
> isolates the ablation to the concept-module crystallization hypothesis."

## Exp 4 — `no_freeze` runtime caveat: memoized ancestor forward

Second-fix `no_freeze` + `sequential` runs were still alive at >2h runtime
(`full` / `no_routing` / `no_crystal` each finished in ~30–45 min). Root
cause: exponential recomputation of shared DAG ancestors.

### Why

With the per-ancestor monkey-patches, each ancestor's forward still called
each of *its* parents' forwards once. In a diamond-shaped DAG, a shared
ancestor like task 2 (out_degree = 9 in Exp 3a) is re-evaluated **once per
incoming edge** on every batch — not once per batch. For a deep child that
pulls task 2 through several distinct paths, task 2's concept module and
every one of *its* ancestors get evaluated multiple times per batch. At 20
tasks with the observed connectivity, the redundant ancestor evaluations
dominate wall-clock.

When parents are frozen (`full`, `no_routing`, `no_crystal`) this is still
wasteful but tolerable because each ancestor call is cheap — no grad graph
is retained. In `no_freeze` the ancestor calls DO retain grad graph, so
the blowup is both a compute cost and a memory cost.

### Third fix (2026-04-13)

Added `forward_dag_memoized(target, x, cnn_no_grad=True)` in
`exp3_growing_dag.py`:

1. BFS from `target` through `parent_models` to collect all reachable
   ancestors (including `target`).
2. Sort the collected set topologically. Because task_ids are assigned in
   growth order and a child's task_id is strictly greater than any of its
   parents' task_ids, sorting by `task_id` ascending is a valid topological
   order.
3. Walk the sorted list, calling each node's *local* forward (`cnn +
   concept_module` for roots, `concept_module(x, parent_outputs=...)` for
   children), caching each output keyed by `id(node)`.
4. Return `cache[id(target)]`.

Each ancestor evaluates exactly once per batch → `O(|reachable DAG|)` per
batch instead of the previous exponential.

`cnn_no_grad=True` keeps the memory-safety fix: root CNN backbones still
run under `torch.no_grad()`. Concept-module calls are NOT wrapped, so
gradients flow into every ancestor concept module attached to the
optimizer.

`_train_node_with_freeze_mode` in `exp4_ablations.py` now patches ONLY the
training node's `forward` with `_make_memoized_no_freeze_forward(node)`.
Ancestor `forward` methods — and their internal `no_grad` wrappers — are
bypassed entirely by the memoized evaluator, so no per-ancestor patching
is needed.

### How to describe this in the writeup

Frame the third fix as an implementation detail, not a scientific choice:
the memoized evaluator produces *mathematically identical* outputs and
gradients to the original per-ancestor patched forward (same graph, same
numerics), it just avoids redundant recomputation of shared ancestors. No
experimental claim is affected; only wall-clock changes. Expected
`no_freeze` runtime drops from ≫2h to ~15–20 min at 20 tasks.

