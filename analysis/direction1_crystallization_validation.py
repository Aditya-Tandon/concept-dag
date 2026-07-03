"""
Direction 1: Crystallization Validation via Partial Correlation
===============================================================
Tests whether the out-degree ↔ weight-drift correlation (ρ = -0.586)
survives after controlling for task similarity.

H₀: Correlation is fully explained by task similarity (selection bias)
H₁: Crystallization is a real mechanism beyond selection bias
Decision: |ρ_partial| > 0.3 AND p < 0.05 → reject H₀

Run: python analysis/direction1_crystallization_validation.py
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

# -------------------------------------------------------------------
# 1. Load experiment data
# -------------------------------------------------------------------
root = Path(__file__).parent.parent

with open(root / "results/exp3/exp3b_results.json") as f:
    exp3b = json.load(f)

with open(root / "results/exp3/exp3a_results.json") as f:
    exp3a = json.load(f)

per_node = exp3b["per_node"]
parent_map = {int(k): v for k, v in exp3a["parent_map"].items()}

out_degrees = np.array([n["out_degree"] for n in per_node])
weight_drifts = np.array([n["weight_drift"] for n in per_node])
n_tasks = len(per_node)

# -------------------------------------------------------------------
# 2. Compute task similarity from CIFAR-100 superclass structure
# -------------------------------------------------------------------
# Standard CIFAR-100 fine-label → superclass mapping
FINE_TO_SUPERCLASS = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7,
    11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15,
    13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16,
    6, 19, 2, 8, 19, 18, 8, 5, 19, 15, 10, 2, 7, 8, 14,
    5, 9, 3, 13, 15, 17, 3, 10, 2, 13,
]

classes_per_task = 5
n_superclasses = 20

# Build superclass distribution vectors
task_vectors = np.zeros((n_tasks, n_superclasses))
for t in range(n_tasks):
    for fl in range(t * classes_per_task, (t + 1) * classes_per_task):
        task_vectors[t, FINE_TO_SUPERCLASS[fl]] += 1

# Cosine similarity matrix
norms = np.linalg.norm(task_vectors, axis=1, keepdims=True)
norms[norms == 0] = 1
task_vectors_normed = task_vectors / norms
sim_matrix = task_vectors_normed @ task_vectors_normed.T

# -------------------------------------------------------------------
# 3. Compute confound: mean child task similarity per node
# -------------------------------------------------------------------
children = {t: [] for t in range(n_tasks)}
for child, parents in parent_map.items():
    for p in parents:
        children[p].append(child)

mean_child_sim = np.array([
    np.mean([sim_matrix[t, c] for c in children[t]]) if children[t] else 0.0
    for t in range(n_tasks)
])

# -------------------------------------------------------------------
# 4. Partial correlation
# -------------------------------------------------------------------
def partial_corr(x, y, z):
    """Pearson partial correlation of x, y controlling for z."""
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    resid_x = x - (slope_xz * z + intercept_xz)
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
    resid_y = y - (slope_yz * z + intercept_yz)
    return stats.pearsonr(resid_x, resid_y)

def partial_corr_spearman(x, y, z):
    """Spearman partial correlation via rank transformation."""
    return partial_corr(stats.rankdata(x), stats.rankdata(y), stats.rankdata(z))

# -------------------------------------------------------------------
# 5. Run all analyses
# -------------------------------------------------------------------
print("=" * 65)
print("DIRECTION 1: CRYSTALLIZATION VALIDATION")
print("=" * 65)

# Raw correlations
r_raw, p_raw = stats.pearsonr(out_degrees, weight_drifts)
rho_raw, p_rho = stats.spearmanr(out_degrees, weight_drifts)
print(f"\nRaw Pearson  r = {r_raw:.4f}  (p = {p_raw:.4f})")
print(f"Raw Spearman ρ = {rho_raw:.4f}  (p = {p_rho:.4f})")

# Partial correlations
r_partial, p_partial = partial_corr(out_degrees, weight_drifts, mean_child_sim)
rho_partial, p_rho_partial = partial_corr_spearman(out_degrees, weight_drifts, mean_child_sim)
print(f"\nPartial Pearson  r = {r_partial:.4f}  (p = {p_partial:.4f})")
print(f"Partial Spearman ρ = {rho_partial:.4f}  (p = {p_rho_partial:.4f})")
print(f"Task similarity explains {1 - r_partial/r_raw:.1%} of raw correlation")

# Permutation test
np.random.seed(42)
n_perm = 10000
perm_rs = np.array([
    partial_corr(np.random.permutation(out_degrees), weight_drifts, mean_child_sim)[0]
    for _ in range(n_perm)
])
p_perm = np.mean(np.abs(perm_rs) >= np.abs(r_partial))
print(f"\nPermutation p-value ({n_perm} perms) = {p_perm:.4f}")

# Bootstrap CI
np.random.seed(42)
n_boot = 10000
boot_rs = np.array([
    partial_corr(
        out_degrees[idx := np.random.choice(n_tasks, n_tasks, replace=True)],
        weight_drifts[idx], mean_child_sim[idx]
    )[0] for _ in range(n_boot)
])
ci_lo, ci_hi = np.percentile(boot_rs, [2.5, 97.5])
print(f"Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

# Leave-one-out
print(f"\nLeave-one-out (flagging Δ > 0.1):")
for i in range(n_tasks):
    mask = np.ones(n_tasks, dtype=bool)
    mask[i] = False
    r_loo, _ = partial_corr(out_degrees[mask], weight_drifts[mask], mean_child_sim[mask])
    if abs(r_loo - r_partial) > 0.1:
        print(f"  Task {i} (deg={out_degrees[i]}): r_partial = {r_loo:.4f}  (Δ = {r_loo - r_partial:+.4f}) ← INFLUENTIAL")

# Without node 2
mask_no2 = np.ones(n_tasks, dtype=bool)
mask_no2[2] = False
r_no2, p_no2 = partial_corr(out_degrees[mask_no2], weight_drifts[mask_no2], mean_child_sim[mask_no2])
print(f"\nWithout node 2 (deg=9): partial r = {r_no2:.4f}  (p = {p_no2:.4f})")

# Decision
print(f"\n{'=' * 65}")
print(f"DECISION")
print(f"  Criterion: |r_partial| > 0.3 AND p < 0.05")
print(f"  Result:    |r_partial| = {abs(r_partial):.4f}, p = {p_partial:.4f}")
if abs(r_partial) > 0.3 and p_partial < 0.05:
    print(f"  → REJECT H₀ (but see robustness caveats)")
else:
    print(f"  → FAIL TO REJECT H₀")
print(f"{'=' * 65}")
print(f"\nCaveats:")
print(f"  - Spearman partial ρ near zero ({rho_partial:.3f})")
print(f"  - Result collapses without node 2")
print(f"  - Bootstrap CI {'includes' if ci_lo <= 0 <= ci_hi else 'excludes'} zero")
print(f"  - Single seed, n=20")
