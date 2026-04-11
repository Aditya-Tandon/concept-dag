"""
Plot suite for Concept DAG experiments.

Generates publication-quality figures from JSON results files produced by
Experiments 3a, 3b, 4, and 5.  All figures are saved as high-res PNGs.

Usage:
    # From individual JSON files:
    python run_experiment.py --exp plot \
        --exp3a results/exp3/exp3a_results.json \
        --exp3b results/exp3/exp3b_results.json \
        [--exp4  results/exp4/exp4_all_results.json] \
        [--exp5a results/exp5/exp5a_results.json] \
        [--exp5b results/exp5/exp5b_results.json] \
        --out_dir results/figures

    # Auto-discover results under ./results/:
    python run_experiment.py --exp plot --auto_discover

Figures produced:
    fig1_task_accuracy.png       — per-task test accuracy (bar chart)
    fig2_dag_topology.png        — DAG graph with out-degree node sizing
    fig3_degree_vs_drift.png     — out-degree vs weight drift scatter + regression
    fig4_ablation_compare.png    — grouped bar chart across ablation variants
    fig5_orth_scores.png         — subspace orthogonality per node
    fig6_cumulative_aa.png       — cumulative average accuracy as DAG grows
    fig7_degree_distribution.png — out-degree histogram
    fig8_ablation_per_task.png   — per-task accuracy by ablation variant
    fig9_causal_ablation.png     — natural vs forced-hub drift comparison
    fig10_n_parents_sweep.png    — accuracy & orthogonality vs n_parents
    fig11_subspace_k_sweep.png   — accuracy & orthogonality vs subspace_k
"""

import os
import json
import argparse
import numpy as np
import warnings
from typing import Dict, Optional

# ── safe imports for optional plotting dependencies
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend for HPC
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    warnings.warn("matplotlib not installed — pip install matplotlib")

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False
    warnings.warn("networkx not installed — pip install networkx  (needed for fig2)")

try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    warnings.warn("scipy not installed — pip install scipy  (regression line uses numpy fallback)")


# ──────────────────────────────────────────────────────────────────────────────
# Style helpers
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "full":       "#2196F3",   # blue
    "no_routing": "#FF9800",   # orange
    "no_crystal": "#4CAF50",   # green
    "no_freeze":  "#9C27B0",   # purple
    "sequential": "#F44336",   # red
    "accent":     "#E91E63",   # pink (for highlight)
    "neutral":    "#607D8B",   # blue-grey
}

def _savefig(fig, path: str, dpi: int = 180):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path}")


def _linreg(x, y):
    """Return (slope, intercept, r, p, stderr) via scipy or numpy fallback."""
    if _HAS_SCIPY:
        return scipy_stats.linregress(x, y)
    # numpy fallback (no p-value)
    coeffs = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    slope, intercept = coeffs
    return slope, intercept, r, None, None


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — per-task accuracy bar chart
# ──────────────────────────────────────────────────────────────────────────────

def fig1_task_accuracy(r3a: Dict, out_dir: str):
    if not _HAS_MPL:
        return
    test_accs  = r3a["test_accs"]
    parent_map = {int(k): v for k, v in r3a["parent_map"].items()}
    out_degree = {int(k): v for k, v in r3a["out_degree"].items()}
    n = len(test_accs)

    colors = [
        PALETTE["accent"] if out_degree.get(t, 0) >= 4 else PALETTE["neutral"]
        for t in range(n)
    ]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    bars = ax.bar(range(n), test_accs, color=colors, edgecolor="white",
                  linewidth=0.6, zorder=3)

    # Annotate root vs. child
    for t in range(n):
        marker = "R" if not parent_map.get(t) else f"d={out_degree.get(t,0)}"
        ax.text(t, test_accs[t] + 0.008, marker, ha="center", va="bottom",
                fontsize=7.5, color="#333333")

    ax.axhline(np.mean(test_accs), color="black", lw=1.5, ls="--",
               label=f"AA = {np.mean(test_accs):.3f}")
    ax.axhline(0.2, color="#9E9E9E", lw=1, ls=":", label="Chance (20%)")

    ax.set_xlabel("Task index", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title("Exp 3a — Per-task accuracy on Split-CIFAR-100 (20 tasks)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(n))
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.legend(fontsize=10)

    patch_hi = mpatches.Patch(color=PALETTE["accent"], label="Hub node (out-degree ≥ 4)")
    patch_lo = mpatches.Patch(color=PALETTE["neutral"], label="Other node")
    ax.legend(handles=[patch_hi, patch_lo,
                        plt.Line2D([0],[0], color="black", lw=1.5, ls="--",
                                   label=f"AA = {np.mean(test_accs):.3f}"),
                        plt.Line2D([0],[0], color="#9E9E9E", lw=1, ls=":",
                                   label="Chance (20%)")],
              fontsize=9, loc="upper right")

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig1_task_accuracy.png"))


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — DAG topology
# ──────────────────────────────────────────────────────────────────────────────

def fig2_dag_topology(r3a: Dict, out_dir: str):
    if not _HAS_MPL or not _HAS_NX:
        print("  Skipping fig2 (needs matplotlib + networkx)")
        return

    parent_map = {int(k): v for k, v in r3a["parent_map"].items()}
    out_degree = {int(k): v for k, v in r3a["out_degree"].items()}
    test_accs  = r3a["test_accs"]
    n = len(test_accs)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for child, parents in parent_map.items():
        for p in parents:
            G.add_edge(p, child)

    # Hierarchical layout: group nodes by generation (depth from roots)
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        # Fallback: spring layout if graphviz not available
        pos = nx.spring_layout(G, seed=42, k=2.0)

    fig, ax = plt.subplots(figsize=(16, 9))

    node_sizes = [300 + 250 * out_degree.get(t, 0) for t in range(n)]
    node_colors = [
        plt.cm.RdYlGn(test_accs[t])   # green = high acc, red = low
        for t in range(n)
    ]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#BDBDBD",
                           arrows=True, arrowsize=14, width=1.2,
                           connectionstyle="arc3,rad=0.08",
                           node_size=node_sizes)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, edgecolors="#333333",
                           linewidths=0.8)
    # Labels: task id + test accuracy
    labels = {t: f"T{t}\n{test_accs[t]:.2f}" for t in range(n)}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7.5,
                             font_color="white", font_weight="bold")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn,
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.01)
    cbar.set_label("Test accuracy", fontsize=11)

    ax.set_title("Exp 3a — Growing Concept DAG topology\n"
                 "(node size ∝ out-degree; colour = test accuracy)",
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig2_dag_topology.png"))


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — out-degree vs weight drift
# ──────────────────────────────────────────────────────────────────────────────

def fig3_degree_vs_drift(r3b: Dict, out_dir: str, r3b_forced: Optional[Dict] = None):
    """
    Scatter plot with linear regression line.
    If r3b_forced is provided, overlay the forced-hub causal ablation points.
    """
    if not _HAS_MPL:
        return

    per_node = r3b["per_node"]
    degrees  = np.array([r["out_degree"]   for r in per_node], dtype=float)
    drifts   = np.array([r["weight_drift"] for r in per_node], dtype=float)
    corr_nat = r3b.get("corr_out_degree_drift", np.corrcoef(degrees, drifts)[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left panel: degree vs drift (natural run)
    ax = axes[0]
    ax.scatter(degrees, drifts, color=PALETTE["full"], s=80, alpha=0.8,
               edgecolors="white", linewidths=0.5, zorder=3, label="Natural run")

    # Regression line
    slope, intercept, r, p_val, _ = _linreg(degrees.tolist(), drifts.tolist())
    x_line = np.linspace(degrees.min(), degrees.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "--", color="black", lw=1.5, label=f"Linear fit (r={r:.3f})")

    # Annotate high-degree nodes
    for rec in per_node:
        if rec["out_degree"] >= 3:
            ax.annotate(f"T{rec['task']}\n(d={rec['out_degree']})",
                        (rec["out_degree"], rec["weight_drift"]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color="#333333")

    p_str = f"p={p_val:.4f}" if p_val is not None else ""
    ax.set_title(f"Exp 3b — Out-degree vs. Weight Drift\n(Natural run, ρ={corr_nat:.3f}, {p_str})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Node out-degree", fontsize=11)
    ax.set_ylabel("Weight drift ‖Δθ‖₂", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Right panel: degree vs accuracy drop
    acc_drops = np.array([r["acc_drop"] for r in per_node], dtype=float)
    ax2 = axes[1]
    ax2.scatter(degrees, acc_drops, color=PALETTE["no_crystal"], s=80, alpha=0.8,
                edgecolors="white", linewidths=0.5, zorder=3)
    ax2.axhline(0, color="#9E9E9E", lw=1.0, ls="--")
    slope2, intercept2, r2, p_val2, _ = _linreg(degrees.tolist(), acc_drops.tolist())
    y_line2 = slope2 * x_line + intercept2
    ax2.plot(x_line, y_line2, "--", color="black", lw=1.5,
             label=f"Linear fit (r={r2:.3f})")

    corr_drop = r3b.get("corr_out_degree_drop", np.corrcoef(degrees, acc_drops)[0, 1])
    ax2.set_title(f"Exp 3b — Out-degree vs. Accuracy Drop\n(ρ={corr_drop:.3f}  — noise-level, as expected)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("Node out-degree", fontsize=11)
    ax2.set_ylabel("Accuracy drop (base − post)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # ── Overlay forced-hub if available
    if r3b_forced:
        pert = r3b_forced.get("perturbation", [])
        forced_hub_task = r3b_forced.get("forced_hub_task", 0)
        fh_deg    = np.array([r["out_degree"]   for r in pert], dtype=float)
        fh_drift  = np.array([r["weight_drift"] for r in pert], dtype=float)
        corr_fh   = r3b_forced.get("corr_out_degree_drift", np.nan)

        axes[0].scatter(fh_deg, fh_drift, color=PALETTE["accent"], s=80,
                        alpha=0.7, edgecolors="white", marker="D",
                        linewidths=0.5, zorder=4, label="Forced-hub run")
        # Highlight the forced hub point
        fh_point = next(r for r in pert if r["task"] == forced_hub_task)
        axes[0].scatter([fh_point["out_degree"]], [fh_point["weight_drift"]],
                         color=PALETTE["accent"], s=160, edgecolors="black",
                         marker="*", zorder=5,
                         label=f"Forced hub T{forced_hub_task} (d={fh_point['out_degree']})")
        axes[0].set_title(
            f"Degree vs. Drift (natural ρ={corr_nat:.3f}; forced-hub ρ={corr_fh:.3f})",
            fontsize=11, fontweight="bold")
        axes[0].legend(fontsize=8.5)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig3_degree_vs_drift.png"))


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — ablation comparison
# ──────────────────────────────────────────────────────────────────────────────

def fig4_ablation_compare(r4_all: Dict, out_dir: str):
    if not _HAS_MPL:
        return

    names = list(r4_all.keys())
    aa_vals  = [r4_all[v]["average_accuracy"]  for v in names]
    bt_vals  = [r4_all[v]["backward_transfer"] for v in names]
    fin_vals = [
        float(np.mean(r4_all[v].get("final_accs", r4_all[v]["test_accs"])))
        for v in names
    ]
    colors = [PALETTE.get(v, PALETTE["neutral"]) for v in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    bar_kw = dict(edgecolor="white", linewidth=0.7)

    # AA
    ax = axes[0]
    bars = ax.bar(names, aa_vals, color=colors, **bar_kw)
    for bar, val in zip(bars, aa_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Average Accuracy (AA)\nat end of each task's training",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, max(aa_vals) * 1.2)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.35)

    # Backward Transfer
    ax = axes[1]
    bt_colors = [PALETTE["accent"] if v < 0 else PALETTE["full"] for v in bt_vals]
    bars = ax.bar(names, bt_vals, color=bt_colors, **bar_kw)
    for bar, val in zip(bars, bt_vals):
        ypos = bar.get_height() + 0.003 if val >= 0 else bar.get_height() - 0.015
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{val:+.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", lw=1.0, ls="--")
    ax.set_title("Backward Transfer (BT)\n(negative = catastrophic forgetting)",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("BT", fontsize=11)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.35)

    # Final AA (re-evaluated after all tasks)
    ax = axes[2]
    bars = ax.bar(names, fin_vals, color=colors, **bar_kw)
    for bar, val in zip(bars, fin_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Final AA (all tasks re-evaluated\nafter full DAG grown)",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, max(fin_vals) * 1.2)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.35)

    fig.suptitle("Exp 4 — Ablation Study: Component Contributions",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig4_ablation_compare.png"))


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 — orthogonality scores
# ──────────────────────────────────────────────────────────────────────────────

def fig5_orth_scores(r3a: Dict, out_dir: str):
    if not _HAS_MPL:
        return

    orth_scores = {int(k): v for k, v in r3a["orth_scores"].items()}
    tasks = sorted(orth_scores.keys())
    vals  = [orth_scores[t] for t in tasks]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(tasks, np.log10(np.maximum(vals, 1e-12)),
           color=PALETTE["full"], edgecolor="white", linewidth=0.6)
    ax.axhline(np.log10(1e-6), color=PALETTE["accent"], ls="--", lw=1.5,
               label="1e-6 threshold (excellent)")
    ax.set_xlabel("Task / node index", fontsize=12)
    ax.set_ylabel("log₁₀ ‖GᵀG − I‖_F  (lower = more orthogonal)", fontsize=11)
    ax.set_title("Exp 3a — Concept Subspace Orthogonality per Node\n"
                 "(all values ~1e-7: subspaces are crystallized)", fontsize=12, fontweight="bold")
    ax.set_xticks(tasks)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig5_orth_scores.png"))


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6 — cumulative average accuracy (AA grows with task count)
# ──────────────────────────────────────────────────────────────────────────────

def fig6_cumulative_aa(r3a: Dict, out_dir: str, r4_all: Optional[Dict] = None):
    if not _HAS_MPL:
        return

    test_accs = r3a["test_accs"]
    n         = len(test_accs)
    cum_aa    = [float(np.mean(test_accs[:t+1])) for t in range(n)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, n+1), cum_aa, color=PALETTE["full"], lw=2.5, marker="o",
            ms=5, label="Full system (Exp 3a)")

    if r4_all:
        for vname, res in r4_all.items():
            if vname == "full":
                continue  # already plotted
            accs = res["test_accs"]
            c_aa = [float(np.mean(accs[:t+1])) for t in range(min(n, len(accs)))]
            ax.plot(range(1, len(c_aa)+1), c_aa, lw=1.5, ls="--",
                    color=PALETTE.get(vname, PALETTE["neutral"]),
                    label=f"Ablation: {vname}")

    ax.axhline(0.2, color="#9E9E9E", lw=1.0, ls=":", label="Chance (20%)")
    ax.set_xlabel("Number of tasks seen", fontsize=12)
    ax.set_ylabel("Cumulative average accuracy", fontsize=12)
    ax.set_title("Cumulative Average Accuracy as DAG Grows", fontsize=13, fontweight="bold")
    ax.set_xlim(1, n)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.35)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig6_cumulative_aa.png"))


# ──────────────────────────────────────────────────────────────────────────────
# Figure 7 — out-degree histogram
# ──────────────────────────────────────────────────────────────────────────────

def fig7_degree_distribution(r3a: Dict, out_dir: str):
    if not _HAS_MPL:
        return

    out_degree = {int(k): v for k, v in r3a["out_degree"].items()}
    vals = list(out_degree.values())
    max_d = max(vals)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = range(0, max_d + 2)
    ax.hist(vals, bins=bins, align="left", color=PALETTE["full"],
            edgecolor="white", linewidth=0.6, rwidth=0.8)
    ax.set_xlabel("Out-degree", fontsize=12)
    ax.set_ylabel("Number of nodes", fontsize=12)
    ax.set_title("Concept DAG — Out-degree Distribution\n"
                 "(power-law-like: one hub dominates)", fontsize=12, fontweight="bold")
    ax.set_xticks(range(0, max_d + 1))
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig7_degree_distribution.png"))


# ──────────────────────────────────────────────────────────────────────────────
# Figure 8 — Ablation per-task accuracy comparison (line plot)
# ──────────────────────────────────────────────────────────────────────────────

def fig8_ablation_per_task(r4_all: Dict, out_dir: str):
    if not _HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    for vname, res in r4_all.items():
        accs = res["test_accs"]
        ax.plot(range(len(accs)), accs, lw=2.0, marker="o", ms=4,
                color=PALETTE.get(vname, PALETTE["neutral"]),
                label=vname, alpha=0.85)

    ax.axhline(0.2, color="#9E9E9E", lw=1.0, ls=":", label="Chance")
    ax.set_xlabel("Task index", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title("Exp 4 — Per-task accuracy across ablation variants", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.35)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig8_ablation_per_task.png"))


# ──────────────────────────────────────────────────────────────────────────────
# Figure 9 — Drift comparison: natural hub vs. forced hub (causal ablation)
# ──────────────────────────────────────────────────────────────────────────────

def fig9_causal_ablation(r3b: Dict, r3b_forced: Dict, out_dir: str):
    if not _HAS_MPL:
        return

    nat   = {r["task"]: r for r in r3b["per_node"]}
    frc   = {r["task"]: r for r in r3b_forced["perturbation"]}
    tasks = sorted(set(nat.keys()) & set(frc.keys()))

    nat_drift = [nat[t]["weight_drift"] for t in tasks]
    frc_drift = [frc[t]["weight_drift"] for t in tasks]

    x = np.arange(len(tasks))
    w = 0.4

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w/2, nat_drift, width=w, color=PALETTE["full"],
           label="Natural run", edgecolor="white")
    ax.bar(x + w/2, frc_drift, width=w, color=PALETTE["accent"],
           label="Forced-hub run", edgecolor="white", alpha=0.85)

    forced_hub_task = r3b_forced.get("forced_hub_task", 0)
    ax.axvline(tasks.index(forced_hub_task), color="black", lw=1.5, ls="--",
               label=f"Forced hub (T{forced_hub_task})")

    ax.set_xticks(x)
    ax.set_xticklabels([f"T{t}" for t in tasks], fontsize=9)
    ax.set_xlabel("Task / node", fontsize=12)
    ax.set_ylabel("Weight drift ‖Δθ‖₂", fontsize=12)
    ax.set_title("Exp 4f — Causal Ablation: Natural vs. Forced-Hub Drift\n"
                 "(if forced hub drifts less → structural effect is causal)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig9_causal_ablation.png"))



    print(f"\nDone — {len(os.listdir(out_dir))} files in {out_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 10 — n_parents sweep
# ──────────────────────────────────────────────────────────────────────────────

def fig10_n_parents_sweep(r5a: Dict, out_dir: str):
    """
    Two-panel figure:
      Left  — AA vs n_parents (line + markers)
      Right — per-task accuracy heatmap (tasks × n_parents values)
    """
    if not _HAS_MPL:
        return
    sweep = r5a["sweep"]
    n_par_vals  = [s["n_parents"]        for s in sweep]
    aa_vals     = [s["average_accuracy"] for s in sweep]
    orth_vals   = [s["avg_orth"]         for s in sweep]
    # per-task matrix: rows = n_parents values, cols = tasks
    task_matrix = np.array([s["test_accs"] for s in sweep])  # (n_configs, n_tasks)
    n_tasks     = task_matrix.shape[1]

    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Top-left: AA vs n_parents
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(n_par_vals, aa_vals, color=PALETTE["full"], lw=2.5, marker="o",
             ms=8, zorder=3)
    for x, y in zip(n_par_vals, aa_vals):
        ax1.text(x, y + 0.003, f"{y:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.set_xlabel("n_parents", fontsize=12)
    ax1.set_ylabel("Average Accuracy", fontsize=12)
    ax1.set_title("AA vs. number of parents per child", fontsize=11, fontweight="bold")
    ax1.set_xticks(n_par_vals)
    ax1.set_ylim(max(0, min(aa_vals) - 0.05), min(1, max(aa_vals) + 0.06))
    ax1.grid(alpha=0.35)

    # ── Top-right: avg_orth vs n_parents (log scale)
    ax2 = fig.add_subplot(gs[0, 1])
    log_orth = np.log10(np.maximum(orth_vals, 1e-12))
    ax2.plot(n_par_vals, log_orth, color=PALETTE["no_crystal"], lw=2.5, marker="s",
             ms=8, zorder=3)
    for x, y, raw in zip(n_par_vals, log_orth, orth_vals):
        ax2.text(x, y + 0.02, f"{raw:.1e}", ha="center", va="bottom", fontsize=8.5)
    ax2.set_xlabel("n_parents", fontsize=12)
    ax2.set_ylabel("log₁₀ avg ‖GᵀG − I‖_F", fontsize=11)
    ax2.set_title("Subspace orthogonality vs. n_parents\n(lower = better crystallization)",
                  fontsize=11, fontweight="bold")
    ax2.set_xticks(n_par_vals)
    ax2.grid(alpha=0.35)

    # ── Bottom: per-task accuracy heatmap
    ax3 = fig.add_subplot(gs[1, :])
    im = ax3.imshow(task_matrix, aspect="auto", cmap="RdYlGn",
                    vmin=0.0, vmax=1.0, interpolation="nearest")
    ax3.set_yticks(range(len(n_par_vals)))
    ax3.set_yticklabels([f"n_par={v}" for v in n_par_vals], fontsize=10)
    ax3.set_xticks(range(n_tasks))
    ax3.set_xticklabels([f"T{t}" for t in range(n_tasks)], fontsize=8.5)
    ax3.set_xlabel("Task index", fontsize=12)
    ax3.set_title("Per-task accuracy heat map (green = higher)", fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax3, orientation="vertical", pad=0.01, shrink=0.85)
    cbar.set_label("Test accuracy", fontsize=10)
    # Annotate cells
    for row in range(len(n_par_vals)):
        for col in range(n_tasks):
            val = task_matrix[row, col]
            ax3.text(col, row, f"{val:.2f}", ha="center", va="center",
                     fontsize=6.5, color="black" if 0.3 < val < 0.8 else "white")

    fig.suptitle("Exp 5a — Sensitivity to n_parents\n"
                 f"(subspace_k fixed = {r5a['fixed_subspace_k']}, "
                 f"{r5a['n_tasks']} tasks, Split-CIFAR-100)",
                 fontsize=13, fontweight="bold")
    _savefig(fig, os.path.join(out_dir, "fig10_n_parents_sweep.png"))


# ──────────────────────────────────────────────────────────────────────────────
# Figure 11 — subspace_k sweep
# ──────────────────────────────────────────────────────────────────────────────

def fig11_subspace_k_sweep(r5b: Dict, out_dir: str):
    """
    Two-panel figure:
      Left  — AA vs subspace_k (line + markers), log x-axis
      Right — per-task accuracy heatmap (tasks × k values)
    Also shows: avg orthogonality score vs k, since k directly affects how many
    directions SoftPCA must keep orthogonal inside concept_dim space.
    """
    if not _HAS_MPL:
        return
    sweep   = r5b["sweep"]
    k_vals  = [s["subspace_k"]        for s in sweep]
    aa_vals = [s["average_accuracy"]  for s in sweep]
    orth_vals = [s["avg_orth"]        for s in sweep]
    task_matrix = np.array([s["test_accs"] for s in sweep])  # (n_configs, n_tasks)
    n_tasks = task_matrix.shape[1]

    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Top-left: AA vs subspace_k
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(k_vals, aa_vals, color=PALETTE["full"], lw=2.5, marker="o", ms=8, zorder=3)
    for x, y in zip(k_vals, aa_vals):
        ax1.text(x, y + 0.003, f"{y:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(k_vals)
    ax1.set_xticklabels([str(k) for k in k_vals])
    ax1.set_xlabel("subspace_k  (log₂ scale)", fontsize=12)
    ax1.set_ylabel("Average Accuracy", fontsize=12)
    ax1.set_title("AA vs. subspace_k (concept vectors per node)", fontsize=11, fontweight="bold")
    ax1.set_ylim(max(0, min(aa_vals) - 0.05), min(1, max(aa_vals) + 0.06))
    ax1.grid(alpha=0.35, which="both")

    # ── Top-right: orth vs subspace_k
    ax2 = fig.add_subplot(gs[0, 1])
    log_orth = np.log10(np.maximum(orth_vals, 1e-12))
    ax2.plot(k_vals, log_orth, color=PALETTE["no_freeze"], lw=2.5, marker="D", ms=8, zorder=3)
    for x, y, raw in zip(k_vals, log_orth, orth_vals):
        ax2.text(x, y + 0.02, f"{raw:.1e}", ha="center", va="bottom", fontsize=8.5)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(k_vals)
    ax2.set_xticklabels([str(k) for k in k_vals])
    ax2.set_xlabel("subspace_k  (log₂ scale)", fontsize=12)
    ax2.set_ylabel("log₁₀ avg ‖GᵀG − I‖_F", fontsize=11)
    ax2.set_title("Orthogonality vs. subspace_k\n(degrades as k → concept_dim: too many vectors to orthogonalise)",
                  fontsize=11, fontweight="bold")
    ax2.grid(alpha=0.35, which="both")

    # ── Bottom: per-task accuracy heatmap
    ax3 = fig.add_subplot(gs[1, :])
    im = ax3.imshow(task_matrix, aspect="auto", cmap="RdYlGn",
                    vmin=0.0, vmax=1.0, interpolation="nearest")
    ax3.set_yticks(range(len(k_vals)))
    ax3.set_yticklabels([f"k={v}" for v in k_vals], fontsize=10)
    ax3.set_xticks(range(n_tasks))
    ax3.set_xticklabels([f"T{t}" for t in range(n_tasks)], fontsize=8.5)
    ax3.set_xlabel("Task index", fontsize=12)
    ax3.set_title("Per-task accuracy heat map (green = higher)", fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax3, orientation="vertical", pad=0.01, shrink=0.85)
    cbar.set_label("Test accuracy", fontsize=10)
    for row in range(len(k_vals)):
        for col in range(n_tasks):
            val = task_matrix[row, col]
            ax3.text(col, row, f"{val:.2f}", ha="center", va="center",
                     fontsize=6.5, color="black" if 0.3 < val < 0.8 else "white")

    fig.suptitle("Exp 5b — Sensitivity to subspace_k (concept subspace dimensionality)\n"
                 f"(n_parents fixed = {r5b['fixed_n_parents']}, "
                 f"{r5b['n_tasks']} tasks, Split-CIFAR-100)",
                 fontsize=13, fontweight="bold")
    _savefig(fig, os.path.join(out_dir, "fig11_subspace_k_sweep.png"))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers to load JSON results
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: Optional[str]) -> Optional[Dict]:
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"  [warn] File not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _auto_discover(root: str = "./results") -> Dict[str, Optional[str]]:
    """Walk ./results/ and find the most recent JSON files for each experiment."""
    found: Dict[str, Optional[str]] = {
        "exp3a": None, "exp3b": None,
        "exp4":  None, "exp4f": None,
        "exp5a": None, "exp5b": None,
    }
    for dirpath, _, files in os.walk(root):
        for fname in files:
            full = os.path.join(dirpath, fname)
            if   "exp3a_results"  in fname: found["exp3a"] = full
            elif "exp3b_results"  in fname: found["exp3b"] = full
            elif "exp4_all"       in fname: found["exp4"]  = full
            elif "forced_hub"     in fname: found["exp4f"] = full
            elif "exp5a_results"  in fname: found["exp5a"] = full
            elif "exp5b_results"  in fname: found["exp5b"] = full
    return found


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_plots(
    exp3a_path:    Optional[str] = None,
    exp3b_path:    Optional[str] = None,
    exp4_path:     Optional[str] = None,
    exp4f_path:    Optional[str] = None,
    exp5a_path:    Optional[str] = None,
    exp5b_path:    Optional[str] = None,
    out_dir:       str = "./results/figures",
    auto_discover: bool = False,
    results_root:  str = "./results",
):
    if not _HAS_MPL:
        print("ERROR: matplotlib is required.  Run: pip install matplotlib")
        return

    if auto_discover:
        paths = _auto_discover(results_root)
        exp3a_path = exp3a_path or paths["exp3a"]
        exp3b_path = exp3b_path or paths["exp3b"]
        exp4_path  = exp4_path  or paths["exp4"]
        exp4f_path = exp4f_path or paths["exp4f"]
        exp5a_path = exp5a_path or paths["exp5a"]
        exp5b_path = exp5b_path or paths["exp5b"]

    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating figures → {out_dir}/")

    r3a  = _load_json(exp3a_path)
    r3b  = _load_json(exp3b_path)
    r4   = _load_json(exp4_path)
    r4f  = _load_json(exp4f_path)
    r5a  = _load_json(exp5a_path)
    r5b  = _load_json(exp5b_path)

    if r3a:
        print("\n[Fig 1] Per-task accuracy (exp3a)")
        fig1_task_accuracy(r3a, out_dir)

        print("[Fig 2] DAG topology (exp3a)")
        fig2_dag_topology(r3a, out_dir)

        print("[Fig 5] Orthogonality scores (exp3a)")
        fig5_orth_scores(r3a, out_dir)

        print("[Fig 7] Degree distribution (exp3a)")
        fig7_degree_distribution(r3a, out_dir)

    if r3b:
        print("[Fig 3] Degree vs drift (exp3b)")
        fig3_degree_vs_drift(r3b, out_dir, r3b_forced=r4f)

    if r3a:
        print("[Fig 6] Cumulative AA (exp3a + ablation overlay)")
        fig6_cumulative_aa(r3a, out_dir, r4_all=r4)

    if r4:
        print("[Fig 4] Ablation comparison (exp4)")
        fig4_ablation_compare(r4, out_dir)

        print("[Fig 8] Per-task accuracy by variant (exp4)")
        fig8_ablation_per_task(r4, out_dir)

    if r3b and r4f:
        print("[Fig 9] Causal ablation: natural vs forced hub (exp3b + exp4f)")
        fig9_causal_ablation(r3b, r4f, out_dir)

    if r5a:
        print("[Fig 10] n_parents sweep (exp5a)")
        fig10_n_parents_sweep(r5a, out_dir)

    if r5b:
        print("[Fig 11] subspace_k sweep (exp5b)")
        fig11_subspace_k_sweep(r5b, out_dir)

    n_figs = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
    print(f"\nDone — {n_figs} figures in {out_dir}/")


# Allow typing import for Optional/Dict at module level without importing from __future__
from typing import Optional, Dict
