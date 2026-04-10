"""
Experiment 1 — Crystallization mechanism comparison.

Tests the core claim: does eigenvector-based aggregation (SVD / SoftPCA)
produce better composed representations than standard methods (concat, mean, attention)?

Sub-experiments:
  1a. Aggregation comparison on SyntheticComposition (no download needed).
      Three modules: parent_A (shape encoder), parent_B (colour encoder),
      child (composition classifier with one of the 5 aggregators).
      Measures: accuracy, representation orthogonality, gradient norms.

  1b. SVD gradient stability analysis.
      Same setup, but tracks gradient norms through the SVD layer over time.
      Tests mitigations: exact SVD vs. SVD+eps vs. SoftPCA.

Run from project root:
    python -m concept_dag.experiments.exp1_crystallization

Results are saved to results/exp1/
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from dataclasses import dataclass, asdict

from ..modules.concept_module import ConceptModule
from ..modules.aggregation import build_aggregator, AGGREGATORS
from ..models.baselines import SimpleMLP, LinearHead
from ..utils.metrics import (
    accuracy,
    representation_orthogonality,
    variance_explained,
    GradientNormTracker,
    CLMetricsTracker,
)
from ..data.loaders import make_synthetic_loaders


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Exp1Config:
    # Synthetic dataset
    n_shape_classes: int = 4
    n_colour_classes: int = 4
    shape_dim: int = 32
    colour_dim: int = 32
    n_samples: int = 6000
    batch_size: int = 128
    # Model
    hidden_dim: int = 64
    out_dim: int = 64
    n_mlp_layers: int = 2
    top_k: int = 2             # k for SVD/SoftPCA — SVD is capped at min(n_parents, parent_dim)=2; SoftPCA can go higher
    # Training
    n_epochs: int = 30
    lr: float = 1e-3
    orth_weight: float = 0.01
    # Misc
    seed: int = 42
    device: str = "cpu"
    results_dir: str = "results/exp1"


# ---------------------------------------------------------------------------
# Experiment 1a — Aggregation comparison
# ---------------------------------------------------------------------------

def run_exp1a(cfg: Exp1Config) -> Dict:
    """
    Compare all 5 aggregators on the synthetic composition task.
    Each aggregator trains a child module that fuses two parent encoders
    (shape and colour) to classify the composed label.
    """
    print("\n" + "="*70)
    print("Experiment 1a: Aggregation strategy comparison")
    print("="*70)

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    os.makedirs(cfg.results_dir, exist_ok=True)

    # Create datasets
    # Parent A: shape task
    train_shape, val_shape, test_shape, n_shape = make_synthetic_loaders(
        task="shape", n_shape_classes=cfg.n_shape_classes, n_colour_classes=cfg.n_colour_classes,
        shape_dim=cfg.shape_dim, colour_dim=cfg.colour_dim,
        n_samples=cfg.n_samples, batch_size=cfg.batch_size, seed=cfg.seed,
    )
    # Parent B: colour task
    train_colour, val_colour, test_colour, n_colour = make_synthetic_loaders(
        task="colour", n_shape_classes=cfg.n_shape_classes, n_colour_classes=cfg.n_colour_classes,
        shape_dim=cfg.shape_dim, colour_dim=cfg.colour_dim,
        n_samples=cfg.n_samples, batch_size=cfg.batch_size, seed=cfg.seed + 1,
    )
    # Child: composition task
    train_comp, val_comp, test_comp, n_comp = make_synthetic_loaders(
        task="composition", n_shape_classes=cfg.n_shape_classes, n_colour_classes=cfg.n_colour_classes,
        shape_dim=cfg.shape_dim, colour_dim=cfg.colour_dim,
        n_samples=cfg.n_samples, batch_size=cfg.batch_size, seed=cfg.seed + 2,
    )

    in_dim = cfg.shape_dim + cfg.colour_dim

    # -----------------------------------------------------------------------
    # Train the two parent modules (shared across all aggregator conditions)
    # -----------------------------------------------------------------------
    print("\n[Step 1] Training parent modules...")

    parent_A = ConceptModule(
        module_id="parent_shape",
        in_dim=in_dim, hidden_dim=cfg.hidden_dim, out_dim=cfg.out_dim,
        n_layers=cfg.n_mlp_layers, n_parents=0,
    ).to(device)
    head_A = LinearHead(cfg.out_dim, n_shape).to(device)

    parent_B = ConceptModule(
        module_id="parent_colour",
        in_dim=in_dim, hidden_dim=cfg.hidden_dim, out_dim=cfg.out_dim,
        n_layers=cfg.n_mlp_layers, n_parents=0,
    ).to(device)
    head_B = LinearHead(cfg.out_dim, n_colour).to(device)

    _train_module(parent_A, head_A, train_shape, cfg, device, "parent_shape")
    _train_module(parent_B, head_B, train_colour, cfg, device, "parent_colour")

    # Evaluate parents
    parent_A_acc = _eval_module(parent_A, head_A, test_shape, device)
    parent_B_acc = _eval_module(parent_B, head_B, test_colour, device)
    print(f"  Parent A (shape)  test acc: {parent_A_acc:.4f}")
    print(f"  Parent B (colour) test acc: {parent_B_acc:.4f}")

    # Freeze parents
    parent_A.freeze()
    parent_B.freeze()

    # Compute and cache concept subspaces
    _collect_subspace(parent_A, test_shape, device)
    _collect_subspace(parent_B, test_colour, device)

    # -----------------------------------------------------------------------
    # For each aggregator, train a child module
    # -----------------------------------------------------------------------
    print("\n[Step 2] Training child modules with each aggregator...")

    aggregators_to_test = list(AGGREGATORS.keys())
    results = {
        "parent_A_acc": parent_A_acc,
        "parent_B_acc": parent_B_acc,
        "child_results": {},
    }

    for agg_name in aggregators_to_test:
        print(f"\n  --- Aggregator: {agg_name} ---")
        torch.manual_seed(cfg.seed + 99)

        # SVD rank is capped at min(n_parents, parent_dim) = min(2, out_dim).
        # SoftPCA projects the concatenated vector so can use a higher k.
        if agg_name == "svd":
            agg_kwargs = {"top_k": min(cfg.top_k, 2)}   # hard cap: 2 parents → max rank 2
        elif agg_name == "soft_pca":
            agg_kwargs = {"top_k": max(cfg.top_k, 8)}   # SoftPCA can go wider
        else:
            agg_kwargs = {}

        child = ConceptModule(
            module_id=f"child_{agg_name}",
            in_dim=cfg.out_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.out_dim,
            n_layers=cfg.n_mlp_layers,
            n_parents=2,
            aggregation=agg_name,
            agg_kwargs=agg_kwargs,
        ).to(device)
        head_child = LinearHead(cfg.out_dim, n_comp).to(device)

        # Gradient tracker on the aggregator
        grad_tracker = GradientNormTracker(child.aggregator, name=agg_name) if child.aggregator else None

        history = _train_child(
            child=child,
            head=head_child,
            parent_A=parent_A,
            parent_B=parent_B,
            train_loader=train_comp,
            cfg=cfg,
            device=device,
            orth_weight=cfg.orth_weight if agg_name == "soft_pca" else 0.0,
        )

        # Final evaluation
        test_acc = _eval_child(child, head_child, parent_A, parent_B, test_comp, device)

        # Representation quality metrics
        orth_score, var_exp = _measure_representation(child, parent_A, parent_B, val_comp, device)

        # Gradient stability
        grad_stats = {}
        if grad_tracker:
            grad_stats = {
                "max_grad_norm": grad_tracker.max_norm(),
                "mean_grad_norm": grad_tracker.mean_norm(),
                "had_explosion": grad_tracker.had_explosion(threshold=100.0),
            }
            grad_tracker.remove()

        result = {
            "test_accuracy": test_acc,
            "representation_orthogonality": orth_score,
            "variance_explained": var_exp,
            "train_history": history,
            "gradient_stats": grad_stats,
        }
        results["child_results"][agg_name] = result

        print(f"    Test acc: {test_acc:.4f}")
        print(f"    Repr. orthogonality (lower=better): {orth_score:.4f}")
        print(f"    Variance explained by top-{cfg.top_k}: {var_exp:.4f}")
        if grad_stats:
            print(f"    Gradient stats: {grad_stats}")

    # Save results
    out_path = os.path.join(cfg.results_dir, "exp1a_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_safe)
    print(f"\nResults saved to {out_path}")

    _print_summary_table(results)
    return results


# ---------------------------------------------------------------------------
# Experiment 1b — SVD gradient stability
# ---------------------------------------------------------------------------

def run_exp1b(cfg: Exp1Config) -> Dict:
    """
    Track gradient norms through the SVD layer at every step.
    Compare: exact SVD vs. SVD+eps vs. SoftPCA.
    """
    print("\n" + "="*70)
    print("Experiment 1b: SVD gradient stability analysis")
    print("="*70)

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_comp, val_comp, test_comp, n_comp = make_synthetic_loaders(
        task="composition",
        n_shape_classes=cfg.n_shape_classes, n_colour_classes=cfg.n_colour_classes,
        shape_dim=cfg.shape_dim, colour_dim=cfg.colour_dim,
        n_samples=cfg.n_samples, batch_size=cfg.batch_size, seed=cfg.seed,
    )
    in_dim = cfg.shape_dim + cfg.colour_dim

    # Re-train shared parents
    train_shape, _, _, n_shape = make_synthetic_loaders("shape", cfg.n_shape_classes, cfg.n_colour_classes,
                                                         cfg.shape_dim, cfg.colour_dim, cfg.n_samples,
                                                         cfg.batch_size, seed=cfg.seed)
    train_colour, _, _, n_colour = make_synthetic_loaders("colour", cfg.n_shape_classes, cfg.n_colour_classes,
                                                           cfg.shape_dim, cfg.colour_dim, cfg.n_samples,
                                                           cfg.batch_size, seed=cfg.seed + 1)

    parent_A = ConceptModule("pa", in_dim=in_dim, hidden_dim=cfg.hidden_dim, out_dim=cfg.out_dim,
                              n_layers=cfg.n_mlp_layers, n_parents=0).to(device)
    parent_B = ConceptModule("pb", in_dim=in_dim, hidden_dim=cfg.hidden_dim, out_dim=cfg.out_dim,
                              n_layers=cfg.n_mlp_layers, n_parents=0).to(device)
    _train_module(parent_A, LinearHead(cfg.out_dim, n_shape).to(device), train_shape, cfg, device, "pa")
    _train_module(parent_B, LinearHead(cfg.out_dim, n_colour).to(device), train_colour, cfg, device, "pb")
    parent_A.freeze(); parent_B.freeze()

    results = {}
    svd_k   = min(cfg.top_k, 2)         # SVD rank ≤ min(n_parents=2, parent_dim)
    spca_k  = max(cfg.top_k, 8)         # SoftPCA can go wider (projects concat vector)
    conditions = {
        "svd_exact": {"aggregation": "svd",      "agg_kwargs": {"top_k": svd_k,  "eps": 0.0}},
        "svd_eps":   {"aggregation": "svd",      "agg_kwargs": {"top_k": svd_k,  "eps": 1e-3}},
        "soft_pca":  {"aggregation": "soft_pca", "agg_kwargs": {"top_k": spca_k, "orth_weight": 0.01}},
    }

    for cond_name, cond_cfg in conditions.items():
        print(f"\n  --- Condition: {cond_name} ---")
        torch.manual_seed(cfg.seed + 200)

        child = ConceptModule(
            module_id=f"child_{cond_name}",
            in_dim=cfg.out_dim, hidden_dim=cfg.hidden_dim, out_dim=cfg.out_dim,
            n_layers=cfg.n_mlp_layers, n_parents=2,
            aggregation=cond_cfg["aggregation"],
            agg_kwargs=cond_cfg["agg_kwargs"],
        ).to(device)
        head_child = LinearHead(cfg.out_dim, n_comp).to(device)

        grad_norms_per_step = []

        def hook_fn(module, grad_in, grad_out):
            for g in grad_out:
                if g is not None:
                    grad_norms_per_step.append(float(g.norm().item()))

        hook = child.aggregator.register_full_backward_hook(hook_fn)

        history = _train_child(child, head_child, parent_A, parent_B, train_comp, cfg, device,
                                orth_weight=cfg.orth_weight if cond_name == "soft_pca" else 0.0)

        hook.remove()

        test_acc = _eval_child(child, head_child, parent_A, parent_B, test_comp, device)

        results[cond_name] = {
            "test_accuracy": test_acc,
            "grad_norms_per_step": grad_norms_per_step,
            "max_grad_norm": max(grad_norms_per_step) if grad_norms_per_step else 0.0,
            "n_explosions": sum(1 for g in grad_norms_per_step if g > 100),
            "train_history": history,
        }
        print(f"    Test acc:       {test_acc:.4f}")
        print(f"    Max grad norm:  {results[cond_name]['max_grad_norm']:.2f}")
        print(f"    # explosions:   {results[cond_name]['n_explosions']}")

    out_path = os.path.join(cfg.results_dir, "exp1b_results.json")
    os.makedirs(cfg.results_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_safe)
    print(f"\nResults saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Helper training functions
# ---------------------------------------------------------------------------

def _train_module(module, head, loader, cfg, device, name):
    params = list(module.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)
    module.train(); head.train()
    for epoch in range(1, cfg.n_epochs + 1):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.cross_entropy(head(module(x)), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sched.step()
    module.eval(); head.eval()


def _eval_module(module, head, loader, device):
    module.eval(); head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = head(module(x)).argmax(-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def _train_child(child, head, parent_A, parent_B, train_loader, cfg, device, orth_weight=0.0):
    params = list(child.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)
    child.train(); head.train()
    history = {"loss": [], "accuracy": []}

    for epoch in range(1, cfg.n_epochs + 1):
        epoch_loss, epoch_acc, n = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                p_a = parent_A(x)
                p_b = parent_B(x)
            child_out = child(x, parent_outputs=[p_a, p_b])
            logits = head(child_out)
            loss = nn.functional.cross_entropy(logits, y)
            if orth_weight > 0:
                loss = loss + orth_weight * child.orth_loss()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            epoch_loss += loss.item()
            epoch_acc += accuracy(logits, y)
            n += 1
        sched.step()
        history["loss"].append(epoch_loss / max(n, 1))
        history["accuracy"].append(epoch_acc / max(n, 1))
        if epoch % 10 == 0 or epoch == 1:
            print(f"      epoch {epoch:3d}/{cfg.n_epochs}  loss={history['loss'][-1]:.4f}  acc={history['accuracy'][-1]:.3f}")

    child.eval(); head.eval()
    return history


def _eval_child(child, head, parent_A, parent_B, loader, device):
    child.eval(); head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p_a, p_b = parent_A(x), parent_B(x)
            preds = head(child(x, parent_outputs=[p_a, p_b])).argmax(-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def _collect_subspace(module, loader, device):
    module.eval()
    module.clear_activation_buffer()
    module._collecting_subspace = True
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            module(x)
    module._collecting_subspace = False
    module.compute_concept_subspace(top_k=8)


def _measure_representation(child, parent_A, parent_B, loader, device):
    child.eval()
    acts = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            p_a, p_b = parent_A(x), parent_B(x)
            z = child(x, parent_outputs=[p_a, p_b])
            acts.append(z.cpu())
    A = torch.cat(acts, dim=0)
    orth = representation_orthogonality(A, top_k=min(8, A.size(1)))
    var  = variance_explained(A, top_k=min(8, A.size(1)))
    return orth, var


def _print_summary_table(results):
    print("\n" + "="*70)
    print("SUMMARY — Aggregation comparison")
    print(f"{'Aggregator':<15} {'Test Acc':>10} {'Repr Orth':>12} {'Var Exp':>10} {'Max GradN':>12}")
    print("-"*70)
    for agg_name, r in results["child_results"].items():
        max_grad = r.get("gradient_stats", {}).get("max_grad_norm", 0.0)
        print(
            f"{agg_name:<15} {r['test_accuracy']:>10.4f} "
            f"{r['representation_orthogonality']:>12.4f} "
            f"{r['variance_explained']:>10.4f} "
            f"{max_grad:>12.2f}"
        )
    print("="*70)


def _json_safe(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 1: Crystallization comparison")
    parser.add_argument("--sub",     type=str, default="1a", choices=["1a", "1b", "all"])
    parser.add_argument("--device",  type=str, default="cpu")
    parser.add_argument("--epochs",  type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="results/exp1")
    args = parser.parse_args()

    cfg = Exp1Config(device=args.device, n_epochs=args.epochs, results_dir=args.out_dir)

    if args.sub in ("1a", "all"):
        run_exp1a(cfg)
    if args.sub in ("1b", "all"):
        run_exp1b(cfg)
