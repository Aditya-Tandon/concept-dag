# Must be set before any imports — fixes OpenMP conflict on macOS with MPS + NumPy
import os
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# SVD and certain linalg ops fall back to CPU on MPS — expected, not an error
warnings.filterwarnings("ignore", message=".*aten::linalg_svd.*")
warnings.filterwarnings("ignore", message=".*aten::linalg_eigh.*")

"""
Top-level experiment runner.

Usage:
    # Phase 1: Test crystallization (no download needed)
    python run_experiment.py --exp 1a
    python run_experiment.py --exp 1b

    # Phase 2: Test routing (needs CIFAR-10)
    python run_experiment.py --exp 2a --data_root ./data

    # Phase 3: Full DAG (needs CIFAR-10 or CIFAR-100)
    python run_experiment.py --exp 3a --data_root ./data

    # Download datasets (explicit permission required — pass --download flag)
    python run_experiment.py --download cifar10 --data_root ./data
    python run_experiment.py --download cifar100 --data_root ./data
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Concept DAG experiment runner")
    parser.add_argument("--exp",      type=str,   default="1a",
                        choices=["1a", "1b", "2a", "2b", "3a", "3b",
                                 "4", "4f", "5a", "5b", "5", "6", "plot"],
                        help="Which experiment to run")
    parser.add_argument("--device",   type=str,   default="auto",
                        help="Device: 'cpu', 'cuda', 'mps', or 'auto'")
    parser.add_argument("--epochs",   type=int,   default=30,
                        help="Number of training epochs per module")
    parser.add_argument("--data_root",type=str,   default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--out_dir",  type=str,   default="./results",
                        help="Output directory for results")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--download", type=str,   default=None,
                        choices=["cifar10", "cifar100"],
                        help="Download a dataset (requires explicit permission)")
    parser.add_argument("--batch_size",type=int,   default=128,
                        help="Batch size for training (default: 128)")
    parser.add_argument("--n_tasks",   type=int,   default=None,
                        help="Override number of tasks (e.g. 10 for fast ablation)")
    # Plot-mode arguments
    parser.add_argument("--exp3a",     type=str,   default=None,
                        help="Path to exp3a_results.json (plot mode)")
    parser.add_argument("--exp3b",     type=str,   default=None,
                        help="Path to exp3b_results.json (plot mode)")
    parser.add_argument("--exp4",      type=str,   default=None,
                        help="Path to exp4_all_results.json (plot mode)")
    parser.add_argument("--exp4f",     type=str,   default=None,
                        help="Path to forced_hub results JSON (plot mode)")
    parser.add_argument("--exp5a",     type=str,   default=None,
                        help="Path to exp5a_results.json (plot mode)")
    parser.add_argument("--exp5b",     type=str,   default=None,
                        help="Path to exp5b_results.json (plot mode)")
    parser.add_argument("--auto_discover", action="store_true",
                        help="Auto-discover result JSONs under --out_dir")
    # Exp 5 sweep overrides
    parser.add_argument("--n_parents_sweep", type=int, nargs="+", default=None,
                        help="n_parents values to sweep in exp5a (e.g. 1 2 3 4 5)")
    parser.add_argument("--subspace_k_sweep", type=int, nargs="+", default=None,
                        help="subspace_k values to sweep in exp5b (e.g. 2 4 8 16 32)")
    # Exp 4 variant filter
    parser.add_argument("--variants", type=str, nargs="+", default=None,
                        help="Subset of exp4 variants to run (e.g. --variants no_freeze sequential). "
                             "If omitted, all 5 variants run.")
    # Exp 4f aggregation override (force-include-task-0 across aggregators)
    parser.add_argument("--aggregation", type=str, default=None,
                        choices=["concat", "mean", "attention", "svd", "soft_pca", "cross_attention"],
                        help="Override aggregation for exp 4f (forced-hub). Default soft_pca. "
                             "Use 'cross_attention' to test the task-0 backbone confound.")
    # SSL backbone arguments (DINO swap)
    parser.add_argument("--backbone", type=str, default="smallcnn",
                        choices=["smallcnn", "dinov2_vits14", "clip_vitb16", "resnet50"],
                        help="Feature extractor backbone. 'smallcnn' = original task-trained CNN. "
                             "Other options use a frozen SSL encoder + feature caching.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory for cached SSL features (default: <data_root>/features_<backbone>). "
                             "Only used when --backbone != smallcnn.")
    # Exp 6 confirmation-run arguments
    parser.add_argument("--best_n_parents",      type=int, default=None,
                        help="Explicit best n_parents for exp6 (otherwise loaded from --exp5a)")
    parser.add_argument("--best_subspace_k",     type=int, default=None,
                        help="Explicit best subspace_k for exp6 (otherwise loaded from --exp5b)")
    parser.add_argument("--confirm_concept_dim", type=int, default=256,
                        help="concept_dim to use in the exp6 confirmation run (default 256)")
    parser.add_argument("--baseline_concept_dim",type=int, default=128,
                        help="Baseline concept_dim that k was tuned at (default 128)")
    parser.add_argument("--exp3a_baseline",      type=str, default=None,
                        help="Path to exp3a_results.json baseline for AA comparison in exp6")
    parser.add_argument("--no_perturbation",     action="store_true",
                        help="Skip the exp3b-style perturbation test inside exp6")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # ── Backbone / feature-cache setup ──────────────────────────────────────
    def _prepare_tasks_with_backbone(raw_tasks, backbone: str, cache_dir, data_root):
        """
        If backbone != 'smallcnn', build the frozen encoder, cache features,
        and return feature-tensor tasks + feature_dim.
        If backbone == 'smallcnn', return raw_tasks unchanged + feature_dim=None.
        """
        if backbone == "smallcnn":
            return raw_tasks, None
        from concept_dag.models.root_encoder import build_encoder
        from concept_dag.data.feature_cache import cache_features
        if cache_dir is None:
            cache_dir = os.path.join(data_root, f"features_{backbone}")
        print(f"\n[backbone] Building encoder: {backbone}")
        encoder = build_encoder(backbone, device=device)
        print(f"[backbone] Feature dim: {encoder.feature_dim}  |  Cache: {cache_dir}")
        tasks = cache_features(encoder, raw_tasks, cache_dir=cache_dir, device=device)
        del encoder  # free GPU memory before training starts
        import gc; gc.collect()
        if device in ("cuda", "mps"):
            torch.cuda.empty_cache() if device == "cuda" else None
        return tasks, tasks[0]["feature_dim"]

    # Download mode
    if args.download:
        from concept_dag.data.loaders import download_cifar10, download_cifar100
        if args.download == "cifar10":
            download_cifar10(args.data_root)
        elif args.download == "cifar100":
            download_cifar100(args.data_root)
        return

    # Experiment dispatch
    if args.exp in ("1a", "1b"):
        from concept_dag.experiments.exp1_crystallization import Exp1Config, run_exp1a, run_exp1b
        cfg = Exp1Config(
            device=device,
            n_epochs=args.epochs,
            results_dir=f"{args.out_dir}/exp1",
            seed=args.seed,
            batch_size=args.batch_size,
        )
        if args.exp == "1a":
            run_exp1a(cfg)
        else:
            run_exp1b(cfg)

    elif args.exp in ("2a", "2b"):
        from concept_dag.experiments.exp2_routing import Exp2Config, run_exp2a
        cfg = Exp2Config(
            data_root=args.data_root,
            device=device,
            root_epochs=args.epochs,
            child_epochs=args.epochs,
            results_dir=f"{args.out_dir}/exp2",
            seed=args.seed,
            batch_size=args.batch_size,
        )
        run_exp2a(cfg)

    elif args.exp in ("3a", "3b"):
        from concept_dag.experiments.exp3_growing_dag import Exp3Config, run_exp3a, run_exp3b
        from concept_dag.data.loaders import make_split_cifar100
        n_tasks = args.n_tasks if args.n_tasks is not None else 20
        cfg = Exp3Config(
            data_root   = args.data_root,
            device      = device,
            root_epochs = args.epochs,
            child_epochs= args.epochs,
            results_dir = f"{args.out_dir}/exp3",
            seed        = args.seed,
            batch_size  = args.batch_size,
            n_tasks     = n_tasks,
            backbone    = args.backbone,
            cache_dir   = args.cache_dir,
        )
        raw_tasks = make_split_cifar100(
            data_root=args.data_root, n_tasks=n_tasks,
            batch_size=args.batch_size, seed=args.seed,
        )
        tasks, feature_dim = _prepare_tasks_with_backbone(
            raw_tasks, args.backbone, args.cache_dir, args.data_root)
        if feature_dim is not None:
            cfg.feature_dim = feature_dim
        if args.exp == "3a":
            run_exp3a(cfg, tasks=tasks)
        else:
            print("Running 3a first to build the DAG, then running 3b...")
            results_3a, nodes, heads, parent_map, _ = run_exp3a(cfg, tasks=tasks)
            run_exp3b(cfg, nodes, heads, parent_map, tasks)

    elif args.exp in ("4", "4f"):
        from concept_dag.experiments.exp4_ablations import (
            Exp4Config, run_all_ablations, run_forced_hub_causal,
        )
        from concept_dag.data.loaders import make_split_cifar100

        n_tasks = args.n_tasks if args.n_tasks is not None else 20
        epochs  = args.epochs  # use --epochs to override (default 30 in parser but
                                # Exp4Config defaults to 25 — explicit wins)
        cfg = Exp4Config(
            data_root    = args.data_root,
            device       = device,
            root_epochs  = epochs,
            child_epochs = epochs,
            results_dir  = f"{args.out_dir}/exp4",
            seed         = args.seed,
            batch_size   = args.batch_size,
            n_tasks      = n_tasks,
            backbone     = args.backbone,
            cache_dir    = args.cache_dir,
        )
        # Load raw data once, then optionally swap to cached SSL features
        raw_tasks = make_split_cifar100(
            data_root=args.data_root, n_tasks=n_tasks,
            batch_size=args.batch_size, seed=args.seed,
        )
        tasks, feature_dim = _prepare_tasks_with_backbone(
            raw_tasks, args.backbone, args.cache_dir, args.data_root)
        if feature_dim is not None:
            cfg.feature_dim = feature_dim

        if args.exp == "4":
            if args.variants:
                from concept_dag.experiments.exp4_ablations import (
                    VARIANTS, run_ablation_variant, _save_variant, Exp4Config,
                )
                requested = set(args.variants)
                known     = {v[0] for v in VARIANTS}
                unknown   = requested - known
                if unknown:
                    raise ValueError(
                        f"Unknown variants {unknown}. Valid: {sorted(known)}"
                    )
                print(f"\n[variants filter] running only: {sorted(requested)}")
                os.makedirs(cfg.results_dir, exist_ok=True)
                for (vname, routing, agg, freeze, seq) in VARIANTS:
                    if vname not in requested:
                        continue
                    vcfg = Exp4Config(
                        **{k: v for k, v in vars(cfg).items()
                           if k not in ("routing_mode","aggregation","freeze_parents",
                                        "is_sequential","variant_name")},
                        routing_mode   = routing,
                        aggregation    = agg,
                        freeze_parents = freeze,
                        is_sequential  = seq,
                        variant_name   = vname,
                    )
                    result = run_ablation_variant(vcfg, tasks)
                    _save_variant(result, cfg.results_dir, vname)
            else:
                run_all_ablations(cfg, tasks=tasks)
        else:
            # exp 4f — causal forced-hub ablation
            if args.aggregation is not None:
                cfg.aggregation = args.aggregation
                print(f"[exp 4f] aggregation override: {cfg.aggregation}")
            import json, os
            result = run_forced_hub_causal(cfg, tasks)
            os.makedirs(cfg.results_dir, exist_ok=True)
            agg_tag = f"_{cfg.aggregation}" if args.aggregation is not None else ""
            out_path = os.path.join(cfg.results_dir, f"exp4f_forced_hub{agg_tag}_results.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nForced-hub results saved to {out_path}")

    elif args.exp in ("5a", "5b", "5"):
        from concept_dag.experiments.exp5_sensitivity import Exp5Config, run_exp5a, run_exp5b
        from dataclasses import field as _field

        n_tasks = args.n_tasks if args.n_tasks is not None else 20
        cfg5 = Exp5Config(
            data_root    = args.data_root,
            device       = device,
            root_epochs  = args.epochs,
            child_epochs = args.epochs,
            results_dir  = f"{args.out_dir}/exp5",
            seed         = args.seed,
            batch_size   = args.batch_size,
            n_tasks      = n_tasks,
        )
        if args.n_parents_sweep is not None:
            cfg5.n_parents_sweep = args.n_parents_sweep
        if args.subspace_k_sweep is not None:
            cfg5.subspace_k_sweep = args.subspace_k_sweep

        if args.exp in ("5a", "5"):
            run_exp5a(cfg5)
        if args.exp in ("5b", "5"):
            run_exp5b(cfg5)

    elif args.exp == "6":
        from concept_dag.experiments.exp6_confirmation import run_exp6
        n_tasks = args.n_tasks if args.n_tasks is not None else 20
        run_exp6(
            data_root           = args.data_root,
            device              = device,
            epochs              = args.epochs,
            batch_size          = args.batch_size,
            seed                = args.seed,
            out_dir             = f"{args.out_dir}/exp6",
            best_n_parents      = args.best_n_parents,
            best_subspace_k     = args.best_subspace_k,
            exp5a_path          = args.exp5a,
            exp5b_path          = args.exp5b,
            exp3a_baseline_path = args.exp3a_baseline,
            confirm_concept_dim = args.confirm_concept_dim,
            baseline_concept_dim= args.baseline_concept_dim,
            n_tasks             = n_tasks,
            run_perturbation    = not args.no_perturbation,
        )

    elif args.exp == "plot":
        from concept_dag.experiments.plot_results import run_plots
        run_plots(
            exp3a_path    = args.exp3a,
            exp3b_path    = args.exp3b,
            exp4_path     = args.exp4,
            exp4f_path    = args.exp4f,
            exp5a_path    = args.exp5a,
            exp5b_path    = args.exp5b,
            out_dir       = f"{args.out_dir}/figures",
            auto_discover = args.auto_discover,
            results_root  = args.out_dir,
        )

    else:
        print(f"Unknown experiment: {args.exp}")


if __name__ == "__main__":
    main()
