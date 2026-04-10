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
                        choices=["1a", "1b", "2a", "2b", "3a", "3b"],
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
        )
        run_exp2a(cfg)

    elif args.exp in ("3a", "3b"):
        from concept_dag.experiments.exp3_growing_dag import Exp3Config, run_exp3a, run_exp3b
        cfg = Exp3Config(
            data_root   = args.data_root,
            device      = device,
            root_epochs = args.epochs,
            child_epochs= args.epochs,
            results_dir = f"{args.out_dir}/exp3",
            seed        = args.seed,
        )
        if args.exp == "3a":
            run_exp3a(cfg)
        else:
            # 3b builds on top of 3a — run 3a first then immediately do 3b
            print("Running 3a first to build the DAG, then running 3b...")
            results_3a, nodes, heads, parent_map, tasks = run_exp3a(cfg)
            run_exp3b(cfg, nodes, heads, parent_map, tasks)

    else:
        print(f"Unknown experiment: {args.exp}")


if __name__ == "__main__":
    main()
