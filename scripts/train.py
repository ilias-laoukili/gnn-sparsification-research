#!/usr/bin/env python3
"""Command-line interface for GNN sparsification ablation studies.

Supports two modes:
1. Single run: Train one model configuration
2. Ablation study: Run all 4 scenarios (Full/Sparse × Binary/Weighted)

Example usage:
    # Run single training
    python scripts/train.py --dataset cora --model gcn --metric jaccard --retention 0.6

    # Run full 4-scenario ablation study
    python scripts/train.py --dataset cora --model gcn --metric jaccard --retention 0.6 --ablation

    # Multiple retention ratios with ablation
    python scripts/train.py --dataset cora --model gcn --metric jaccard --retention 0.4 0.6 0.8 --ablation

    # Force CPU and set custom seed
    python scripts/train.py --dataset cora --model gcn --metric jaccard --device cpu --seed 123
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DatasetLoader
from src.experiments.ablation import AblationStudy
from src.models.gnn import get_model
from src.sparsification.core import GraphSparsifier
from src.training.trainer import GNNTrainer
from src.utils.seeds import set_global_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="GNN Sparsification Training & Ablation Study CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cora", "citeseer", "pubmed", "flickr"],
        help="Dataset to use for training",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gcn", "sage", "gat"],
        help="GNN model architecture",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="jaccard",
        choices=["jaccard", "adamic_adar", "random"],
        help="Similarity metric for sparsification",
    )

    parser.add_argument(
        "--retention",
        type=float,
        nargs="+",
        default=[1.0],
        help="Edge retention ratio(s) (0.0 to 1.0). Multiple values run separate experiments.",
    )

    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run full 4-scenario ablation study instead of single training",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum training epochs (default: 200)",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)",
    )

    parser.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="Hidden layer dimension (default: 128)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Weight decay (default: 5e-4)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu/cuda/mps). Auto-detects if not specified.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress during training",
    )

    return parser.parse_args()


def get_device(requested: Optional[str] = None) -> torch.device:
    """Determine the best available device.

    Args:
        requested: Specific device requested by user, or None for auto-detect.

    Returns:
        PyTorch device object.
    """
    if requested:
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_single_training(
    args: argparse.Namespace,
    data,
    num_features: int,
    num_classes: int,
    device: torch.device,
    retention_ratio: float,
) -> None:
    """Run single model training.

    Args:
        args: Parsed command-line arguments.
        data: PyTorch Geometric data object.
        num_features: Number of input features.
        num_classes: Number of output classes.
        device: PyTorch device.
        retention_ratio: Edge retention ratio.
    """
    if retention_ratio < 1.0:
        print(f"\n[2/4] Sparsifying graph ({args.metric}, {retention_ratio:.0%} retention)...")
        sparsifier = GraphSparsifier(data, str(device))
        data = sparsifier.sparsify(args.metric, retention_ratio)
        print(f"      Sparse edges: {data.edge_index.size(1):,}")
    else:
        print("\n[2/4] Using full graph (retention=1.0)")

    print(f"\n[3/4] Initializing {args.model.upper()} model...")
    model = get_model(
        args.model,
        in_channels=num_features,
        hidden_channels=args.hidden,
        out_channels=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = GNNTrainer(model, optimizer, device=str(device))

    print(f"\n[4/4] Training (max {args.epochs} epochs, patience {args.patience})...")
    start_time = time.time()
    test_acc, history = trainer.train_and_evaluate(
        data, epochs=args.epochs, patience=args.patience
    )
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Test Accuracy:    {test_acc:.4f}")
    print(f"Best Val Acc:     {history['best_val_acc']:.4f}")
    print(f"Epochs Trained:   {history['epochs_trained']}")
    print(f"Training Time:    {elapsed:.2f}s")
    print("=" * 60)


def run_ablation_study(
    args: argparse.Namespace,
    data,
    num_features: int,
    num_classes: int,
    device: torch.device,
    retention_ratio: float,
) -> None:
    """Run 4-scenario ablation study.

    Args:
        args: Parsed command-line arguments.
        data: PyTorch Geometric data object.
        num_features: Number of input features.
        num_classes: Number of output classes.
        device: PyTorch device.
        retention_ratio: Edge retention ratio.
    """
    print(f"\n{'='*70}")
    print(f"Running ablation study with {retention_ratio:.0%} edge retention")
    print("=" * 70)

    study = AblationStudy(
        data=data,
        num_features=num_features,
        num_classes=num_classes,
        device=str(device),
    )

    results_df = study.run_full_study(
        model_name=args.model,
        metric=args.metric,
        retention_ratio=retention_ratio,
        hidden_channels=args.hidden,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )

    print("\n" + "=" * 70)
    print(f"ABLATION STUDY RESULTS (Retention: {retention_ratio:.0%})")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)


def main() -> None:
    """Main entry point for CLI."""
    args = parse_args()

    set_global_seed(args.seed)
    device = get_device(args.device)

    print("=" * 60)
    print("GNN SPARSIFICATION EXPERIMENT")
    print("=" * 60)
    print(f"Dataset:    {args.dataset}")
    print(f"Model:      {args.model}")
    print(f"Metric:     {args.metric}")
    print(f"Retention:  {args.retention}")
    print(f"Mode:       {'Ablation Study' if args.ablation else 'Single Training'}")
    print(f"Device:     {device}")
    print(f"Seed:       {args.seed}")
    print("=" * 60)

    print("\n[1/4] Loading dataset...")
    loader = DatasetLoader(root="./data")
    data, num_features, num_classes = loader.get_dataset(args.dataset, str(device))
    print(f"      Nodes: {data.num_nodes:,}")
    print(f"      Edges: {data.edge_index.size(1):,}")
    print(f"      Features: {num_features}, Classes: {num_classes}")

    for retention_ratio in args.retention:
        if args.ablation:
            run_ablation_study(args, data.clone(), num_features, num_classes, device, retention_ratio)
        else:
            run_single_training(args, data.clone(), num_features, num_classes, device, retention_ratio)

    print("\n✓ Experiment complete")


if __name__ == "__main__":
    main()
