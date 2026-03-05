#!/usr/bin/env python3
"""Sequential experiment runner for graph sparsification ablation study.

Six experiment types, each answering a distinct question:

  1a. threshold_sweep  — Deterministic top-k sparsification by metric score.
  1b. random_baseline  — Random edge removal (control).
  1c. inverse_threshold — Keep WORST edges (proves metrics capture real structure).
  2.  backbone          — APSP-based metric backbone (structure-determined retention).
  3.  sampled           — Probabilistic sampling proportional to score (soft boundaries).
  4.  degree_aware      — Threshold with per-node minimum edge guarantee (prevents isolation).
  5.  feature_metric    — Threshold using feature cosine similarity (feature-aware, not topology).

Usage:
    python scripts/run_ablation.py --all
    python scripts/run_ablation.py --dataset cora
    python scripts/run_ablation.py --list
"""

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch

from src.data import SAFE_DATASETS, DatasetLoader
from src.experiments.ablation import AblationStudy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METRICS = ["jaccard", "approx_er", "adamic_adar"]
RANDOM_METRIC = ["random"]
MODELS = ["gcn", "sage", "gat"]
RETENTION_RATIOS = [i / 10 for i in range(1, 11)]  # 0.1, 0.2, ..., 1.0
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

DATASETS_NO_FEATURES = ["polblogs"]
DATASETS_TOO_LARGE = ["flickr", "physics", "cs", "corafull", "ppi"]

RESULTS_DIR = Path(__file__).parent.parent / "results"

COMMON_TRAIN_KWARGS = dict(hidden_channels=64, epochs=200, patience=20)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_single_dataset_experiments(dataset_name: str, device: str, results_dir: Path):
    """Run all experiments for a single dataset."""
    print(f"\n{'#' * 70}")
    print(f"# DATASET: {dataset_name.upper()}  |  Device: {device}")
    print(f"{'#' * 70}")

    loader = DatasetLoader(root=str(Path(__file__).parent.parent / "data"))
    data, num_features, num_classes = loader.get_dataset(dataset_name, device)
    print(f"Loaded: {data.num_nodes:,} nodes, {data.edge_index.shape[1]:,} edges")
    print(f"Features: {num_features}, Classes: {num_classes}")

    study = AblationStudy(
        data=data,
        num_features=num_features,
        num_classes=num_classes,
        device=device,
    )

    all_results = []

    # ------------------------------------------------------------------
    # 1) Threshold sweep — all models, varying retention ratios
    #    Question: how does accuracy degrade as we remove more edges?
    #    Includes metric-based methods + random baseline.
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"EXP 1/5: Threshold sweep ({len(MODELS)} models × {len(RETENTION_RATIOS)} retentions)")
    print(f"  Metrics: {METRICS} + random + inverse")
    print(f"{'=' * 60}")

    # 1a) Metric-based threshold (Jaccard, AA, Approx ER)
    print("\n  Metric-based sparsification...")
    df = study.run_multi_config_study(
        model_names=MODELS,
        metrics=METRICS,
        retention_ratios=RETENTION_RATIOS,
        seeds=SEEDS,
        use_metric_backbone=False,
        **COMMON_TRAIN_KWARGS,
    )
    df["SparsificationType"] = "Threshold"
    df["Dataset"] = dataset_name
    df["ExperimentType"] = "threshold_sweep"
    all_results.append(df)

    # 1b) Random baseline (A+B only — random weights are meaningless for C/D)
    print("\n  Random baseline...")
    df = study.run_multi_config_study(
        model_names=MODELS,
        metrics=RANDOM_METRIC,
        retention_ratios=RETENTION_RATIOS,
        seeds=SEEDS,
        use_metric_backbone=False,
        skip_weighted=True,
        **COMMON_TRAIN_KWARGS,
    )
    df["SparsificationType"] = "Random"
    df["Dataset"] = dataset_name
    df["ExperimentType"] = "threshold_sweep"
    all_results.append(df)

    # 1c) Inverse threshold — keep LEAST important edges (control experiment)
    #     A+B only: proves metrics capture real structure (should perform worse than random)
    print("\n  Inverse threshold (control)...")
    df = study.run_multi_config_study(
        model_names=MODELS,
        metrics=METRICS,
        retention_ratios=RETENTION_RATIOS,
        seeds=SEEDS,
        use_metric_backbone=False,
        skip_weighted=True,
        keep_lowest=True,
        **COMMON_TRAIN_KWARGS,
    )
    df["SparsificationType"] = "InverseThreshold"
    df["Dataset"] = dataset_name
    df["ExperimentType"] = "threshold_sweep"
    all_results.append(df)

    # ------------------------------------------------------------------
    # 2) Metric backbone — all models, retention determined by graph
    #    Question: how much does the backbone prune, and at what cost?
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"EXP 2/5: Metric backbone ({len(MODELS)} models × {len(METRICS)} metrics)")
    print(f"{'=' * 60}")

    df = study.run_multi_config_study(
        model_names=MODELS,
        metrics=METRICS,
        retention_ratios=[1.0],  # ignored by backbone
        seeds=SEEDS,
        use_metric_backbone=True,
        **COMMON_TRAIN_KWARGS,
    )
    df["SparsificationType"] = "MetricBackbone"
    df["Dataset"] = dataset_name
    df["ExperimentType"] = "backbone"
    all_results.append(df)

    # ------------------------------------------------------------------
    # 3) Sampled sparsification — probabilistic edge sampling
    #    Question: does soft (probabilistic) selection outperform hard top-k?
    #
    #    MOTIVATION: Deterministic threshold always removes the same edges,
    #    creating a hard cutoff. Bridge edges between communities often have
    #    low structural similarity (few shared neighbors) but are critical
    #    for information flow across the graph. Probabilistic sampling gives
    #    every edge a chance proportional to its score, preserving more
    #    structural diversity and reducing sensitivity to score noise.
    #
    #    If sampled > threshold: metric scores are noisy and soft selection
    #    avoids over-fitting to imprecise rankings.
    #    If sampled ≈ threshold: the hard cutoff is fine; scores are reliable.
    #    If sampled < threshold: randomness hurts; the metric ranking is accurate.
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"EXP 3/5: Sampled sparsification ({len(MODELS)} models × {len(RETENTION_RATIOS)} retentions)")
    print(f"  Metrics: {METRICS}")
    print(f"{'=' * 60}")

    df = study.run_multi_config_study(
        model_names=MODELS,
        metrics=METRICS,
        retention_ratios=RETENTION_RATIOS,
        seeds=SEEDS,
        use_metric_backbone=False,
        skip_weighted=True,  # A+B only: weighting is orthogonal to sampling strategy
        sparsification_mode="sampled",
        **COMMON_TRAIN_KWARGS,
    )
    df["SparsificationType"] = "Sampled"
    df["Dataset"] = dataset_name
    df["ExperimentType"] = "sampled_sweep"
    all_results.append(df)

    # ------------------------------------------------------------------
    # 4) Degree-aware sparsification — per-node minimum edge guarantee
    #    Question: does preventing node isolation improve accuracy?
    #
    #    MOTIVATION: Standard threshold treats all edges equally in a global
    #    ranking. Low-degree nodes (common in sparse citation graphs) can
    #    lose ALL their edges at aggressive retention ratios, becoming
    #    isolated. An isolated node can only be classified by its features,
    #    losing all benefit of the GNN's message passing.
    #
    #    Degree-aware sparsification guarantees at least 1 edge per node
    #    before filling the remaining budget from the global ranking. This
    #    preserves graph connectivity while still removing redundant edges
    #    from high-degree hubs.
    #
    #    If degree_aware > threshold: node isolation is a real problem and
    #    the guarantee prevents accuracy loss from disconnected nodes.
    #    If degree_aware ≈ threshold: most nodes already keep edges under
    #    global ranking (graph has high minimum degree).
    #    If degree_aware < threshold: the guarantee forces keeping bad edges
    #    from low-degree nodes, displacing globally better ones.
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"EXP 4/5: Degree-aware sparsification ({len(MODELS)} models × {len(RETENTION_RATIOS)} retentions)")
    print(f"  Metrics: {METRICS}")
    print(f"{'=' * 60}")

    df = study.run_multi_config_study(
        model_names=MODELS,
        metrics=METRICS,
        retention_ratios=RETENTION_RATIOS,
        seeds=SEEDS,
        use_metric_backbone=False,
        skip_weighted=True,  # A+B only: focus on structural effect
        sparsification_mode="degree_aware",
        **COMMON_TRAIN_KWARGS,
    )
    df["SparsificationType"] = "DegreeAware"
    df["Dataset"] = dataset_name
    df["ExperimentType"] = "degree_aware_sweep"
    all_results.append(df)

    # ------------------------------------------------------------------
    # 5) Feature cosine metric — threshold using feature similarity
    #    Question: does a feature-aware metric outperform topology-only metrics?
    #
    #    MOTIVATION: Jaccard and Adamic-Adar measure structural neighborhood
    #    overlap but ignore the actual NODE FEATURES that GNNs aggregate.
    #    Feature cosine similarity directly measures how aligned two nodes'
    #    feature vectors are. On homophilous graphs, high-cosine edges
    #    connect same-class nodes and carry "useful" signal for classification.
    #
    #    This is the simplest possible feature-aware metric. If it outperforms
    #    structural metrics, it suggests that feature alignment is more
    #    important than topology for determining edge importance in GNNs.
    #
    #    If feature_cosine > structural: features are a better guide than
    #    topology for sparsification — makes sense because GNNs aggregate
    #    features, not just structure.
    #    If feature_cosine ≈ structural: both signals are roughly equivalent.
    #    If feature_cosine < structural: topology captures something about
    #    edge importance that raw feature similarity misses (e.g., bridges).
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"EXP 5/5: Feature cosine metric ({len(MODELS)} models × {len(RETENTION_RATIOS)} retentions)")
    print(f"{'=' * 60}")

    df = study.run_multi_config_study(
        model_names=MODELS,
        metrics=["feature_cosine"],
        retention_ratios=RETENTION_RATIOS,
        seeds=SEEDS,
        use_metric_backbone=False,
        skip_weighted=True,  # A+B only
        **COMMON_TRAIN_KWARGS,
    )
    df["SparsificationType"] = "Threshold"
    df["Dataset"] = dataset_name
    df["ExperimentType"] = "feature_metric_sweep"
    all_results.append(df)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    results_df = pd.concat(all_results, ignore_index=True)
    output_file = results_dir / f"{dataset_name}_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(results_df)} results to {output_file}")

    del data, study
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run sparsification experiments")
    parser.add_argument("--dataset", type=str, help="Dataset name to run")
    parser.add_argument("--all", action="store_true", help="Run all datasets sequentially")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    valid_datasets = [
        d for d in SAFE_DATASETS if d not in DATASETS_NO_FEATURES and d not in DATASETS_TOO_LARGE
    ]

    if args.list:
        print("Available datasets (safe for 24GB RAM):")
        for d in valid_datasets:
            print(f"  - {d}")
        skipped = [d for d in SAFE_DATASETS if d in DATASETS_TOO_LARGE]
        if skipped:
            print(f"\nSkipped (too large): {skipped}")
        return

    if not args.dataset and not args.all:
        parser.print_help()
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    start_time = time.time()

    if args.all:
        datasets_to_run = valid_datasets
    else:
        if args.dataset.lower() not in [d.lower() for d in valid_datasets]:
            print(f"Error: Unknown dataset '{args.dataset}'")
            print(f"Valid datasets: {valid_datasets}")
            return
        datasets_to_run = [args.dataset.lower()]

    for dataset_name in datasets_to_run:
        try:
            run_single_dataset_experiments(dataset_name, device, RESULTS_DIR)
        except Exception as e:
            print(f"ERROR running {dataset_name}: {e}")
            import traceback

            traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"COMPLETED in {elapsed/60:.1f} minutes")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
