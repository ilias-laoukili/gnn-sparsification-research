"""Controlled ablation study framework for sparsification experiments.

This module implements the four-scenario ablation methodology to disentangle
the effects of graph structure (sparsification) from edge information (weighting).
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
import torch
from torch import Tensor
from torch_geometric.data import Data

from ..models.gnn import EDGE_WEIGHT_MODELS, BaseGNN, get_model
from ..sparsification.core import GraphSparsifier
from ..sparsification.metrics import compute_topology_metrics
from ..training.trainer import GNNTrainer
from ..utils.seeds import set_global_seed


def _get_process_memory_mb() -> float:
    """Get current process RSS memory in MB (works on all platforms)."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


class ExperimentScenario(Enum):
    """Defines the four controlled experimental scenarios.

    Each scenario represents a unique combination of graph topology
    (full vs. sparse) and edge weighting (binary vs. weighted).

    Attributes:
        A_FULL_BINARY: Full graph with unweighted edges (baseline).
        B_SPARSE_BINARY: Sparsified graph with unweighted edges.
        C_FULL_WEIGHTED: Full graph with Jaccard-weighted edges.
        D_SPARSE_WEIGHTED: Sparsified graph with Jaccard-weighted edges.
    """

    A_FULL_BINARY = "A: Full + Binary"
    B_SPARSE_BINARY = "B: Sparse + Binary"
    C_FULL_WEIGHTED = "C: Full + Weighted"
    D_SPARSE_WEIGHTED = "D: Sparse + Weighted"


@dataclass
class ExperimentResult:
    """Container for a single experiment run's results."""

    scenario: ExperimentScenario
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    epochs_trained: int
    best_val_acc: float
    num_edges: int
    preprocessing_time_sec: float
    sparsification_time_sec: float
    weight_computation_time_sec: float
    training_time_sec: float
    total_time_sec: float
    peak_memory_mb: float
    actual_retention: float = 1.0
    clustering_coeff: float = 0.0
    algebraic_connectivity: float = 0.0


class AblationStudy:
    """Orchestrates the four-scenario ablation study.

    Runs controlled experiments comparing the effects of sparsification
    (structural pruning) versus weighting (edge information) on GNN
    performance.

    Args:
        data: PyG Data object with the full graph.
        num_features: Input feature dimensionality.
        num_classes: Number of target classes.
        device: Computation device string.
        use_metric_backbone: If True, use global metric backbone for sparsification.

    Attributes:
        data: Original full graph data.
        sparsifier: GraphSparsifier instance for edge pruning.
        results: List of ExperimentResult objects from runs.

    Example:
        >>> study = AblationStudy(data, 1433, 7, "cuda")
        >>> results_df = study.run_full_study(
        ...     model_name="gcn",
        ...     metric="jaccard",
        ...     retention_ratio=0.6
        ... )
    """

    def __init__(
        self,
        data: Data,
        num_features: int,
        num_classes: int,
        device: str,
        use_metric_backbone: bool = False,
    ) -> None:
        self.data = data
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.sparsifier = GraphSparsifier(data, device)
        self.use_metric_backbone = use_metric_backbone
        self.results: List[ExperimentResult] = []
        self.verbose: bool = False

    def compute_edge_weights(
        self,
        data: Data,
        metric: str = "jaccard",
    ) -> Tensor:
        """Compute normalized edge weights using the specified metric.

        Args:
            data: PyG Data object to compute weights for.
            metric: Edge scoring metric ('jaccard' or 'adamic_adar').

        Returns:
            Tensor of normalized edge weights, shape (num_edges,).
        """
        temp_sparsifier = GraphSparsifier(data, self.device)
        scores = temp_sparsifier.compute_scores(metric)

        scores_min = scores.min()
        scores_max = scores.max()
        if scores_max > scores_min:
            normalized = (scores - scores_min) / (scores_max - scores_min)
        else:
            normalized = np.ones_like(scores)

        normalized = np.clip(normalized, 0.1, 1.0)

        return torch.tensor(normalized, dtype=torch.float32, device=self.device)

    def _run_single_experiment(
        self,
        scenario: ExperimentScenario,
        exp_data: Data,
        edge_weight: Optional[Tensor],
        model_name: str,
        hidden_channels: int,
        epochs: int,
        patience: int,
        seed: int,
        sparsification_time_sec: float = 0.0,
        weight_computation_time_sec: float = 0.0,
        actual_retention: float = 1.0,
        clustering_coeff: float = 0.0,
        algebraic_connectivity: float = 0.0,
        **model_kwargs,
    ) -> ExperimentResult:
        """Execute a single experimental scenario."""
        set_global_seed(seed)

        model = get_model(
            model_name,
            self.num_features,
            hidden_channels,
            self.num_classes,
            **model_kwargs,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        trainer = GNNTrainer(model, optimizer, device=self.device)

        # Track memory: use psutil (works on all platforms)
        mem_before = _get_process_memory_mb()

        # Measure training wall time
        train_start = time.time()
        accuracy, history = trainer.train_and_evaluate(
            exp_data,
            epochs=epochs,
            patience=patience,
            edge_weight=edge_weight,
        )
        train_end = time.time()
        training_time_sec = train_end - train_start

        mem_after = _get_process_memory_mb()
        peak_memory_mb = max(mem_after - mem_before, 0.0)

        preprocessing_time_sec = sparsification_time_sec + weight_computation_time_sec
        total_time_sec = preprocessing_time_sec + training_time_sec

        test_metrics = history.get("test_metrics", {})

        return ExperimentResult(
            scenario=scenario,
            accuracy=accuracy,
            macro_f1=test_metrics.get("macro_f1", 0.0),
            macro_precision=test_metrics.get("macro_precision", 0.0),
            macro_recall=test_metrics.get("macro_recall", 0.0),
            epochs_trained=history["epochs_trained"],
            best_val_acc=history["best_val_acc"],
            num_edges=exp_data.edge_index.size(1),
            preprocessing_time_sec=preprocessing_time_sec,
            sparsification_time_sec=sparsification_time_sec,
            weight_computation_time_sec=weight_computation_time_sec,
            training_time_sec=training_time_sec,
            total_time_sec=total_time_sec,
            peak_memory_mb=peak_memory_mb,
            actual_retention=actual_retention,
            clustering_coeff=clustering_coeff,
            algebraic_connectivity=algebraic_connectivity,
        )

    def _do_sparsify(
        self,
        metric: str,
        retention_ratio: float,
        *,
        use_metric_backbone: bool = False,
        keep_lowest: bool = False,
        sparsification_mode: str = "threshold",
        seed: int = 42,
    ) -> Tuple[Data, float]:
        """Dispatch to the appropriate sparsification method.

        Centralizes the sparsification call so that run_full_study and
        run_multi_config_study share the same routing logic.

        Args:
            metric: Edge scoring metric.
            retention_ratio: Target retention (ignored for backbone).
            use_metric_backbone: Use APSP-based metric backbone.
            keep_lowest: Keep lowest-scoring edges (inverse control).
            sparsification_mode: One of "threshold", "sampled", "degree_aware".
            seed: Random seed (used by sampled mode).

        Returns:
            (sparse_data, actual_retention_ratio)
        """
        original_num_edges = self.data.edge_index.size(1)

        if use_metric_backbone:
            sparse_data, backbone_stats = self.sparsifier.sparsify_metric_backbone(
                metric=metric,
            )
            actual_retention = backbone_stats["retention_ratio"]
        elif sparsification_mode == "sampled":
            sparse_data = self.sparsifier.sparsify_sampled(
                metric, retention_ratio, seed=seed,
            )
            actual_retention = sparse_data.edge_index.size(1) / original_num_edges
        elif sparsification_mode == "degree_aware":
            sparse_data = self.sparsifier.sparsify_degree_aware(
                metric, retention_ratio, min_edges_per_node=1,
            )
            actual_retention = sparse_data.edge_index.size(1) / original_num_edges
        else:
            # Default: deterministic threshold (top-k / bottom-k)
            sparse_data = self.sparsifier.sparsify(
                metric, retention_ratio, keep_lowest=keep_lowest,
            )
            actual_retention = sparse_data.edge_index.size(1) / original_num_edges

        return sparse_data, actual_retention

    def run_full_study(
        self,
        model_name: str = "gcn",
        metric: str = "jaccard",
        retention_ratio: float = 0.6,
        hidden_channels: int = 64,
        epochs: int = 200,
        patience: int = 20,
        seed: int = 42,
        use_metric_backbone: Optional[bool] = None,
        skip_weighted: bool = False,
        keep_lowest: bool = False,
        sparsification_mode: str = "threshold",
        **model_kwargs,
    ) -> pd.DataFrame:
        """Execute scenarios of the ablation study.

        Args:
            model_name: GNN architecture ('gcn', 'sage', 'gat').
            metric: Edge metric for sparsification and weighting.
            retention_ratio: Fraction of edges to keep in sparse scenarios.
                            Ignored if use_metric_backbone=True (backbone determines retention).
            hidden_channels: Hidden layer size for the model.
            epochs: Maximum training epochs per scenario.
            patience: Early stopping patience.
            seed: Random seed for reproducibility.
            use_metric_backbone: Override instance setting for metric backbone usage.
            skip_weighted: If True, skip scenarios C and D (weighted).
                          Use for Random baseline where weights are meaningless.
            keep_lowest: If True, keep lowest-scoring edges (control experiment).
            **model_kwargs: Additional model arguments (e.g., heads for GAT).

        Returns:
            DataFrame with columns: Scenario, Accuracy, Epochs, BestValAcc, Edges.
        """
        self.results = []

        # Auto-skip weighted scenarios for models that don't support edge_weight
        if model_name.lower() not in EDGE_WEIGHT_MODELS:
            skip_weighted = True

        backbone_flag = (
            self.use_metric_backbone if use_metric_backbone is None else use_metric_backbone
        )

        # Measure sparsification time separately
        sparsify_start = time.time()
        sparse_data, actual_retention = self._do_sparsify(
            metric=metric,
            retention_ratio=retention_ratio,
            use_metric_backbone=backbone_flag,
            keep_lowest=keep_lowest,
            sparsification_mode=sparsification_mode,
            seed=seed,
        )
        if self.verbose and backbone_flag:
            print(f"Metric backbone retention: {actual_retention:.1%}")
        sparsification_time_sec = time.time() - sparsify_start

        # Compute edge weights only when needed (scenarios C/D)
        weight_start = time.time()
        if not skip_weighted:
            full_weights = self.compute_edge_weights(self.data, metric)
            sparse_weights = self.compute_edge_weights(sparse_data, metric)
        else:
            full_weights = None
            sparse_weights = None
        weight_end = time.time()
        weight_computation_time_sec = weight_end - weight_start

        # Compute topology metrics for full and sparse graphs (once per study)
        full_topo = compute_topology_metrics(self.sparsifier.adj)
        sparse_ei = sparse_data.edge_index.cpu().numpy()
        sparse_adj = sp.csr_matrix(
            (np.ones(sparse_ei.shape[1]), (sparse_ei[0], sparse_ei[1])),
            shape=(self.data.num_nodes, self.data.num_nodes),
        )
        sparse_topo = compute_topology_metrics(sparse_adj)

        scenarios_config = [
            (ExperimentScenario.A_FULL_BINARY, self.data, None, 1.0, full_topo),
            (ExperimentScenario.B_SPARSE_BINARY, sparse_data, None, actual_retention, sparse_topo),
        ]
        if not skip_weighted:
            scenarios_config.extend(
                [
                    (ExperimentScenario.C_FULL_WEIGHTED, self.data, full_weights, 1.0, full_topo),
                    (
                        ExperimentScenario.D_SPARSE_WEIGHTED,
                        sparse_data,
                        sparse_weights,
                        actual_retention,
                        sparse_topo,
                    ),
                ]
            )

        for scenario, exp_data, edge_weight, scenario_retention, topo in scenarios_config:
            if self.verbose:
                print(f"Running {scenario.value}...")
            result = self._run_single_experiment(
                scenario=scenario,
                exp_data=exp_data,
                edge_weight=edge_weight,
                model_name=model_name,
                hidden_channels=hidden_channels,
                epochs=epochs,
                patience=patience,
                seed=seed,
                sparsification_time_sec=sparsification_time_sec,
                weight_computation_time_sec=weight_computation_time_sec,
                actual_retention=scenario_retention,
                clustering_coeff=topo.get("clustering_coefficient", 0.0),
                algebraic_connectivity=topo.get("algebraic_connectivity", 0.0),
                **model_kwargs,
            )
            self.results.append(result)
            if self.verbose:
                print(f"  -> Accuracy: {result.accuracy:.4f}")

        return self.results_to_dataframe()

    def results_to_dataframe(self) -> pd.DataFrame:
        """Convert results list to a formatted DataFrame.

        Returns:
            DataFrame with experiment results for easy comparison.
        """
        records = [
            {
                "Scenario": r.scenario.value,
                "Accuracy": r.accuracy,
                "MacroF1": r.macro_f1,
                "MacroPrecision": r.macro_precision,
                "MacroRecall": r.macro_recall,
                "Epochs": r.epochs_trained,
                "BestValAcc": r.best_val_acc,
                "Edges": r.num_edges,
                "ActualRetention": r.actual_retention,
                "PreprocessSec": r.preprocessing_time_sec,
                "SparsifySec": r.sparsification_time_sec,
                "WeightSec": r.weight_computation_time_sec,
                "TrainSec": r.training_time_sec,
                "TotalTimeSec": r.total_time_sec,
                "PeakMemMB": r.peak_memory_mb,
                "ClusteringCoeff": r.clustering_coeff,
                "AlgebraicConnectivity": r.algebraic_connectivity,
            }
            for r in self.results
        ]
        return pd.DataFrame(records)

    def run_training_curves(
        self,
        model_name: str = "gcn",
        metric: str = "jaccard",
        retention_ratio: float = 0.6,
        hidden_channels: int = 64,
        epochs: int = 100,
        patience: Optional[int] = None,
        seed: int = 42,
        **model_kwargs,
    ) -> Dict[str, List[float]]:
        """Run training for all four scenarios and collect validation accuracy history.

        This method trains models on each ablation scenario and returns the
        validation accuracy progression over epochs, enabling comparison of
        training dynamics across different configurations.

        Args:
            model_name: GNN architecture ('gcn', 'sage', 'gat').
            metric: Edge metric for sparsification and weighting.
            retention_ratio: Fraction of edges to keep in sparse scenarios.
            hidden_channels: Hidden layer size for the model.
            epochs: Maximum training epochs per scenario.
            patience: Early stopping patience. If None, trains for full epochs.
            seed: Random seed for reproducibility.
            **model_kwargs: Additional model arguments (e.g., heads for GAT).

        Returns:
            Dictionary mapping scenario names to lists of validation accuracies.
            Example: {"A: Full + Binary": [0.72, 0.75, ...], ...}

        Example:
            >>> curves = study.run_training_curves(
            ...     model_name="gcn",
            ...     metric="jaccard",
            ...     retention_ratio=0.6,
            ...     epochs=100,
            ...     seed=42
            ... )
            >>> plt.plot(curves["A: Full + Binary"], label="Full + Binary")
        """
        # Generate sparse graph and edge weights once
        sparse_data = self.sparsifier.sparsify(metric, retention_ratio)
        full_weights = self.compute_edge_weights(self.data, metric)
        sparse_weights = self.compute_edge_weights(sparse_data, metric)

        # Define the 4 configurations
        scenarios_config = [
            (ExperimentScenario.A_FULL_BINARY, self.data, None),
            (ExperimentScenario.B_SPARSE_BINARY, sparse_data, None),
            (ExperimentScenario.C_FULL_WEIGHTED, self.data, full_weights),
            (ExperimentScenario.D_SPARSE_WEIGHTED, sparse_data, sparse_weights),
        ]

        training_curves = {}

        for scenario, exp_data, edge_weight in scenarios_config:
            print(f"Training {scenario.value}...")

            # Set seed for reproducibility
            set_global_seed(seed)

            # Initialize fresh model
            model = get_model(
                model_name,
                self.num_features,
                hidden_channels,
                self.num_classes,
                **model_kwargs,
            ).to(self.device)

            # Train model
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            trainer = GNNTrainer(model, optimizer, device=self.device)

            history = trainer.train(
                exp_data,
                epochs=epochs,
                patience=patience,
                edge_weight=edge_weight,
            )

            # Store validation accuracy history
            training_curves[scenario.value] = history["val_acc"]
            print(
                f"  -> Final val acc: {history['val_acc'][-1]:.4f} "
                f"(epochs: {history['epochs_trained']})"
            )

        return training_curves

    def run_multi_config_study(
        self,
        model_names: List[str],
        metrics: List[str],
        retention_ratios: List[float],
        hidden_channels: int = 64,
        epochs: int = 200,
        patience: int = 20,
        seeds: List[int] = [42],
        use_metric_backbone: Optional[bool] = None,
        skip_weighted: bool = False,
        keep_lowest: bool = False,
        sparsification_mode: str = "threshold",
    ) -> pd.DataFrame:
        """Run ablation study across multiple configurations and seeds.

        Precomputes sparsification, weights, and topology per (metric, retention)
        to avoid redundant computation across models and seeds.

        Args:
            model_names: List of model architectures to test.
            metrics: List of edge metrics to evaluate.
            retention_ratios: List of retention ratios to test.
                             Ignored if use_metric_backbone=True.
            hidden_channels: Hidden layer size.
            epochs: Maximum training epochs.
            patience: Early stopping patience.
            seeds: List of random seeds for reproducibility.
            use_metric_backbone: If True, use global metric backbone (ignores retention_ratios).
            skip_weighted: If True, skip scenarios C and D (weighted).
            keep_lowest: If True, keep lowest-scoring edges (control experiment).

        Returns:
            DataFrame with all results including config columns.
        """
        all_results = []
        backbone_flag = (
            self.use_metric_backbone if use_metric_backbone is None else use_metric_backbone
        )
        original_num_edges = self.data.edge_index.size(1)

        # Full-graph topology (constant across all configs)
        full_topo = compute_topology_metrics(self.sparsifier.adj)

        for metric in metrics:
            # Full-graph weights depend only on metric (compute once)
            full_weights_cache = None

            for retention in retention_ratios:
                # --- Precompute per (metric, retention) ---
                sparsify_start = time.time()
                sparse_data, actual_retention = self._do_sparsify(
                    metric=metric,
                    retention_ratio=retention,
                    use_metric_backbone=backbone_flag,
                    keep_lowest=keep_lowest,
                    sparsification_mode=sparsification_mode,
                    seed=seeds[0] if seeds else 42,
                )
                sparsification_time_sec = time.time() - sparsify_start

                # Sparse topology (same for all models/seeds at this retention)
                sparse_ei = sparse_data.edge_index.cpu().numpy()
                sparse_adj = sp.csr_matrix(
                    (np.ones(sparse_ei.shape[1]), (sparse_ei[0], sparse_ei[1])),
                    shape=(self.data.num_nodes, self.data.num_nodes),
                )
                sparse_topo = compute_topology_metrics(sparse_adj)

                for model_name in model_names:
                    # Determine if this model needs weighted scenarios
                    model_skip_weighted = skip_weighted or (
                        model_name.lower() not in EDGE_WEIGHT_MODELS
                    )

                    # Compute weights only when needed, cache full-graph weights
                    weight_start = time.time()
                    if not model_skip_weighted:
                        if full_weights_cache is None:
                            full_weights_cache = self.compute_edge_weights(self.data, metric)
                        full_weights = full_weights_cache
                        sparse_weights = self.compute_edge_weights(sparse_data, metric)
                    else:
                        full_weights = None
                        sparse_weights = None
                    weight_computation_time_sec = time.time() - weight_start

                    # Build scenario list
                    scenarios_config = [
                        (ExperimentScenario.A_FULL_BINARY, self.data, None, 1.0, full_topo),
                        (
                            ExperimentScenario.B_SPARSE_BINARY,
                            sparse_data,
                            None,
                            actual_retention,
                            sparse_topo,
                        ),
                    ]
                    if not model_skip_weighted:
                        scenarios_config.extend(
                            [
                                (
                                    ExperimentScenario.C_FULL_WEIGHTED,
                                    self.data,
                                    full_weights,
                                    1.0,
                                    full_topo,
                                ),
                                (
                                    ExperimentScenario.D_SPARSE_WEIGHTED,
                                    sparse_data,
                                    sparse_weights,
                                    actual_retention,
                                    sparse_topo,
                                ),
                            ]
                        )

                    for seed in seeds:
                        print(f"\n{'='*60}")
                        print(f"Config: {model_name} | {metric} | {retention:.0%} | Seed: {seed}")
                        print(f"{'='*60}")

                        self.results = []
                        for scenario, exp_data, edge_weight, s_ret, topo in scenarios_config:
                            result = self._run_single_experiment(
                                scenario=scenario,
                                exp_data=exp_data,
                                edge_weight=edge_weight,
                                model_name=model_name,
                                hidden_channels=hidden_channels,
                                epochs=epochs,
                                patience=patience,
                                seed=seed,
                                sparsification_time_sec=sparsification_time_sec,
                                weight_computation_time_sec=weight_computation_time_sec,
                                actual_retention=s_ret,
                                clustering_coeff=topo.get("clustering_coefficient", 0.0),
                                algebraic_connectivity=topo.get("algebraic_connectivity", 0.0),
                            )
                            self.results.append(result)

                        df = self.results_to_dataframe()
                        df["Model"] = model_name
                        df["Metric"] = metric
                        df["Retention"] = retention
                        df["Seed"] = seed
                        all_results.append(df)

        return pd.concat(all_results, ignore_index=True)
