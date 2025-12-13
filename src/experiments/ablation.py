"""Controlled ablation study framework for sparsification experiments.

This module implements the four-scenario ablation methodology to disentangle
the effects of graph structure (sparsification) from edge information (weighting).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data
import time

from ..models.gnn import BaseGNN, get_model
from ..sparsification.core import GraphSparsifier
from ..training.trainer import GNNTrainer
from ..utils.seeds import set_global_seed


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
    """Container for a single experiment run's results.

    Args:
        scenario: The experimental scenario (A, B, C, or D).
        accuracy: Test set accuracy achieved.
        epochs_trained: Number of epochs before stopping.
        best_val_acc: Best validation accuracy during training.
        num_edges: Number of edges in the graph used.
        preprocessing_time_sec: Time spent preparing graph/weights for this run.
        training_time_sec: Wall-clock training time for this run.
        peak_memory_mb: Peak GPU memory allocated (if CUDA), else 0.
    """

    scenario: ExperimentScenario
    accuracy: float
    epochs_trained: int
    best_val_acc: float
    num_edges: int
    preprocessing_time_sec: float
    training_time_sec: float
    peak_memory_mb: float


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
    ) -> None:
        self.data = data
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.sparsifier = GraphSparsifier(data, device)
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
        **model_kwargs,
    ) -> ExperimentResult:
        """Execute a single experimental scenario."""
        set_global_seed(seed)

        # Measure preprocessing time for this scenario (edge weights already computed externally)
        prep_start = time.time()
        # No heavy preprocessing here beyond passing data/weights, but keep hook
        prep_end = time.time()
        preprocessing_time_sec = prep_end - prep_start

        model = get_model(
            model_name,
            self.num_features,
            hidden_channels,
            self.num_classes,
            **model_kwargs,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.01, weight_decay=5e-4
        )
        trainer = GNNTrainer(model, optimizer, device=self.device)

        # Track peak CUDA memory
        peak_memory_mb = 0.0
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

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

        if torch.cuda.is_available() and self.device.startswith("cuda"):
            peak_bytes = torch.cuda.max_memory_allocated()
            peak_memory_mb = float(peak_bytes) / (1024.0 * 1024.0)

        return ExperimentResult(
            scenario=scenario,
            accuracy=accuracy,
            epochs_trained=history["epochs_trained"],
            best_val_acc=history["best_val_acc"],
            num_edges=exp_data.edge_index.size(1),
            preprocessing_time_sec=preprocessing_time_sec,
            training_time_sec=training_time_sec,
            peak_memory_mb=peak_memory_mb,
        )

    def run_full_study(
        self,
        model_name: str = "gcn",
        metric: str = "jaccard",
        retention_ratio: float = 0.6,
        hidden_channels: int = 64,
        epochs: int = 200,
        patience: int = 20,
        seed: int = 42,
        **model_kwargs,
    ) -> pd.DataFrame:
        """Execute all four scenarios of the ablation study.

        Args:
            model_name: GNN architecture ('gcn', 'sage', 'gat').
            metric: Edge metric for sparsification and weighting.
            retention_ratio: Fraction of edges to keep in sparse scenarios.
            hidden_channels: Hidden layer size for the model.
            epochs: Maximum training epochs per scenario.
            patience: Early stopping patience.
            seed: Random seed for reproducibility.
            **model_kwargs: Additional model arguments (e.g., heads for GAT).

        Returns:
            DataFrame with columns: Scenario, Accuracy, Epochs, BestValAcc, Edges.
        """
        self.results = []

        # Measure sparsification and weight computation times
        prep_start = time.time()
        sparse_data = self.sparsifier.sparsify(metric, retention_ratio)
        full_weights = self.compute_edge_weights(self.data, metric)
        sparse_weights = self.compute_edge_weights(sparse_data, metric)
        prep_end = time.time()
        common_prep_time = prep_end - prep_start

        scenarios_config = [
            (ExperimentScenario.A_FULL_BINARY, self.data, None),
            (ExperimentScenario.B_SPARSE_BINARY, sparse_data, None),
            (ExperimentScenario.C_FULL_WEIGHTED, self.data, full_weights),
            (ExperimentScenario.D_SPARSE_WEIGHTED, sparse_data, sparse_weights),
        ]

        for scenario, exp_data, edge_weight in scenarios_config:
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
                **model_kwargs,
            )
            # Add common preprocessing time (sparsify + compute weights once) to each scenario
            result.preprocessing_time_sec += common_prep_time
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
                "Epochs": r.epochs_trained,
                "BestValAcc": r.best_val_acc,
                "Edges": r.num_edges,
                "PreprocessSec": r.preprocessing_time_sec,
                "TrainSec": r.training_time_sec,
                "PeakMemMB": r.peak_memory_mb,
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
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.01, weight_decay=5e-4
            )
            trainer = GNNTrainer(model, optimizer, device=self.device)
            
            history = trainer.train(
                exp_data,
                epochs=epochs,
                patience=patience,
                edge_weight=edge_weight,
            )

            # Store validation accuracy history
            training_curves[scenario.value] = history["val_acc"]
            print(f"  -> Final val acc: {history['val_acc'][-1]:.4f} "
                  f"(epochs: {history['epochs_trained']})")

        return training_curves

    def run_multi_config_study(
        self,
        model_names: List[str],
        metrics: List[str],
        retention_ratios: List[float],
        hidden_channels: int = 64,
        epochs: int = 200,
        patience: int = 20,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Run ablation study across multiple configurations.

        Args:
            model_names: List of model architectures to test.
            metrics: List of edge metrics to evaluate.
            retention_ratios: List of retention ratios to test.
            hidden_channels: Hidden layer size.
            epochs: Maximum training epochs.
            patience: Early stopping patience.
            seed: Random seed.

        Returns:
            DataFrame with all results including config columns.
        """
        all_results = []

        for model_name in model_names:
            for metric in metrics:
                for retention in retention_ratios:
                    print(f"\n{'='*60}")
                    print(f"Config: {model_name} | {metric} | {retention:.0%}")
                    print(f"{'='*60}")

                    df = self.run_full_study(
                        model_name=model_name,
                        metric=metric,
                        retention_ratio=retention,
                        hidden_channels=hidden_channels,
                        epochs=epochs,
                        patience=patience,
                        seed=seed,
                    )
                    df["Model"] = model_name
                    df["Metric"] = metric
                    df["Retention"] = retention
                    all_results.append(df)

        return pd.concat(all_results, ignore_index=True)
