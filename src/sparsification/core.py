"""Core graph sparsification engine.

This module provides the main GraphSparsifier class that combines edge
metrics with quantile-based thresholding to produce sparsified graphs.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from .metrics import (
    calculate_adamic_adar_scores,
    calculate_jaccard_scores,
    calculate_effective_resistance_scores,
    calculate_approx_effective_resistance_scores,
)
from .metric_backbone import metric_backbone_sparsify


class GraphSparsifier:
    """Engine for graph sparsification via edge metric thresholding.

    Computes edge importance scores using various metrics and applies
    quantile-based filtering to retain only the most significant edges.

    Args:
        data: PyG Data object containing the graph structure.
        device: Target device for tensor operations.

    Attributes:
        data: Original graph data (immutable reference).
        device: Computation device string.
        num_nodes: Number of nodes in the graph.
        num_edges: Number of edges (directed count).
        adj: Sparse adjacency matrix in CSR format.

    Example:
        >>> sparsifier = GraphSparsifier(data, "cuda")
        >>> sparse_data = sparsifier.sparsify("jaccard", retention_ratio=0.5)
    """

    SUPPORTED_METRICS = {
        "jaccard",
        "adamic-adar",
        "adamic_adar",
        "aa",
        "effective_resistance",
        "effective-resistance",
        "er",
        "approx_effective_resistance",
        "approx_er",
        "random",
        "rand",
        "degree",
    }

    def __init__(self, data: Data, device: str) -> None:
        self.data = data
        self.device = device
        self.num_nodes = data.num_nodes
        self.num_edges = data.edge_index.size(1)
        self.verbose: bool = False

        edge_index_cpu = data.edge_index.cpu().numpy()
        self.adj = sp.csr_matrix(
            (np.ones(self.num_edges), (edge_index_cpu[0], edge_index_cpu[1])),
            shape=(self.num_nodes, self.num_nodes),
        )

        self._score_cache: Dict[str, np.ndarray] = {}

    def _scores_to_dissimilarity(self, scores: np.ndarray) -> np.ndarray:
        """Convert similarity-like scores to dissimilarity distances.

        Metric backbone pruning is defined for distances, so we normalize
        arbitrary scores to [0, 1] and flip them so that larger values
        represent weaker connections.
        """
        score_min = scores.min()
        score_max = scores.max()
        if score_max > score_min:
            normalized = (scores - score_min) / (score_max - score_min)
        else:
            normalized = np.zeros_like(scores)

        # Small epsilon keeps distances strictly positive for stability
        return 1.0 - normalized + 1e-6

    def _normalize_metric_name(self, metric: str) -> str:
        """Standardize metric name and validate."""
        metric_lower = metric.lower().replace("-", "_").replace(" ", "_")
        if metric_lower in {"jaccard"}:
            return "jaccard"
        if metric_lower in {"adamic_adar", "aa"}:
            return "adamic_adar"
        if metric_lower in {"effective_resistance", "er"}:
            return "effective_resistance"
        if metric_lower in {"approx_effective_resistance", "approx_er"}:
            return "approx_effective_resistance"
        if metric_lower in {"random", "rand"}:
            return "random"
        if metric_lower in {"degree"}:
            return "degree"
        
        raise ValueError(
            f"Metric '{metric}' not supported. "
            f"Choose from: {self.SUPPORTED_METRICS}"
        )

    def compute_scores(self, metric: str) -> np.ndarray:
        """Compute or retrieve cached edge scores for a given metric.

        Args:
            metric: Metric name.

        Returns:
            Array of edge scores aligned with edge_index columns.

        Raises:
            ValueError: If metric is not supported.
        """
        metric_key = self._normalize_metric_name(metric)

        if metric_key in self._score_cache:
            return self._score_cache[metric_key]

        if metric_key == "jaccard":
            scores = calculate_jaccard_scores(self.adj)
        elif metric_key == "adamic_adar":
            scores = calculate_adamic_adar_scores(self.adj)
        elif metric_key == "effective_resistance":
            scores = calculate_effective_resistance_scores(self.adj)
        elif metric_key == "approx_effective_resistance":
            scores = calculate_approx_effective_resistance_scores(self.adj)
        elif metric_key == "random":
            scores = np.random.rand(self.adj.nnz)
        elif metric_key == "degree":
            degrees = np.asarray(self.adj.sum(axis=1)).flatten()
            rows, cols = self.adj.nonzero()
            # Score is product of degrees of incident nodes.
            # Edges connecting high-degree nodes get higher scores.
            scores = degrees[rows] * degrees[cols]
        else:
            # This branch should be unreachable if _normalize_metric_name is correct
            raise ValueError(f"Internal error: Unhandled metric '{metric_key}'")

        self._score_cache[metric_key] = scores
        return scores

    def sparsify(
        self,
        metric: str,
        retention_ratio: float,
        return_mask: bool = False,
    ) -> Data | Tuple[Data, torch.Tensor]:
        """Create a sparsified copy of the graph.

        Uses argsort-based selection to keep exactly the top-k% of edges
        ranked by the specified similarity metric.

        Args:
            metric: Edge scoring method ('jaccard' or 'adamic-adar').
            retention_ratio: Fraction of edges to retain, in (0, 1].
            return_mask: If True, also return the boolean edge mask.

        Returns:
            If return_mask is False:
                Sparsified Data object with reduced edge_index.
            If return_mask is True:
                Tuple of (sparsified Data, edge mask tensor).

        Raises:
            ValueError: If retention_ratio is not in (0, 1].
        """
        if not 0 < retention_ratio <= 1:
            raise ValueError(f"retention_ratio must be in (0, 1], got {retention_ratio}")

        if retention_ratio == 1.0:
            if return_mask:
                return self.data.clone(), torch.ones(self.num_edges, dtype=torch.bool)
            return self.data.clone()

        scores = self.compute_scores(metric)

        # Use argsort to select exactly top-k edges (handles ties correctly)
        num_keep = int(self.num_edges * retention_ratio)
        top_indices = np.argsort(scores)[-num_keep:]

        mask = np.zeros(self.num_edges, dtype=bool)
        mask[top_indices] = True

        sparse_edge_index = self.data.edge_index[:, mask].to(self.device)

        sparse_data = self.data.clone()
        sparse_data.edge_index = sparse_edge_index

        if return_mask:
            return sparse_data, torch.from_numpy(mask)
        return sparse_data

    def sparsify_metric_backbone(
        self,
        metric: str,
        target_retention: float,
        alpha: float | None = None,
        alpha_bounds: tuple[float, float] = (1.0, 6.0),
        max_iter: int = 8,
        tol: float = 0.02,
    ) -> tuple[Data, dict]:
        """Sparsify using the metric backbone (RTI) to hit a retention target.

        Args:
            metric: Edge scoring method (similarity). Converted internally to distances.
            target_retention: Desired fraction of edges to keep.
            alpha: Optional fixed alpha. If provided, binary search is skipped.
            alpha_bounds: Search interval for alpha when tuning toward target_retention.
            max_iter: Max binary-search iterations.
            tol: Acceptable absolute error on retention ratio.

        Returns:
            (sparse_data_on_device, stats_dict) where stats contain retention_ratio.
        """
        if not 0 < target_retention <= 1:
            raise ValueError(
                f"target_retention must be in (0, 1], got {target_retention}"
            )

        if alpha_bounds[0] >= alpha_bounds[1]:
            raise ValueError("alpha_bounds must be increasing (low, high)")

        distances = self._scores_to_dissimilarity(self.compute_scores(metric))

        def run(alpha_value: float) -> tuple[Data, dict]:
            sparse, stats = metric_backbone_sparsify(
                self.data, distances, alpha=alpha_value, verbose=self.verbose
            )
            return sparse, stats

        if alpha is not None:
            sparse_data, stats = run(alpha)
            return sparse_data.to(self.device), stats

        low, high = alpha_bounds
        best_sparse = None
        best_stats = None

        for _ in range(max_iter):
            mid = (low + high) / 2.0
            sparse_data, stats = run(mid)
            retention = stats["retention_ratio"]

            if best_stats is None or abs(retention - target_retention) < abs(
                best_stats["retention_ratio"] - target_retention
            ):
                best_sparse, best_stats = sparse_data, stats

            if abs(retention - target_retention) <= tol:
                break

            if retention > target_retention:
                low = mid  # need more pruning -> increase alpha
            else:
                high = mid  # need fewer removals -> decrease alpha

        assert best_sparse is not None and best_stats is not None
        return best_sparse.to(self.device), best_stats

    def get_retention_curve_data(
        self,
        metric: str,
        retention_rates: list[float],
    ) -> list[Data]:
        """Generate multiple sparsified graphs for retention rate sweeps.

        Useful for sensitivity analysis experiments that compare model
        performance across different sparsification levels.

        Args:
            metric: Edge scoring method to use.
            retention_rates: List of retention ratios to evaluate.

        Returns:
            List of sparsified Data objects, one per retention rate.
        """
        return [self.sparsify(metric, rate) for rate in retention_rates]

    @property
    def stats(self) -> dict:
        """Summary statistics about the graph structure."""
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "density": self.num_edges / (self.num_nodes * (self.num_nodes - 1)),
            "avg_degree": self.num_edges / self.num_nodes,
        }
