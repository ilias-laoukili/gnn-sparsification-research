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
        "approx_er",
        "approx_effective_resistance",
        "approx-er",
        "random",
        "rand",
    }

    def __init__(self, data: Data, device: str) -> None:
        self.data = data
        self.device = device
        self.num_nodes = data.num_nodes
        self.num_edges = data.edge_index.size(1)

        edge_index_cpu = data.edge_index.cpu().numpy()
        self.adj = sp.csr_matrix(
            (np.ones(self.num_edges), (edge_index_cpu[0], edge_index_cpu[1])),
            shape=(self.num_nodes, self.num_nodes),
        )

        self._score_cache: Dict[str, np.ndarray] = {}

    def _normalize_metric_name(self, metric: str) -> str:
        """Standardize metric name to canonical form."""
        metric_lower = metric.lower().replace("-", "_")
        if metric_lower in {"adamic_adar", "aa"}:
            return "adamic_adar"
        if metric_lower in {"effective_resistance", "er"}:
            return "effective_resistance"
        if metric_lower in {"approx_er", "approx_effective_resistance"}:
            return "approx_er"
        if metric_lower in {"random", "rand"}:
            return "random"
        return metric_lower

    def compute_scores(self, metric: str) -> np.ndarray:
        """Compute or retrieve cached edge scores for a given metric.

        Args:
            metric: Metric name ('jaccard', 'adamic-adar', or 'aa').

        Returns:
            Array of edge scores aligned with edge_index columns.

        Raises:
            ValueError: If metric is not supported.
        """
        metric_key = self._normalize_metric_name(metric)

        if metric_key not in {"jaccard", "adamic_adar", "effective_resistance", "approx_er", "random"}:
            raise ValueError(
                f"Metric '{metric}' not supported. "
                f"Choose from: {self.SUPPORTED_METRICS}"
            )

        if metric_key in self._score_cache:
            return self._score_cache[metric_key]

        if metric_key == "jaccard":
            scores = calculate_jaccard_scores(self.adj)
        elif metric_key == "adamic_adar":
            scores = calculate_adamic_adar_scores(self.adj)
        elif metric_key == "effective_resistance":
            scores = calculate_effective_resistance_scores(self.adj)
        elif metric_key == "approx_er":
            scores = calculate_approx_effective_resistance_scores(self.adj)
        else:  # random baseline
            scores = np.random.rand(self.num_edges)

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
            raise ValueError(
                f"retention_ratio must be in (0, 1], got {retention_ratio}"
            )

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
