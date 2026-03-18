"""Core graph sparsification engine.

This module provides the main GraphSparsifier class that combines edge
metrics with quantile-based thresholding to produce sparsified graphs.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from .metric_backbone import compute_metric_backbone
from .metrics import (
    calculate_adamic_adar_scores,
    calculate_approx_effective_resistance_scores,
    calculate_effective_resistance_scores,
    calculate_feature_cosine_scores,
    calculate_jaccard_scores,
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
        "approx_effective_resistance",
        "approx_er",
        "random",
        "rand",
        "degree",
        "feature_cosine",
        "feature-cosine",
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

    # Metrics where higher score = more distant (distance-like).
    # All others are treated as similarity metrics (higher = more similar).
    _DISTANCE_METRICS = {"effective_resistance", "approx_effective_resistance"}

    def _scores_to_cost(self, scores: np.ndarray, metric: str) -> np.ndarray:
        """Convert edge scores to distance costs for the metric backbone.

        Follows the proximity-to-distance transformation from Simas et al.:
            1. Normalize scores to proximity in (0, 1]
            2. Apply  distance = 1/proximity - 1

        For distance-like metrics (effective resistance), scores are first
        inverted to similarity before normalization.

        References:
            Simas, T. et al. "The metric backbone preserves community structure."
        """
        metric_key = self._normalize_metric_name(metric)

        # Distance metrics: invert to similarity first
        if metric_key in self._DISTANCE_METRICS:
            safe = np.maximum(scores, 1e-10)
            similarity = 1.0 / safe
        else:
            similarity = scores.copy()

        # Normalize to proximity in (0, 1]
        s_max = similarity.max()
        if s_max <= 0:
            return np.ones_like(scores)
        proximity = similarity / s_max

        # Replace zeros with a small fraction of the minimum non-zero value
        nonzero = proximity[proximity > 0]
        floor = (nonzero.min() * 0.01) if len(nonzero) > 0 else 1e-6
        proximity[proximity <= 0] = floor

        # Proximity → distance: d = 1/p − 1  (standard isomorphic transformation)
        return 1.0 / proximity - 1.0

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
        if metric_lower in {"feature_cosine"}:
            return "feature_cosine"

        raise ValueError(
            f"Metric '{metric}' not supported. " f"Choose from: {self.SUPPORTED_METRICS}"
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
        elif metric_key == "feature_cosine":
            # Feature-based metric: cosine similarity between node features.
            # Rationale: structural metrics (Jaccard, AA) only use topology.
            # Feature cosine captures whether connected nodes are actually
            # similar in feature space, which is what GNNs aggregate.
            # On homophilous graphs, high-cosine edges connect same-class
            # nodes; on heterophilous graphs, low-cosine edges may be more
            # informative — but either way this gives the GNN a different
            # signal than pure topology.
            features = self.data.x.cpu().numpy() if self.data.x is not None else None
            if features is None:
                raise ValueError("feature_cosine requires node features (data.x)")
            scores = calculate_feature_cosine_scores(self.adj, features)
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
        keep_lowest: bool = False,
    ) -> Data | Tuple[Data, torch.Tensor]:
        """Create a sparsified copy of the graph.

        Uses argsort-based selection to keep exactly the top-k% (or bottom-k%)
        of edges ranked by the specified similarity metric.

        Args:
            metric: Edge scoring method ('jaccard', 'adamic_adar', etc.).
            retention_ratio: Fraction of edges to retain, in (0, 1].
            return_mask: If True, also return the boolean edge mask.
            keep_lowest: If True, keep the lowest-scoring edges instead of
                        the highest. Useful as a control experiment.

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

        # Use argsort to select exactly top-k or bottom-k edges
        num_keep = int(self.num_edges * retention_ratio)
        sorted_indices = np.argsort(scores)
        if keep_lowest:
            selected_indices = sorted_indices[:num_keep]
        else:
            selected_indices = sorted_indices[-num_keep:]

        mask = np.zeros(self.num_edges, dtype=bool)
        mask[selected_indices] = True

        sparse_edge_index = self.data.edge_index[:, mask].to(self.device)

        sparse_data = self.data.clone()
        sparse_data.edge_index = sparse_edge_index

        if return_mask:
            return sparse_data, torch.from_numpy(mask)
        return sparse_data

    def sparsify_metric_backbone(
        self,
        metric: str,
        epsilon: float = 1e-9,
    ) -> tuple[Data, dict]:
        """Sparsify using the Global Metric Backbone (APSP-based).

        The Metric Backbone preserves exact geodesic distances by keeping only
        edges that lie on shortest paths. Unlike threshold-based methods, the
        retention ratio is determined by the graph structure, not a parameter.

        Args:
            metric: Edge scoring method (similarity). Converted to distances.
            epsilon: Floating-point tolerance for edge classification.

        Returns:
            (sparse_data_on_device, stats_dict) where stats contain retention_ratio.

        Note:
            The retention ratio is determined by the graph structure and cannot
            be tuned. For target-retention sparsification, use sparsify() instead.
        """
        distances = self._scores_to_cost(self.compute_scores(metric), metric)

        sparse_data, stats = compute_metric_backbone(
            self.data, distances, epsilon=epsilon, verbose=self.verbose
        )

        return sparse_data.to(self.device), stats

    def sparsify_sampled(
        self,
        metric: str,
        retention_ratio: float,
        seed: int = 42,
        return_mask: bool = False,
    ) -> Data | Tuple[Data, torch.Tensor]:
        """Sparsify by probabilistic sampling proportional to edge scores.

        WHY THIS EXISTS:
        Standard threshold sparsification (sparsify()) is deterministic — it
        always removes the same bottom-k edges. This creates a hard cutoff
        that can sever important local connections that happen to have low
        global scores. For example, a bridge edge between two communities
        might have low Jaccard similarity (few common neighbors) but is
        critical for information flow.

        Probabilistic sampling addresses this by giving every edge a CHANCE
        of being kept, proportional to its score. This has two benefits:
          1. Diversity: different seeds produce different subgraphs, enabling
             ensemble-like effects and reducing sensitivity to score noise.
          2. Soft boundaries: low-scoring edges aren't guaranteed to be cut,
             so local structure is better preserved in expectation.

        HOW IT WORKS:
        1. Compute edge scores using the specified metric.
        2. Normalize scores to probabilities (softmax-like: score / sum).
        3. Sample edges WITHOUT replacement, with probability proportional
           to score. This gives exactly retention_ratio * num_edges edges.
        4. The result is a valid subgraph (same nodes, fewer edges).

        This is inspired by Spielman-Srivastava spectral sparsification,
        which samples edges proportional to effective resistance, but here
        we generalize to any edge metric.

        Args:
            metric: Edge scoring method ('jaccard', 'adamic_adar', etc.).
            retention_ratio: Fraction of edges to retain, in (0, 1].
            seed: Random seed for reproducible sampling.
            return_mask: If True, also return the boolean edge mask.

        Returns:
            Sparsified Data object (and optionally the edge mask).
        """
        if not 0 < retention_ratio <= 1:
            raise ValueError(f"retention_ratio must be in (0, 1], got {retention_ratio}")

        if retention_ratio == 1.0:
            if return_mask:
                return self.data.clone(), torch.ones(self.num_edges, dtype=torch.bool)
            return self.data.clone()

        rng = np.random.default_rng(seed)
        scores = self.compute_scores(metric)

        # Convert scores to sampling probabilities.
        # Add a small floor so that even zero-score edges have a nonzero
        # chance of being sampled (prevents guaranteed removal).
        floor = 1e-8
        scores = np.nan_to_num(scores, nan=floor, posinf=floor, neginf=floor)
        probs = np.maximum(scores, floor)
        probs = probs / probs.sum()

        # Sample exactly num_keep edges WITHOUT replacement, weighted by score.
        num_keep = int(self.num_edges * retention_ratio)
        selected = rng.choice(self.num_edges, size=num_keep, replace=False, p=probs)

        mask = np.zeros(self.num_edges, dtype=bool)
        mask[selected] = True

        sparse_edge_index = self.data.edge_index[:, mask].to(self.device)
        sparse_data = self.data.clone()
        sparse_data.edge_index = sparse_edge_index

        if return_mask:
            return sparse_data, torch.from_numpy(mask)
        return sparse_data

    def sparsify_degree_aware(
        self,
        metric: str,
        retention_ratio: float,
        min_edges_per_node: int = 1,
        return_mask: bool = False,
    ) -> Data | Tuple[Data, torch.Tensor]:
        """Sparsify with a per-node minimum edge budget.

        WHY THIS EXISTS:
        Standard threshold sparsification treats all edges equally in a
        global ranking. This means low-degree nodes (which have few edges
        to begin with) can lose ALL their edges at aggressive retention
        ratios, effectively isolating them from the graph. Isolated nodes
        contribute nothing to GNN message passing and are classified based
        solely on their features, hurting accuracy.

        In contrast, high-degree hub nodes have many redundant edges and
        can afford to lose most of them without losing connectivity.

        This method enforces a MINIMUM number of edges per node before
        doing global thresholding on the remaining budget. This ensures:
          1. No node becomes isolated (preserves graph connectivity).
          2. Low-degree nodes keep their most important connections.
          3. The overall retention ratio is still approximately respected.

        HOW IT WORKS:
        1. For each node, guarantee its top-k edges (by score) are kept,
           where k = min(min_edges_per_node, node_degree).
        2. Count how many edges the guarantee phase already selected.
        3. Fill the remaining budget from the global ranked list, skipping
           edges already guaranteed.
        4. The total number of kept edges = retention_ratio * num_edges
           (or slightly more if the guarantee phase exceeds the budget).

        This is motivated by the observation that degree distribution in
        real graphs follows a power law — a small number of hubs have
        most edges while many peripheral nodes have very few.

        Args:
            metric: Edge scoring method.
            retention_ratio: Target fraction of edges to retain, in (0, 1].
            min_edges_per_node: Minimum edges to guarantee per node.
            return_mask: If True, also return the boolean edge mask.

        Returns:
            Sparsified Data object (and optionally the edge mask).
        """
        if not 0 < retention_ratio <= 1:
            raise ValueError(f"retention_ratio must be in (0, 1], got {retention_ratio}")

        if retention_ratio == 1.0:
            if return_mask:
                return self.data.clone(), torch.ones(self.num_edges, dtype=torch.bool)
            return self.data.clone()

        scores = self.compute_scores(metric)
        edge_index_np = self.data.edge_index.cpu().numpy()
        num_keep = int(self.num_edges * retention_ratio)

        # Phase 1: Guarantee min_edges_per_node for each node.
        # For each node, find its incident edges and keep the top-k by score.
        guaranteed = set()
        for node in range(self.num_nodes):
            # Find all edges where this node is the source (directed graph
            # stores both directions, so source covers outgoing).
            incident_mask = edge_index_np[0] == node
            incident_indices = np.where(incident_mask)[0]

            if len(incident_indices) == 0:
                continue

            # Keep the top-k scoring edges for this node
            k = min(min_edges_per_node, len(incident_indices))
            incident_scores = scores[incident_indices]
            top_k = incident_indices[np.argsort(incident_scores)[-k:]]
            guaranteed.update(top_k.tolist())

        # Phase 2: Fill remaining budget from global ranking.
        # Sort all edges by score descending; add non-guaranteed edges until
        # we reach the target budget.
        mask = np.zeros(self.num_edges, dtype=bool)
        for idx in guaranteed:
            mask[idx] = True

        if mask.sum() < num_keep:
            # Still have budget left — fill from global ranking
            sorted_indices = np.argsort(scores)[::-1]  # descending
            for idx in sorted_indices:
                if mask.sum() >= num_keep:
                    break
                if not mask[idx]:
                    mask[idx] = True
        # If guarantee phase already exceeded budget, we keep all guaranteed
        # edges (better to keep slightly more than isolate nodes).

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
