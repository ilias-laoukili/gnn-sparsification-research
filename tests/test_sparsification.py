"""Unit tests for sparsification methods."""

import pytest
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
import networkx as nx

from src.sparsification.metric_backbone import (
    metric_backbone_sparsify,
    validate_alpha_behavior,
    compute_triangle_detours,
)
from src.sparsification.metrics import (
    calculate_jaccard_scores,
    calculate_adamic_adar_scores,
    calculate_effective_resistance_scores,
)
from src.sparsification.baselines import SparsificationBaselines


class TestMetricBackbone:
    """Test metric backbone sparsification."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple test graph (triangle)."""
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long)
        data = Data(edge_index=edge_index, num_nodes=3)
        return data

    @pytest.fixture
    def karate_graph(self):
        """Create Karate Club graph."""
        G = nx.karate_club_graph()
        edge_list = list(G.edges())
        edge_index = torch.tensor(
            [[u, v] for u, v in edge_list] + [[v, u] for u, v in edge_list], dtype=torch.long
        ).t()
        data = Data(edge_index=edge_index, num_nodes=G.number_of_nodes())
        return data

    def test_alpha_one_keeps_all_edges(self, karate_graph):
        """α=1.0 should not remove edges (RTI is triangle inequality)."""
        # Use uniform distances
        num_edges = karate_graph.edge_index.size(1)
        metric_scores = np.ones(num_edges)

        sparse_data, stats = metric_backbone_sparsify(
            karate_graph, metric_scores, alpha=1.0, verbose=False
        )

        assert sparse_data.edge_index.size(1) == num_edges
        assert stats["retention_ratio"] == 1.0

    def test_alpha_infinity_keeps_all_edges(self, karate_graph):
        """α→∞ should keep all edges (no constraint)."""
        num_edges = karate_graph.edge_index.size(1)
        metric_scores = np.random.rand(num_edges)

        sparse_data, stats = metric_backbone_sparsify(
            karate_graph, metric_scores, alpha=100.0, verbose=False
        )

        # Very large alpha should keep most/all edges
        assert stats["retention_ratio"] >= 0.95

    def test_monotonic_alpha_behavior(self, simple_graph):
        """Larger alpha should never remove more edges."""
        num_edges = simple_graph.edge_index.size(1)
        metric_scores = np.array([1.0, 1.5, 0.8, 1.5, 0.8, 1.0])

        results = validate_alpha_behavior(
            simple_graph, metric_scores, alpha_values=[1.0, 1.5, 2.0, 5.0]
        )

        # Check monotonicity
        alphas = sorted(results.keys())
        for i in range(len(alphas) - 1):
            assert results[alphas[i]] <= results[alphas[i + 1]]

    def test_handles_disconnected_components(self):
        """Should handle graphs with multiple components."""
        # Two separate triangles
        edge_index = torch.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], [1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4]],
            dtype=torch.long,
        )
        data = Data(edge_index=edge_index, num_nodes=6)

        num_edges = data.edge_index.size(1)
        metric_scores = np.ones(num_edges)

        sparse_data, stats = metric_backbone_sparsify(data, metric_scores, alpha=1.5, verbose=False)

        # Should not crash
        assert sparse_data.edge_index.size(1) > 0


class TestMetrics:
    """Test edge metric calculations."""

    @pytest.fixture
    def triangle_adj(self):
        """Create adjacency matrix for a triangle."""
        return sp.csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

    def test_jaccard_scores_range(self, triangle_adj):
        """Jaccard scores should be in [0, 1]."""
        scores = calculate_jaccard_scores(triangle_adj)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_jaccard_handles_isolated_nodes(self):
        """Jaccard should handle isolated nodes without errors."""
        # Graph with isolated node
        adj = sp.csr_matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        scores = calculate_jaccard_scores(adj)
        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))

    def test_adamic_adar_non_negative(self, triangle_adj):
        """Adamic-Adar scores should be non-negative."""
        scores = calculate_adamic_adar_scores(triangle_adj)
        assert np.all(scores >= 0.0)

    def test_adamic_adar_handles_degree_one(self):
        """Adamic-Adar should handle degree-1 nodes."""
        # Star graph (center node + 3 leaves)
        adj = sp.csr_matrix([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
        scores = calculate_adamic_adar_scores(adj)
        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))

    def test_effective_resistance_symmetry(self, triangle_adj):
        """Effective resistance should be consistent for undirected graphs."""
        scores = calculate_effective_resistance_scores(triangle_adj)

        # All edges in triangle should have same resistance
        # (by symmetry)
        assert np.allclose(scores, scores[0], rtol=1e-3)


class TestBaselines:
    """Test baseline sparsification methods."""

    @pytest.fixture
    def test_data(self):
        """Create test graph."""
        edge_index = torch.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]], dtype=torch.long
        )
        return Data(edge_index=edge_index, num_nodes=4)

    def test_random_determinism(self, test_data):
        """Same seed should give same result."""
        sparse1, _ = SparsificationBaselines.random_sparsify(
            test_data, retention_ratio=0.5, seed=42
        )
        sparse2, _ = SparsificationBaselines.random_sparsify(
            test_data, retention_ratio=0.5, seed=42
        )

        assert torch.equal(sparse1.edge_index, sparse2.edge_index)

    def test_random_different_seeds(self, test_data):
        """Different seeds should give different results (with high prob)."""
        sparse1, _ = SparsificationBaselines.random_sparsify(
            test_data, retention_ratio=0.5, seed=42
        )
        sparse2, _ = SparsificationBaselines.random_sparsify(
            test_data, retention_ratio=0.5, seed=123
        )

        # Should be different (with high probability)
        # Allowing for rare collision
        assert not torch.equal(sparse1.edge_index, sparse2.edge_index)

    def test_degree_based_retention(self, test_data):
        """Degree-based should respect retention ratio."""
        retention = 0.5
        sparse, stats = SparsificationBaselines.degree_based_sparsify(
            test_data, retention_ratio=retention, strategy="low_first"
        )

        expected_edges = int(test_data.edge_index.size(1) * retention)
        assert sparse.edge_index.size(1) == expected_edges
        assert stats["retention_ratio"] == retention

    def test_dropedge_is_random(self, test_data):
        """DropEdge should be equivalent to random with inverted ratio."""
        drop_rate = 0.3
        retention = 1.0 - drop_rate

        sparse1, stats1 = SparsificationBaselines.dropedge_sparsify(
            test_data, drop_rate=drop_rate, seed=42
        )
        sparse2, stats2 = SparsificationBaselines.random_sparsify(
            test_data, retention_ratio=retention, seed=42
        )

        assert torch.equal(sparse1.edge_index, sparse2.edge_index)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_graph(self):
        """Handle empty graphs."""
        data = Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=5)

        # Should not crash
        sparse, stats = SparsificationBaselines.random_sparsify(data, retention_ratio=0.5)
        assert sparse.edge_index.size(1) == 0

    def test_single_edge(self):
        """Handle graph with single edge."""
        data = Data(edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), num_nodes=2)

        metric_scores = np.array([1.0, 1.0])
        sparse, stats = metric_backbone_sparsify(data, metric_scores, alpha=1.5)

        # Should keep edge (no triangles)
        assert sparse.edge_index.size(1) == 2

    def test_tree_graph(self):
        """Handle tree graphs (no triangles)."""
        # Binary tree: 0-1, 0-2, 1-3, 1-4
        edge_index = torch.tensor(
            [[0, 0, 1, 1, 1, 2, 3, 4], [1, 2, 0, 3, 4, 0, 1, 1]], dtype=torch.long
        )
        data = Data(edge_index=edge_index, num_nodes=5)

        num_edges = data.edge_index.size(1)
        metric_scores = np.ones(num_edges)

        sparse, stats = metric_backbone_sparsify(data, metric_scores, alpha=1.5)

        # All edges should be kept (no triangles = no detours)
        assert stats["edges_without_triangles"] == num_edges
        assert stats["retention_ratio"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
