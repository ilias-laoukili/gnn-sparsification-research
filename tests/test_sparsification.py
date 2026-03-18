"""Unit tests for sparsification methods."""

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data

from src.sparsification.core import GraphSparsifier
from src.sparsification.metric_backbone import compute_metric_backbone
from src.sparsification.metrics import (
    calculate_adamic_adar_scores,
    calculate_approx_effective_resistance_scores,
    calculate_effective_resistance_scores,
    calculate_jaccard_scores,
)
from src.training.trainer import GNNTrainer


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

    def test_uniform_weights_keep_all_edges(self, karate_graph):
        """Uniform weights: all edges are metric, backbone keeps everything."""
        num_edges = karate_graph.edge_index.size(1)
        edge_weights = np.ones(num_edges)

        sparse_data, stats = compute_metric_backbone(
            karate_graph, edge_weights, epsilon=1e-9, verbose=False
        )

        assert sparse_data.edge_index.size(1) == num_edges
        assert stats["retention_ratio"] == 1.0

    def test_inverse_weights_prune_edges(self, karate_graph):
        """Inverse Jaccard weights should prune some semi-metric edges."""
        sparsifier = GraphSparsifier(karate_graph, "cpu")
        scores = sparsifier.compute_scores("jaccard")
        costs = sparsifier._scores_to_cost(scores, "jaccard")

        sparse_data, stats = compute_metric_backbone(
            karate_graph, costs, epsilon=1e-9, verbose=False
        )

        # Should remove at least some edges
        assert stats["retention_ratio"] < 1.0
        assert sparse_data.edge_index.size(1) > 0

    def test_handles_disconnected_components(self):
        """Should handle graphs with multiple components."""
        # Two separate triangles
        edge_index = torch.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], [1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4]],
            dtype=torch.long,
        )
        data = Data(edge_index=edge_index, num_nodes=6)

        num_edges = data.edge_index.size(1)
        edge_weights = np.ones(num_edges)

        sparse_data, stats = compute_metric_backbone(
            data, edge_weights, epsilon=1e-9, verbose=False
        )

        # Should not crash and keep all edges (uniform weights)
        assert sparse_data.edge_index.size(1) == num_edges

    def test_single_edge(self):
        """Handle graph with single edge."""
        data = Data(edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), num_nodes=2)
        edge_weights = np.array([1.0, 1.0])

        sparse_data, stats = compute_metric_backbone(data, edge_weights, epsilon=1e-9)

        # No alternative paths exist — edge must be kept
        assert sparse_data.edge_index.size(1) == 2

    def test_tree_keeps_all_edges(self):
        """Tree graphs have no detours — all edges must be kept."""
        # Binary tree: 0-1, 0-2, 1-3, 1-4
        edge_index = torch.tensor(
            [[0, 0, 1, 1, 1, 2, 3, 4], [1, 2, 0, 3, 4, 0, 1, 1]], dtype=torch.long
        )
        data = Data(edge_index=edge_index, num_nodes=5)

        num_edges = data.edge_index.size(1)
        edge_weights = np.ones(num_edges)

        sparse_data, stats = compute_metric_backbone(data, edge_weights, epsilon=1e-9)

        assert stats["retention_ratio"] == 1.0


class TestGraphSparsifier:
    """Test the main GraphSparsifier class."""

    @pytest.fixture
    def cora_like_graph(self):
        """Create a small graph for testing."""
        G = nx.karate_club_graph()
        edge_list = list(G.edges())
        edge_index = torch.tensor(
            [[u, v] for u, v in edge_list] + [[v, u] for u, v in edge_list], dtype=torch.long
        ).t()
        return Data(edge_index=edge_index, num_nodes=G.number_of_nodes())

    def test_sparsify_retention(self, cora_like_graph):
        """Threshold sparsification respects retention ratio."""
        sparsifier = GraphSparsifier(cora_like_graph, "cpu")
        original_edges = cora_like_graph.edge_index.size(1)

        sparse_data = sparsifier.sparsify("jaccard", retention_ratio=0.5)
        expected = int(original_edges * 0.5)
        assert sparse_data.edge_index.size(1) == expected

    def test_sparsify_full_retention(self, cora_like_graph):
        """retention_ratio=1.0 returns all edges."""
        sparsifier = GraphSparsifier(cora_like_graph, "cpu")
        sparse_data = sparsifier.sparsify("jaccard", retention_ratio=1.0)
        assert sparse_data.edge_index.size(1) == cora_like_graph.edge_index.size(1)

    def test_scores_to_cost_proximity_transform(self, cora_like_graph):
        """_scores_to_cost should apply 1/proximity - 1 transformation."""
        sparsifier = GraphSparsifier(cora_like_graph, "cpu")
        # Scores: 0.5, 1.0, 0.25 → normalized proximity: 0.5, 1.0, 0.25
        # distance = 1/proximity - 1 → 1.0, 0.0, 3.0
        scores = np.array([0.5, 1.0, 0.25])
        costs = sparsifier._scores_to_cost(scores, "jaccard")
        np.testing.assert_allclose(costs, [1.0, 0.0, 3.0])

    def test_random_metric(self, cora_like_graph):
        """Random metric should produce valid scores."""
        sparsifier = GraphSparsifier(cora_like_graph, "cpu")
        scores = sparsifier.compute_scores("random")
        assert len(scores) == cora_like_graph.edge_index.size(1)
        assert np.all(scores >= 0) and np.all(scores <= 1)


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
        adj = sp.csr_matrix([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
        scores = calculate_adamic_adar_scores(adj)
        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))

    def test_effective_resistance_symmetry(self, triangle_adj):
        """All edges in triangle should have same resistance (by symmetry)."""
        scores = calculate_effective_resistance_scores(triangle_adj)
        assert np.allclose(scores, scores[0], rtol=1e-3)

    def test_approx_effective_resistance_positive(self, triangle_adj):
        """Approximate effective resistance scores should be positive."""
        scores = calculate_approx_effective_resistance_scores(triangle_adj)
        assert np.all(scores > 0)

    def test_approx_effective_resistance_deterministic(self, triangle_adj):
        """Same seed should give same results."""
        scores1 = calculate_approx_effective_resistance_scores(triangle_adj, seed=42)
        scores2 = calculate_approx_effective_resistance_scores(triangle_adj, seed=42)
        assert np.allclose(scores1, scores2)

    def test_approx_effective_resistance_correlation(self):
        """Approximate ER should correlate with exact ER."""
        G = nx.karate_club_graph()
        adj = nx.to_scipy_sparse_array(G, format="csr")

        exact = calculate_effective_resistance_scores(adj)
        approx = calculate_approx_effective_resistance_scores(adj, epsilon=0.3, seed=42)

        from scipy.stats import spearmanr

        corr, _ = spearmanr(exact, approx)
        assert corr > 0.5


class TestGNNTrainer:
    """Test GNNTrainer training loop and best-model restoration."""

    @pytest.fixture
    def tiny_data(self):
        """Create a minimal node-classification dataset for fast tests."""
        torch.manual_seed(0)
        # 20 nodes, 2 features, 2 classes — simple ring graph
        n = 20
        src = torch.arange(n)
        dst = (src + 1) % n
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src]),
        ], dim=0)
        x = torch.randn(n, 2)
        y = (torch.arange(n) % 2).long()

        # 10 train / 5 val / 5 test
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask   = torch.zeros(n, dtype=torch.bool)
        test_mask  = torch.zeros(n, dtype=torch.bool)
        train_mask[:10] = True
        val_mask[10:15] = True
        test_mask[15:]  = True

        return Data(x=x, edge_index=edge_index, y=y,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                    num_nodes=n)

    @pytest.fixture
    def simple_model(self):
        """Two-layer MLP (no graph convolution) for speed."""
        return nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.LogSoftmax(dim=1),
        )

    def _make_trainer(self, model):
        opt = Adam(model.parameters(), lr=0.01)
        return GNNTrainer(model, opt, device="cpu")

    # ── wrap the sequential model so GNNTrainer's forward(data, edge_weight) works ──

    class _WrapModel(nn.Module):
        """Adapter: forward(data, edge_weight=None) → node log-probs."""
        def __init__(self, mlp):
            super().__init__()
            self.mlp = mlp

        def forward(self, data, edge_weight=None):
            return self.mlp(data.x)

    def test_best_val_acc_matches_history_max(self, tiny_data):
        """history['best_val_acc'] must equal max(history['val_acc'])."""
        model   = self._WrapModel(
            nn.Sequential(nn.Linear(2, 8), nn.ReLU(),
                          nn.Linear(8, 2), nn.LogSoftmax(dim=1))
        )
        trainer = self._make_trainer(model)
        history = trainer.train(tiny_data, epochs=30, patience=5)

        assert history["best_val_acc"] == pytest.approx(
            max(history["val_acc"]), abs=1e-6
        ), ("best_val_acc should equal the maximum val_acc seen during training, "
            "not the accuracy at the final epoch.")

    def test_best_model_restored_after_early_stopping(self, tiny_data):
        """After training, model weights should correspond to the best val epoch.

        We verify this by re-evaluating via compute_metrics and checking that
        the reported accuracy matches history['best_val_acc'] (within float noise).
        """
        model   = self._WrapModel(
            nn.Sequential(nn.Linear(2, 8), nn.ReLU(),
                          nn.Linear(8, 2), nn.LogSoftmax(dim=1))
        )
        trainer = self._make_trainer(model)
        # Short patience forces early stopping well before max epochs
        history = trainer.train(tiny_data, epochs=200, patience=5)

        # The model should have been restored to the best-epoch checkpoint.
        # Re-running evaluate() on val_mask must give back best_val_acc.
        restored_val_acc = trainer._evaluate(tiny_data, tiny_data.val_mask)
        assert restored_val_acc == pytest.approx(
            history["best_val_acc"], abs=1e-6
        ), ("After training, model should hold best-epoch weights. "
            "If weights were left at the final (possibly degraded) epoch, "
            "restored_val_acc < best_val_acc.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
