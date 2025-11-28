"""Unit tests for sparsification methods."""

import pytest
import torch
from torch_geometric.data import Data

from src.sparsification.base import BaseSparsifier
from src.sparsification.random import RandomSparsifier
from src.sparsification.spectral import SpectralSparsifier
from src.sparsification.metric import MetricSparsifier


@pytest.fixture
def sample_graph():
    """Create a simple test graph."""
    x = torch.randn(10, 8)  # 10 nodes, 8 features
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8]
    ])
    y = torch.randint(0, 3, (10,))
    
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=10)


def test_random_sparsifier_initialization():
    """Test RandomSparsifier initialization."""
    sparsifier = RandomSparsifier(sparsification_ratio=0.5, seed=42)
    assert sparsifier.sparsification_ratio == 0.5
    assert sparsifier.seed == 42


def test_random_sparsifier_reduces_edges(sample_graph):
    """Test that RandomSparsifier reduces the number of edges."""
    sparsifier = RandomSparsifier(sparsification_ratio=0.5, seed=42)
    sparse_graph = sparsifier.sparsify(sample_graph)
    
    original_edges = sample_graph.edge_index.shape[1]
    sparse_edges = sparse_graph.edge_index.shape[1]
    
    assert sparse_edges < original_edges
    assert sparse_edges == int(original_edges * 0.5)


def test_sparsifier_preserves_node_count(sample_graph):
    """Test that sparsification preserves node count."""
    sparsifier = RandomSparsifier(sparsification_ratio=0.5)
    sparse_graph = sparsifier.sparsify(sample_graph)
    
    assert sparse_graph.num_nodes == sample_graph.num_nodes


def test_sparsifier_invalid_ratio():
    """Test that invalid sparsification ratios raise ValueError."""
    with pytest.raises(ValueError):
        RandomSparsifier(sparsification_ratio=1.5)
    
    with pytest.raises(ValueError):
        RandomSparsifier(sparsification_ratio=-0.5)


def test_metric_sparsifier_jaccard(sample_graph):
    """Test MetricSparsifier with Jaccard similarity."""
    sparsifier = MetricSparsifier(
        sparsification_ratio=0.5,
        metric="jaccard"
    )
    sparse_graph = sparsifier.sparsify(sample_graph)
    
    original_edges = sample_graph.edge_index.shape[1]
    sparse_edges = sparse_graph.edge_index.shape[1]
    
    assert sparse_edges <= original_edges


def test_metric_sparsifier_cosine(sample_graph):
    """Test MetricSparsifier with cosine similarity."""
    sparsifier = MetricSparsifier(
        sparsification_ratio=0.5,
        metric="cosine"
    )
    sparse_graph = sparsifier.sparsify(sample_graph)
    
    assert sparse_graph.num_nodes == sample_graph.num_nodes


def test_get_sparsification_stats(sample_graph):
    """Test sparsification statistics computation."""
    sparsifier = RandomSparsifier(sparsification_ratio=0.5, seed=42)
    sparse_graph = sparsifier.sparsify(sample_graph)
    
    stats = sparsifier.get_sparsification_stats(sample_graph, sparse_graph)
    
    assert "original_edges" in stats
    assert "sparsified_edges" in stats
    assert "edges_removed" in stats
    assert "actual_sparsification_ratio" in stats
    assert stats["original_edges"] == sample_graph.edge_index.shape[1]
