"""Example script demonstrating sparsification methods.

This script loads the Cora dataset and applies different sparsification
methods, comparing their effects on the graph structure.

Run this script to verify that the sparsification module is working correctly.

Usage:
    python scripts/demo_sparsification.py
"""

import torch
from torch_geometric.datasets import Planetoid

from src.sparsification.random import RandomSparsifier
from src.sparsification.metric import MetricSparsifier
from src.sparsification.spectral import SpectralSparsifier


def print_graph_stats(data, title="Graph Statistics"):
    """Print statistics about a graph."""
    print(f"\n{title}")
    print("=" * 60)
    print(f"  Number of nodes: {data.num_nodes}")
    print(f"  Number of edges: {data.num_edges}")
    print(f"  Number of features: {data.num_features}")
    print(f"  Number of classes: {data.y.max().item() + 1}")
    print(f"  Average node degree: {data.num_edges / data.num_nodes:.2f}")
    print("=" * 60)


def demo_sparsification():
    """Demonstrate different sparsification methods."""
    print("\n" + "=" * 60)
    print("GNN Sparsification Demo")
    print("=" * 60)
    
    # Load Cora dataset
    print("\nLoading Cora dataset...")
    dataset = Planetoid(root='data/', name='Cora')
    data = dataset[0]
    
    print_graph_stats(data, "Original Graph")
    
    # Test Random Sparsification
    print("\n" + "-" * 60)
    print("Testing Random Sparsification")
    print("-" * 60)
    
    sparsifier = RandomSparsifier(sparsification_ratio=0.5, seed=42)
    sparse_data = sparsifier.sparsify(data)
    
    print_graph_stats(sparse_data, "Randomly Sparsified Graph (50%)")
    
    stats = sparsifier.get_sparsification_stats(data, sparse_data)
    print("\nSparsification Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test Jaccard Sparsification
    print("\n" + "-" * 60)
    print("Testing Jaccard Similarity Sparsification")
    print("-" * 60)
    
    sparsifier = MetricSparsifier(
        sparsification_ratio=0.5,
        metric="jaccard"
    )
    sparse_data = sparsifier.sparsify(data)
    
    print_graph_stats(sparse_data, "Jaccard Sparsified Graph (50%)")
    
    stats = sparsifier.get_sparsification_stats(data, sparse_data)
    print("\nSparsification Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test Cosine Sparsification
    print("\n" + "-" * 60)
    print("Testing Cosine Similarity Sparsification")
    print("-" * 60)
    
    sparsifier = MetricSparsifier(
        sparsification_ratio=0.5,
        metric="cosine"
    )
    sparse_data = sparsifier.sparsify(data)
    
    print_graph_stats(sparse_data, "Cosine Sparsified Graph (50%)")
    
    stats = sparsifier.get_sparsification_stats(data, sparse_data)
    print("\nSparsification Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test Spectral Sparsification (placeholder)
    print("\n" + "-" * 60)
    print("Testing Spectral Sparsification (Placeholder)")
    print("-" * 60)
    
    sparsifier = SpectralSparsifier(sparsification_ratio=0.5)
    sparse_data = sparsifier.sparsify(data)
    
    print_graph_stats(sparse_data, "Spectrally Sparsified Graph (50%)")
    
    stats = sparsifier.get_sparsification_stats(data, sparse_data)
    print("\nSparsification Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Compare different sparsification ratios
    print("\n" + "=" * 60)
    print("Comparing Different Sparsification Ratios (Random)")
    print("=" * 60)
    
    for ratio in [0.2, 0.5, 0.8]:
        sparsifier = RandomSparsifier(sparsification_ratio=ratio, seed=42)
        sparse_data = sparsifier.sparsify(data)
        
        print(f"\nRatio: {ratio:.1f}")
        print(f"  Original edges: {data.num_edges}")
        print(f"  Sparsified edges: {sparse_data.num_edges}")
        print(f"  Edges removed: {data.num_edges - sparse_data.num_edges}")
        print(f"  Actual ratio: {sparse_data.num_edges / data.num_edges:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Implement GCN/GAT models in src/models/")
    print("2. Complete training loop in scripts/train.py")
    print("3. Run full experiments: python scripts/train.py")
    print("\nSee docs/QUICKSTART.md for more information.")
    print()


if __name__ == "__main__":
    demo_sparsification()
