"""Global Metric Backbone sparsification preserving geodesic distances.

The Metric Backbone B ⊆ G is defined such that for every pair of nodes (u,v),
the shortest path distance in B equals the shortest path distance in G.

Edge retention condition:
    Keep edge (u,v) iff: w_uv <= d_G(u,v) + epsilon

Where:
    - w_uv is the direct edge weight
    - d_G(u,v) is the shortest path distance via APSP
    - epsilon is floating-point tolerance

References:
    Serrano, M.Á., Boguñá, M., & Vespignani, A. (2009).
    "Extracting the multiscale backbone of complex weighted networks." PNAS.
"""

from typing import Dict, Tuple
import numpy as np
from numpy.typing import NDArray
import networkx as nx
from torch_geometric.data import Data
import torch


def compute_metric_backbone(
    data: Data,
    edge_weights: NDArray[np.float64],
    epsilon: float = 1e-9,
    verbose: bool = True,
) -> Tuple[Data, Dict]:
    """Compute the Global Metric Backbone using All-Pairs Shortest Paths (APSP).

    Retains edge (u,v) if and only if: w_uv <= d_G(u,v) + epsilon

    Args:
        data: PyTorch Geometric Data object
        edge_weights: Edge weights (dissimilarity/distance - higher = weaker).
                      Must be aligned with data.edge_index ordering.
        epsilon: Floating-point tolerance (default: 1e-9)
        verbose: Print progress and statistics

    Returns:
        Tuple of (sparsified_data, statistics_dict)

    Complexity:
        Time:  O(n * E * log(n)) for Dijkstra from all sources
        Space: O(n²) for storing pairwise distances

        For large graphs (n > 10,000), consider approximate methods.

    Example:
        >>> dissimilarity = 1.0 - similarity_scores
        >>> sparse_data, stats = compute_metric_backbone(data, dissimilarity)
        >>> print(f"Retained {stats['retention_ratio']:.1%} of edges")
    """
    edge_index = data.edge_index.cpu().numpy()
    rows, cols = edge_index[0], edge_index[1]
    n_nodes = data.num_nodes
    n_edges = len(rows)

    if verbose:
        print(f"Computing Global Metric Backbone")
        print(f"  Nodes: {n_nodes:,}, Edges: {n_edges:,}")
        print(f"  Epsilon: {epsilon}")

    # Step 1: Build weighted NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    for idx, (u, v) in enumerate(zip(rows, cols)):
        weight = edge_weights[idx]
        if u < v:  # Undirected: add each edge once
            if G.has_edge(u, v):
                G[u][v]['weight'] = min(G[u][v]['weight'], weight)
            else:
                G.add_edge(u, v, weight=weight)

    if verbose:
        print(f"  Unique undirected edges: {G.number_of_edges():,}")
        print(f"  Computing APSP via Dijkstra... ", end="", flush=True)

    # Step 2: Compute All-Pairs Shortest Paths
    dist_matrix = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    if verbose:
        print("Done!")
        print(f"  Classifying edges...")

    # Step 3: Classify edges as metric or semi-metric
    keep_mask = np.zeros(n_edges, dtype=bool)
    edges_metric = 0
    edges_semi_metric = 0

    for idx, (u, v) in enumerate(zip(rows, cols)):
        direct_weight = edge_weights[idx]
        shortest_dist = dist_matrix.get(u, {}).get(v, np.inf)

        if shortest_dist == np.inf:
            # Disconnected nodes - keep edge (bridge)
            keep_mask[idx] = True
            edges_metric += 1
        elif direct_weight <= shortest_dist + epsilon:
            # Metric edge: lies on a shortest path
            keep_mask[idx] = True
            edges_metric += 1
        else:
            # Semi-metric edge: shorter alternative exists
            edges_semi_metric += 1

    # Step 4: Create sparsified graph
    sparse_edge_index = torch.from_numpy(edge_index[:, keep_mask])
    sparse_data = data.clone()
    sparse_data.edge_index = sparse_edge_index
    sparse_weights = edge_weights[keep_mask]

    stats = {
        "original_edges": n_edges,
        "retained_edges": int(keep_mask.sum()),
        "removed_edges": int((~keep_mask).sum()),
        "retention_ratio": float(keep_mask.sum() / n_edges),
        "edges_metric": edges_metric,
        "edges_semi_metric": edges_semi_metric,
        "epsilon": epsilon,
        "sparse_weights": sparse_weights,
        "keep_mask": keep_mask,
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"Metric Backbone Results")
        print(f"{'='*50}")
        print(f"  Original edges:        {stats['original_edges']:,}")
        print(f"  Metric (retained):     {stats['retained_edges']:,} ({stats['retention_ratio']:.1%})")
        print(f"  Semi-metric (removed): {stats['removed_edges']:,}")

    return sparse_data, stats


def verify_geodesic_preservation(
    original_data: Data,
    sparse_data: Data,
    original_weights: NDArray[np.float64],
    sparse_weights: NDArray[np.float64],
    n_samples: int = 500,
    epsilon: float = 1e-6,
    seed: int = 42,
) -> Dict:
    """Verify that the Metric Backbone preserves all geodesic distances.

    For a valid Metric Backbone: d_B(u,v) = d_G(u,v) for all node pairs.

    Args:
        original_data: Original PyG Data object
        sparse_data: Sparsified Metric Backbone Data object
        original_weights: Edge weights for original graph
        sparse_weights: Edge weights for backbone
        n_samples: Number of random node pairs to test
        epsilon: Tolerance for floating-point comparison
        seed: Random seed for reproducibility

    Returns:
        Dictionary with verification results:
        - pairs_tested: Number of node pairs tested
        - verified_equal: Pairs with preserved distances
        - violations: Pairs where distances differ
        - geodesic_preserved: Boolean indicating success
    """
    def build_nx_graph(data: Data, weights: NDArray[np.float64]) -> nx.Graph:
        """Build weighted NetworkX graph from PyG data."""
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        edge_index = data.edge_index.cpu().numpy()
        for idx, (u, v) in enumerate(zip(edge_index[0], edge_index[1])):
            if u < v:
                w = weights[idx]
                if G.has_edge(u, v):
                    G[u][v]['weight'] = min(G[u][v]['weight'], w)
                else:
                    G.add_edge(u, v, weight=w)
        return G

    G_original = build_nx_graph(original_data, original_weights)
    G_backbone = build_nx_graph(sparse_data, sparse_weights)

    # Sample random node pairs
    rng = np.random.default_rng(seed)
    n = original_data.num_nodes
    pairs = set()
    while len(pairs) < n_samples:
        u, v = rng.integers(0, n, size=2)
        if u != v:
            pairs.add((min(u, v), max(u, v)))
    pairs = list(pairs)

    violations = []
    verified = 0
    unreachable_original = 0
    unreachable_backbone = 0

    for u, v in pairs:
        try:
            d_orig = nx.shortest_path_length(G_original, u, v, weight='weight')
        except nx.NetworkXNoPath:
            unreachable_original += 1
            continue

        try:
            d_back = nx.shortest_path_length(G_backbone, u, v, weight='weight')
        except nx.NetworkXNoPath:
            unreachable_backbone += 1
            violations.append((u, v, d_orig, float('inf'), float('inf')))
            continue

        diff = abs(d_orig - d_back)
        if diff > epsilon:
            violations.append((u, v, d_orig, d_back, diff))
        else:
            verified += 1

    return {
        "pairs_tested": len(pairs),
        "verified_equal": verified,
        "violations": len(violations),
        "unreachable_original": unreachable_original,
        "unreachable_backbone": unreachable_backbone,
        "max_violation": max([v[4] for v in violations]) if violations else 0.0,
        "geodesic_preserved": len(violations) == 0,
        "violation_details": violations[:5],
    }
