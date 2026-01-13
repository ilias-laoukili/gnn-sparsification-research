"""Topological analysis tools for graph sparsification."""

import networkx as nx
import numpy as np
from typing import Dict, Any, List, Tuple
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def analyze_sparsification_impact(
    data_original: Data, data_sparse: Data, detailed: bool = False
) -> Dict[str, Any]:
    """Compute topological changes from sparsification.

    Args:
        data_original: Original PyTorch Geometric Data object
        data_sparse: Sparsified Data object
        detailed: Whether to compute expensive metrics (diameter, betweenness)

    Returns:
        Dictionary with topological metrics and their changes
    """
    # Convert to NetworkX for analysis
    G_original = to_networkx(data_original, to_undirected=True)
    G_sparse = to_networkx(data_sparse, to_undirected=True)

    metrics = {
        # Basic metrics
        "nodes_original": G_original.number_of_nodes(),
        "nodes_sparse": G_sparse.number_of_nodes(),
        "edges_original": G_original.number_of_edges(),
        "edges_sparse": G_sparse.number_of_edges(),
        "edge_retention": G_sparse.number_of_edges() / G_original.number_of_edges(),
        # Connectivity
        "components_original": nx.number_connected_components(G_original),
        "components_sparse": nx.number_connected_components(G_sparse),
        "is_connected_original": nx.is_connected(G_original),
        "is_connected_sparse": nx.is_connected(G_sparse),
        # Clustering
        "clustering_original": nx.average_clustering(G_original),
        "clustering_sparse": nx.average_clustering(G_sparse),
        # Degree statistics
        "avg_degree_original": np.mean([d for n, d in G_original.degree()]),
        "avg_degree_sparse": np.mean([d for n, d in G_sparse.degree()]),
        "max_degree_original": max([d for n, d in G_original.degree()]),
        "max_degree_sparse": max([d for n, d in G_sparse.degree()]),
        # Density
        "density_original": nx.density(G_original),
        "density_sparse": nx.density(G_sparse),
    }

    # Compute changes
    metrics["clustering_change"] = metrics["clustering_sparse"] - metrics["clustering_original"]
    metrics["avg_degree_change"] = metrics["avg_degree_sparse"] - metrics["avg_degree_original"]
    metrics["density_change"] = metrics["density_sparse"] - metrics["density_original"]

    # Expensive metrics (only if detailed=True)
    if detailed:
        # Diameter (only for connected graphs)
        if nx.is_connected(G_original):
            metrics["diameter_original"] = nx.diameter(G_original)
        else:
            metrics["diameter_original"] = None

        if nx.is_connected(G_sparse):
            metrics["diameter_sparse"] = nx.diameter(G_sparse)
        else:
            metrics["diameter_sparse"] = None

        # Average shortest path length
        if nx.is_connected(G_original):
            metrics["avg_path_length_original"] = nx.average_shortest_path_length(G_original)
        else:
            # Compute for largest component
            largest_cc = max(nx.connected_components(G_original), key=len)
            metrics["avg_path_length_original"] = nx.average_shortest_path_length(
                G_original.subgraph(largest_cc)
            )

        if nx.is_connected(G_sparse):
            metrics["avg_path_length_sparse"] = nx.average_shortest_path_length(G_sparse)
        else:
            largest_cc = max(nx.connected_components(G_sparse), key=len)
            metrics["avg_path_length_sparse"] = nx.average_shortest_path_length(
                G_sparse.subgraph(largest_cc)
            )

    return metrics


def compute_degree_distribution(data: Data, bins: int = 20) -> Dict[str, np.ndarray]:
    """Compute degree distribution of a graph.

    Args:
        data: PyTorch Geometric Data object
        bins: Number of bins for histogram

    Returns:
        Dictionary with degree distribution statistics
    """
    G = to_networkx(data, to_undirected=True)
    degrees = np.array([d for n, d in G.degree()])

    hist, bin_edges = np.histogram(degrees, bins=bins)

    return {
        "degrees": degrees,
        "histogram": hist,
        "bin_edges": bin_edges,
        "mean": np.mean(degrees),
        "median": np.median(degrees),
        "std": np.std(degrees),
        "min": np.min(degrees),
        "max": np.max(degrees),
        "q25": np.percentile(degrees, 25),
        "q75": np.percentile(degrees, 75),
    }


def compute_edge_overlap(data1: Data, data2: Data) -> Dict[str, float]:
    """Compute edge overlap between two graphs.

    Args:
        data1: First PyTorch Geometric Data object
        data2: Second Data object

    Returns:
        Dictionary with overlap statistics
    """
    # Convert edge indices to sets of tuples
    edges1 = set(tuple(sorted([u.item(), v.item()])) for u, v in data1.edge_index.t())
    edges2 = set(tuple(sorted([u.item(), v.item()])) for u, v in data2.edge_index.t())

    intersection = edges1 & edges2
    union = edges1 | edges2

    return {
        "edges_1": len(edges1),
        "edges_2": len(edges2),
        "intersection": len(intersection),
        "union": len(union),
        "jaccard": len(intersection) / len(union) if len(union) > 0 else 0.0,
        "overlap_with_1": len(intersection) / len(edges1) if len(edges1) > 0 else 0.0,
        "overlap_with_2": len(intersection) / len(edges2) if len(edges2) > 0 else 0.0,
    }


def compute_spectral_properties(data: Data, k: int = 10) -> Dict[str, Any]:
    """Compute spectral properties of graph Laplacian.

    Args:
        data: PyTorch Geometric Data object
        k: Number of eigenvalues to compute

    Returns:
        Dictionary with spectral properties
    """
    from scipy.sparse.linalg import eigsh

    # Convert to adjacency matrix
    edge_index = data.edge_index.cpu().numpy()
    n = data.num_nodes
    adj = sp.csr_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(n, n)
    )

    # Compute Laplacian
    degrees = np.array(adj.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    L = D - adj

    # Compute smallest k eigenvalues
    k = min(k, n - 2)  # eigsh requires k < n-1

    try:
        eigenvalues, eigenvectors = eigsh(L, k=k, which="SM")

        return {
            "eigenvalues": eigenvalues.tolist(),
            "spectral_gap": float(eigenvalues[1] - eigenvalues[0]),
            "algebraic_connectivity": float(eigenvalues[1]),  # Fiedler value
            "num_zero_eigenvalues": int(np.sum(np.abs(eigenvalues) < 1e-6)),
        }
    except Exception as e:
        return {
            "error": str(e),
            "eigenvalues": None,
            "spectral_gap": None,
            "algebraic_connectivity": None,
            "num_zero_eigenvalues": None,
        }


def compute_community_structure(data: Data, method: str = "louvain") -> Dict[str, Any]:
    """Analyze community structure of a graph.

    Args:
        data: PyTorch Geometric Data object
        method: Community detection method ('louvain', 'label_propagation')

    Returns:
        Dictionary with community structure metrics
    """
    G = to_networkx(data, to_undirected=True)

    if method == "louvain":
        try:
            import community as community_louvain

            partition = community_louvain.best_partition(G)
            modularity = community_louvain.modularity(partition, G)
        except ImportError:
            # Fallback to greedy modularity
            from networkx.algorithms import community

            communities = community.greedy_modularity_communities(G)
            partition = {node: i for i, comm in enumerate(communities) for node in comm}
            modularity = community.modularity(G, communities)

    elif method == "label_propagation":
        from networkx.algorithms import community

        communities = list(community.label_propagation_communities(G))
        partition = {node: i for i, comm in enumerate(communities) for node in comm}
        modularity = community.modularity(G, communities)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute statistics
    num_communities = len(set(partition.values()))
    community_sizes = [sum(1 for v in partition.values() if v == i) for i in range(num_communities)]

    return {
        "num_communities": num_communities,
        "modularity": float(modularity),
        "avg_community_size": float(np.mean(community_sizes)),
        "largest_community_size": int(max(community_sizes)),
        "smallest_community_size": int(min(community_sizes)),
        "community_size_std": float(np.std(community_sizes)),
    }


def compare_topological_properties(
    data_original: Data, sparsified_data_list: List[Tuple[str, Data]], detailed: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Compare topological properties across multiple sparsification methods.

    Args:
        data_original: Original graph
        sparsified_data_list: List of (method_name, sparsified_data) tuples
        detailed: Whether to compute expensive metrics

    Returns:
        Dictionary mapping method names to their topological analysis
    """
    results = {
        "original": analyze_sparsification_impact(data_original, data_original, detailed=detailed)
    }

    for method_name, data_sparse in sparsified_data_list:
        results[method_name] = analyze_sparsification_impact(
            data_original, data_sparse, detailed=detailed
        )

    return results
