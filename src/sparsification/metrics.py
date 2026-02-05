"""Edge similarity metrics for graph sparsification.

This module provides efficient O(E) implementations of common edge metrics
using sparse matrix operations. All functions operate on scipy CSR matrices
for memory efficiency on large graphs.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray
import networkx as nx


def calculate_jaccard_scores(adj: sp.csr_matrix) -> NDArray[np.float64]:
    """Compute Jaccard similarity for all edges in a graph.

    The Jaccard coefficient measures neighborhood overlap between connected
    nodes: J(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

    Implementation uses sparse matrix multiplication for O(E) complexity:
    - Intersection: (A @ A)[u,v] gives |N(u) ∩ N(v)|
    - Union: computed as deg(u) + deg(v) - intersection

    Args:
        adj: Sparse adjacency matrix in CSR format. Must be symmetric
            (undirected graph) with binary entries.

    Returns:
        Array of Jaccard scores for each edge, ordered by CSR indices.
        Shape: (num_edges,). Values in range [0, 1].

    Edge Cases:
        - Isolated nodes: Returns 0.0 (no neighbors to compare)
        - Degree-1 nodes: May return 1.0 if neighbors are identical

    Example:
        >>> adj = sp.csr_matrix([[0,1,1], [1,0,1], [1,1,0]])
        >>> scores = calculate_jaccard_scores(adj)
    """
    adj_binary = (adj > 0).astype(np.float64)
    degrees = np.asarray(adj_binary.sum(axis=1)).flatten()

    intersection = adj_binary @ adj_binary
    rows, cols = adj_binary.nonzero()

    intersection_counts = np.asarray(intersection[rows, cols]).flatten()
    union_counts = degrees[rows] + degrees[cols] - intersection_counts

    # **FIX:** Handle zero-degree nodes (isolated nodes)
    # If union is 0, both nodes are isolated → Jaccard = 0
    scores = np.divide(
        intersection_counts,
        union_counts,
        out=np.zeros_like(intersection_counts, dtype=np.float64),
        where=union_counts > 0,
    )

    # Ensure no NaN or inf values from numerical errors
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores


def calculate_adamic_adar_scores(adj: sp.csr_matrix) -> NDArray[np.float64]:
    """Compute Adamic-Adar index for all edges in a graph.

    The Adamic-Adar index weights common neighbors by their inverse log degree:
    AA(u,v) = Σ_{w ∈ N(u) ∩ N(v)} 1 / log(deg(w))

    Implementation uses the square-root decomposition trick for O(E) complexity:
    - Let D = diag(1/sqrt(log(deg)))
    - Then AA = (A @ D) @ (D @ A) = (A @ D)^2 evaluated at edge positions

    This avoids explicit neighbor enumeration by leveraging matrix multiplication.

    Args:
        adj: Sparse adjacency matrix in CSR format. Must be symmetric
            (undirected graph) with binary entries.

    Returns:
        Array of Adamic-Adar scores for each edge, ordered by CSR indices.
        Shape: (num_edges,). Higher values indicate more low-degree common neighbors.

    Edge Cases:
        - No common neighbors: Returns 0.0
        - Common neighbors with degree 1: Uses log(2) to avoid log(1)=0

    Note:
        **FIX:** Uses log(degree + 1) to avoid log(1) = 0 and division by zero.
        This handles degree-1 nodes gracefully.

    Example:
        >>> adj = sp.csr_matrix([[0,1,1], [1,0,1], [1,1,0]])
        >>> scores = calculate_adamic_adar_scores(adj)
    """
    adj_binary = (adj > 0).astype(np.float64)
    degrees = np.asarray(adj_binary.sum(axis=1)).flatten()

    # **FIX:** Use log(degree + 1) to avoid log(1) = 0 and division by zero
    # This handles degree-1 nodes gracefully
    log_degrees = np.log(degrees + 1)

    # Create diagonal matrix with 1/sqrt(log(deg+1))
    inv_sqrt_log = 1.0 / np.sqrt(log_degrees)

    # Handle any remaining inf/nan from numerical errors
    inv_sqrt_log[~np.isfinite(inv_sqrt_log)] = 0.0

    diag_weights = sp.diags(inv_sqrt_log, format="csr")
    weighted_adj = adj_binary @ diag_weights

    aa_matrix = weighted_adj @ weighted_adj.T

    rows, cols = adj_binary.nonzero()
    scores = np.asarray(aa_matrix[rows, cols]).flatten()

    # Ensure no NaN or inf values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores


def calculate_effective_resistance_scores(adj: sp.csr_matrix) -> NDArray[np.float64]:
    """Compute EXACT effective resistance for all edges (Dense, O(N³)).

    Effective resistance measures the "electrical distance" between nodes
    when the graph is viewed as an electrical network with unit resistances.
    High R_eff indicates critical edges (bridges, bottlenecks), while low
    R_eff indicates redundant edges (many alternative paths).

    Formula: R_eff(u,v) = L^+[u,u] + L^+[v,v] - 2*L^+[u,v]
    where L^+ is the Moore-Penrose pseudoinverse of the graph Laplacian.

    Args:
        adj: Sparse adjacency matrix in CSR format. Must be symmetric
            (undirected graph) with binary entries.

    Returns:
        Array of effective resistance scores for each edge.
        Shape: (num_edges,). Higher values = more critical edges.

    Warning:
        **EXACT (DENSE) IMPLEMENTATION - Use only for small benchmarks (<5k nodes).**

        This implementation computes the exact pseudoinverse via dense
        matrix inversion with:
        - Time complexity: O(N³)
        - Space complexity: O(N²)

        For larger graphs, consider using NetworkX's `resistance_distance` function
        which can be more efficient for computing individual edge resistances.

    Example:
        >>> adj = sp.csr_matrix([[0,1,1], [1,0,1], [1,1,0]])
        >>> scores = calculate_effective_resistance_scores(adj)
    """
    n = adj.shape[0]
    degrees = np.array(adj.sum(axis=1)).flatten()

    # Construct graph Laplacian: L = D - A
    D = sp.diags(degrees, format="csr")
    L = D - adj

    # Add regularization for numerical stability
    L_reg = L + 1e-10 * sp.eye(n, format="csr")

    # Compute Moore-Penrose pseudoinverse (dense; exact, small-scale only)
    L_pinv = np.linalg.pinv(L_reg.toarray())

    # Compute effective resistance for all edges using vectorized indexing
    rows, cols = adj.nonzero()
    r_eff = L_pinv[rows, rows] + L_pinv[cols, cols] - 2.0 * L_pinv[rows, cols]
    r_eff = np.maximum(r_eff, 1e-10)
    return r_eff.astype(np.float64)


def calculate_approx_effective_resistance_scores(
    adj: sp.csr_matrix,
    epsilon: float = 0.3,
    seed: int = 42,
    max_cg_iters: int = 500,
    cg_tol: float = 1e-6,
) -> NDArray[np.float64]:
    """Compute APPROXIMATE effective resistance using Spielman-Srivastava spectral sketching.

    This implements the edge-based Johnson-Lindenstrauss projection method that
    correctly approximates effective resistance R_eff = b^T L^+ b (not biharmonic
    distance which would be b^T (L^+)^2 b).

    The key insight is projecting through the incidence matrix B:
        Z = L^+ B R  →  ||Z[u] - Z[v]||² ≈ (e_u-e_v)^T L^+ B B^T L^+ (e_u-e_v)
                                         = (e_u-e_v)^T L^+ L L^+ (e_u-e_v)
                                         = (e_u-e_v)^T L^+ (e_u-e_v) = R_eff(u,v)

    Algorithm:
        1. Construct sparse incidence matrix B (n × m) for undirected edges
           where B[u,e] = +1, B[v,e] = -1 for edge e = (u,v)
        2. Generate random Gaussian R (m × k) scaled by 1/√k
        3. Compute Y = B @ R (project edges to node space)
        4. Solve k linear systems L @ Z = Y using sparse conjugate gradient
        5. R_eff(u,v) ≈ ||Z[u] - Z[v]||² (squared Euclidean distance)

    Time Complexity: O(m · k · CG_iters) ≈ O(m · log(n) / ε²)
    Space Complexity: O(n · k + m) = O(n · log(n) / ε² + m)

    Args:
        adj: Sparse adjacency matrix in CSR format. Must be symmetric
            (undirected graph) with binary entries.
        epsilon: Approximation error parameter. Smaller values give better
            accuracy but require more projections.
        seed: Random seed for reproducibility.
        max_cg_iters: Maximum conjugate gradient iterations per solve.
        cg_tol: Convergence tolerance for conjugate gradient.

    Returns:
        Array of approximate effective resistance scores for each edge.
        Shape matches adj.nonzero(). Higher values = more critical edges.

    Note:
        For graphs with <5k nodes, the exact method may be faster due to
        CG overhead. Use this for larger graphs where O(N³) is infeasible.

    References:
        - Spielman & Srivastava (2011): Graph Sparsification by Effective Resistances
        - Spielman & Teng (2004): Nearly-linear time algorithms for graph partitioning

    Example:
        >>> adj = sp.csr_matrix([[0,1,1], [1,0,1], [1,1,0]])
        >>> scores = calculate_approx_effective_resistance_scores(adj, epsilon=0.3)
    """
    rng = np.random.default_rng(seed)
    n = adj.shape[0]

    # Get all edges from adjacency matrix (includes both directions for undirected)
    all_rows, all_cols = adj.nonzero()

    # For incidence matrix, use only undirected edges (i < j to avoid duplicates)
    mask = all_rows < all_cols
    u_edges = all_rows[mask]
    v_edges = all_cols[mask]
    m = len(u_edges)  # number of undirected edges

    if m == 0:
        return np.zeros(len(all_rows), dtype=np.float64)

    # JL dimension: k = 24 * log(n) / ε² ensures (1±ε) approximation w.h.p.
    k = max(int(24 * np.log(max(n, 2)) / (epsilon**2)), 1)

    # Construct graph Laplacian: L = D - A
    degrees = np.array(adj.sum(axis=1)).flatten()
    D = sp.diags(degrees, format="csr")
    L = D - adj

    # Add small regularization for numerical stability
    L_reg = L + 1e-10 * sp.eye(n, format="csr")

    # Construct sparse incidence matrix B (n × m)
    # For edge i connecting (u, v): B[u, i] = +1, B[v, i] = -1
    B = sp.csr_matrix(
        (
            np.concatenate([np.ones(m), -np.ones(m)]),
            (
                np.concatenate([u_edges, v_edges]),
                np.concatenate([np.arange(m), np.arange(m)]),
            ),
        ),
        shape=(n, m),
    )

    # Generate random Gaussian projection R (m × k), scaled by 1/√k
    R = rng.standard_normal((m, k)) / np.sqrt(k)

    # Project through incidence matrix: Y = B @ R, shape (n, k)
    Y = B @ R

    # Solve L @ Z = Y using sparse CG, giving Z = L^+ @ B @ R
    Z = np.zeros((n, k), dtype=np.float64)
    for i in range(k):
        z, _ = spla.cg(L_reg, Y[:, i], maxiter=max_cg_iters, rtol=cg_tol)
        Z[:, i] = z

    # Compute effective resistance: R_eff(u,v) ≈ ||Z[u] - Z[v]||²
    diff = Z[all_rows] - Z[all_cols]
    r_eff = np.sum(diff**2, axis=1)

    # Ensure positive values
    r_eff = np.maximum(r_eff, 1e-10)
    return r_eff.astype(np.float64)


def compute_geodesic_preservation(
    original_adj: sp.csr_matrix,
    sparse_adj: sp.csr_matrix,
    n_samples: int = 500,
    seed: int = 42,
) -> Dict:
    """Compute geodesic (shortest path) preservation ratio between original and sparse graphs.

    Measures what fraction of sampled node pairs have preserved shortest path distances
    after sparsification. This works with unweighted graphs (hop distance).

    Args:
        original_adj: Original adjacency matrix (sparse CSR).
        sparse_adj: Sparsified adjacency matrix (sparse CSR).
        n_samples: Number of random node pairs to sample.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with:
        - preservation_ratio: Fraction of pairs with preserved distances [0, 1]
        - pairs_tested: Number of node pairs tested
        - preserved_count: Number of pairs with same distance
        - increased_count: Number of pairs with increased distance
        - disconnected_count: Number of pairs disconnected in sparse graph
        - avg_distance_increase: Average increase in distance for non-preserved pairs
    """
    n = original_adj.shape[0]
    rng = np.random.default_rng(seed)

    # Build NetworkX graphs for shortest path computation
    G_orig = nx.from_scipy_sparse_array(original_adj)
    G_sparse = nx.from_scipy_sparse_array(sparse_adj)

    # Sample random node pairs
    pairs = set()
    max_attempts = n_samples * 10
    attempts = 0
    while len(pairs) < n_samples and attempts < max_attempts:
        u, v = rng.integers(0, n, size=2)
        if u != v:
            pairs.add((min(u, v), max(u, v)))
        attempts += 1
    pairs = list(pairs)

    preserved = 0
    increased = 0
    disconnected = 0
    distance_increases = []

    for u, v in pairs:
        # Get original distance
        try:
            d_orig = nx.shortest_path_length(G_orig, u, v)
        except nx.NetworkXNoPath:
            # Already disconnected in original - skip
            continue

        # Get sparse distance
        try:
            d_sparse = nx.shortest_path_length(G_sparse, u, v)
        except nx.NetworkXNoPath:
            disconnected += 1
            continue

        if d_sparse == d_orig:
            preserved += 1
        else:
            increased += 1
            distance_increases.append(d_sparse - d_orig)

    total_valid = preserved + increased + disconnected
    preservation_ratio = preserved / total_valid if total_valid > 0 else 0.0

    return {
        "preservation_ratio": preservation_ratio,
        "pairs_tested": len(pairs),
        "preserved_count": preserved,
        "increased_count": increased,
        "disconnected_count": disconnected,
        "avg_distance_increase": np.mean(distance_increases) if distance_increases else 0.0,
        "max_distance_increase": max(distance_increases) if distance_increases else 0,
    }


def compute_topology_metrics(adj: sp.csr_matrix) -> Dict:
    """Compute topological metrics for a graph.

    Args:
        adj: Adjacency matrix (sparse CSR).

    Returns:
        Dictionary with:
        - num_nodes: Number of nodes
        - num_edges: Number of edges (undirected count)
        - avg_degree: Average node degree
        - clustering_coefficient: Global clustering coefficient
        - algebraic_connectivity: Second smallest eigenvalue of Laplacian (Fiedler value)
        - num_connected_components: Number of connected components
        - largest_component_ratio: Fraction of nodes in largest component
    """
    n = adj.shape[0]
    G = nx.from_scipy_sparse_array(adj)

    # Basic stats
    num_edges = G.number_of_edges()
    degrees = np.array([d for _, d in G.degree()])
    avg_degree = degrees.mean() if len(degrees) > 0 else 0.0

    # Clustering coefficient
    clustering = nx.average_clustering(G)

    # Connected components
    components = list(nx.connected_components(G))
    num_components = len(components)
    largest_component_size = max(len(c) for c in components) if components else 0
    largest_component_ratio = largest_component_size / n if n > 0 else 0.0

    # Algebraic connectivity (Fiedler value) - only for connected graphs
    # For disconnected graphs, use the largest connected component
    algebraic_connectivity = 0.0
    if num_components == 1 and n > 1:
        try:
            algebraic_connectivity = nx.algebraic_connectivity(G, method='tracemin_lu')
        except Exception:
            # Fall back to computing from Laplacian eigenvalues
            try:
                L = nx.laplacian_matrix(G).astype(np.float64)
                # Get second smallest eigenvalue
                eigenvalues = spla.eigsh(L, k=min(2, n-1), which='SM', return_eigenvectors=False)
                algebraic_connectivity = float(sorted(eigenvalues)[1]) if len(eigenvalues) > 1 else 0.0
            except Exception:
                algebraic_connectivity = 0.0
    elif num_components > 1:
        # For disconnected graphs, algebraic connectivity is 0
        algebraic_connectivity = 0.0

    return {
        "num_nodes": n,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "clustering_coefficient": clustering,
        "algebraic_connectivity": algebraic_connectivity,
        "num_connected_components": num_components,
        "largest_component_ratio": largest_component_ratio,
    }


def compute_topology_preservation(
    original_adj: sp.csr_matrix,
    sparse_adj: sp.csr_matrix,
) -> Dict:
    """Compute how well topology is preserved after sparsification.

    Args:
        original_adj: Original adjacency matrix (sparse CSR).
        sparse_adj: Sparsified adjacency matrix (sparse CSR).

    Returns:
        Dictionary with preservation ratios and absolute changes:
        - edge_retention: Fraction of edges retained
        - clustering_preservation: sparse_cc / original_cc
        - connectivity_preservation: sparse_ac / original_ac (or 0 if disconnected)
        - component_change: Change in number of connected components
        - original_metrics: Full metrics for original graph
        - sparse_metrics: Full metrics for sparse graph
    """
    orig_metrics = compute_topology_metrics(original_adj)
    sparse_metrics = compute_topology_metrics(sparse_adj)

    # Edge retention
    edge_retention = sparse_metrics["num_edges"] / orig_metrics["num_edges"] if orig_metrics["num_edges"] > 0 else 0.0

    # Clustering preservation
    clustering_preservation = (
        sparse_metrics["clustering_coefficient"] / orig_metrics["clustering_coefficient"]
        if orig_metrics["clustering_coefficient"] > 0 else 1.0
    )

    # Algebraic connectivity preservation
    connectivity_preservation = (
        sparse_metrics["algebraic_connectivity"] / orig_metrics["algebraic_connectivity"]
        if orig_metrics["algebraic_connectivity"] > 0 else 0.0
    )

    # Component change
    component_change = sparse_metrics["num_connected_components"] - orig_metrics["num_connected_components"]

    return {
        "edge_retention": edge_retention,
        "clustering_preservation": clustering_preservation,
        "connectivity_preservation": connectivity_preservation,
        "component_change": component_change,
        "original_metrics": orig_metrics,
        "sparse_metrics": sparse_metrics,
    }
