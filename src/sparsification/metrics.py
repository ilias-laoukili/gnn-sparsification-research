"""Edge similarity metrics for graph sparsification.

This module provides efficient O(E) implementations of common edge metrics
using sparse matrix operations. All functions operate on scipy CSR matrices
for memory efficiency on large graphs.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray


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
