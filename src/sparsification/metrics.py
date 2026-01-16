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
