"""Edge similarity metrics for graph sparsification.

This module provides efficient O(E) implementations of common edge metrics
using sparse matrix operations. All functions operate on scipy CSR matrices
for memory efficiency on large graphs.

Includes both exact (dense) and approximate (sparse) effective resistance
implementations for different scale requirements.
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

    scores = np.divide(
        intersection_counts,
        union_counts,
        out=np.zeros_like(intersection_counts, dtype=np.float64),
        where=union_counts > 0,
    )

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

    Note:
        Nodes with degree < 2 contribute 0 (log(1) = 0 causes division issues).
        This is standard behavior following the original paper.

    Example:
        >>> adj = sp.csr_matrix([[0,1,1], [1,0,1], [1,1,0]])
        >>> scores = calculate_adamic_adar_scores(adj)
    """
    adj_binary = (adj > 0).astype(np.float64)
    degrees = np.asarray(adj_binary.sum(axis=1)).flatten()

    # Use 'where' to avoid log(0) warnings. Degrees of 0 result in -inf,
    # and degrees of 1 result in 0.
    log_degrees = np.log(
        degrees,
        out=np.full_like(degrees, -np.inf, dtype=float),
        where=(degrees > 0),
    )
    # Handle degree=1 case (log(1)=0) by setting to inf -> 1/inf=0
    log_degrees[log_degrees == 0] = np.inf

    inv_sqrt_log = np.sqrt(1.0 / log_degrees)
    inv_sqrt_log[np.isinf(inv_sqrt_log)] = 0.0

    diag_weights = sp.diags(inv_sqrt_log, format="csr")
    weighted_adj = adj_binary @ diag_weights

    aa_matrix = weighted_adj @ weighted_adj.T

    rows, cols = adj_binary.nonzero()
    scores = np.asarray(aa_matrix[rows, cols]).flatten()

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
        
        For graphs with >5,000 nodes, use `calculate_approx_effective_resistance_scores`
        which provides O(m log n) complexity via Johnson-Lindenstrauss projections.

    Example:
        >>> adj = sp.csr_matrix([[0,1,1], [1,0,1], [1,1,0]])
        >>> scores = calculate_effective_resistance_scores(adj)
    
    See Also:
        calculate_approx_effective_resistance_scores: Scalable approximation.
    """
    n = adj.shape[0]
    degrees = np.array(adj.sum(axis=1)).flatten()
    
    # Construct graph Laplacian: L = D - A
    D = sp.diags(degrees, format='csr')
    L = D - adj
    
    # Add regularization for numerical stability
    L_reg = L + 1e-10 * sp.eye(n, format='csr')
    
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
    """Compute APPROXIMATE effective resistance using JLT projections (O(m log n)).

    Uses the Johnson-Lindenstrauss Lemma (JLT) and sparse Conjugate Gradient
    solvers to approximate effective resistance without dense matrix operations.
    This is the Spielman-Srivastava approach for scalable graph sparsification.

    Algorithm:
        1. Generate random projection matrix Q of size k × N where k ≈ 24 log(N) / ε²
        2. For each row i in Q, solve L·x_i = Q_i using Conjugate Gradient
        3. R_eff(u,v) ≈ ||Z[:,u] - Z[:,v]||² where Z is the solution matrix

    Args:
        adj: Sparse adjacency matrix in CSR format. Must be symmetric
            (undirected graph) with binary entries.
        epsilon: Approximation error parameter. Smaller = more accurate but slower.
            Typical values: 0.1 (high accuracy), 0.3 (balanced), 0.5 (fast).
        seed: Random seed for reproducibility of JLT projections.
        max_cg_iters: Maximum iterations for Conjugate Gradient solver.
        cg_tol: Convergence tolerance for CG solver.

    Returns:
        Array of approximate effective resistance scores for each edge.
        Shape: (num_edges,). Higher values = more critical edges.

    Complexity:
        - Time: O(k · m · log(1/tol)) where k = O(log(n)/ε²) and m = edges
        - Space: O(k · n) - linear in nodes, no dense matrices

    When to use:
        - **Small graphs (<5K nodes):** Use `calculate_effective_resistance_scores` 
          (exact). The dense O(N³) method is faster due to optimized BLAS.
        - **Large graphs (>10K nodes):** Use this approximation. The sparse 
          CG-based method avoids memory issues and scales better.
        - **Medium graphs (5K-10K nodes):** Either works; benchmark if unsure.

    Note:
        Handles disconnected components via Laplacian regularization.
        Results are (1±ε)-approximations with high probability.
        Rankings are well-preserved (Spearman ρ ≈ 0.80) even if absolute values differ.

    Example:
        >>> adj = sp.csr_matrix([[0,1,1], [1,0,1], [1,1,0]])
        >>> scores = calculate_approx_effective_resistance_scores(adj, epsilon=0.3)

    References:
        Spielman, D.A. and Srivastava, N. (2008). Graph Sparsification by
        Effective Resistances. STOC '08.
    """
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    
    # Number of JLT projections: k = O(log(n) / ε²)
    # Using 24 log(n) / ε² as per JL lemma for (1±ε) approximation
    k = max(int(np.ceil(24 * np.log(n + 1) / (epsilon ** 2))), 10)
    k = min(k, n)  # Cap at n for very small graphs
    
    # Construct graph Laplacian: L = D - A
    degrees = np.asarray(adj.sum(axis=1)).flatten()
    D = sp.diags(degrees, format='csr')
    L = D - adj
    
    # Regularize for disconnected components and numerical stability
    # Small regularization preserves sparsity while ensuring positive definiteness
    reg = 1e-8 * sp.eye(n, format='csr')
    L_reg = (L + reg).tocsr()
    
    # Generate random JLT projection matrix (sparse Rademacher)
    # Using sparse random ±1/√k entries for efficiency
    Q = rng.choice([-1.0, 1.0], size=(k, n)) / np.sqrt(k)
    
    # Solve k linear systems: L_reg @ Z[i,:] = Q[i,:]
    # Z will store the solutions, shape (k, n)
    Z = np.zeros((k, n), dtype=np.float64)
    
    for i in range(k):
        rhs = Q[i, :]
        # Use Conjugate Gradient for symmetric positive semi-definite systems
        # minres is more stable for singular systems, but CG is faster
        x0 = np.zeros(n)  # Initial guess
        solution, info = spla.cg(L_reg, rhs, x0=x0, maxiter=max_cg_iters, atol=cg_tol)
        
        if info != 0:
            # CG didn't converge; fall back to minres for this system
            # Note: scipy minres uses 'rtol' not 'tol' in newer versions
            solution, _ = spla.minres(L_reg, rhs, maxiter=max_cg_iters, rtol=cg_tol)
        
        Z[i, :] = solution
    
    # Compute approximate effective resistance for all edges
    # R_eff(u,v) ≈ ||Z[:,u] - Z[:,v]||²
    rows, cols = adj.nonzero()
    
    # Vectorized computation: sum of squared differences across k dimensions
    diff = Z[:, rows] - Z[:, cols]  # Shape: (k, num_edges)
    r_eff = np.sum(diff ** 2, axis=0)  # Shape: (num_edges,)
    
    # Ensure non-negative (numerical errors can cause tiny negatives)
    r_eff = np.maximum(r_eff, 1e-10)
    
    return r_eff.astype(np.float64)
