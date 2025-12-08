"""Edge similarity metrics for graph sparsification.

This module provides efficient O(E) implementations of common edge metrics
using sparse matrix operations. All functions operate on scipy CSR matrices
for memory efficiency on large graphs.
"""

import numpy as np
import scipy.sparse as sp
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

    log_degrees = np.log(degrees)
    log_degrees[log_degrees == 0] = np.inf

    inv_sqrt_log = np.sqrt(1.0 / log_degrees)
    inv_sqrt_log[np.isinf(inv_sqrt_log)] = 0.0

    diag_weights = sp.diags(inv_sqrt_log, format="csr")
    weighted_adj = adj_binary @ diag_weights

    aa_matrix = weighted_adj @ weighted_adj.T

    rows, cols = adj_binary.nonzero()
    scores = np.asarray(aa_matrix[rows, cols]).flatten()

    return scores
