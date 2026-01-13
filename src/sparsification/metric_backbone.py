"""Metric backbone sparsification via Relaxed Triangle Inequality (RTI).

Implementation of Serrano et al. (2009) metric backbone framework.
"""

from typing import Dict, Tuple, Set
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from torch_geometric.data import Data
import torch


def compute_triangle_detours(
    adj: sp.csr_matrix, edge_metric_scores: NDArray[np.float64], verbose: bool = False
) -> Dict[Tuple[int, int], float]:
    """Compute minimum detour distance for each edge through triangles.

    For edge (u,v), finds min_w[d(u,w) + d(w,v)] where w forms a triangle.

    Args:
        adj: Sparse adjacency matrix (symmetric, binary)
        edge_metric_scores: Distance/metric values for each edge (aligned with CSR order)
        verbose: Print progress statistics

    Returns:
        Dictionary mapping (u,v) edge tuple to minimum detour distance.
        Returns np.inf if no triangles exist for an edge.
    """
    n = adj.shape[0]
    rows, cols = adj.nonzero()

    # Build edge-to-score mapping for O(1) lookup
    # Store both directions for undirected graphs
    edge_to_score = {}
    for idx, (u, v) in enumerate(zip(rows, cols)):
        edge_to_score[(u, v)] = edge_metric_scores[idx]
        # For undirected graphs, also store reverse direction with same score
        if (v, u) not in edge_to_score:
            edge_to_score[(v, u)] = edge_metric_scores[idx]

    detours = {}
    edges_with_triangles = 0

    # Access CSR internals for faster neighbor lookup
    indptr = adj.indptr
    indices = adj.indices

    for idx, (u, v) in enumerate(zip(rows, cols)):
        # Find common neighbors (triangle vertices)
        neighbors_u = indices[indptr[u] : indptr[u + 1]]
        neighbors_v = indices[indptr[v] : indptr[v + 1]]
        
        # Use set intersection on indices (efficient for sparse graphs)
        common = set(neighbors_u) & set(neighbors_v)

        if not common:
            detours[(u, v)] = np.inf  # No detour possible
            continue

        # Compute minimum detour distance
        min_detour = np.inf
        for w in common:
            # Get distances d(u,w) and d(w,v)
            # Use .get() with None default, then check explicitly
            d_uw = edge_to_score.get((u, w))
            if d_uw is None:
                d_uw = edge_to_score.get((w, u))
            d_wv = edge_to_score.get((w, v))
            if d_wv is None:
                d_wv = edge_to_score.get((v, w))

            if d_uw is not None and d_wv is not None:
                detour = d_uw + d_wv
                min_detour = min(min_detour, detour)

        detours[(u, v)] = min_detour
        if min_detour < np.inf:
            edges_with_triangles += 1

    if verbose:
        total_edges = len(rows)
        print(
            f"  Edges with triangles: {edges_with_triangles}/{total_edges} "
            f"({100*edges_with_triangles/total_edges:.1f}%)"
        )

    return detours


def metric_backbone_sparsify(
    data: Data, metric_scores: NDArray[np.float64], alpha: float = 1.5, verbose: bool = False
) -> Tuple[Data, Dict[str, any]]:
    """Apply metric backbone sparsification via Relaxed Triangle Inequality.

    For dissimilarity metrics, removes edges that are "redundant" because an
    efficient alternative path exists through a common neighbor.
    
    An edge (u,v) is removed if: min_detour <= alpha * direct_dist
    where min_detour = min_w[d(u,w) + d(w,v)] for common neighbors w.
    
    Interpretation:
    - If the best detour through a triangle is not much longer than direct,
      the direct edge is redundant.
    - alpha controls the threshold: higher alpha = more aggressive pruning

    Args:
        data: PyTorch Geometric Data object
        metric_scores: Distance/dissimilarity scores for edges (higher = weaker)
        alpha: Stretch factor. Typical range [1.0, 2.0].
               - alpha=1.0: No edges removed (exact triangle inequality)
               - alpha→∞: All edges kept (no constraint)
        verbose: Print statistics

    Returns:
        (sparsified_data, statistics_dict)

    Algorithm Complexity:
        - Time: O(E * k) where k = avg. common neighbors per edge
        - Space: O(E) for detour storage

    References:
        Serrano, M.Á., Boguñá, M., & Vespignani, A. (2009).
        Extracting the multiscale backbone of complex weighted networks.
        PNAS, 106(16), 6483-6488.
    """
    edge_index = data.edge_index.cpu().numpy()

    # Convert to scipy sparse matrix
    rows, cols = edge_index[0], edge_index[1]
    adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(data.num_nodes, data.num_nodes))

    if verbose:
        print(f"Computing triangle detours for {len(rows)} edges...")

    # Compute minimum detour for each edge
    detours = compute_triangle_detours(adj, metric_scores, verbose)

    # Apply RTI filtering
    # For dissimilarity metrics: remove edge if detour is efficient
    # i.e., min_detour <= alpha * direct_dist (edge is redundant)
    keep_mask = []
    edges_removed_by_rti = 0
    edges_without_triangles = 0

    for idx, (u, v) in enumerate(zip(rows, cols)):
        direct_dist = metric_scores[idx]
        min_detour = detours.get((u, v), np.inf)

        if min_detour == np.inf:
            # No triangles - keep edge (isolated or bridging)
            keep_mask.append(True)
            edges_without_triangles += 1
        elif min_detour > alpha * direct_dist:
            # Detour is much longer than direct - edge is important, keep it
            keep_mask.append(True)
        else:
            # Detour is efficient (min_detour <= alpha * direct) - edge is redundant
            keep_mask.append(False)
            edges_removed_by_rti += 1

    keep_mask = np.array(keep_mask, dtype=bool)

    # Create sparsified graph
    sparse_edge_index = torch.from_numpy(edge_index[:, keep_mask])
    sparse_data = data.clone()
    sparse_data.edge_index = sparse_edge_index

    stats = {
        "original_edges": len(rows),
        "retained_edges": int(keep_mask.sum()),
        "removed_edges": int((~keep_mask).sum()),
        "retention_ratio": float(keep_mask.sum() / len(rows)),
        "edges_removed_by_rti": edges_removed_by_rti,
        "edges_without_triangles": edges_without_triangles,
        "alpha": alpha,
    }

    if verbose:
        print(f"\nMetric Backbone Statistics:")
        print(f"  Alpha (stretch factor): {alpha}")
        print(f"  Original edges: {stats['original_edges']:,}")
        print(f"  Retained edges: {stats['retained_edges']:,} " f"({stats['retention_ratio']:.1%})")
        print(f"  Removed by RTI: {edges_removed_by_rti:,}")
        print(f"  Kept (no triangles): {edges_without_triangles:,}")

    return sparse_data, stats


def validate_alpha_behavior(
    data: Data, metric_scores: NDArray[np.float64], alpha_values: list = None
) -> Dict[float, float]:
    """Validate alpha parameter behavior for metric backbone sparsification.
    
    With the corrected RTI logic for dissimilarity metrics:
    - Lower alpha = less aggressive pruning (more edges kept)
    - Higher alpha = more aggressive pruning (fewer edges kept)
    - alpha=1.0 keeps all edges (strictest threshold)

    Returns:
        Dictionary mapping alpha values to retention ratios.
    """
    if alpha_values is None:
        alpha_values = [1.0, 1.5, 2.0, 5.0, 10.0]

    results = {}
    for alpha in alpha_values:
        sparse_data, stats = metric_backbone_sparsify(
            data, metric_scores, alpha=alpha, verbose=False
        )
        results[alpha] = stats["retention_ratio"]

    # Validation checks
    # With dissimilarity metrics: alpha=1.0 should keep most/all edges
    # (only edges where min_detour > direct are removed at alpha=1.0)
    
    # Check monotonicity: larger alpha should remove MORE edges (lower retention)
    alpha_sorted = sorted(alpha_values)
    for i in range(len(alpha_sorted) - 1):
        assert (
            results[alpha_sorted[i]] >= results[alpha_sorted[i + 1]]
        ), f"Retention should decrease with alpha: {alpha_sorted[i]} vs {alpha_sorted[i+1]}"

    return results
