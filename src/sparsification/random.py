"""Symmetric random sparsification helpers.

Assigns the same random score to both directions (u,v) and (v,u) of each
undirected edge, ensuring the sparsified graph stays undirected. The fixed
ranking from `seed` is reused across all retention rates so that r=0.6 is
always a subset of r=0.8 — matching the nesting property of score-based
metrics like Jaccard.
"""

import numpy as np
from torch_geometric.data import Data


def precompute_random_scores(data: Data, seed: int = 42):
    """Assign a reproducible, symmetric random score to every undirected edge.

    Returns
    -------
    undirected_scores : np.ndarray of shape (n_unique_undirected_edges,)
    inverse_idx : np.ndarray mapping each directed edge to its
        undirected-edge index in `undirected_scores`.
    """
    ei = data.edge_index.cpu().numpy()
    src, dst = ei[0], ei[1]
    n = data.num_nodes
    u = np.minimum(src, dst)
    v = np.maximum(src, dst)
    keys = u.astype(np.int64) * (n + 1) + v.astype(np.int64)
    _, inverse_idx = np.unique(keys, return_inverse=True)
    n_undirected = int(inverse_idx.max()) + 1
    rng = np.random.default_rng(seed)
    undirected_scores = rng.random(n_undirected)
    return undirected_scores, inverse_idx


def random_sparsify(data: Data, undirected_scores, inverse_idx,
                    retention_ratio: float, device: str) -> Data:
    """Return a symmetrically sparsified copy of *data* (random baseline).

    Keeps the top-*retention_ratio* fraction of undirected edges by their
    pre-assigned random scores.
    """
    if retention_ratio == 1.0:
        return data.clone()
    n_undirected = len(undirected_scores)
    n_keep = max(1, int(n_undirected * retention_ratio))
    keep_undir = np.zeros(n_undirected, dtype=bool)
    keep_undir[np.argsort(undirected_scores)[-n_keep:]] = True
    mask = keep_undir[inverse_idx]
    sparse = data.clone()
    sparse.edge_index = data.edge_index[:, mask].to(device)
    return sparse
