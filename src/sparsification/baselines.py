"""Baseline sparsification methods for comparison.

Provides standard baseline methods to validate that metric-based approaches
offer improvements over simple heuristics.
"""

from typing import Tuple, Dict
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
import networkx as nx


class SparsificationBaselines:
    """Baseline sparsification methods for experimental comparison."""

    @staticmethod
    def random_sparsify(
        data: Data, retention_ratio: float = 0.5, seed: int = 42
    ) -> Tuple[Data, Dict]:
        """Random edge removal baseline.

        Args:
            data: PyTorch Geometric Data object
            retention_ratio: Fraction of edges to keep [0, 1]
            seed: Random seed for reproducibility

        Returns:
            (sparsified_data, statistics_dict)
        """
        rng = np.random.default_rng(seed)
        num_edges = data.edge_index.size(1)
        num_keep = int(num_edges * retention_ratio)

        # Random edge selection
        indices = rng.choice(num_edges, size=num_keep, replace=False)
        mask = np.zeros(num_edges, dtype=bool)
        mask[indices] = True

        sparse_data = data.clone()
        sparse_data.edge_index = data.edge_index[:, mask]

        stats = {
            "method": "random",
            "original_edges": num_edges,
            "retained_edges": num_keep,
            "retention_ratio": retention_ratio,
            "seed": seed,
        }

        return sparse_data, stats

    @staticmethod
    def degree_based_sparsify(
        data: Data, retention_ratio: float = 0.5, strategy: str = "low_first"
    ) -> Tuple[Data, Dict]:
        """Remove edges connected to low-degree nodes first.

        Args:
            data: PyTorch Geometric Data object
            retention_ratio: Fraction of edges to keep
            strategy: 'low_first' (remove low-degree edges) or
                     'high_first' (remove hub edges)

        Returns:
            (sparsified_data, statistics_dict)
        """
        edge_index = data.edge_index.cpu().numpy()
        num_edges = edge_index.shape[1]

        # Compute node degrees
        degrees = np.bincount(edge_index.flatten(), minlength=data.num_nodes)

        # Compute edge scores (sum of endpoint degrees)
        edge_scores = degrees[edge_index[0]] + degrees[edge_index[1]]

        # Select edges based on strategy
        num_keep = int(num_edges * retention_ratio)
        if strategy == "low_first":
            # Keep high-degree edges (remove low-degree edges)
            top_indices = np.argsort(edge_scores)[-num_keep:]
        elif strategy == "high_first":
            # Keep low-degree edges (remove hub edges)
            top_indices = np.argsort(edge_scores)[:num_keep]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        mask = np.zeros(num_edges, dtype=bool)
        mask[top_indices] = True

        sparse_data = data.clone()
        sparse_data.edge_index = torch.from_numpy(edge_index[:, mask])

        stats = {
            "method": f"degree_{strategy}",
            "original_edges": num_edges,
            "retained_edges": num_keep,
            "retention_ratio": retention_ratio,
        }

        return sparse_data, stats

    @staticmethod
    def betweenness_based_sparsify(
        data: Data, retention_ratio: float = 0.5, sample_fraction: float = 0.1
    ) -> Tuple[Data, Dict]:
        """Remove edges with lowest betweenness centrality.

        Note: Exact betweenness is O(VE), so we use sampling for large graphs.

        Args:
            data: PyTorch Geometric Data object
            retention_ratio: Fraction of edges to keep
            sample_fraction: Fraction of nodes to sample for centrality estimation

        Returns:
            (sparsified_data, statistics_dict)
        """
        # Convert to NetworkX
        edge_index = data.edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        G.add_edges_from(zip(edge_index[0], edge_index[1]))

        # Compute edge betweenness (with sampling for large graphs)
        if data.num_nodes > 1000:
            k = max(10, int(data.num_nodes * sample_fraction))
            edge_bc = nx.edge_betweenness_centrality(G, k=k, normalized=True)
        else:
            edge_bc = nx.edge_betweenness_centrality(G, normalized=True)

        # Map edge betweenness to edge_index order
        edge_scores = np.array(
            [
                edge_bc.get((u, v), edge_bc.get((v, u), 0.0))
                for u, v in zip(edge_index[0], edge_index[1])
            ]
        )

        # Keep high-betweenness edges (remove redundant ones)
        num_keep = int(len(edge_scores) * retention_ratio)
        top_indices = np.argsort(edge_scores)[-num_keep:]

        mask = np.zeros(len(edge_scores), dtype=bool)
        mask[top_indices] = True

        sparse_data = data.clone()
        sparse_data.edge_index = torch.from_numpy(edge_index[:, mask])

        stats = {
            "method": "betweenness",
            "original_edges": len(edge_scores),
            "retained_edges": num_keep,
            "retention_ratio": retention_ratio,
            "sample_fraction": sample_fraction,
        }

        return sparse_data, stats

    @staticmethod
    def dropedge_sparsify(data: Data, drop_rate: float = 0.5, seed: int = 42) -> Tuple[Data, Dict]:
        """DropEdge baseline (Rong et al., 2020).

        Random edge dropping used as data augmentation in GNN training.

        Args:
            data: PyTorch Geometric Data object
            drop_rate: Fraction of edges to drop [0, 1]
            seed: Random seed

        Returns:
            (augmented_data, statistics_dict)

        Reference:
            Rong, Y., Huang, W., Xu, T., & Huang, J. (2020).
            DropEdge: Towards Deep Graph Convolutional Networks on Node Classification.
            ICLR 2020.
        """
        sparse_data, stats = SparsificationBaselines.random_sparsify(
            data, retention_ratio=1.0 - drop_rate, seed=seed
        )
        stats["method"] = "dropedge"
        stats["drop_rate"] = drop_rate
        return sparse_data, stats
