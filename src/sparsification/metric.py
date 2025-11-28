"""Metric-based graph sparsification methods.

This module implements sparsification strategies based on edge similarity metrics
such as Jaccard similarity, cosine similarity, or other distance measures.
"""

from typing import Optional

import torch
from torch_geometric.data import Data

from .base import BaseSparsifier


class MetricSparsifier(BaseSparsifier):
    """Metric-based edge sparsification.
    
    This sparsifier uses similarity or distance metrics between node features
    to determine which edges to keep. Common metrics include Jaccard similarity,
    cosine similarity, and Euclidean distance.
    
    Example:
        >>> sparsifier = MetricSparsifier(
        ...     sparsification_ratio=0.5,
        ...     metric="jaccard"
        ... )
        >>> sparse_data = sparsifier.sparsify(data)
    """
    
    def __init__(
        self,
        sparsification_ratio: float = 0.5,
        preserve_node_features: bool = True,
        metric: str = "jaccard",
        threshold: Optional[float] = None,
        **kwargs
    ):
        """Initialize the metric sparsifier.
        
        Args:
            sparsification_ratio: Fraction of edges to retain (default: 0.5)
            preserve_node_features: Whether to keep node features (default: True)
            metric: Similarity metric to use - "jaccard", "cosine", or "euclidean"
            threshold: Optional similarity threshold (overrides ratio if set)
            **kwargs: Additional configuration parameters
        """
        super().__init__(sparsification_ratio, preserve_node_features, **kwargs)
        self.metric = metric
        self.threshold = threshold
        
        if metric not in ["jaccard", "cosine", "euclidean"]:
            raise ValueError(
                f"Unsupported metric: {metric}. "
                f"Choose from: 'jaccard', 'cosine', 'euclidean'"
            )
    
    def compute_jaccard_similarity(self, data: Data) -> torch.Tensor:
        """Compute Jaccard similarity for edges based on neighborhood overlap.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Tensor of Jaccard similarity scores for each edge
        """
        edge_index = data.edge_index
        row, col = edge_index
        
        # Build adjacency list
        adj_list = [set() for _ in range(data.num_nodes)]
        for i in range(edge_index.shape[1]):
            adj_list[row[i].item()].add(col[i].item())
        
        # Compute Jaccard similarity for each edge
        jaccard_scores = []
        for i in range(edge_index.shape[1]):
            src, dst = row[i].item(), col[i].item()
            neighbors_src = adj_list[src]
            neighbors_dst = adj_list[dst]
            
            intersection = len(neighbors_src & neighbors_dst)
            union = len(neighbors_src | neighbors_dst)
            
            jaccard = intersection / union if union > 0 else 0.0
            jaccard_scores.append(jaccard)
        
        return torch.tensor(jaccard_scores, dtype=torch.float)
    
    def compute_cosine_similarity(self, data: Data) -> torch.Tensor:
        """Compute cosine similarity between node features.
        
        Args:
            data: PyTorch Geometric Data object with node features
            
        Returns:
            Tensor of cosine similarity scores for each edge
        """
        if data.x is None:
            raise ValueError("Cosine similarity requires node features")
        
        edge_index = data.edge_index
        row, col = edge_index
        
        # Normalize features
        x_norm = torch.nn.functional.normalize(data.x, p=2, dim=1)
        
        # Compute cosine similarity for each edge
        cosine_scores = (x_norm[row] * x_norm[col]).sum(dim=1)
        
        return cosine_scores
    
    def sparsify(self, data: Data) -> Data:
        """Apply metric-based sparsification to the graph.
        
        Args:
            data: PyTorch Geometric Data object containing the graph
            
        Returns:
            Sparsified Data object with edges selected based on similarity metric
        """
        # Compute edge scores based on metric
        if self.metric == "jaccard":
            edge_scores = self.compute_jaccard_similarity(data)
        elif self.metric == "cosine":
            edge_scores = self.compute_cosine_similarity(data)
        elif self.metric == "euclidean":
            if data.x is None:
                raise ValueError("Euclidean distance requires node features")
            row, col = data.edge_index
            edge_scores = -torch.norm(data.x[row] - data.x[col], p=2, dim=1)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Select edges to keep
        if self.threshold is not None:
            # Use threshold-based selection
            edge_mask = edge_scores >= self.threshold
            edge_indices_to_keep = torch.nonzero(edge_mask).squeeze()
        else:
            # Use ratio-based selection
            num_edges = data.edge_index.shape[1]
            num_edges_to_keep = int(num_edges * self.sparsification_ratio)
            _, top_indices = torch.topk(edge_scores, num_edges_to_keep)
            edge_indices_to_keep = top_indices
        
        # Create new data object
        sparse_data = Data(
            x=data.x if self.preserve_node_features else None,
            edge_index=data.edge_index[:, edge_indices_to_keep],
            y=data.y if hasattr(data, 'y') else None,
            num_nodes=data.num_nodes,
        )
        
        # Copy additional attributes
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            sparse_data.edge_attr = data.edge_attr[edge_indices_to_keep]
        
        if hasattr(data, 'train_mask'):
            sparse_data.train_mask = data.train_mask
        if hasattr(data, 'val_mask'):
            sparse_data.val_mask = data.val_mask
        if hasattr(data, 'test_mask'):
            sparse_data.test_mask = data.test_mask
        
        return sparse_data
    
    def __repr__(self) -> str:
        """String representation of the metric sparsifier."""
        return (
            f"MetricSparsifier("
            f"sparsification_ratio={self.sparsification_ratio}, "
            f"metric={self.metric}, "
            f"threshold={self.threshold})"
        )
