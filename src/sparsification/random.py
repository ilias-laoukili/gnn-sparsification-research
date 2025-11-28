"""Random edge sparsification method.

This module implements a baseline sparsification strategy that randomly
removes edges from the graph. Useful as a baseline for comparison.
"""

from typing import Optional

import torch
from torch_geometric.data import Data

from .base import BaseSparsifier


class RandomSparsifier(BaseSparsifier):
    """Random edge removal sparsification.
    
    This sparsifier randomly selects edges to keep based on the specified
    sparsification ratio. It serves as a baseline method for comparison
    with more sophisticated sparsification strategies.
    
    Example:
        >>> sparsifier = RandomSparsifier(sparsification_ratio=0.5, seed=42)
        >>> sparse_data = sparsifier.sparsify(data)
    """
    
    def __init__(
        self,
        sparsification_ratio: float = 0.5,
        preserve_node_features: bool = True,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize the random sparsifier.
        
        Args:
            sparsification_ratio: Fraction of edges to retain (default: 0.5)
            preserve_node_features: Whether to keep node features (default: True)
            seed: Random seed for reproducibility (default: None)
            **kwargs: Additional configuration parameters
        """
        super().__init__(sparsification_ratio, preserve_node_features, **kwargs)
        self.seed = seed
    
    def sparsify(self, data: Data) -> Data:
        """Apply random sparsification to the graph.
        
        Args:
            data: PyTorch Geometric Data object containing the graph
            
        Returns:
            Sparsified Data object with randomly reduced edges
        """
        # Set random seed for reproducibility
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        else:
            generator = None
        
        # Get number of edges
        num_edges = data.edge_index.shape[1]
        num_edges_to_keep = int(num_edges * self.sparsification_ratio)
        
        # Randomly select edges to keep
        if generator is not None:
            perm = torch.randperm(num_edges, generator=generator)
        else:
            perm = torch.randperm(num_edges)
        
        edge_indices_to_keep = perm[:num_edges_to_keep]
        
        # Create new data object
        sparse_data = Data(
            x=data.x if self.preserve_node_features else None,
            edge_index=data.edge_index[:, edge_indices_to_keep],
            y=data.y if hasattr(data, 'y') else None,
            num_nodes=data.num_nodes,
        )
        
        # Copy additional attributes if they exist
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
        """String representation of the random sparsifier."""
        return (
            f"RandomSparsifier("
            f"sparsification_ratio={self.sparsification_ratio}, "
            f"seed={self.seed})"
        )
