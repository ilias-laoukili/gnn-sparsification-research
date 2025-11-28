"""Abstract base class for graph sparsification methods.

This module defines the interface that all sparsification methods must implement,
ensuring consistency across different sparsification strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from torch_geometric.data import Data


class BaseSparsifier(ABC):
    """Abstract base class for graph sparsification methods.
    
    All sparsification methods should inherit from this class and implement
    the `sparsify` method. This ensures a consistent interface for comparing
    different sparsification strategies.
    
    Attributes:
        sparsification_ratio: Target ratio of edges to keep (0.0 to 1.0)
        preserve_node_features: Whether to preserve original node features
        config: Additional configuration parameters
    """
    
    def __init__(
        self,
        sparsification_ratio: float = 0.5,
        preserve_node_features: bool = True,
        **kwargs
    ):
        """Initialize the base sparsifier.
        
        Args:
            sparsification_ratio: Fraction of edges to retain (default: 0.5)
            preserve_node_features: Whether to keep node features (default: True)
            **kwargs: Additional configuration parameters
        """
        if not 0.0 <= sparsification_ratio <= 1.0:
            raise ValueError(
                f"sparsification_ratio must be between 0.0 and 1.0, "
                f"got {sparsification_ratio}"
            )
        
        self.sparsification_ratio = sparsification_ratio
        self.preserve_node_features = preserve_node_features
        self.config = kwargs
    
    @abstractmethod
    def sparsify(self, data: Data) -> Data:
        """Apply sparsification to a PyG Data object.
        
        This method must be implemented by all concrete sparsification classes.
        It should return a new Data object with reduced edges while preserving
        the graph's essential structure.
        
        Args:
            data: PyTorch Geometric Data object containing the graph
            
        Returns:
            Sparsified Data object with reduced edge_index
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the sparsify() method"
        )
    
    def get_sparsification_stats(self, original_data: Data, sparsified_data: Data) -> Dict[str, Any]:
        """Compute statistics about the sparsification operation.
        
        Args:
            original_data: Original graph data
            sparsified_data: Sparsified graph data
            
        Returns:
            Dictionary containing sparsification statistics
        """
        orig_edges = original_data.edge_index.shape[1]
        sparse_edges = sparsified_data.edge_index.shape[1]
        
        return {
            "original_edges": orig_edges,
            "sparsified_edges": sparse_edges,
            "edges_removed": orig_edges - sparse_edges,
            "actual_sparsification_ratio": sparse_edges / orig_edges if orig_edges > 0 else 0.0,
            "target_sparsification_ratio": self.sparsification_ratio,
        }
    
    def __repr__(self) -> str:
        """String representation of the sparsifier."""
        return (
            f"{self.__class__.__name__}("
            f"sparsification_ratio={self.sparsification_ratio}, "
            f"preserve_node_features={self.preserve_node_features})"
        )
