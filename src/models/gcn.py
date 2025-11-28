"""Placeholder for GCN model implementation.

TODO: Implement Graph Convolutional Network model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Graph Convolutional Network for node classification.
    
    TODO: Complete this implementation based on your research needs.
    
    Example:
        >>> model = GCN(in_channels=1433, hidden_channels=64, out_channels=7)
        >>> out = model(data.x, data.edge_index)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        normalize: bool = True,
    ):
        """Initialize GCN model.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden units
            out_channels: Number of output classes
            num_layers: Number of GCN layers
            dropout: Dropout rate
            normalize: Whether to normalize adjacency
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # TODO: Implement layer initialization
        # self.convs = nn.ModuleList()
        # self.convs.append(GCNConv(in_channels, hidden_channels, normalize=normalize))
        # for _ in range(num_layers - 2):
        #     self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=normalize))
        # self.convs.append(GCNConv(hidden_channels, out_channels, normalize=normalize))
        
        raise NotImplementedError("GCN model not yet implemented")
    
    def forward(self, x, edge_index):
        """Forward pass.
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)
            
        Returns:
            Output logits (num_nodes, out_channels)
        """
        # TODO: Implement forward pass
        # for i, conv in enumerate(self.convs[:-1]):
        #     x = conv(x, edge_index)
        #     x = F.relu(x)
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.convs[-1](x, edge_index)
        # return x
        
        raise NotImplementedError("GCN forward pass not yet implemented")
