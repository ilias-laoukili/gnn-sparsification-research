"""Placeholder for GAT model implementation.

TODO: Implement Graph Attention Network model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """Graph Attention Network for node classification.
    
    TODO: Complete this implementation based on your research needs.
    
    Example:
        >>> model = GAT(in_channels=1433, hidden_channels=64, out_channels=7, heads=8)
        >>> out = model(data.x, data.edge_index)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 8,
        output_heads: int = 1,
        dropout: float = 0.6,
        concat: bool = True,
    ):
        """Initialize GAT model.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden units per head
            out_channels: Number of output classes
            num_layers: Number of GAT layers
            heads: Number of attention heads
            output_heads: Number of attention heads in output layer
            dropout: Dropout rate
            concat: Whether to concatenate or average attention heads
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # TODO: Implement layer initialization
        # self.convs = nn.ModuleList()
        # self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=concat))
        # for _ in range(num_layers - 2):
        #     in_dim = hidden_channels * heads if concat else hidden_channels
        #     self.convs.append(GATConv(in_dim, hidden_channels, heads=heads, concat=concat))
        # in_dim = hidden_channels * heads if concat else hidden_channels
        # self.convs.append(GATConv(in_dim, out_channels, heads=output_heads, concat=False))
        
        raise NotImplementedError("GAT model not yet implemented")
    
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
        #     x = F.elu(x)
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.convs[-1](x, edge_index)
        # return x
        
        raise NotImplementedError("GAT forward pass not yet implemented")
