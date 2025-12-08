"""Graph Neural Network architectures for node classification."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class BaseGNN(nn.Module, ABC):
    """Abstract base class for all GNN architectures.

    Provides common interface and parameter reset functionality for
    consistent model initialization across experiments.
    """

    @abstractmethod
    def forward(
        self, data: Data, edge_weight: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass through the network.

        Args:
            data: PyG Data object containing node features and edge index.
            edge_weight: Optional edge weights for message passing.

        Returns:
            Log-softmax predictions of shape (num_nodes, num_classes).
        """
        pass

    def reset_parameters(self) -> None:
        """Reinitialize all learnable parameters.

        Iterates through all submodules and calls their reset_parameters
        method if available. Essential for fair multi-run comparisons.
        """
        for module in self.modules():
            if hasattr(module, "reset_parameters") and module is not self:
                module.reset_parameters()


class GCN(BaseGNN):
    """Two-layer Graph Convolutional Network.

    Implements the architecture from Kipf & Welling (2017) with
    ReLU activation and dropout regularization. Supports edge weights.

    Args:
        in_channels: Dimensionality of input node features.
        hidden_channels: Number of hidden units.
        out_channels: Number of output classes.
        dropout: Dropout probability during training.

    Example:
        >>> model = GCN(1433, 64, 7)
        >>> out = model(data)  # (num_nodes, 7)
        >>> out = model(data, edge_weight=weights)  # With edge weights
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(
        self, data: Data, edge_weight: Optional[Tensor] = None
    ) -> Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)


class GraphSAGE(BaseGNN):
    """Two-layer GraphSAGE network with mean aggregation.

    Implements the inductive learning framework from Hamilton et al. (2017)
    designed for scalable node embedding generation.

    Args:
        in_channels: Dimensionality of input node features.
        hidden_channels: Number of hidden units.
        out_channels: Number of output classes.
        dropout: Dropout probability during training.

    Note:
        GraphSAGE does not natively support edge weights in its mean
        aggregation. Edge weights are ignored if passed.

    Example:
        >>> model = GraphSAGE(1433, 64, 7)
        >>> out = model(data)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(
        self, data: Data, edge_weight: Optional[Tensor] = None
    ) -> Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(BaseGNN):
    """Two-layer Graph Attention Network.

    Implements the attention mechanism from Veličković et al. (2018)
    with configurable multi-head attention.

    Args:
        in_channels: Dimensionality of input node features.
        hidden_channels: Number of hidden units per attention head.
        out_channels: Number of output classes.
        heads: Number of attention heads in first layer.
        dropout: Dropout probability for both features and attention.

    Note:
        GAT learns attention weights internally. External edge weights
        can be passed but are multiplied with learned attention scores.
        The second layer uses single-head attention with concatenation
        disabled to produce final class predictions.

    Example:
        >>> model = GAT(1433, 64, 7, heads=4)
        >>> out = model(data)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 2,
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=heads, dropout=dropout
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

    def forward(
        self, data: Data, edge_weight: Optional[Tensor] = None
    ) -> Tensor:
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


MODEL_REGISTRY = {
    "gcn": GCN,
    "graphsage": GraphSAGE,
    "sage": GraphSAGE,
    "gat": GAT,
}


def get_model(
    name: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    **kwargs,
) -> BaseGNN:
    """Factory function to instantiate GNN models by name.

    Args:
        name: Model identifier ('gcn', 'graphsage', 'sage', or 'gat').
        in_channels: Input feature dimensionality.
        hidden_channels: Hidden layer size.
        out_channels: Number of output classes.
        **kwargs: Additional model-specific arguments (e.g., heads for GAT).

    Returns:
        Instantiated GNN model.

    Raises:
        ValueError: If model name is not registered.
    """
    name_lower = name.lower()
    if name_lower not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name_lower](
        in_channels, hidden_channels, out_channels, **kwargs
    )
