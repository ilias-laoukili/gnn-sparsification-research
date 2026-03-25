"""Variable-depth GNN models for HPO experiments.

These models support configurable depth (num_layers), hidden size, and dropout,
unlike the fixed 2-layer models in gnn.py. Used by all HPO and transfer scripts.
"""

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class FlexibleGCN(nn.Module):
    """Variable-depth GCN with configurable hidden size and dropout."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, edge_weight=None):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)


class FlexibleSAGE(nn.Module):
    """Variable-depth GraphSAGE with mean aggregation."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, edge_weight=None):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class FlexibleGAT(nn.Module):
    """Variable-depth GAT with multi-head attention (heads=2 on hidden layers)."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout, heads=2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(
                GATConv(in_channels, out_channels, heads=1,
                        concat=False, dropout=dropout))
        else:
            self.convs.append(
                GATConv(in_channels, hidden_channels,
                        heads=heads, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_channels * heads, hidden_channels,
                            heads=heads, dropout=dropout))
            self.convs.append(
                GATConv(hidden_channels * heads, out_channels,
                        heads=1, concat=False, dropout=dropout))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, edge_weight=None):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class FlexibleMLP(nn.Module):
    """Pure MLP — identical HP space as FlexibleGCN, ignores edge_index.

    Zero-edge baseline: can only use node features.
    GCN(sparse) beating this proves retained edges carry structural signal.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(in_channels, out_channels))
        else:
            self.layers.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.layers.append(nn.Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data, edge_weight=None):
        x = data.x  # edge_index intentionally ignored
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return F.log_softmax(x, dim=1)


FLEX_REGISTRY = {
    "gcn":       FlexibleGCN,
    "graphsage": FlexibleSAGE,
    "gat":       FlexibleGAT,
    "mlp":       FlexibleMLP,
}
