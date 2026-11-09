import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class DirSAGELayer(nn.Module):
    # One layer = one round of message passing, from drivers AND from readers.
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.from_drivers = SAGEConv(dim_in, dim_out)                      # includes W_self
        self.from_readers = SAGEConv(dim_in, dim_out, root_weight=False)   # W_self only once

    def forward(self, h, edge_index):
        a = self.from_drivers(h, edge_index)            # along the wires
        b = self.from_readers(h, edge_index.flip(0))    # against the wires
        return torch.relu(a + b)


class DirGraphSAGE(nn.Module):
    def __init__(self, n_features, hidden, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        dim = n_features
        for i in range(n_layers):
            self.layers.append(DirSAGELayer(dim, hidden))
            dim = hidden
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
        return torch.sigmoid(self.head(h)).squeeze(-1)