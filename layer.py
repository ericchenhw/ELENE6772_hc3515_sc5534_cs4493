import torch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, Linear, global_mean_pool
import torch.nn as nn
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.data import Data
import networkx as nx
from community import community_louvain
import torch.nn.functional as F
from torch.nn import Parameter


class MultiScaleGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, com_num, heads=1):
        super(MultiScaleGNN, self).__init__()
        # Ensemble of different GNN layers
        self.gcn = GCNConv(in_channels, out_channels)
        self.gat = GATConv(in_channels, out_channels, heads=heads, concat=False)
        self.sage = SAGEConv(in_channels, out_channels)
        self.gin = GINConv(Linear(in_channels, out_channels))
        self.linear = Linear(in_channels, out_channels)

        # Multi-scale learning components
        self.global_feature = global_mean_pool
        self.community_transform = Linear(in_channels, out_channels)
        self.global_transform = Linear(in_channels, out_channels)
        self.community_weight = Parameter(torch.Tensor(com_num, 1))
        nn.init.kaiming_normal_(self.community_weight)

    def forward(self, x, edge_index, batch):
        # Feature extraction using different GNN methods
        gcn_out = self.gcn(x, edge_index)
        gat_out = self.gat(x, edge_index)
        sage_out = self.sage(x, edge_index)
        gin_out = self.gin(x, edge_index)
        linear_out = self.linear(x)

        # Global-level features
        whole_batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        global_out = global_mean_pool(x, whole_batch).repeat(x.size(0), 1)
        global_out = self.global_transform(global_out)

        # Community-level features
        community_out = global_mean_pool(x, batch)
        community_out = self.community_transform(community_out)
        community_out = community_out[batch]

        # Multi-scale feature aggregation (60/30/10 ratio)
        out = 0.6 * ((F.relu(gcn_out) + F.relu(gat_out) + F.relu(sage_out) + F.relu(gin_out) + F.relu(linear_out)) / 5) \
              + 0.3 * F.relu(community_out) \
              + 0.1 * F.relu(global_out)

        return out