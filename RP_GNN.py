import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from layer import MultiScaleGNN


class RP_GNN(nn.Module):
    def __init__(self, com_num=14, robustness_length=19, node_num=500):
        super(RP_GNN, self).__init__()
        self.Feature_Extraction_Module = Feature_Extraction_Module(com_num)
        self.Connectivity_Branch = Connectivity_Branch(robustness_length, node_num)
        self.Controllability_Branch = Controllability_Branch(robustness_length, node_num)

        physical_bias = 0.001 * torch.ones(robustness_length, requires_grad=True)
        physical_bias[(robustness_length // 2):] = 0
        self.physical_bias = nn.Parameter(physical_bias)

    def forward(self, X, edge_index, com_div):
        X = self.Feature_Extraction_Module(X, edge_index, com_div)
        X_embedding, connectivity = self.Connectivity_Branch(X, edge_index)
        X = torch.cat((X, X_embedding.unsqueeze(1)), dim=1)
        controllability = self.Controllability_Branch(X)
        
        connectivity = connectivity + self.physical_bias
        controllability = controllability + self.physical_bias

        return connectivity, controllability.squeeze()


class Residual_Block(nn.Module):
    def __init__(self, input_dim, output_dim, com_num):
        super(Residual_Block, self).__init__()
        self.gnn1 = MultiScaleGNN(input_dim, output_dim, com_num)
        self.gnn2 = MultiScaleGNN(output_dim, output_dim, com_num)

    def forward(self, X, edge_index, com_div):
        X1 = self.gnn1(X, edge_index, com_div)
        X2 = self.gnn2(X1, edge_index, com_div)
        X_res = X1 + X2  # Residual connection
        return X_res


class Feature_Extraction_Module(nn.Module):
    def __init__(self, com_num):
        super(Feature_Extraction_Module, self).__init__()
        self.block1 = Residual_Block(9, 16, com_num)
        self.block_res = Residual_Block(16, 16, com_num)
        self.block2 = Residual_Block(16, 8, com_num)

    def forward(self, X, edge_index, com_div):
        X1 = self.block1(X, edge_index, com_div)
        X_res = self.block_res(X1, edge_index, com_div)
        X1 = X1 + X_res  # Residual connection
        X2 = self.block2(X1, edge_index, com_div)
        return X2


class Connectivity_Branch(nn.Module):
    def __init__(self, robustness_length, node_num):
        super(Connectivity_Branch, self).__init__()
        self.sage = SAGEConv(8, 1)
        self.sage1 = SAGEConv(8, 1, aggr='max')
        self.linear = nn.Linear(node_num, 1000)
        self.linear1 = nn.Linear(1000, 1000)
        self.linear2 = nn.Linear(1000, robustness_length)

    def forward(self, X, edge_index):
        X_mean = F.relu(self.sage(X, edge_index))
        X_max = F.relu(self.sage1(X, edge_index))
        X = X_mean + X_max
        X = X.view(-1, 1)
        X = X.squeeze()
        
        X_embedding = X
        X = F.relu(self.linear(X))
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))

        return X_embedding, X


class Controllability_Branch(nn.Module):
    def __init__(self, robustness_length, node_num):
        super(Controllability_Branch, self).__init__()
        self.linear1 = nn.Linear(9, 16)
        self.linear2 = nn.Linear(16, 16)
        self.linear3 = nn.Linear(16 * node_num, robustness_length)

    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = X.view(1, -1)
        X = F.relu(self.linear3(X))
        return X