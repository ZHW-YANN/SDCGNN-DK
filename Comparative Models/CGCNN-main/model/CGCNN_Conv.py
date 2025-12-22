import torch.nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    CGConv,
)


class ConvLayer(torch.nn.Module):
    def __init__(self, args):
        super(ConvLayer, self).__init__()
        self.linear_f = Linear(128+args.cs_edge_features_dim, 64, bias=True)
        self.linear_s = Linear(128+args.cs_edge_features_dim, 64, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.softplus1 = torch.nn.Softplus()
        self.softplus2 = torch.nn.Softplus()
        self.bn = torch.nn.BatchNorm1d(64)

    def forward(self, x, edge_source, edge_target, edge_attr):
        z = torch.cat([x[edge_source], x[edge_target], edge_attr], dim=-1)
        message = scatter_sum(self.sigmoid(self.linear_f(z)) * self.softplus1(self.linear_s(z)), edge_source, dim=0)
        message = self.bn(message)
        x = self.softplus2(x+message)
        return x


class CS_CGCNN(torch.nn.Module):
    def __init__(self, args):
        super(CS_CGCNN, self).__init__()
        self.embedding = Linear(args.cs_x_features_dim, 64)
        self.convs = torch.nn.ModuleList([ConvLayer(args) for _ in range(3)])
        self.linear = Linear(64, 1)

    def forward(self, x, edge_source, edge_target, edge_attr, node_batch):
        x = self.embedding(x)
        for conv_func in self.convs:
            x = conv_func(x, edge_source, edge_target, edge_attr)
        x = global_mean_pool(x, node_batch)
        x = self.linear(x)
        return x.squeeze()
