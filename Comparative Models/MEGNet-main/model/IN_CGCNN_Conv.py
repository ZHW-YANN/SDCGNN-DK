import torch.nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Sigmoid
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter, scatter_sum
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    MetaLayer,
)


class Node_Convolution(torch.nn.Module):
    def __init__(self, args):
        super(Node_Convolution, self).__init__()
        self.linear_f = Linear(128 * 3, 128, bias=True)
        self.linear_s = Linear(128 * 3, 128, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.softplus1 = torch.nn.Softplus()
        self.softplus2 = torch.nn.Softplus()
        self.bn = BatchNorm1d(128)

    def forward(self, x, edge_attr, edge_source, edge_target):
        z = torch.cat([x[edge_source], x[edge_target], edge_attr], dim=-1)
        message = scatter_sum(self.sigmoid(self.linear_f(z)) * self.softplus1(self.linear_s(z)), edge_source, dim=0)
        message = self.bn(message)
        x = self.softplus2(x + message)
        return x


class Edge_Convolution(torch.nn.Module):
    def __init__(self, args):
        super(Edge_Convolution, self).__init__()
        self.message_linear = Linear(args.nhid * 3, args.nhid, bias=True)
        self.edge_linear = Linear(args.nhid, args.nhid, bias=True)

    def forward(self, x, edge_attr, edge_index):
        # comb = scatter_mean(edge_attr, edge_index[0, :], dim=0)
        # edge_message = F.relu(self.message_linear(comb))
        # edge_attr = edge_attr + edge_message
        # edge_attr = F.relu(self.edge_linear(edge_attr))
        # return edge_attr
        row, col = edge_index
        comb = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_message = F.relu(self.message_linear(comb))
        # edge_message = scatter_mean(comb, edge_index[0, :], dim=0)
        edge_attr = edge_attr + edge_message
        edge_attr = F.relu(self.edge_linear(edge_attr))
        return edge_attr


class Node_Edge_Pooling(torch.nn.Module):
    def __init__(self, args):
        super(Node_Edge_Pooling, self).__init__()
        self.graph_linear = Linear(args.nhid, args.nhid, bias=True)

    def forward(self, x, edge_attr, node_batch):
        # graph_res = torch.cat([global_add_pool(x, node_batch), global_add_pool(edge_attr, edge_batch)], dim=1)
        graph_res = global_add_pool(x, node_batch)
        graph_res = self.graph_linear(graph_res)
        return graph_res


class IN_CGCNN_Conv(torch.nn.Module):
    def __init__(self, args):
        super(IN_CGCNN_Conv, self).__init__()
        self.node_embedding = Sequential(
            Linear(args.in_x_features_dim, args.nhid, bias=True), ReLU()
        )
        self.edge_embedding = Sequential(
            Linear(args.in_edge_features_dim, args.nhid, bias=True), ReLU()
        )
        self.node_conv = torch.nn.ModuleList()
        self.edge_conv = torch.nn.ModuleList()
        self.node_edge_pooling = torch.nn.ModuleList()
        for i in range(5):
            self.node_conv.append(Node_Convolution(args))
        for i in range(5):
            self.edge_conv.append(Edge_Convolution(args))
        for i in range(5):
            self.node_edge_pooling.append(Node_Edge_Pooling(args))

        # final linear regression
        self.readout = Sequential(
            Linear(args.nhid, 64), ReLU(),
            Linear(64, 32), ReLU(),
            Linear(32, 16), ReLU(),
            Linear(16, 1), ReLU()
        )

    def forward(self, x, edge_source, edge_target, edge_attr, node_batch):
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        pool_results = []
        for i in range(5):
            x = self.node_conv[i](x, edge_attr, edge_source, edge_target)
            # edge_attr = self.edge_conv[i](x, edge_attr, edge_index)
            pool = self.node_edge_pooling[i](x, edge_attr, node_batch)
            pool_results.append(pool)
        y = torch.sum(torch.stack(pool_results), dim=0)
        y = self.readout(y)
        return y



