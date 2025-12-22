import torch.nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Sigmoid
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    MetaLayer,
)


class MEGNet_EdgeConvolution(torch.nn.Module):

    def __init__(self):
        super(MEGNet_EdgeConvolution, self).__init__()
        self.act = 'relu'
        self.edge_mlp = torch.nn.ModuleList()
        self.edge_bn_list = torch.nn.ModuleList()
        self.dropout_rate = 0.0
        for i in range(3):
            if i == 0:
                self.edge_mlp.append(Linear(256, 64))
            else:
                self.edge_mlp.append(Linear(64, 64))
            self.edge_bn_list.append(BatchNorm1d(64, track_running_stats=True))

    def forward(self, src, dest, edge_attr, global_attr, batch):
        comb = torch.cat([src, dest, edge_attr, global_attr[batch]], dim=1)
        for i in range(3):
            if i == 0:
                out = self.edge_mlp[i](comb)
            else:
                out = self.edge_mlp[i](out)
            out = getattr(F, self.act)(out)
            out = self.edge_bn_list[i](out)
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        return out


class MEGNet_NodeConvolution(torch.nn.Module):

    def __init__(self):
        super(MEGNet_NodeConvolution, self).__init__()
        self.act = 'relu'
        self.node_mlp = torch.nn.ModuleList()
        self.node_bn_list = torch.nn.ModuleList()
        self.dropout_rate = 0.0
        for i in range(3):
            if i == 0:
                self.node_mlp.append(Linear(192, 64))
            else:
                self.node_mlp.append(Linear(64, 64))
            self.node_bn_list.append(BatchNorm1d(64, track_running_stats=True))

    def forward(self, x, edge_index, edge_attr, global_attr, batch):
        v_e = scatter_mean(edge_attr, edge_index[0, :], dim=0)
        comb = torch.cat([x, v_e, global_attr[batch]], dim=1)
        for i in range(3):
            if i == 0:
                out = self.node_mlp[i](comb)
            else:
                out = self.node_mlp[i](out)
            out = getattr(F, self.act)(out)
            out = self.node_bn_list[i](out)
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        return out


class MEGNet_GlobalConvolution(torch.nn.Module):
    def __init__(self):
        super(MEGNet_GlobalConvolution, self).__init__()
        self.act = 'relu'
        self.global_mlp = torch.nn.ModuleList()
        self.global_bn_list = torch.nn.ModuleList()
        self.dropout_rate = 0.0
        for i in range(3):
            if i == 0:
                self.global_mlp.append(Linear(192, 64))
            else:
                self.global_mlp.append(Linear(64, 64))
            self.global_bn_list.append(BatchNorm1d(64, track_running_stats=True))

    def forward(self, x, edge_index, edge_attr, global_attr, batch):
        u_e = scatter_mean(edge_attr, edge_index[0, :], dim=0)
        u_e = scatter_mean(u_e, batch, dim=0)
        u_v = scatter_mean(x, batch, dim=0)
        comb = torch.cat([u_e, u_v, global_attr], dim=1)
        for i in range(3):
            if i == 0:
                out = self.global_mlp[i](comb)
            else:
                out = self.global_mlp[i](out)
            out = getattr(F, self.act)(out)
            out = self.global_bn_list[i](out)
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        return out


class MEGNet(torch.nn.Module):
    def __init__(self, args):
        super(MEGNet, self).__init__()
        self.pre_lin_list = Linear(args.cs_x_features_dim, 64)
        self.act = 'relu'
        self.edge_embedding_list = torch.nn.ModuleList()
        self.node_embedding_list = torch.nn.ModuleList()
        self.global_embedding_list = torch.nn.ModuleList()
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        self.set2set_x = Set2Set(64, processing_steps=3)
        self.set2set_e = Set2Set(64, processing_steps=3)
        self.post_lin_list = Linear(320, 64)
        self.final_mlp = Linear(64, 1)
        for i in range(3):
            if i == 0:
                edge_embedding = Sequential(Linear(args.cs_edge_features_dim, 64), ReLU(), Linear(64, 64), ReLU())
                node_emdedding = Sequential(Linear(64, 64), ReLU(), Linear(64, 64), ReLU())
                global_embedding = Sequential(Linear(args.cs_global_features_dim, 64), ReLU(), Linear(64, 64), ReLU())
            else:
                edge_embedding = Sequential(Linear(64, 64), ReLU(), Linear(64, 64), ReLU())
                node_emdedding = Sequential(Linear(64, 64), ReLU(), Linear(64, 64), ReLU())
                global_embedding = Sequential(Linear(64, 64), ReLU(), Linear(64, 64), ReLU())
            self.edge_embedding_list.append(edge_embedding)
            self.node_embedding_list.append(node_emdedding)
            self.global_embedding_list.append(global_embedding)
            self.conv_list.append(
                MetaLayer(
                    MEGNet_EdgeConvolution(),
                    MEGNet_NodeConvolution(),
                    MEGNet_GlobalConvolution()
                )
            )

    def forward(self, x, edge_index, edge_attr, global_attr, node_batch):
        # GNN dense layers
        out = self.pre_lin_list(x)
        out = getattr(F, self.act)(out)

        # GNN conv layers
        for i in range(3):
            if i == 0:
                e_temp = self.edge_embedding_list[i](edge_attr)
                x_temp = self.node_embedding_list[i](out)
                u_temp = self.global_embedding_list[i](global_attr)
                x_out, e_out, u_out = self.conv_list[i](
                    x_temp, edge_index, e_temp, u_temp, node_batch
                )
                x = torch.add(x_out, x_temp)
                e = torch.add(e_out, e_temp)
                u = torch.add(u_out, u_temp)
            else:
                e_temp = self.edge_embedding_list[i](e)
                x_temp = self.node_embedding_list[i](x)
                u_temp = self.global_embedding_list[i](u)
                x_out, e_out, u_out = self.conv_list[i](
                    x_temp, edge_index, e_temp, u_temp, node_batch
                )
                x = torch.add(x_out, x)
                e = torch.add(e_out, e)
                u = torch.add(u_out, u)
        # GNN post layers
        x_pool = self.set2set_x(x, node_batch)
        e = scatter(e, edge_index[0, :], dim=0, reduce="mean")
        e_pool = self.set2set_e(e, node_batch)
        out = torch.cat([x_pool, e_pool, u], dim=1)
        out = self.post_lin_list(out)
        out = getattr(F, self.act)(out)
        out = self.final_mlp(out)
        return out.view(-1)

