import torch
import torch.nn.functional as F
import torch_geometric
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, Parameter
from torch_geometric.utils    import softmax
from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.utils    import softmax as tg_softmax
from torch_geometric.nn.inits import glorot, zeros
torch.cuda.empty_cache()

class GATGNN_AGAT_LAYER(MessagePassing):
    def __init__(self, args, **kwargs):
        super(GATGNN_AGAT_LAYER, self).__init__(aggr='add', flow='target_to_source', **kwargs)

        self.act = 'softplus'
        self.fc_layers = 2
        self.batch_track_stats = True
        self.batch_norm = 'True'
        self.dropout_rate = 0.0
        self.dim = 64
        # FIXED-lines ------------------------------------------------------------
        self.heads = 4
        self.add_bias = True
        self.neg_slope = 0.2

        self.bn1 = BatchNorm1d(self.heads)
        self.W = Parameter(torch.Tensor(self.dim * 2, self.heads * self.dim))
        self.att = Parameter(torch.Tensor(1, self.heads, 2 * self.dim))

        if self.add_bias:
            self.bias = Parameter(torch.Tensor(self.dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # FIXED-lines -------------------------------------------------------------

    def reset_parameters(self):
        glorot(self.W)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        out_i = torch.cat([x_i, edge_attr], dim=-1)
        out_j = torch.cat([x_j, edge_attr], dim=-1)
        # print("out_i: ", out_i.shape)
        # print("out_j: ", out_j.shape)

        out_i = getattr(F, self.act)(torch.matmul(out_i, self.W))
        # print("matmul out_i: ", out_i.shape)
        out_j = getattr(F, self.act)(torch.matmul(out_j, self.W))
        # print("matmul out_j: ", out_j.shape)
        out_i = out_i.view(-1, self.heads, self.dim)
        # print("view out_i: ", out_i.shape)
        out_j = out_j.view(-1, self.heads, self.dim)
        # print("view out_j: ", out_j.shape)

        alpha = getattr(F, self.act)((torch.cat([out_i, out_j], dim=-1) * self.att).sum(dim=-1))
        # print("sum alpha: ", alpha.shape)
        alpha = getattr(F, self.act)(self.bn1(alpha))
        alpha = tg_softmax(alpha, edge_index_i)
        # print(alpha)
        # print("softmax alpha: ", alpha.shape)

        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)
        out_j = (out_j * alpha.view(-1, self.heads, 1)).transpose(0, 1)
        return out_j

    def update(self, aggr_out):
        out = aggr_out.mean(dim=0)
        if self.bias is not None:  out = out + self.bias
        # print("bias: ", self.bias)
        return out


class Composition_Attention(torch.nn.Module):
    def __init__(self, neurons):
        super(Composition_Attention, self).__init__()
        self.node_layer1    = Linear(neurons+103, 32)
        self.atten_layer    = Linear(32, 1)

    def forward(self,x, node_batch, global_fea):
        counts = torch.unique(node_batch, return_counts=True)[-1]
        graph_embed = global_fea
        graph_embed = torch.repeat_interleave(graph_embed, counts, dim=0)
        chunk = torch.cat([x, graph_embed], dim=-1)
        x = F.softplus(self.node_layer1(chunk))
        x = self.atten_layer(x)
        weights = softmax(x, node_batch)
        return weights


class GATGNN(torch.nn.Module):
    def __init__(self, args):
        super(GATGNN, self).__init__()
        self.x_embedding = Linear(args.cs_x_features_dim, 64)
        self.e_embedding = Linear(args.cs_edge_features_dim, 64)
        self.neg_slope = 0.2
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        self.post_lin_list = torch.nn.ModuleList()
        self.batch_track_stats = 'True'
        self.dropout_rate = 0.0
        self.pool = "global_add_pool"
        self.act = "softplus"
        self.comp_atten = Composition_Attention(64)
        self.post_fc_count = 1
        self.lin_out = Linear(64, 1)
        for i in range(3):
            conv = GATGNN_AGAT_LAYER(args)
            self.conv_list.append(conv)
            bn = BatchNorm1d(64, track_running_stats=self.batch_track_stats)
            self.bn_list.append(bn)
        for i in range(self.post_fc_count):
            lin = Linear(64, 64)
            self.post_lin_list.append(lin)

    def forward(self, x, edge_source, edge_target, edge_attr, global_fea, node_batch):
        x = self.x_embedding(x)
        edge_attr = F.leaky_relu(self.e_embedding(edge_attr), self.neg_slope)
        edge_index = torch.cat((edge_source, edge_target), dim=-1).reshape(2, -1)
        # Node Attention
        for i in range(3):
            x = self.conv_list[i](x, edge_index, edge_attr)
            x = self.bn_list[i](x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # print(x.shape)
        # Global Attention
        ag = self.comp_atten(x, node_batch, global_fea)
        x = (x) * ag

        x = getattr(torch_geometric.nn, self.pool)(x, node_batch)

        for i in range(0, len(self.post_lin_list)):
            x = self.post_lin_list[i](x)
            x = getattr(F, self.act)(x)
        x = self.lin_out(x)
        return x.view(-1)

        # if x.shape[1] == 1:
        #     return x.view(-1)
        # else:
        #     return x

