import torch
import torch.nn.functional as F
import torch_geometric
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, Parameter
from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.utils    import softmax
from torch_geometric.nn.inits import glorot, zeros


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
        self.W1 = Parameter(torch.Tensor(self.dim, self.heads * self.dim))
        # self.W2 = Parameter(torch.Tensor(1, self.heads * 2)) # if mask weight matrix of energy
        self.W2 = Parameter(torch.Tensor(2, self.heads * 2))
        self.att1 = Parameter(torch.Tensor(1, self.heads, 2 * self.dim))
        self.att2 = Parameter(torch.Tensor(1, self.heads, 2))

        if self.add_bias:
            self.bias = Parameter(torch.Tensor(self.dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # FIXED-lines -------------------------------------------------------------

    def reset_parameters(self):
        glorot(self.W1)
        glorot(self.W2)
        glorot(self.att1)
        glorot(self.att2)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        out_i = x_i
        out_j = x_j

        out_i = getattr(F, self.act)(torch.matmul(out_i, self.W1))
        # print("matmul out_i: ", out_i.shape)
        out_j = getattr(F, self.act)(torch.matmul(out_j, self.W1))
        # print("matmul out_j: ", out_j.shape)
        edge_attr = getattr(F, self.act)(torch.matmul(edge_attr, self.W2))
        out_i = out_i.view(-1, self.heads, self.dim)
        # print("view out_i: ", out_i.shape)
        out_j = out_j.view(-1, self.heads, self.dim)
        # print("view out_j: ", out_j.shape)
        edge_attr = edge_attr.view(-1, self.heads, 2)
        alpha = getattr(F, self.act)((torch.cat([out_i, out_j], dim=-1) * self.att1).sum(dim=-1) + (edge_attr * self.att2).sum(dim=-1))
        # print("sum alpha: ", alpha.shape)
        # print("sum alpha: ", alpha)
        alpha = getattr(F, self.act)(self.bn1(alpha))
        alpha = softmax(alpha, edge_index_i)
        # print(alpha)
        # print("softmax alpha: ", alpha.shape)

        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)
        out_j = (out_j * alpha.view(-1, self.heads, 1)).transpose(0, 1)
        # print("final out_j", out_j.shape)
        return out_j

    def update(self, aggr_out):
        # print("aggr_out: ", aggr_out.shape)
        out = aggr_out.mean(dim=0)
        # print("pout: ", out)
        # print("pout.size: ", out.shape)
        if self.bias is not None:  out = out + self.bias
        # print("bias: ", self.bias)
        return out

class GlobalAttention(torch.nn.Module):
    def __init__(self, neurons):
        super(GlobalAttention, self).__init__()
        self.global_layer1 = Linear(3, 64)
        self.node_layer1 = Linear(128, 64)
        self.atten_layer = Linear(64, 1)

    def forward(self, x, node_batch, global_attr):
        global_attr = self.global_layer1(global_attr)
        counts = torch.unique(node_batch, return_counts=True)[-1]
        graph_embed = global_attr
        graph_embed = torch.repeat_interleave(graph_embed, counts, dim=0)
        chunk = torch.cat([x, graph_embed], dim=-1)
        x = F.softplus(self.node_layer1(chunk))
        x = self.atten_layer(x)
        weights = softmax(x, node_batch)
        return weights


class GATGNN(torch.nn.Module):
    def __init__(self, args):
        super(GATGNN, self).__init__()
        self.x_embedding = Linear(args.in_x_features_dim, 64)
        self.e_embedding = Linear(args.in_edge_features_dim, 64)
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        self.post_lin_list = torch.nn.ModuleList()
        self.batch_track_stats = 'True'
        self.dropout_rate = 0.0
        self.pool = "global_add_pool"
        self.act = "softplus"
        self.post_fc_count = 1
        self.global_attention = GlobalAttention(64)
        self.lin_out = Linear(64, 1)
        for i in range(6):
            conv = GATGNN_AGAT_LAYER(args)
            self.conv_list.append(conv)
            bn = BatchNorm1d(64, track_running_stats=self.batch_track_stats)
            self.bn_list.append(bn)
        for i in range(self.post_fc_count):
            lin = Linear(64, 64)
            self.post_lin_list.append(lin)

    def forward(self, x, edge_source, edge_target, edge_attr, global_attr, node_batch):
        x = self.x_embedding(x)
        # edge_attr = self.e_embedding(edge_attr)
        # print(x.shape)
        # print(edge_attr.shape)
        edge_index = torch.cat((edge_source, edge_target), dim=-1).reshape(2, -1)
        # Node Attention
        for i in range(6):
            x = self.conv_list[i](x, edge_index, edge_attr)
            # print(x.shape)
            x = self.bn_list[i](x)
            # print(x.shape)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # print(x.shape)
        # Global Attention
        ag = self.global_attention(x, node_batch, global_attr)
        x = (x) * ag
        x = getattr(torch_geometric.nn, self.pool)(x, node_batch)

        for i in range(0, len(self.post_lin_list)):
            x = self.post_lin_list[i](x)
            x = getattr(F, self.act)(x)
        # x = self.lin_out(x)

        return x

        # if x.shape[1] == 1:
        #     return x.view(-1)
        # else:
        #     return x

