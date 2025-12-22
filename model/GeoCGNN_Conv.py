import re

import torch.nn
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, Parameter
from torch.nn import ( Linear, Bilinear, Sigmoid, Softplus, ELU, ReLU, SELU,
                       CELU, BatchNorm1d, ModuleList, Sequential,Tanh)

def get_activation(name):
    act_name = name.lower()
    m = re.match(r"(\w+)\((\d+\.\d+)\)", act_name)
    if m is not None:
        act_name, alpha = m.groups()
        alpha = float(alpha)
        print(act_name, alpha)
    else:
        alpha = 1.0
    if act_name == 'softplus':
        return Softplus()
    # elif act_name == 'ssp':
    #     return SSP()
    elif act_name == 'elu':
        return ELU(alpha)
    elif act_name == 'relu':
        return ReLU()
    elif act_name == 'selu':
        return SELU()
    elif act_name == 'celu':
        return CELU(alpha)
    elif act_name == 'sigmoid':
        return Sigmoid()
    elif act_name == 'tanh':
        return Tanh()
    else:
        raise NameError("Not supported activation: {}".format(name))

def _bn_act(num_features, activation, use_batch_norm):
    if use_batch_norm:
        if activation is None:
            return BatchNorm1d(num_features)
        else:
            return Sequential(BatchNorm1d(num_features), activation)
    else:
        return activation

class NodeEmbedding(torch.nn.Module):
    def __init__(self, in_features, out_features, activation=Sigmoid(), use_batch_norm=False, bias=False):
        super(NodeEmbedding, self).__init__()
        self.linear = Linear(in_features, out_features, bias=bias)
        self.activation = _bn_act(out_features, activation, use_batch_norm)

    def forward(self, x):
        output = self.activation(self.linear(x))
        return output


class GatedGraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, n_grid_K, n_Gauusian,
                 gate_activation, MLP_activation, use_node_batch_norm, use_edge_batch_norm, bias):
        super(GatedGraphConvolution, self).__init__()
        k1 = n_Gauusian
        k2 = n_grid_K ** 3
        self.linear1_vector = Linear(k1, out_features, bias=bias)
        self.linear1_vector_gate = Linear(k1, out_features, bias=bias)
        self.activation1_vector_gate = _bn_act(out_features, gate_activation, use_edge_batch_norm)
        self.linear2_vector = Linear(k2, out_features, bias=bias)
        self.linear2_vector_gate = Linear(k2, k2, bias=bias)
        self.activation2_vector_gate = _bn_act(k2, gate_activation, use_edge_batch_norm)

        self.linear_gate = Linear(in_features, out_features, bias=bias)
        self.activation_gate = _bn_act(out_features, gate_activation, use_edge_batch_norm)

        self.linear_MLP = Linear(in_features, out_features, bias=bias)
        self.activation_MLP = _bn_act(out_features, MLP_activation, use_edge_batch_norm)


    def forward(self, input, edge_sources, edge_targets, rij, combine_sets, plane_wave, cutoff):
        ni = input[edge_sources].contiguous()
        nj = input[edge_targets].contiguous()
        rij = rij.reshape(-1, 1).contiguous()
        mask = rij < cutoff
        delta = (ni - nj) / rij
        final_fe = torch.cat([ni, nj, delta], dim=1)
        del ni, nj, delta
        torch.cuda.empty_cache()

        e_gate = self.activation_gate(self.linear_gate(final_fe))
        e_MLP = self.activation_MLP(self.linear_MLP(final_fe))
        z1 = self.linear1_vector(combine_sets)
        gate = self.activation2_vector_gate(self.linear2_vector_gate(plane_wave))
        z2 = self.linear2_vector(plane_wave * gate)
        z = e_gate * e_MLP * (z1 + z2) * mask
        del z1, z2, e_gate, e_MLP
        torch.cuda.empty_cache()
        output = input.clone()
        output.index_add_(0, edge_sources, z)
        return output

class Gated_pooling(torch.nn.Module):
    def __init__(self, in_features, out_features, activation, use_batch_norm, bias):
        super(Gated_pooling, self).__init__()
        self.linear1 = Linear(in_features, out_features, bias=bias)
        self.activation1 = _bn_act(out_features, activation, use_batch_norm)
        self.linear2 = Linear(in_features, out_features, bias=bias)
        self.activation2 = _bn_act(out_features, activation, use_batch_norm)


    def forward(self, input, graph_indices, node_counts):
        z = self.activation1(self.linear1(input) * self.linear2(input))
        graphcount = len(node_counts)
        device = z.device
        blank = torch.zeros(graphcount, z.shape[1]).to(device)
        blank.index_add_(0, graph_indices, z) / node_counts.unsqueeze(1)
        return blank

class OLP(torch.nn.Module):
    def __init__(self, in_features, out_features, activation, use_batch_norm, bias):
        super(OLP, self).__init__()
        self.linear = Linear(in_features, out_features, bias=bias)
        self.activation = _bn_act(out_features, activation, use_batch_norm)

    def forward(self, input):
        z = self.linear(input)
        if self.activation:
            z = self.activation(z)
        return z


class GeoCGNN(torch.nn.Module):
    def __init__(self, args):
        super(GeoCGNN, self).__init__()
        self.N_block = args.N_block
        self.cutoff = args.cutoff
        self.n_hidden_feat = args.n_hidden_feat
        self.n2v_concatent_feat = self.n_hidden_feat * 3
        self.n_grid_K = args.n_grid_K
        self.n_Gaussian = args.n_Gaussian
        self.n_MLP_LR = args.n_MLP_LR
        self.embedding = NodeEmbedding(args.cs_x_features_dim, self.n_hidden_feat)
        node_activation = get_activation(args.node_activation)
        MLP_activation = get_activation(args.MLP_activation)
        self.conv = [GatedGraphConvolution(self.n2v_concatent_feat, self.n_hidden_feat, self.n_grid_K, self.n_Gaussian,
                                            gate_activation=node_activation,
                                            MLP_activation=MLP_activation,
                                            use_node_batch_norm=args.use_node_batch_norm,
                                            use_edge_batch_norm=args.use_edge_batch_norm,
                                            bias=args.conv_bias)]
        self.conv += [GatedGraphConvolution(self.n2v_concatent_feat, self.n_hidden_feat, self.n_grid_K, self.n_Gaussian,
                                            gate_activation=node_activation,
                                            MLP_activation=MLP_activation,
                                            use_node_batch_norm=args.use_node_batch_norm,
                                            use_edge_batch_norm=args.use_edge_batch_norm,
                                            bias=args.conv_bias) for _ in range(self.N_block-1)]
        self.conv = torch.nn.ModuleList(self.conv)
        self.gated_pooling = [Gated_pooling(self.n_hidden_feat, self.n_hidden_feat,
                                            activation=MLP_activation,
                                            use_batch_norm=args.use_node_batch_norm,
                                            bias=args.conv_bias) for _ in range(self.N_block)]
        self.gated_pooling = torch.nn.ModuleList(self.gated_pooling)
        self.MLP_psi2n = [OLP(self.n_hidden_feat, self.n_hidden_feat,
                              activation=MLP_activation,
                              use_batch_norm=args.use_node_batch_norm,
                              bias=args.conv_bias) for _ in range(self.N_block)]
        self.MLP_psi2n = torch.nn.ModuleList(self.MLP_psi2n)

        self.linear_regression = [
            OLP(int(self.n_hidden_feat / 2 ** (i - 1)), int(self.n_hidden_feat / 2 ** i),
                activation=MLP_activation,
                use_batch_norm=args.use_node_batch_norm,
                bias=args.conv_bias) for i in range(1, self.n_MLP_LR)]
        # self.linear_regression += [
        #     OLP(int(self.n_hidden_feat / 2 ** (self.n_MLP_LR - 1)), 1, activation=None, use_batch_norm=None, bias=args.conv_bias)]
        self.linear_regression = torch.nn.ModuleList(self.linear_regression)


    def forward(self, x, edge_sources, edge_targets, edge_distance, graph_indices, node_counts, combine_sets, plane_wave):
        x = self.embedding(x)
        PoolingResults = []
        for i in range(self.N_block):
            x = self.conv[i](x, edge_sources, edge_targets, edge_distance, combine_sets, plane_wave, self.cutoff)
            poo = self.gated_pooling[i](x, graph_indices, node_counts)
            PoolingResults.append(poo)
            x = self.MLP_psi2n[i](x)

        y = torch.sum(torch.stack(PoolingResults), dim=0)
        # y = graph_vec
        for lr in self.linear_regression:
            y = lr(y)
        return y
        # if output_graph:
        #     return y.squeeze(), graph_vec
        # else:
        #     return y.squeeze()

