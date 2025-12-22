import torch.nn
import torch.nn.functional as F
from model import CS_Conv, IN_Conv, Merge_Model, CGCNN_Conv, IN_CGCNN_Conv, GATGNN_Conv, GeoCGNN_Conv

class CryStal_Conv(torch.nn.Module):
    def __init__(self):
        super(CryStal_Conv, self).__init__()

    def forward(self, x, edge_index, edge_attr, batch):
        return 0


class SDCGNN(torch.nn.Module):
    def __init__(self, args):
        super(SDCGNN, self).__init__()
        self.args = args
        # self.cs_net = CS_Conv.CS_Conv(args)
        # self.in_net = IN_Conv.IN_Conv(args)
        self.merge_net = Merge_Model.Merge_Net(args)
        # self.cgcnn = CGCNN_Conv.CS_CGCNN(args)
        # self.in_cgcnn = IN_CGCNN_Conv.IN_CGCNN_Conv(args)
        self.gatgnn = GATGNN_Conv.GATGNN(args)
        self.geocgnn = GeoCGNN_Conv.GeoCGNN(args)

    # CGCNN + 间隙网络
    # def forward(self, cs_x, in_x, cs_edge_source, cs_edge_target, in_edge_source, in_edge_target, cs_edge_attr, in_edge_attr, cs_node_batch, in_node_batch):
    #     cs_out = self.cs_net(cs_x, cs_edge_source, cs_edge_target, cs_edge_attr, cs_node_batch)
    #     cs_out = self.cgcnn(cs_x, cs_edge_source, cs_edge_target, cs_edge_attr, cs_node_batch)
    #     in_out = self.in_net(in_x, in_edge_source, in_edge_target, in_edge_attr, in_node_batch)
    #     merge_out = torch.cat([cs_out, in_out], dim=-1)
    #     merge_out = self.merge_net(merge_out)
    #     cs_out = self.gatgnn(cs_x, cs_edge_source, cs_edge_target, cs_edge_attr, cs_node_batch)
    #     return merge_out.squeeze()
    #     in_out = self.in_net(in_x, in_edge_source, in_edge_target, in_edge_attr, in_node_batch)
    #     merge_out = torch.cat([cs_out, in_out], dim=-1)
    #     merge_out = self.merge_net(merge_out)
    #     print('merge_out: ', merge_out)
    #     return cs_out.squeeze(), in_out.squeeze()

    # GeoCGNN + 间隙网络
    # def forward(self, cs_x, cs_edge_sources, cs_edge_targets, cs_edge_distance, cs_node_batch, cs_node_counts, cs_combine_sets, cs_plane_wave):
    #
    #     cs_out = self.geocgnn(cs_x, cs_edge_sources, cs_edge_targets, cs_edge_distance, cs_node_batch, cs_node_counts, cs_combine_sets, cs_plane_wave)
    #     return cs_out

    # GeoCGNN + GATGNN + 内融合
    def forward(self, cs_x, cs_edge_sources, cs_edge_targets, cs_edge_distance, cs_node_batch, cs_node_counts, cs_combine_sets, cs_plane_wave,
                in_x, in_edge_sources, in_edge_targets, in_edge_attr, in_node_batch):

        cs_out = self.geocgnn(cs_x, cs_edge_sources, cs_edge_targets, cs_edge_distance, cs_node_batch, cs_node_counts, cs_combine_sets, cs_plane_wave)

        in_out = self.gatgnn(in_x, in_edge_sources, in_edge_targets, in_edge_attr, in_node_batch)

        merge_out = torch.cat([cs_out, in_out], dim=-1)

        final_out = self.merge_net(merge_out)

        if final_out.shape[1] == 1:
            return final_out.view(-1)
        else:
            return final_out



