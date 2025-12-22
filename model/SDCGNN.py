import torch.nn
import torch.nn.functional as F
from model import CS_Conv, IN_Conv, Merge_Model, CGCNN_Conv, IN_CGCNN_Conv, GATGNN_Conv, GeoCGNN_Conv, MEGNET_Conv


class CryStal_Conv(torch.nn.Module):
    """
    Crystal structure convolution module placeholder
    """

    def __init__(self):
        super(CryStal_Conv, self).__init__()

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass placeholder that returns zero

        Args:
            x: Node features
            edge_index: Edge connectivity information
            edge_attr: Edge attributes
            batch: Batch information

        Returns:
            Zero tensor (placeholder implementation)
        """
        return 0


class SDCGNN(torch.nn.Module):
    """
    Spatial Dual Crystal Graph Neural Network combining crystal structure and interstice network
    """

    def __init__(self, args):
        super(SDCGNN, self).__init__()
        self.args = args
        # self.cs_net = CS_Conv.CS_Conv(args)
        # self.in_net = IN_Conv.IN_Conv(args)
        self.merge_net = Merge_Model.Merge_Net5(args)
        # self.cgcnn = CGCNN_Conv.CS_CGCNN(args)
        # self.in_cgcnn = IN_CGCNN_Conv.IN_CGCNN_Conv(args)
        self.gatgnn = GATGNN_Conv.GATGNN(args)
        self.geocgnn = GeoCGNN_Conv.GeoCGNN(args)
        self.megnet = MEGNET_Conv.MEGNet(args)

    # Crystal structure + Interstice network + Internal fusion
    def forward(self, cs_x, in_x, cs_edge_index, in_edge_sources, in_edge_targets, cs_edge_attr, in_edge_attr,
                global_attr, cs_node_batch, in_node_batch):
        """
        Forward pass for the dual crystal graph neural network

        Args:
            cs_x: Crystal structure node features
            in_x: Interstice network node features
            cs_edge_index: Crystal structure edge indices
            in_edge_sources: Interstice network edge source indices
            in_edge_targets: Interstice network edge target indices
            cs_edge_attr: Crystal structure edge attributes
            in_edge_attr: Interstice network edge attributes
            global_attr: Global attributes
            cs_node_batch: Crystal structure node batch information
            in_node_batch: Interstice network node batch information

        Returns:
            Final output predictions from the merged crystal structure and interstice network features
        """

        in_out = self.gatgnn(in_x, in_edge_sources, in_edge_targets, in_edge_attr, global_attr, in_node_batch)
        cs_out = self.megnet(cs_x, cs_edge_index, cs_edge_attr, global_attr, cs_node_batch)

        final_out, _ = self.merge_net(in_out, cs_out)  # if FFS-5
        # final_out = self.merge_net(in_out, cs_out)  # if FFS-1, FFS-2, FFS-3, FFS-4

        if final_out.shape[1] == 1:
            return final_out.view(-1)
        else:
            return final_out
