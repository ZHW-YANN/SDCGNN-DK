import torch.nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Sigmoid, Parameter


# class Merge_Net(torch.nn.Module):
#     def __init__(self, args):
#         super(Merge_Net, self).__init__()
#         self.merge_fc_layer = 1
#         self.merge_lin_list = torch.nn.ModuleList()
#         self.act = 'relu'
#         self.lin_out = Linear(128, 1)
#         for i in range(self.merge_fc_layer):
#             lin = Linear(128, 128)
#             self.merge_lin_list.append(lin)
#
#
#     def forward(self, x):
#         for i in range(len(self.merge_lin_list)):
#             x = self.merge_lin_list[i](x)
#             x = getattr(F, self.act)(x)
#         x = self.lin_out(x)
#
#         return x
class Merge_Net(torch.nn.Module):
    def __init__(self, args):
        super(Merge_Net, self).__init__()
        self.weight = Parameter(torch.ones(2))
        self.act = 'relu'
        self.linear1 = Linear(128, 64)
        self.linear2 = Linear(64, 32)
        self.linear3 = Linear(32, 16)
        self.linear4 = Linear(16, 1)


    def forward(self, a, b):

        weight = F.softmax(self.weight, dim=0)
        x = torch.cat([a * weight[0], b * weight[1]], dim=1)
        x = getattr(F, self.act)(self.linear1(x))
        x = getattr(F, self.act)(self.linear2(x))
        x = getattr(F, self.act)(self.linear3(x))
        x = self.linear4(x)
        return x.view(-1)

