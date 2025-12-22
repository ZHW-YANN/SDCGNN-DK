import torch.nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Sigmoid, Parameter


# Feature-level fusion method
class Merge_Net1(torch.nn.Module):
    def __init__(self, args):
        super(Merge_Net1, self).__init__()
        # self.weight = Parameter(torch.ones(2))
        self.merge_fc_layer = Linear(128, 64)
        self.act = 'relu'
        self.merge_lin_layer = Linear(64, 1)

    def forward(self, a, b):
        # weight = F.softmax(self.weight, dim=0)
        x = torch.cat([a, b], dim=1)
        x = self.merge_fc_layer(x)
        x = getattr(F, self.act)(x)
        x = self.merge_lin_layer(x)
        return x


# Decision-level fusion
class Merge_Net2(torch.nn.Module):
    def __init__(self, args):
        super(Merge_Net2, self).__init__()
        self.lin1 = Linear(64, 1)
        self.lin2 = Linear(64, 1)
        self.merge_fc_layer = Linear(2, 1)
        self.act = 'relu'
        # self.merge_lin_layer = Linear(64, 1)

    def forward(self, a, b):
        a_x = self.lin1(a)
        b_x = self.lin2(b)
        x = torch.cat([a_x, b_x], dim=1)
        x = self.merge_fc_layer(x)
        # weight = F.softmax(torch.cat([a_x, b_x], dim=1), dim=1)
        # x = torch.cat([weight[:, 0].view(-1, 1) * a, weight[:, 1].view(-1, 1) * b], dim=1)
        # x = self.merge_fc_layer(x)
        # x = getattr(F, self.act)(x)
        # x = self.merge_lin_layer(x)
        return x

# Hybrid-level fusion
class Merge_Net3(torch.nn.Module):
    def __init__(self, args):
        super(Merge_Net3, self).__init__()
        self.lin1 = Linear(64, 1)
        self.lin2 = Linear(64, 1)
        self.merge_fc_layer = Linear(130, 64)
        self.act = 'relu'
        self.merge_lin_layer = Linear(64, 1)

    def forward(self, a, b):
        a_x = self.lin1(a)
        b_x = self.lin2(b)
        a = torch.cat([a, a_x], dim=1)
        b = torch.cat([b, b_x], dim=1)
        x = torch.cat([a, b], dim=1)
        x = self.merge_fc_layer(x)
        x = getattr(F, self.act)(x)
        x = self.merge_lin_layer(x)
        return x


# Model-level fusion
class Merge_Net4(torch.nn.Module):
    def __init__(self, args):
        super(Merge_Net4, self).__init__()
        self.lstm1 = torch.nn.LSTM(64, 64, 3)
        self.lstm2 = torch.nn.LSTM(64, 64, 3)
        self.merge_fc_layer = 1
        self.merge_lin_list = torch.nn.ModuleList()
        self.act = 'relu'
        self.lin_out = Linear(32, 1)
        for i in range(self.merge_fc_layer):
            lin = Linear(64, 32)
            self.merge_lin_list.append(lin)

    def forward(self, a, b):
        x, (xx1, xx2) = self.lstm1(a)
        y, _ = self.lstm2(x + b, (xx1, xx2))
        for i in range(len(self.merge_lin_list)):
            x = self.merge_lin_list[i](x)
            x = getattr(F, self.act)(x)
        x = self.lin_out(x)
        return x

# Hybrid feature fusion strategy based on structural importance
class Merge_Net5(torch.nn.Module):
    def __init__(self, args):
        super(Merge_Net5, self).__init__()
        self.lin1 = Linear(64, 1)
        self.lin2 = Linear(64, 1)
        self.merge_fc_layer = Linear(128, 64)
        self.act = 'relu'
        self.merge_lin_layer = Linear(64, 1)

    def forward(self, a, b):
        a_x = self.lin1(a)
        b_x = self.lin2(b)
        weight = F.softmax(torch.cat([a_x, b_x], dim=1), dim=1)
        x = torch.cat([weight[:, 0].view(-1, 1) * a, weight[:, 1].view(-1, 1) * b], dim=1)
        x = self.merge_fc_layer(x)
        x = getattr(F, self.act)(x)
        x = self.merge_lin_layer(x)
        return x, weight


# Hybrid feature fusion strategy with fixed feature importance
class Merge_Net_fix(torch.nn.Module):
    def __init__(self, args):
        super(Merge_Net_fix, self).__init__()
        self.lin1 = Linear(64, 1)
        self.lin2 = Linear(64, 1)
        self.merge_fc_layer = Linear(128, 64)
        self.act = 'relu'
        self.merge_lin_layer = Linear(64, 1)

    def forward(self, a, b):
        weight = [0.6, 0.4]
        x = torch.cat([weight[0] * a, weight[1] * b], dim=1)
        x = self.merge_fc_layer(x)
        x = getattr(F, self.act)(x)
        x = self.merge_lin_layer(x)
        return x
