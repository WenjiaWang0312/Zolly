from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
'''
this folder and code is modified base on SemGCN code,
https://github.com/garyzhao/SemGCN
the SemGCN model.
'''


class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(
            torch.zeros(size=(2, in_features, out_features),
                        dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.m = (self.adj > 0)
        self.e = nn.Parameter(
            torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(
            self.in_features) + ' -> ' + str(self.out_features) + ')'


class _GraphConv(nn.Module):

    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):

    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):

    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        # self.nonlocal = GraphNonLocal(hid_dim, sub_sample=group_size) # keywords nonlocal
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        # out = self.nonlocal(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out


class SemGCN(nn.Module):

    def __init__(self,
                 adj,
                 hid_dim,
                 coords_dim=(2, 3),
                 num_layers=4,
                 nodes_group=None,
                 p_dropout=None):
        super(SemGCN, self).__init__()

        _gconv_input = [
            _GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)
        ]
        _gconv_layers = []

        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(
                    _ResGraphConv(adj,
                                  hid_dim,
                                  hid_dim,
                                  hid_dim,
                                  p_dropout=p_dropout))
        else:
            group_size = len(nodes_group[0])
            assert group_size > 1

            grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
            restored_order = [0] * len(grouped_order)
            for i in range(len(restored_order)):
                for j in range(len(grouped_order)):
                    if grouped_order[j] == i:
                        restored_order[i] = j
                        break

            _gconv_input.append(
                _GraphNonLocal(hid_dim, grouped_order, restored_order,
                               group_size))
            for i in range(num_layers):
                _gconv_layers.append(
                    _ResGraphConv(adj,
                                  hid_dim,
                                  hid_dim,
                                  hid_dim,
                                  p_dropout=p_dropout))
                _gconv_layers.append(
                    _GraphNonLocal(hid_dim, grouped_order, restored_order,
                                   group_size))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = SemGraphConv(hid_dim, coords_dim[1], adj)

    def forward(self, x):
        """
        input: bx16x2 / bx32
        output: bx16x3
        """
        if len(x.shape) == 2:
            x = x.view(x.shape[0], 16, 2)
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)
        return out
