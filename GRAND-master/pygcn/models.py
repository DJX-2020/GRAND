import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import numpy as np
from pygcn.layers import GraphConvolution, MLPLayer


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn=False):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.bn1 = nn.BatchNorm1d(nfeat) if use_bn else None
        self.bn2 = nn.BatchNorm1d(nhid) if use_bn else None
        self.use_bn = use_bn

    def forward(self, x, adj):

        if self.use_bn:
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        
        x = F.relu(self.gc1(x, adj))
        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.gc2(x, adj)

        # 他这里返回的 x，只是一个 logits，没做 softmax，我给他补上
        return F.log_softmax(x, dim=1)


class GCNC(nn.Module):
    def __init__(self, in_feat, n_class, n_hid=32, n_emb=64, drop_rate=0.1):
        super(GCNC, self).__init__()
        self.gc1 = GraphConvolution(in_feat, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_emb)
        # 这个 linear 层就用作我的分类器，也就是说我的分类器就是一层的全连接网络
        self.linear = nn.Linear(n_emb, n_class)
        self.drop_rate = drop_rate

    def forward(self, x, adj):
        # 第一层
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.drop_rate, training=self.training)

        # 第二层
        x = self.gc2(x, adj)
        z = x  # 保存当前结果为图嵌入向量
        x = F.relu(z)
        x = F.dropout(x, self.drop_rate, training=self.training)

        return F.log_softmax(self.linear(x), dim=1), z


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, is_cuda=True, use_bn =False):
        super(MLP, self).__init__()

        self.layer1 = MLPLayer(nfeat, nhid)
        self.layer2 = MLPLayer(nhid, nclass)

        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.is_cuda = is_cuda
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        
    def forward(self, x):
         
        if self.use_bn: 
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        x = F.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.layer2(x)

        return F.log_softmax(x, dim=1)

