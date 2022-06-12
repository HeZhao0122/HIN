import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import Graphsn_GCN


class GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()

        self.gc1 = Graphsn_GCN(nfeat, nhid)
        self.gc2 = Graphsn_GCN(nhid, nhid)
        self.decoder = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # F.celu(self.gc1(x, adj), alpha=2e-5)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        out = self.decoder(F.tanh(x))

        return x, out


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, device):
        super(GCN, self).__init__()
        self.linear = nn.Linear(nfeat, nhid)
        self.classifier = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.use_cuda = True if device == 'cuda' else False
        print("GCN initialized !")
        nn.init.xavier_normal_(self.linear.weight.data)

    def process_adj(self, A_hat, features):
        AX = torch.mm(A_hat, features)
        return AX

    def forward(self, feature):
        embs = F.dropout(feature, self.dropout)
        embs = F.tanh(self.linear(embs))
        out = self.classifier(embs)
        return embs, out
