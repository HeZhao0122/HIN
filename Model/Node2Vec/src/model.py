import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

from torch_geometric.utils.num_nodes import maybe_num_nodes

try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

EPS = 1e-15


class GCNConv_dgl(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g):
        with g.local_scope():
            g.ndata['h'] = self.linear(x)
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            return g.ndata['h']


class MyGCN(nn.Module):
    def __init__(self, nfeats, hidden_size, features):
        super(MyGCN, self).__init__()
        self.embeddings = nn.Embedding(features.shape[0], features.shape[1])
        self.embedding_dim = hidden_size
        self.embeddings.weight.requires_grad_(False)
        self.embeddings.weight = nn.Parameter(features)
        self.linear = nn.Linear(nfeats, hidden_size)

    def forward(self, pos_rw, neg_rw):
        # x = F.tanh(self.linear(x))
        # x = F.relu(x)
        return self.loss(pos_rw, neg_rw)
    '''
    def loss(self, embedding, pos_rw, neg_rw):
        # Positive loss.
        embedding_dim = embedding.shape[1]
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = embedding[start].view(pos_rw.size(0), 1, embedding_dim)
        h_rest = embedding[rest.view(-1)].view(pos_rw.size(0), -1, embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = embedding[start].view(neg_rw.size(0), 1, embedding_dim)
        h_rest = embedding[rest.view(-1)].view(neg_rw.size(0), -1, embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss
    '''

    def embedding(self, idx):
        return self.linear(F.tanh(self.embeddings(idx)))


    def loss(self, pos_rw, neg_rw):
        r"""Computes the loss given positive and negative random walks."""

        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

    def output(self):
        return self.linear(self.embeddings.weight)