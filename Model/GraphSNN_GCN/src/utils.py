import numpy as np
import pandas as pd
import networkx as nx
import os
import random
import torch
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from link_loader import data_loader
import os
import scipy.sparse as sp
from collections import Counter, defaultdict
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def load_node(path):
    """
    Load the old ID of the node and the characteristics of the node. If the node has no characteristics,
    it will be initialized directly at random
    """
    node2attr={}
    with open(path) as f:
        for line in f:
            data=line.strip().split('\t')
            node2attr[int(data[0])]=np.array([float(i) for i in data[2].split(',')],dtype=np.float32)
    return node2attr


def load_data(dataset, use_features, base_path, use_cuda):
    '''
    create adj_matrix and feature X, both tensor matrix
    '''
    node2type_file = 'node2type.txt'
    link_file = 'link.dat'
    train_label_node = 'label.dat.train'
    val_label_node = 'label.dat.val'
    mapping_file = 'name2id.txt'
    # org_data_root = f'../../../Data/{dataset}'

    node2type = dict()
    type2id = dict()
    id2name = dict()
    train2label = dict()
    val2label = dict()
    name2id = dict()
    label_class = []
    dl = data_loader(f'{base_path}/{dataset}', symetric=False if dataset=='ACM2' else True)

    with open(f'{base_path}/{dataset}/{train_label_node}', 'r') as train_lf:
        for line in train_lf:
            line = line.strip().split('\t')
            train2label[int(line[0])] = int(line[1])
            if int(line[1]) not in label_class:
                label_class.append(int(line[1]))
    with open(f'{base_path}/{dataset}/{val_label_node}', 'r') as val_lf:
        for line in val_lf:
            line = line.strip().split('\t')
            val2label[int(line[0])] = int(line[1])
            if int(line[1]) not in label_class:
                label_class.append(int(line[1]))
    num_class = len(label_class)

    with open(f'{base_path}/{dataset}/{node2type_file}', 'r')as node_file:
        for line in node_file:
            line = line.strip().split('\t')
            node2type[int(line[0])] = int(line[1])
            if int(line[1]) not in type2id:
                type2id[int(line[1])] = []
            type2id[int(line[1])].append(int(line[0]))
    type2id = {type: sorted(nodes) for type, nodes in type2id.items()}

    with open(f'{base_path}/{dataset}/{mapping_file}', 'r') as mapping_f:
        for line in mapping_f:
            line = line.strip().split('\t')
            node, nid = int(line[0]), int(line[1])
            name2id[node] = nid
            id2name[nid] = node
    link_file = f'{base_path}/{dataset}/{link_file}'
    row = []
    col = []
    with open(link_file, 'r') as linkf:
        for line in linkf:
            line = line.strip().split('\t')
            left, right = line[0], line[1]
            row.append(int(left))
            col.append(int(right))
            if dataset == 'ACM2':
                row.append(int(right))
                col.append(int(left))

    link_num = len(row)
    np_row = np.array(row+col)
    np_col = np.array(col+row)
    indices = np.vstack((np_row, np_col)).T
    weight = np.ones(len(np_row))
    np.random.shuffle(indices)
    split_ratio = 0.9
    train_edge_index = torch.tensor(indices.T[:, :int(link_num*split_ratio)])
    val_edge_index = torch.tensor(indices.T[:,int(link_num*split_ratio):])

    # adjM = sum(dl.links['data'].values()).tocoo()
    adjM = torch.sparse_coo_tensor(torch.LongTensor(indices.T), torch.tensor(weight), size=(len(node2type), len(node2type)))
    # train_pos, train_neg = dl.train_pos, dl.train_neg
    # val_pos, val_neg = dl.valid_pos, dl.valid_neg
    # import pdb;pdb.set_trace()
    simMs = {}
    simM_path = f'{base_path}/{dataset}/simMs.pickle'
    try:
        simMs = pickle.load(open(simM_path, 'rb'))
    except:
        simMs = get_batch_simM(adjM, type2id, node2type, dl, use_cuda)
        pickle.dump(simMs, open(simM_path, 'wb'))
    # get features
    features = None
    if use_features:
        features = []
        name2attr = load_node(path=f'{base_path}/{dataset}/node.dat')
        for i in range(adjM.shape[0]):
            feature = name2attr[i]
            features.append(feature)
        features = np.array(features, dtype=np.float32)
        features = torch.tensor(features)
    return torch.tensor(adjM.to_dense()), features, simMs, type2id, id2name, train2label, val2label, num_class,\
           [train_edge_index, val_edge_index]


def get_org_simM(adjM, type2id, node2type, dl, use_cuda):
    simMs = {}
    row = adjM.row
    col = adjM.col
    values = torch.tensor(np.ones(len(row)))
    for tid, _ in type2id.items():
        nodes = type2id[tid]
        idx = []
        for i in range(len(row)):
            if node2type[row[i]] == tid:
                idx.append(i)
        new_row = row[idx] - dl.nodes['shift'][tid]
        new_col = col[idx]
        new_val = values[idx]
        tadjM = torch.sparse_coo_tensor(torch.LongTensor(np.vstack((new_row, new_col))), new_val,
                                        size=(len(nodes), dl.nodes['total']))

        d = torch.sparse.sum(tadjM, dim=1).to_dense()
        t_simM = torch.zeros(len(nodes), len(nodes))
        for tid2, _ in type2id.items():
            t_row, t_col = new_row, new_col
            t_values = new_val
            nodes2 = type2id[tid2]
            tidx = []
            for j in range(len(t_values)):
                if node2type[t_col[j]] == tid2:
                    tidx.append(j)
            t_indices = torch.LongTensor(np.vstack((t_row[tidx], t_col[tidx]-dl.nodes['shift'][tid2])))
            t_values = t_values[tidx]
            tt_adjM = torch.sparse_coo_tensor(t_indices, t_values, size=(len(nodes), len(nodes2))).to_dense()

            tt_adjM.requires_grad = False
            tt_adjM = (tt_adjM.t() / torch.sqrt(d)).t()
            tt_simM = torch.zeros(len(nodes), len(nodes))
            shape = tt_adjM.shape
            if use_cuda:
                tt_adjM = tt_adjM.cuda()
                tt_simM = tt_simM.cuda()

            for i in tqdm(range(len(nodes))):
                tmp = torch.sum((tt_adjM[i].expand(shape) - tt_adjM) * (tt_adjM[i].expand(shape) - tt_adjM), axis=1)
                tt_simM[i] = tmp
            t_simM += tt_simM.cpu().detach()
        simMs[tid] = t_simM
        # import pdb;pdb.set_trace()
    return simMs


def get_batch_simM(adjM, type2id, node2type, dl, use_cuda):
    simMs = {}
    row = adjM.row
    col = adjM.col
    values = torch.tensor(np.ones(len(row)))
    for tid, _ in type2id.items():
        nodes = type2id[tid]
        idx = []
        for i in range(len(row)):
            if node2type[row[i]] == tid:
                idx.append(i)
        new_row = row[idx] - dl.nodes['shift'][tid]
        new_col = col[idx]
        new_val = values[idx]
        # import pdb;pdb.set_trace()
        tadjM = torch.sparse_coo_tensor(torch.LongTensor(np.vstack((new_row, new_col))), new_val,
                                        size=(len(nodes), dl.nodes['total']))

        d = torch.sparse.sum(tadjM, dim=1).to_dense()
        t_simM = torch.zeros(len(nodes), len(nodes))

        batch_size = 5000
        epochs = int(dl.nodes['total']/batch_size) + 1

        for epoch in tqdm(range(epochs)):
            shift = epoch * batch_size
            t_row, t_col = new_row, new_col
            t_values = new_val
            if shift >= dl.nodes['total']:
                break
            if shift+batch_size <= dl.nodes['total']:
                nodes2 = [i for i in range(batch_size)]
            else:
                nodes2 = [i for i in range(dl.nodes['total']-shift)]
            tidx = []
            for j in range(len(t_values)):
                if 0 <= t_col[j]-shift < len(nodes2):
                    tidx.append(j)
            t_indices = torch.LongTensor(np.vstack((t_row[tidx], t_col[tidx]-shift)))
            t_values = t_values[tidx]
            tt_adjM = torch.sparse_coo_tensor(t_indices, t_values, size=(len(nodes), len(nodes2))).to_dense()

            tt_adjM.requires_grad = False
            tt_adjM = (tt_adjM.t() / torch.sqrt(d)).t()
            tt_simM = torch.zeros(len(nodes), len(nodes))
            shape = tt_adjM.shape
            if use_cuda:
                tt_adjM = tt_adjM.cuda()
                tt_simM = tt_simM.cuda()

            for i in range(len(nodes)):
                tmp = torch.sum((tt_adjM[i].expand(shape) - tt_adjM) * (tt_adjM[i].expand(shape) - tt_adjM), axis=1)
                tt_simM[i] = tmp
            t_simM += tt_simM.cpu().detach()
        simMs[tid] = t_simM
        # import pdb;pdb.set_trace()
    return simMs


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sim_loss(simMs, embeddings, type2id, device='cuda'):
    bce_loss = torch.nn.BCEWithLogitsLoss()
    embeddings = F.normalize(embeddings, p=2, dim=1)
    loss = 0.0
    for tid in type2id:
        simM = simMs[tid].to(device)
        idx = type2id[tid]
        t_emb = embeddings[idx]
        t_sim = torch.mm(t_emb, t_emb.t())
        t_simM = simM*0.5
        t_simM = torch.ones(t_simM.shape).to(device)-t_simM
        loss += bce_loss(t_sim, t_simM)

    return loss


def output(args, embeddings, id2name):
    '''
    print('output data')
    if need_handle:
        for idx, emb in enumerate(embeddings):
            if type(emb) != np.ndarray:
                embeddings[idx] = emb.cpu().detach().numpy()
    # embeddings = sum(embeddings)
    output_path = f'../data/{args.dataset}/{args.output}'
    embeddings = embeddings.cpu().detach().numpy()
    with open(output_path, 'w') as file:
        file.write(
            f'size={args.size}, dropout={args.dropout}, ,topk:{args.topk}, lr={args.lr}, batch-size={args.batch_size}, epochs={args.epochs}, attributed={args.attributed}, supervised={args.supervised}\n')
        for nid, name in id2name.items():
            file.write('{}\t{}\n'.format(name, embeddings[nid]))
    '''
    embeddings = embeddings.cpu().detach().numpy()
    output_path = f'../data/{args.dataset}/{args.output}'
    with open(output_path, 'w') as file:
        file.write(
            f'size={args.size}, dropout={args.dropout}, ,topk:{args.topk}, lr={args.lr}, batch-size={args.batch_size}, epochs={args.epochs}, attributed={args.attributed}, supervised={args.supervised}\n')
        for nid, name in id2name.items():
            file.write('{}\t{}\n'.format(str(name), ' '.join([str(i) for i in embeddings[nid]])))


def pairwise_loss(embedding, pos_rw, neg_rw, device='cuda'):
    embedding = F.normalize(embedding, p=2, dim=1)
    pair_loss = torch.nn.MarginRankingLoss(margin=1)
    '''
    pair_loss = torch.nn.MarginRankingLoss(margin=1)
    loss = 0.0
    for rid in pos_rw:
        pos_score = torch.sum(embedding[pos_rw[rid][0]] * embedding[pos_rw[rid][1]], dim=1).view(1,-1).squeeze(0)
        neg_score = torch.sum(embedding[neg_rw[rid][0]] * embedding[neg_rw[rid][1]], dim=1).view(1,-1).squeeze(0)
        target = torch.ones(len(pos_score)).to(device)
        loss += pair_loss(pos_score, neg_score, target)
    '''
    EPS = 1e-15
    embedding_dim = embedding.shape[1]
    # Positive loss.
    start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

    h_start = embedding[start].view(pos_rw.size(0), 1, embedding_dim)
    h_rest = embedding[rest.view(-1)].view(pos_rw.size(0), -1, embedding_dim)

    pos_out = (h_start * h_rest).sum(dim=-1).view(-1)
    # pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

    # Negative loss.
    start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

    h_start = embedding[start].view(neg_rw.size(0), 1, embedding_dim)
    h_rest = embedding[rest.view(-1)].view(neg_rw.size(0), -1, embedding_dim)

    neg_out = (h_start * h_rest).sum(dim=-1).view(-1)
    target = torch.ones(len(pos_out)).cuda()
    # neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()
    return pair_loss(pos_out, neg_out, target)

    # return pos_loss + neg_loss

    #return loss/len(pos_rw)


def plot_loss(losses, base_path, name):
    x = [i for i in range(len(losses))]
    y = [loss.detach().cpu().numpy() for loss in losses]
    plt.plot(x, y)
    plt.title(name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{base_path}/{name}.png')


def dense2sparse(tensor):
    idx = torch.nonzero(tensor).T
    data = tensor[idx[0], idx[1]]
    sparse = torch.sparse_coo_tensor(idx, data, tensor.shape)
    return sparse

def process_adj(adj, features, a):
    features = features.requires_grad_(False).cuda()
    adj = adj.requires_grad_(False).cuda()
    self_loop = dense2sparse(torch.eye(adj.shape[0]).cuda())
    adj = dense2sparse(adj)
    A_hat = torch.add(self_loop, adj*a)
    d = torch.sparse.sum(adj, dim=1).to_dense()
    tmp = torch.ones(d.shape, dtype=torch.double).cuda()
    # import pdb;pdb.set_trace()
    d = torch.where(d==0.0, tmp, d)
    # A_hat = dense2sparse(A_hat)
    A_hat = torch.sparse.mm(dense2sparse(torch.diag(d.pow(-0.5))), A_hat)
    A_hat = torch.sparse.mm(A_hat, dense2sparse(torch.diag(d.pow(-0.5))))
    new_features = torch.sparse.mm(A_hat.float(), features).cpu()
    # import pdb;pdb.set_trace()

    return new_features
