import os
import numpy as np
import torch
import dgl

def load_data(dataset, base_path):
    link_file = f'{base_path}/{dataset}/link.dat'
    node_file = f'{base_path}/{dataset}/node.dat'
    row = []
    col = []
    with open(link_file, 'r') as linkf:
        for line in linkf:
            line = line.strip().split('\t')
            left, right = line[0], line[1]
            row.append(int(left))
            col.append(int(right))

    id2features = {}
    with open(node_file, 'r') as nodef:
        for line in nodef:
            line = line.strip().split('\t')
            nid = int(line[0])
            feature = np.array([float(i) for i in line[2].split(',')], dtype=np.float32)
            id2features[nid] = feature
    sorted(id2features)
    features = [f for _, f in id2features.items()]
    features = torch.FloatTensor(features)

    row = np.array(row)
    col = np.array(col)
    weight = torch.tensor(np.ones(len(row)), dtype=torch.double)
    edge_index = torch.tensor(np.vstack((row, col)))
    adj = torch.sparse_coo_tensor(indices=torch.LongTensor(np.vstack((row, col))), values=weight, size=(features.shape[0], features.shape[0]))
    # g = dgl.graph((torch.tensor(row).cuda(), torch.tensor(col).cuda()), num_nodes=features.shape[0], device='cuda')
    # g.edata['w'] = weight.cuda()
    features = process_adj(adj, features, a=0.1)
    return edge_index, features.cuda()

def output(args, embeddings, output_path):
    embeddings = embeddings.cpu().detach().numpy()
    with open(output_path, 'w') as file:
        file.write(
            f'size={args.size}, dropout={args.dropout}, ,topk:{args.topk}, lr={args.lr}, batch-size={args.batch_size}, epochs={args.epochs}, attributed={args.attributed}, supervised={args.supervised}\n')
        for nid in range(len(embeddings)):
            file.write('{}\t{}\n'.format(str(nid), ' '.join([str(i) for i in embeddings[nid]])))



def dense2sparse(tensor):
    idx = torch.nonzero(tensor).T
    data = tensor[idx[0], idx[1]]
    sparse = torch.sparse_coo_tensor(idx, data, tensor.shape)
    return sparse

def process_adj(adj, features, a):
    features = features.requires_grad_(False)
    adj = adj.requires_grad_(False)
    self_loop = dense2sparse(torch.eye(features.shape[0]))
    # adj = dense2sparse(adj)
    A_hat = torch.add(self_loop, adj*a)
    d = torch.sparse.sum(adj, dim=1).to_dense()
    tmp = torch.ones(d.shape, dtype=torch.double)
    # import pdb;pdb.set_trace()
    d = torch.where(d==0.0, tmp, d)
    # A_hat = dense2sparse(A_hat)
    A_hat = torch.sparse.mm(dense2sparse(torch.diag(d.pow(-0.5))), A_hat)
    A_hat = torch.sparse.mm(A_hat, dense2sparse(torch.diag(d.pow(-0.5))))
    new_features = torch.sparse.mm(A_hat.float(), features).cpu()
    # import pdb;pdb.set_trace()

    return new_features


