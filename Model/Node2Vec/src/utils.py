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
    features = torch.FloatTensor(features).cuda()

    row = np.array(row)
    col = np.array(col)
    weight = torch.FloatTensor(np.ones(len(row)))
    edge_index = torch.tensor(np.vstack((row, col)))
    g = dgl.graph((torch.tensor(row).cuda(), torch.tensor(col).cuda()), num_nodes=features.shape[0], device='cuda')
    g.edata['w'] = weight.cuda()
    return edge_index, g, features

def output(args, embeddings, output_path):
    embeddings = embeddings.cpu().detach().numpy()
    with open(output_path, 'w') as file:
        file.write(
            f'size={args.size}, dropout={args.dropout}, ,topk:{args.topk}, lr={args.lr}, batch-size={args.batch_size}, epochs={args.epochs}, attributed={args.attributed}, supervised={args.supervised}\n')
        for nid in range(len(embeddings)):
            file.write('{}\t{}\n'.format(str(nid), ' '.join([str(i) for i in embeddings[nid]])))


