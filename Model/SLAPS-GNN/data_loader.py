# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import warnings

import torch
import numpy as np
import random
import os

from citation_networks import load_citation_network, sample_mask

warnings.simplefilter("ignore")


def load_ogb_data(dataset_str):
    from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(dataset_str)

    data = dataset[0]
    features = data.x
    nfeats = data.num_features
    nclasses = dataset.num_classes
    labels = data.y

    split_idx = dataset.get_idx_split()

    train_mask = sample_mask(split_idx['train'], data.x.shape[0])
    val_mask = sample_mask(split_idx['valid'], data.x.shape[0])
    test_mask = sample_mask(split_idx['test'], data.x.shape[0])

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels).view(-1)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask


def load_data(args):
    dataset_str = args.dataset

    if dataset_str.startswith('ogb'):
        return load_ogb_data(dataset_str)

    if dataset_str in ["ACM2", "ACM", "DBLP", "PubMed", "DBLP2"]:
        return

    return load_citation_network(dataset_str)


def load_hin_data(args):
    data_folder = f'../../Data/{args.dataset}'
    node_file = 'node.dat'
    label_file = 'label.dat'
    label_test_file = 'label.dat.test'

    id2type = {}
    id2feature = {}
    with open(os.path.join(data_folder, node_file)) as nf:
        for line in nf:
            line = line.strip().split('\t')
            nid, tid, feature = int(line[0]), int(line[2]), np.array([float(i) for i in line[3].split(',')], dtype=np.float32)
            id2type[nid] = tid
            id2feature[nid] = feature
    sorted(id2feature)
    features = np.array([f for nid, f in id2feature.items()])

    id2label = {}
    with open(os.path.join(data_folder, label_file)) as lf:
        for line in lf:
            line = line.strip().split('\t')
            nid, label = int(line[0]), int(line[-1])
            id2label[nid] = label

    yl = [nid for nid in id2label]
    random.shuffle(yl)
    labels = np.array([id2label[l] for l in yl])
    ratio = 0.3
    train_idx = yl[:int(ratio*len(yl))]
    val_idx = yl[int(ratio*len(yl)):]

    test_idx = []
    test_labels = []
    with open(os.path.join(data_folder, label_test_file), 'r') as tlf:
        for line in tlf:
            line = line.strip().split('\t')
            nid, label = int(line[0]), int(line[-1])
            test_idx.append(nid)
            test_labels.append(label)

    labels = np.concatenate((labels, np.array(test_labels)))
    train_mask = sample_mask(range(int(ratio*len(yl))), labels.shape[0])
    val_mask = sample_mask(range(int(ratio*len(yl)), len(yl)), labels.shape[0])
    test_mask = sample_mask(range(len(yl), len(labels)), labels.shape[0])

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    nfeats = features.shape[1]
    nclasses = torch.max(labels).item() + 1

    # import pdb;pdb.set_trace()

    return features, nfeats, labels, nclasses, train_mask, val_mask,test_mask, train_idx, val_idx, test_idx



