import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
from model import *
from utils import *
from datetime import datetime
from scipy.linalg import fractional_matrix_power
torch.backends.cudnn.benchmark = True
from torch_geometric.nn import Node2Vec
from sklearn.metrics import roc_auc_score,average_precision_score
import math
from tqdm import tqdm



def parse_args():
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--output', type=str, required=False,help='emb file',default='emb.dat')
    parser.add_argument('--ltype', type=str, default="1,4")

    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--size', type=int, default=50)#feature size
    parser.add_argument('--dim', type=int, default=30)  # output emb size
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--negative_cnt', type=int, default=5)
    parser.add_argument('--sample_times', type=int, default=1)
    parser.add_argument('--topk', type=int, default=20)

    parser.add_argument('--supervised', type=str, default="False")
    parser.add_argument('--neigh-por', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--stop_cnt', type=int, default=5)
    parser.add_argument('--global-weight',type=float,default=0.05)
    parser.add_argument('--eval_every', type=int, default=5)


    parser.add_argument('--attributed', type=str, default="True")


    parser.add_argument('--dataset', type=str, help='', default='ACM2')
    parser.add_argument('--closs', type=float, help='Weight of the classification loss', default=0)
    parser.add_argument('--ploss', type=float, help='Weight of the pairwise loss', default=1)
    parser.add_argument('--sloss', type=float, help='Weight of the similarity loss', default=0.0)
    parser.add_argument('--a', type=int, default=0.5)
    args=parser.parse_args()

    if args.dataset=='ACM2':
        args.dim=50
        args.ltype = '0,1'
        args.lr = 0.01
        args.topk = 15
        args.epochs = 400
        args.stop_cnt = 100
        args.batch_size=128
    elif args.dataset=='ACM':
        args.dim=30
        args.ltype='0,2,4,6'
        args.lr=0.01
        args.topk=20 #25
        args.epochs=100
        args.stop_cnt=100
        args.global_weight=0.05
        args.batch_size=6
        args.seed = 7
    elif args.dataset=='DBLP2':
        args.dim=300
        args.ltype = '1,3,4,5'
        #args.lr = 0.00001
        args.lr = 0.00002
        args.topk = 35
        args.epochs = 60
        args.stop_cnt = 100
        args.global_weight = 0.12
        args.batch_size=2
        args.seed = 11
    elif args.dataset=='DBLP':
        args.ltype='0,1,2'
        args.lr=0.00005
        args.topk=30 #25
        args.epochs=200
        args.stop_cnt=100
        args.global_weight=0.1
    elif args.dataset=='Freebase':
        args.dim=200
        args.ltype='0,1,2,3,4'
        args.lr=0.00002
        args.topk=35 #25
        args.epochs=150
        args.stop_cnt=100
        args.global_weight=0.15
        args.batch_size=2
    elif args.dataset=='PubMed':
        args.dim=200
        args.ltype='1,2,4'
        args.lr=0.0002
        args.topk=25
        args.epochs=60
        args.stop_cnt=100
        args.global_weight=0.1
        args.batch_size=2
    return args


def main():
    print(f'start time:{datetime.now()}')

    args = parse_args()
    print(f'emb size:{args.dim}')

    print(
        f'dataset:{args.dataset},attributed:{args.attributed},ltypes:{args.ltype},'
        f'topk:{args.topk},lr:{args.lr},batch-size:{args.batch_size},stop_cnt:{args.stop_cnt},epochs:{args.epochs}')
    print(f'global weight:{args.global_weight}')

    base_path = f'../data/'
    adj, features, simMs, type2id, id2name, train2label, val2label, num_class, edge_index = \
        load_data(args.dataset, True if args.attributed == "True" else False, base_path, torch.cuda.is_available())
    node_num = len(id2name)
    print(f'load data finish:{datetime.now()}')
    print('node num:', node_num)

    # import pdb;pdb.set_trace()
    if args.attributed != "True":
        features = torch.tensor(np.random.randn(adj.shape[0], args.size).astype(np.float32), dtype=torch.double)

    train_idx = [nid for nid, _ in train2label.items()]
    train_edge_index, val_edge_index = edge_index
    train_labels = torch.tensor([label for _, label in train2label.items()])
    val_idx = [nid for nid, _ in val2label.items()]
    val_labels = torch.tensor([label for _, label in val2label.items()])
    device = torch.device('cuda')if torch.cuda.is_available() else 'cpu'

    new_features = process_adj(adj, features, args.a).to(device)

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=args.dim,
                nclass=num_class,
                dropout=args.dropout,
                device=device)
    train_labels = train_labels.to(device)
    val_labels = val_labels.to(device)
    model = model.to(device)

    train_node2vec = Node2Vec(train_edge_index, embedding_dim=args.dim, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, num_nodes=new_features.shape[0], sparse=True)
    train_loader = train_node2vec.loader(batch_size=64, shuffle=True, num_workers=4)

    val_node2vec = Node2Vec(val_edge_index, embedding_dim=args.dim, walk_length=20,
                              context_size=10, walks_per_node=10,
                              num_negative_samples=1, p=1, q=1, num_nodes=new_features.shape[0], sparse=True)
    val_loader = val_node2vec.loader(batch_size=64, shuffle=True, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    cross_entropy = nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    print("Start training!")
    slosses, plosses, closses = [], [], []
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        nids = [i for i in range(adj.shape[0])]

        optimizer.zero_grad()
        embeddings, outs = [], []
        for i in range(math.ceil(adj.shape[0] / args.batch_size)):
            try:
                nid = nids[i * args.batch_size:(i + 1) * args.batch_size]
            except:
                nid = nids[i * args.batch_size: adj.shape[0]]
            embedding, out = model(new_features[nid])
            embeddings.append(embedding)
            outs.append(out)
        embeddings = torch.cat(embeddings, dim=0)
        outs = torch.cat(outs, dim=0)

        closs = cross_entropy(outs[train_idx], train_labels)*args.closs
        '''
        closs = 0.0
        for i in range(math.ceil(len(train_idx)/args.batch_size)):
            try:
                nid = train_idx[i*args.batch_size:(i+1)*args.batch_size]
                batch_label = train_labels[i*args.batch_size:(i+1)*args.batch_size]
            except:
                nid = train_idx[i*args.batch_size: adj.shape[0]]
                batch_label = train_labels[i*args.batch_size: adj.shape[0]]
            c_loss = cross_entropy(outs[nid], batch_label)
            # c_loss.backward(retain_graph=True)
            closs += c_loss
        closs /= math.ceil(len(train_idx)/args.batch_size)
        '''
        # closs.backward(retain_graph=True)
        '''
        ploss = pairwise_loss(embeddings, train_pos, train_neg, device) * args.alpha
        # ploss.backward(retain_graph=True)
        '''
        ploss = 0.0
        cnt = 0
        for pos_rw, neg_rw in train_loader:
            cnt += 1
            skip_gram_loss = pairwise_loss(embeddings, pos_rw.to(device), neg_rw.to(device))*args.ploss
            skip_gram_loss.backward(retain_graph=True)
            ploss += skip_gram_loss.item()
        ploss = ploss/cnt
        # ploss.backward(retain_graph=True)



        sloss = sim_loss(simMs, embeddings, type2id, device) * args.sloss
        # sloss.backward()

        loss = closs + ploss + sloss
        loss.backward()
        optimizer.step()
        print('Epoch: {:04d}|loss_train: {}|sloss: {}|closs: {}|ploss: {}|time: {:.4f}s'.format(epoch+1, loss,
                                                                                                 sloss.item(),
                                                                                                 closs,
                                                                                                 ploss,
                                                                                                 time.time()-t))
        closses.append(closses)
        plosses.append(ploss)
        slosses.append(sloss)
        if (epoch+1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_emb = []
                val_out = []
                for i in range(math.ceil(adj.shape[0] / args.batch_size)):
                    try:
                        nid = nids[i * args.batch_size:(i + 1) * args.batch_size]
                    except:
                        nid = nids[i * args.batch_size: adj.shape[0]]
                    embedding, out = model(new_features[nid])
                    val_emb.append(embedding)
                    val_out.append(out)
                val_emb = torch.cat(val_emb, dim=0)
                val_out = torch.cat(val_out, dim=0)
                val_closs = cross_entropy(val_out[val_idx], val_labels)*args.closs
                val_sloss = sim_loss(simMs, val_emb, type2id, device) * args.sloss
                # val_ploss = pairwise_loss(val_emb, val_pos, val_neg, device) * args.alpha

                val_ploss = 0.0
                cnt=0
                for pos_rw, neg_rw in val_loader:
                    cnt += 1
                    skip_gram_loss = pairwise_loss(val_emb, pos_rw.to(device), neg_rw.to(device)) * args.ploss
                    val_ploss += skip_gram_loss
                val_ploss /= cnt


                val_loss = val_closs + val_sloss + val_ploss
                print('Epoch: {:04d}|val_loss: {}|sloss: {}|closs: {}|ploss: {}'.format(
                    epoch+1, val_loss, val_sloss, val_closs, val_ploss))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    output(args, val_emb, id2name)

        # del embeddings, outs, closs, ploss, sloss, loss
        torch.cuda.empty_cache()

    plot_loss(slosses, f'{base_path}/{args.dataset}', 'topology_loss')
    plot_loss(closses, f'{base_path}/{args.dataset}', 'classification_loss')
    plot_loss(plosses, f'{base_path}/{args.dataset}', 'link_prediction_loss')

if __name__ == '__main__':
    main()