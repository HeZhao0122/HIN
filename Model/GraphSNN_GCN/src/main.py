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
torch.backends.cudnn.enabled = True

from sklearn.metrics import roc_auc_score,average_precision_score
from functools import reduce
from tqdm import tqdm



def parse_args():
    parser = argparse.ArgumentParser(description='ckd')



    parser.add_argument('--output', type=str, required=False,help='emb file',default='emb.dat')
    parser.add_argument('--ltype', type=str, default="1,4")

    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--size', type=int, default=50)#feature size
    parser.add_argument('--dim', type=int, default=30)  # output emb size
    parser.add_argument('--hidden', type=int, default=50)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--negative_cnt', type=int, default=5)
    parser.add_argument('--sample_times', type=int, default=1)
    parser.add_argument('--topk', type=int, default=20)

    parser.add_argument('--supervised', type=str, default="False")
    parser.add_argument('--neigh-por', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--stop_cnt', type=int, default=5)
    parser.add_argument('--global-weight',type=float,default=0.05)
    parser.add_argument('--eval_every', type=int, default=5)


    parser.add_argument('--attributed', type=str, default="True")


    parser.add_argument('--dataset', type=str, help='', default='ACM2')
    parser.add_argument('--alpha', type=float, help='Weight of the pairwise loss', default=0.7)
    parser.add_argument('--beta', type=float, help='Weight of the similarity loss', default=0.1)
    args=parser.parse_args()

    if args.dataset=='ACM2':
        args.dim=50
        args.ltype = '0,1'
        args.lr = 0.01
        args.topk = 15
        args.epochs = 250
        args.stop_cnt = 100
        args.global_weight = 0.1
        args.batch_size=4
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
    '''
    if args.dataset=='PubMed':
        args.ltype='1,2,4'
        args.lr=0.0002
        args.topk=25 #25
        args.epochs=100
        args.stop_cnt=100
        args.global_weight=0.1
    elif args.dataset=='acm3':
        args.ltype = '0,1'
        args.lr = 0.00002
        args.topk = 20
        args.epochs = 40
        args.stop_cnt = 100
        args.global_weight = 0.05
    elif args.dataset=='DBLP2':
        args.ltype='1,3,4,5'
        args.lr=0.00005
        args.topk=25 #25
        args.epochs=150
        args.stop_cnt=100
        args.global_weight=0.1
    return args
    '''


def main():
    print(f'start time:{datetime.now()}')

    args = parse_args()
    print(f'emb size:{args.dim}')

    print(
        f'dataset:{args.dataset},attributed:{args.attributed},ltypes:{args.ltype},'
        f'topk:{args.topk},lr:{args.lr},batch-size:{args.batch_size},stop_cnt:{args.stop_cnt},epochs:{args.epochs}')
    print(f'global weight:{args.global_weight}')

    base_path = f'../data/'
    A, X, simMs, type2id, id2name, train2label, val2label, num_class, train_pos, train_neg,\
        val_pos, val_neg = load_data(args.dataset, True if args.attributed == "True" else False, base_path, torch.cuda.is_available())
    print(f'load data finish:{datetime.now()}')
    print('node num:', A.shape[0])

    # import pdb;pdb.set_trace()
    if args.attributed != "True":
        X = np.random.randn(A.shape[0], args.size).astype(np.float32)

    G = nx.from_numpy_matrix(A.numpy())
    features = torch.FloatTensor(X.numpy())
    A_array = A.numpy()
    sub_graphs = []
    train_idx = [nid for nid, _ in train2label.items()]
    train_labels = torch.tensor([label for _, label in train2label.items()])
    val_idx = [nid for nid, _ in val2label.items()]
    val_labels = torch.tensor([label for _, label in val2label.items()])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    new_adj_path = f'{base_path}/{args.dataset}/new_adj.pickle'
    try:
        adj = pickle.load(open(new_adj_path, 'rb'))
    except:
        for i in tqdm(np.arange(len(A_array))):
            s_indexes = []
            for j in np.arange(len(A_array)):
                s_indexes.append(i)
                if (A_array[i][j] == 1):
                    s_indexes.append(j)
            sub_graphs.append(G.subgraph(s_indexes))

        subgraph_nodes_list = []

        for i in np.arange(len(sub_graphs)):
            subgraph_nodes_list.append(list(sub_graphs[i].nodes))

        sub_graphs_adj = []
        for index in np.arange(len(sub_graphs)):
            # import pdb;pdb.set_trace()
            sub_graphs_adj.append(nx.adjacency_matrix(sub_graphs[index]).toarray())

        sub_graph_edges = []
        for index in np.arange(len(sub_graphs)):
            sub_graph_edges.append(sub_graphs[index].number_of_edges())

        sub_graph_nodes_count = []
        for x in subgraph_nodes_list:
            sub_graph_nodes_count.append(len(x))

        new_adj = torch.zeros(A_array.shape[0], A_array.shape[0])

        for node in tqdm(np.arange(len(subgraph_nodes_list))):
            sub_adj = sub_graphs_adj[node]
            for neighbors in np.arange(len(subgraph_nodes_list[node])):
                index = subgraph_nodes_list[node][neighbors]
                count = torch.tensor(0).float()
                if (index == node):
                    continue
                else:
                    c_neighbors = set(subgraph_nodes_list[node]).intersection(subgraph_nodes_list[index])
                    if index in c_neighbors:
                        nodes_list = subgraph_nodes_list[node]
                        sub_graph_index = nodes_list.index(index)
                        c_neighbors_list = list(c_neighbors)
                        for i, item1 in enumerate(nodes_list):
                            if (item1 in c_neighbors):
                                for item2 in c_neighbors_list:
                                    j = nodes_list.index(item2)
                                    count += sub_adj[i][j]
                    # count 是overlap subgraph边的数量
                    new_adj[node][index] = count / 2
                    new_adj[node][index] = new_adj[node][index] / (len(c_neighbors) * (len(c_neighbors) - 1))
                    new_adj[node][index] = new_adj[node][index] * (len(c_neighbors) ** 1)

        # labels = torch.LongTensor(labels)

        weight = torch.FloatTensor(new_adj)
        weight = weight / weight.sum(1, keepdim=True)

        weight = weight + torch.FloatTensor(A_array)

        coeff = weight.sum(1, keepdim=True)
        coeff = torch.diag((coeff.T)[0])

        weight = weight + coeff

        weight = weight.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        row_sum = np.array(np.sum(weight, axis=1))
        degree_matrix = np.matrix(np.diag(row_sum + 1))

        D = fractional_matrix_power(degree_matrix, -0.5)
        A_tilde_hat = D.dot(weight).dot(D)

        adj = torch.FloatTensor(A_tilde_hat)
        pickle.dump(adj, open(new_adj_path, 'wb'))

    # Model and optimizer
    model = GNN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=num_class,
                dropout=args.dropout)
    if torch.cuda.is_available():
        model = model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        train_labels = train_labels.cuda()
        val_labels = val_labels.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    cross_entropy = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    print("Start training!")
    slosses, plosses, closses = [], [], []
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings, out = model(features, adj)

        closs = cross_entropy(out[train_idx], train_labels)
        sloss = sim_loss(simMs, embeddings, type2id, device)
        ploss = pairwise_loss(embeddings, train_pos, train_neg, device)
        loss = closs + args.alpha * ploss + args.beta * sloss
        loss.backward()
        optimizer.step()
        print('Epoch: {:04d}|loss_train: {}|sloss: {}|closs: {}|ploss: {}|time: {:.4f}s'.format(epoch+1, loss,
                                                                                                 sloss,
                                                                                                 closs,
                                                                                                 ploss,
                                                                                                 time.time()-t))
        closses.append(closs)
        plosses.append(ploss)
        slosses.append(sloss)
        if (epoch+1) % args.eval_every == 0:
            model.eval()
            val_emb, val_out = model(features, adj)
            val_closs = cross_entropy(val_out[val_idx], val_labels)
            val_sloss = sim_loss(simMs, val_emb, type2id, device)
            val_ploss = pairwise_loss(val_emb, val_pos, val_neg, device)
            val_loss = val_closs + args.alpha * val_sloss
            print('Epoch: {:04d}|val_loss: {}|sloss: {}|closs: {}|ploss: {}'.format(
                epoch+1, val_loss, val_sloss, val_closs, val_ploss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                output(args, val_emb, id2name)
    plot_loss(slosses, f'{base_path}/{args.dataset}', 'topology_loss')
    plot_loss(closses, f'{base_path}/{args.dataset}', 'classification_loss')
    plot_loss(plosses, f'{base_path}/{args.dataset}', 'link_prediction_loss')

if __name__ == '__main__':
    main()