import os.path as osp
import argparse
from model import MyGCN

from torch_geometric.nn import Node2Vec
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Node2Vec')

    parser.add_argument('--output', type=str, required=False,help='emb file',default='emb.dat')
    parser.add_argument('--ltype', type=str, default="1,4")

    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--size', type=int, default=50)#feature size
    parser.add_argument('--dim', type=int, default=50)  # output emb size
    parser.add_argument('--hidden', type=int, default=50)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--negative_cnt', type=int, default=5)
    parser.add_argument('--sample_times', type=int, default=1)
    parser.add_argument('--topk', type=int, default=20)

    parser.add_argument('--supervised', type=str, default="False")
    parser.add_argument('--neigh-por', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--stop_cnt', type=int, default=5)
    parser.add_argument('--global-weight',type=float,default=0.05)
    parser.add_argument('--eval_every', type=int, default=5)


    parser.add_argument('--attributed', type=str, default="True")


    parser.add_argument('--dataset', type=str, help='', default='ACM')
    parser.add_argument('--alpha', type=float, help='Weight of the pairwise loss', default=0.7)
    parser.add_argument('--beta', type=float, help='Weight of the similarity loss', default=0.0)
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
        args.dim=100
        args.ltype='0,2,4,6'
        args.lr=0.00001
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
        args.dim=100
        args.ltype='1,2,4'
        args.lr=0.01
        args.topk=25
        args.epochs=60
        args.stop_cnt=100
        args.global_weight=0.1
        args.batch_size=2
    return args


def main(args):
    data_dir = '../data'
    emb_file = 'emb.dat'
    edge_index, g, features = load_data(args.dataset, data_dir)
    path_gen = Node2Vec(edge_index, embedding_dim=100, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True)
    loader = path_gen.loader(batch_size=256, shuffle=True, num_workers=4)
    model = MyGCN(nfeats=features.shape[1], hidden_size=100).cuda()
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        cnt = 0
        embeddings = model(features, g)
        for pos_rw, neg_rw in loader:
            cnt += 1
            loss = model.loss(embeddings, pos_rw.to(args.device), neg_rw.to(args.device))
            total_loss += loss
        total_loss /= cnt
        total_loss.backward()
        optimizer.step()
        output(args, embeddings, f'{data_dir}/{args.dataset}/{emb_file}')
        # import pdb;pdb.set_trace()
        return total_loss.item()


    for epoch in range(1, 151):
        loss = train()
        if epoch%10 == 0:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')


args = parse_args()
main(args)