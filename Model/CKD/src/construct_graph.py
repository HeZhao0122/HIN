import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import networkx as nx
from tqdm import tqdm
from scipy.sparse import lil_matrix
import argparse


def projection_transH(original, norm):
    return original - torch.sum(original * norm, dim=len(original.size()) - 1, keepdim=True) * norm


def projection_transR(original, proj_matrix):  # (batch, k_dim)   (batch, k_dim * dim) k_dim可以与dim相同
    ent_embedding_size = original.shape[1]
    rel_embedding_size = proj_matrix.shape[1] // ent_embedding_size
    original = original.view(-1, ent_embedding_size, 1)  # (batch, k_dim, 1)
    proj_matrix = proj_matrix.view(-1, rel_embedding_size, ent_embedding_size)  # (batch, dim, k_dim)
    return torch.matmul(proj_matrix, original).view(-1, rel_embedding_size)  # (batch, dim, 1) -> (batch, dim)


def calculate_score(h_e, t_e, rel, model_type, L1_flag=True, norm=None, proj=None):
    if model_type == "transe":
        if L1_flag:
            score = torch.exp(-torch.sum(torch.abs(h_e + rel - t_e), 1))
        else:
            score = torch.exp(-torch.sum(torch.abs(h_e + rel - t_e) ** 2, 1))

    elif model_type == "transh":
        proj_h_e = projection_transH(h_e, norm)
        proj_t_e = projection_transH(t_e, norm)
        if L1_flag:
            score = torch.exp(-torch.sum(torch.abs(proj_h_e + rel - proj_t_e), 1))
        else:
            score = torch.exp(-torch.sum(torch.abs(proj_h_e + rel - proj_t_e) ** 2, 1))

    else:
        proj_h_e = projection_transR(h_e, proj)
        proj_t_e = projection_transR(t_e, proj)
        if L1_flag:
            score = torch.exp(-torch.sum(torch.abs(proj_h_e + rel - proj_t_e), 1))
        else:
            score = torch.exp(-torch.sum(torch.abs(proj_h_e + rel - proj_t_e) ** 2, 1))

    return score


def construct_transition_graph(threshold, entity_encoder, rel_emb, heads, tails, model_type='transh', norm=None, proj=None):
    # loc_count = args.loc_coun
    L1_flag = True

    bar = tqdm(total=len(heads))
    bar.set_description('Construct Transition Graph')

    transition_graph = {} #nid ->[topK neighbours]
    final_graph = {}
    score_matrix = torch.zeros(size=(len(heads), len(tails)))
    for i in range(len(heads)):
        transition_graph[heads[i]] = []
        h_e = entity_encoder(torch.LongTensor([heads[i]]))
        t_e = entity_encoder(torch.LongTensor(tails))
        transition_vector = calculate_score(h_e, t_e, rel_emb, model_type, L1_flag, norm, proj)
        score_matrix[heads[i]] = transition_vector
        '''
        indices = torch.argsort(transition_vector, descending=True)[:threshold]
        for index in indices:
            transition_graph[heads[i]].append(tails[int(index)])
        '''
        bar.update(1)

    h2h_matrix = torch.mm(score_matrix, score_matrix.T)
    for i in range(len(heads)):
        final_graph[heads[i]] = []
        transition_vector = h2h_matrix[i]
        indices = torch.argsort(transition_vector, descending=True)[:threshold]
        for index in indices:
            final_graph[heads[i]].append(heads[index])
        if heads[i] not in final_graph[heads[i]]:
            final_graph[heads[i]] = final_graph[heads[i]][:-1]
            final_graph[heads[i]].append(heads[i])
    '''
    t2h_transition_graph = {}
    score_matrix = score_matrix.T
    for j in range(len(tails)):
        t2h_transition_graph[tails[j]] = []
        transition_vector = score_matrix[j]
        indices = torch.argsort(transition_vector, descending=True)[:threshold]
        for index in indices:
            t2h_transition_graph[tails[j]].append(heads[int(index)])
        bar.update(1)
    bar.close()
    # import pdb;pdb.set_trace()

    for hid in heads:
        h_neighbor = transition_graph[hid]
        # 加self—loop
        final_graph[hid] = [hid]
        for tid in h_neighbor:
            final_graph[hid] += t2h_transition_graph[tid]
    '''
    final_graph = {hid: set(neighbor) for hid, neighbor in final_graph.items()}
    '''
    transition_graph = lil_matrix((loc_count, loc_count), dtype=np.float32)  # 有向图
    for i in range(loc_count):
        h_e = entity_encoder(torch.LongTensor([i]))
        t_list = list(range(loc_count))

        # t_e = loc_encoder(torch.LongTensor(list(range(loc_count))))
        t_e = entity_encoder(torch.LongTensor(t_list[:i] + t_list[i + 1:]))
        transition_vector = calculate_score(h_e, t_e, temporal_preference, model_type, L1_flag, norm, proj)

        # indices = torch.argsort(transition_vector, descending=True)[1:threshold + 1]  # 选top_k,第一个index必是自身,即i
        indices = torch.argsort(transition_vector, descending=True)[:threshold]  # 选top_k
        norm = transition_vector[indices].sum()
        for index in indices:
            index = index.item()
            transition_graph[i, index] = (transition_vector[index] / norm).item()

        bar.update(1)
    with open(filename, 'wb') as f:
        pickle.dump(transition_graph, f, protocol=2)
    '''
    return final_graph


def output_graph(file_path, graph, name2id):
    out_file = open(file_path, 'w')
    for node in graph:
        for tid in graph:
            out_file.write(f'{str(name2id[node])}\t{str(name2id[tid])}\n')
    out_file.close()

def get_ckd_graph(heads, tails, relation, pretrained_path, file_path, name2id, topk=5):
    pretrain_model = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    entity_encoder = nn.Embedding.from_pretrained(pretrain_model['ent_embeddings.weight'])
    rel_encoder = nn.Embedding.from_pretrained(pretrain_model['rel_embeddings.weight'])
    relation_emb = rel_encoder(torch.LongTensor([relation]))
    norm_encoder = nn.Embedding.from_pretrained(pretrain_model['norm_vector.weight'])
    norm_emb = norm_encoder(torch.LongTensor([relation]))
    graph = construct_transition_graph(topk, entity_encoder, relation_emb, heads, tails, norm=norm_emb)
    output_graph(file_path, graph, name2id)
    return graph

def get_parser():
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument("--model_type", default="transh", type=str, help="使用哪种KGE方法")
    # parser.add_argument("--pretrain_model", default="../data/gowalla-transe-1637901500.ckpt", type=str, help="加载模型")
    parser.add_argument("--pretrain_model", default="../data/gowalla-transh-1638513816.ckpt", type=str, help="加载模型")
    # parser.add_argument("--pretrain_model", default="../data/gowalla-transr-1639027562.ckpt", type=str, help="加载模型")
    parser.add_argument("--version", default="scheme1", type=str, help="使用哪种版本的KG")
    parser.add_argument("--threshold", default=20, type=int, help="构造稀疏转移graph")
    parser.add_argument("--user_count", default=7768, type=int, help="用户数目")
    parser.add_argument("--loc_count", default=106994, type=int, help="POI数目")
    parser.add_argument("--L1_flag", default=True, type=bool, help="使用L1范数")
    args = parser.parse_args()
    return args


def main():
    args = get_parser()
    pretrain_model = torch.load(args.pretrain_model, map_location=lambda storage, loc: storage)
    user_count = args.user_count
    graph_file = './' + args.version + '_' + args.model_type + '_' + 'user' + '_' + str(args.threshold) + '.pkl'

    print(graph_file)

    user_encoder = nn.Embedding.from_pretrained(
        pretrain_model['ent_embeddings.weight'][:user_count])
    loc_encoder = nn.Embedding.from_pretrained(
        pretrain_model['model_state_dict']['ent_embeddings.weight'][user_count:])
    rel_encoder = nn.Embedding.from_pretrained(pretrain_model['model_state_dict']['rel_embeddings.weight'])

    temporal_preference = rel_encoder(torch.LongTensor([1]))
    friend_preference = rel_encoder(torch.LongTensor([3]))

    if args.model_type == "transh":
        norm_encoder = nn.Embedding.from_pretrained(pretrain_model['model_state_dict']['norm_embeddings.weight'])

        norm_temporal = norm_encoder(torch.LongTensor([1]))
        norm_friend = norm_encoder(torch.LongTensor([3]))
        # construct_transition_graph(args, graph_file, loc_encoder, temporal_preference, norm=norm_temporal)
        construct_transition_graph(args, graph_file, user_encoder, friend_preference, norm=norm_friend)

    elif args.model_type == "transr":
        proj_encoder = nn.Embedding.from_pretrained(pretrain_model['model_state_dict']['proj_embeddings.weight'])

        proj_temporal = proj_encoder(torch.LongTensor([1]))
        proj_friend = proj_encoder(torch.LongTensor([3]))
        # construct_transition_graph(args, graph_file, loc_encoder, temporal_preference, proj=proj_temporal)
        construct_transition_graph(args, graph_file, user_encoder, friend_preference, norm=proj_friend)
    else:
        # construct_transition_graph(args, graph_file, loc_encoder, temporal_preference)
        construct_transition_graph(args, graph_file, user_encoder, friend_preference)

if __name__ == '__main__':
    main()