import torch
import torch.nn as nn
import numpy as np

def output(embeddings, output_path):
    embeddings = embeddings.cpu().detach().numpy()
    with open(output_path, 'w') as file:
        file.write('\n')
        for nid in range(len(embeddings)):
            file.write('{}\t{}\n'.format(str(nid), ' '.join([str(i) for i in embeddings[nid]])))




pretrained_model = torch.load('/home/zhaohe/CKD/Data/PubMed/PubMed_transh.ckpt', map_location=lambda storage, loc: storage)
entity_encoder = nn.Embedding.from_pretrained(
    pretrained_model['ent_embeddings.weight']
)
rel_encoder = nn.Embedding.from_pretrained(pretrained_model['rel_embeddings.weight'])
# import pdb;pdb.set_trace()
norm_encoder = nn.Embedding.from_pretrained(pretrained_model['norm_vector.weight'])
entity_embeddings = entity_encoder.weight

outpath = '../data/PubMed/emb.dat'
output(entity_embeddings, outpath)