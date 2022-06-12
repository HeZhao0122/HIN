#!/bin/bash

# Note: Only 'R-GCN', 'HAN', 'MAGNN', and 'HGT' support attributed='True' or supervised='True'
# Note: Only 'DBLP' and 'PubMed' support attributed='True'

dataset='ACM' # choose from 'DBLP', 'DBLP2', 'Freebase', and 'PubMed','ACM','ACM2'
model='SLAPS-GNN' # choose from 'CKD', 'metapath2vec-ESim', 'Node2Vec', 'HIN2Vec', 'R-GCN', 'HAN', 'MAGNN', 'HGT', 'TransE', and 'ConvE'
task='nc' # choose 'nc' for node classification, 'lp' for link prediction, or 'both' for both tasks
attributed='True' # choose 'True' or 'False'
supervised='True' # choose 'True' or 'False'

python evaluate.py -dataset ${dataset} -model ${model} -task ${task} -attributed ${attributed} -supervised ${supervised}