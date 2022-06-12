#!/bin/bash

# Note: Only 'R-GCN', 'HAN', 'MAGNN', and 'HGT' support attributed='True' or supervised='True'
# Note: Only 'DBLP' and 'PubMed' contain node attributes.

dataset='ACM2' # choose from 'DBLP', 'DBLP2', 'Freebase', and 'PubMed','ACM','ACM2'
model='Node2Vec' # choose from 'metapath2vec-ESim', 'CKD','Node2Vec', 'HIN2Vec', 'R-GCN', 'HAN', 'MAGNN', 'HGT', 'ComplEx', and 'ConvE'
attributed='False' # choose 'True' or 'False'
supervised='False' # choose 'True' or 'False'
version='link'

mkdir ../Model/${model}/data
mkdir ../Model/${model}/data/${dataset}
if [ ${model} = 'CKD' ]
then
  echo "mkdir success"
  mkdir ../Model/${model}/data/${dataset}/${version}
fi

python transform.py -dataset ${dataset} -model ${model} -attributed ${attributed} -supervised ${supervised} -version ${version}