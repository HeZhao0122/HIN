#!/bin/bash


dataset='PubMed'
emb_file='emb.dat'
meta_emb='meta_emb'
link_file='link.txt'

python main.py data/${dataset}/${link_file} data/${dataset}/${emb_file} data/${dataset}/${meta_emb}
