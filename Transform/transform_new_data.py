import numpy as np
from collections import defaultdict
import networkx as nx
import torch
from sklearn.model_selection import train_test_split
data_folder, model_folder = '../Data', '../Model'
node_file, link_file, label_file = 'node.dat', 'link.dat', 'label.dat'
info_file, meta_file = 'info.dat', 'meta.dat'

a = torch.tensor([[1,2],
                  [3,4]])
for i in a:
    print(i)