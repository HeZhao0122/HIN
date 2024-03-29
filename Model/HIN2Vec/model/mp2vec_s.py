#!/usr/bin/python
# -*- encoding: utf8 -*-

import math
from multiprocessing import Process, Pool, Value, Array
import numpy as np
import optparse
import os
import random
import sys
import time
import warnings

from ..ds import mp


__author__ = 'sheep'


class Common(object):

    def __init__(self):
        self.node_vocab = None
        self.path_vocab = None
        self.node2vec = None
        self.path2vec = None

    def train(self, training_fname, seed=None):
        raise NotImplementedError

    def dump_to_file(self, output_fname, type_='node'):
        '''
            input:
                type_: 'node' or 'path'
        '''
        with open(output_fname, 'w') as f:
            if type_ == 'node':
                f.write('%d %d\n' % (len(self.node_vocab), self.size))
                for node, vector in zip(self.node_vocab, self.node2vec):
                    line = ' '.join([str(v) for v in vector])
                    f.write('%s %s\n' % (node.node_id, line))
            else:
                f.write('%d %d\n' % (self.path_vocab.distinct_path_count(),
                                     self.size))
                for path, vector in zip(self.path_vocab, self.path2vec):
                    if path.is_inverse:
                        continue
                    line = ' '.join([str(v) for v in vector])
                    f.write('%s %s\n' % (path.path_id, line))


class MP2Vec(Common):

    def __init__(self, size=100, window=10, neg=5,
                       alpha=0.005, num_processes=1, iterations=1,
                       normed=True, same_w=False,
                       is_no_circle_path=False):
        '''
            size:      Dimensionality of word embeddings
            window:    Max window length
            neg:       Number of negative examples (>0) for
                       negative sampling, 0 for hierarchical softmax
            alpha:     Starting learning rate
            num_processes: Number of processes
            iterations: Number of iterations
            normed:    To normalize the final vectors or not
            same_w:    Same matrix for nodes and context nodes
            is_no_circle_path: Generate training data without circle in the path
        '''
        self.size = size
        self.window = window
        self.neg = neg
        self.alpha = alpha
        self.num_processes = num_processes
        self.iterations = iterations
        self.vocab = None
        self.node2vec = None
        self.path2vec = None
        self.normed = normed
        self.same_w = same_w
        self.is_no_circle_path = is_no_circle_path

    def train(self, g, training_fname, class2node_ids,
                                       seed=None,
                                       edge_class_inverse_mapping=None,
                                       k_hop_neighbors=None,
                                       id2vec_fname=None,
                                       path2vec_fname=None):
        '''
            input:
                training_fname:
                    each line: <node_id> <edge_id> ...
        '''
        def get_training_size(fname):
            with open(fname, 'r') as f:
                for line in f:
                    pass
                return f.tell()

        if seed is not None:
            np.random.seed(seed)

        node_vocab = mp.NodeVocab.load_from_file(training_fname)
        path_vocab = mp.PathVocab.load_from_file(training_fname,
                                self.window,
                                inverse_mapping=edge_class_inverse_mapping)
        for ith, p in enumerate(path_vocab.paths):
            if p.is_inverse:
                continue
            print(ith, p)

        training_size = get_training_size(training_fname)
        print('training bytes: %d' % training_size)
        print('distinct node count: %d' % len(node_vocab))
        print('distinct path count: %d' % path_vocab.distinct_path_count())

        #load pre-trained node and path vectors
        id2vec = None
        if id2vec_fname is not None:
            id2vec = MP2Vec.load_id2vec(id2vec_fname)
        path2vec = None
        if path2vec_fname is not None:
            path2vec = MP2Vec.load_path2vec(path2vec_fname,
                                            path_vocab.path2index)

        #initialize vectors
        Wx, Wy, Wpath = MP2Vec.init_net(self.size,
                                        len(node_vocab),
                                        path_vocab.distinct_path_count(),
                                        id2vec=id2vec,
                                        path2vec=path2vec)

        counter = Value('i', 0)
        tables = {
            'all': UnigramTable(g, node_vocab, uniform=True)
        }

        print('start training')
        if self.num_processes > 1:
            processes = []
            for i in range(self.num_processes):
                start = training_size / self.num_processes * i
                end = training_size / self.num_processes * (i+1)
                if i == self.num_processes-1:
                    end = training_size

                p = Process(target=train_process,
                                   args=(i, node_vocab, path_vocab,
                                         Wx, Wy, Wpath, tables,
                                         self.neg, self.alpha,
                                         self.window, counter,
                                         self.iterations,
                                         training_fname, (start, end),
                                         self.same_w,
                                         k_hop_neighbors,
                                         self.is_no_circle_path))
                processes.append(p)

            start = time.time()
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            end = time.time()
        else:
            start = time.time()
            train_process(0, node_vocab, path_vocab,
                          Wx, Wy, Wpath, tables,
                          self.neg, self.alpha,
                          self.window, counter,
                          self.iterations,
                          training_fname, (0, training_size),
                          self.same_w,
                          k_hop_neighbors,
                          self.is_no_circle_path)
            end = time.time()

        self.node_vocab = node_vocab
        self.path_vocab = path_vocab

        #normalize node and path vectors
        node2vec = []
        if self.normed:
            for vec in Wx:
                vec = np.array(list(vec))
                norm=np.linalg.norm(vec)
                if norm==0:
                    node2vec.append(vec)
                else:
                    node2vec.append(vec/norm)
        else:
            for vec in Wx:
                node2vec.append(np.array(list(vec)))
        self.node2vec = node2vec

        path2vec = []
        if self.normed:
            for vec in Wpath:
                vec = np.array(list(vec))
                norm=np.linalg.norm(vec)
                if norm==0:
                    path2vec.append(vec)
                else:
                    path2vec.append(vec/norm)
        else:
            for vec in Wpath:
                path2vec.append(np.array(list(vec)))
        self.path2vec = path2vec

        print('Finished. Total time: %.2f minutes' %  ((end-start)/60))

    @staticmethod
    def load_id2vec(fname):
        id2vec = {}
        with open(fname, 'r') as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue
                tokens = line.strip().split(' ')
                id_ = int(tokens[0])
                id2vec[id_] = map(float, tokens[1:])
        return id2vec

    @staticmethod
    def load_path2vec(fname, path2index):
        id2vec = {}
        with open(fname, 'r') as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue
                tokens = line.strip().split(' ')
                id_ = int(path2index[tokens[0]])
                id2vec[id_] = map(float, tokens[1:])
        return id2vec

    @staticmethod
    def init_net(dim, node_size, path_size,
                 id2vec=None, path2vec=None):
        '''
            return
                Wx: a |V|*d matrix for input layer to hidden layer
                Wy: a |V|*d matrix for hidden layer to output layer
                Wpath: a |paths|*d matrix for hidden layer to output layer
        '''
        tmp = np.random.uniform(low=-0.5/dim,
                                high=0.5/dim,
                                size=(node_size, dim)).astype(np.float64)
        Wx = np.ctypeslib.as_ctypes(tmp)
        Wx = Array(Wx._type_, Wx, lock=False)

        if id2vec is not None:
            for i, vec in sorted(id2vec.items()):
                for j in range(len(vec)):
                    Wx[i][j] = vec[j]

        tmp = np.random.uniform(low=-0.5/dim,
                                high=0.5/dim,
                                size=(node_size, dim)).astype(np.float64)
        Wy = np.ctypeslib.as_ctypes(tmp)
        Wy = Array(Wy._type_, Wy, lock=False)

        if id2vec is not None:
            for i, vec in sorted(id2vec.items()):
                for j in range(len(vec)):
                    Wy[i][j] = vec[j]

        tmp = np.random.uniform(low=0.0,
                                high=1.0/dim,
                                size=(node_size, dim)).astype(np.float64)
        Wpath = np.ctypeslib.as_ctypes(tmp)
        Wpath = Array(Wpath._type_, Wpath, lock=False)

        if path2vec is not None:
            for i, vec in sorted(path2vec.items()):
                for j in range(len(vec)):
                    Wpath[i][j] = vec[j]

        return Wx, Wy, Wpath


class UnigramTable(object):
    '''
        For negative sampling.
        A list of indices of words in the vocab
        following a power law distribution.
    '''
    def __init__(self, g, vocab, seed=None, size=1000000, times=1, node_ids=None, uniform=False):
        self.table = UnigramTable.generate_table(g, vocab,
                                                 vocab.count() * times,
                                                 node_ids,
                                                 uniform)
        if seed is not None:
            np.random.seed(seed)
        self.randints = np.random.randint(low=0,
                                          high=len(self.table),
                                          size=size)
        self.size = size
        self.index = 0

    @staticmethod
    def generate_table(g, vocab, table_size, node_ids, uniform):
        power = 0.75
        if node_ids is not None:
            if uniform:
                total = len([t for t in vocab
                             if t.node_id in node_ids])
            else:
                total = sum([math.pow(t.count, power) for t in vocab
                             if t.node_id in node_ids])
        else:
            if uniform:
                total = len(vocab)
            else:
                total = sum([math.pow(t.count, power) for t in vocab])

        table = np.zeros(table_size, dtype=np.uint32)
        p = 0
        current = 0
        for index, word in enumerate(vocab.nodes):
            if node_ids is not None and word.node_id not in node_ids:
                continue

            if uniform:
                p += float(1.0)/total
            else:
                p += float(math.pow(word.count, power))/total

            to_ = int(table_size * p)
            if to_ != table_size:
                to_ = to_+1
            for i in range(current, to_):
                table[i] = index
            current = to_
        return table

    def cleanly_sample(self, neighbors, count):
        samples = []
        while True:
            unchecked = self.sample(count)
            for s in unchecked:
                if s in neighbors:
                    continue
                samples.append(s)
                if len(samples) >= count:
                    return samples

    def sample(self, count):
        if count == 0:
            return []

        if self.index + count < self.size:
            samples = [self.table[i] for i
                       in self.randints[self.index:self.index+count]]
            self.index += count
            return samples

        if self.index + count == self.size:
            samples = [self.table[i] for i
                       in self.randints[self.index:]]
            self.index = 0
            self.randints = np.random.randint(low=0,
                                              high=len(self.table),
                                              size=self.size)
            return samples

        self.index = 0
        self.randints = np.random.randint(low=0,
                                          high=len(self.table),
                                          size=self.size)
        return self.sample(count)

#TODO speed up
#TODO the order of edges of the path
def get_context(node_index_walk, edge_walk, walk, path_vocab,
                index, window_size, no_circle=False):
    start = max(index - window_size, 0)
    end = min(index + window_size + 1, len(node_index_walk))
    context = []
    if no_circle:
        x = node_index_walk[index]
        visited = set()
        for i in range(index+1, end):
            y = node_index_walk[i]
            if x == y or y in visited:
                break
            path = path_vocab.path2index[','.join(edge_walk[index:i])]
            context.append((y, path, edge_walk[i-1]))
            visited.add(y)
    else:
        for i in range(index+1, end):
            path = path_vocab.path2index[','.join(edge_walk[index:i])]
            context.append((node_index_walk[i], path, edge_walk[i-1]))
    return context

def sigmoid(x):
    if x > 6:
        return 1.0
    elif x < -6:
        return 0.0
    return 1 / (1 + math.exp(-x))

def train_process(pid, node_vocab, path_vocab, Wx, Wy, Wpath,
                  tables,
                  neg, starting_alpha, win, counter,
                  iterations, training_fname, start_end,
                  same_w, k_hop_neighbors,
                  is_no_circle_path):

    def dev_sigmoid(x):
        ex = math.exp(-x)
        s = 1 / (1 + ex)
        return s * (1-s)

    def get_wp2_wp3(wp):
        wp2 = np.zeros(dim)
        wp3 = np.zeros(dim)
        for i, v in enumerate(wp):
#           v *= 4
            if v > 0:
                wp2[i] = 1
            if -6 <= v <= 6:
                wp3[i] = dev_sigmoid(v)
        return wp2, wp3

    np.seterr(invalid='raise', over ='raise', under='raise')

    #ignore the PEP 3118 buffer warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        Wx = np.ctypeslib.as_array(Wx)
        Wy = np.ctypeslib.as_array(Wy)
        Wpath = np.ctypeslib.as_array(Wpath)

    error_fname = 'error.%d' % pid
    os.system('rm -f %s' % error_fname)

    max_path_id = 0
    for ith, path in enumerate(path_vocab.paths):
        if path.is_inverse is False:
            max_path_id = ith
        else:
            break

    win_index = 0
    step = 10000
    dim = len(Wx[0])
    alpha = starting_alpha
    start, end = start_end

    table = tables['all']
    cur_win = win

    for iteration in range(iterations):
        word_count = 0
        first = True
        with open(training_fname, 'r') as f:
            f.seek(start)
            while f.tell() < end:
                #read a random walk
                walk = f.readline().strip().split()
                if len(walk) <= 2:
                    continue
                # the first element of the walk may be truncated
                if first:
                    if len(walk) % 2 == 1:
                        walk = walk[2:]
                    else:
                        walk = walk[1:]
                    first = False

                node_index_walk = [node_vocab.node2index[x]
                                   for i, x in enumerate(walk)
                                   if i % 2 == 0]
                edge_walk = [x for i, x in enumerate(walk)
                             if i % 2 == 1]

                for i, x in enumerate(node_index_walk):
                    #generate positive training data
                    for pos_y, path, last_edge_id in get_context(node_index_walk,
                                                    edge_walk,
                                                    walk,
                                                    path_vocab,
                                                    i,
                                                    cur_win,
                                                    no_circle=is_no_circle_path):

                        #generate negative training data
                        if k_hop_neighbors is not None:
                            negs = table.cleanly_sample(k_hop_neighbors[x], neg)
                        else:
                            negs = table.sample(neg)

                        #SGD learning
                        for y, path, label in ([(pos_y, path, 1)]
                                            + [(y, path, 0) for y in negs]):

                            if x == y:
                                continue
                            if label == 0 and y == pos_y:
                                continue

                            wx = Wx[x]
                            if same_w:
                                wy = Wx[y]
                            else:
                                wy = Wy[y]
                            wp = Wpath[path]
                            wp2, wp3 = get_wp2_wp3(wp)

                            dot = sum(wp2 * wx * wy)
                            p = sigmoid(dot)
                            g = alpha * (label - p)
                            if g == 0:
                                continue

                            epath = g * wp3 * wx * wy
                            wp2 = g * wp2
                            ex = wp2 * wy
                            wy += wp2 * wx
                            wx += ex
                            wp += epath

                    word_count += 1

                    if word_count % step == 0:
                        counter.value += step
                        ratio = float(counter.value)/node_vocab.node_count
                        ratio = ratio/iterations

                        alpha = starting_alpha * (1-ratio)
                        if alpha < starting_alpha * 0.0001:
                            alpha = starting_alpha * 0.0001

                        sys.stdout.write(("\r%f "
                                          "%d/%d (%.2f%%) "
                                          "" % (alpha,
                                               counter.value,
                                               node_vocab.node_count*iterations,
                                               ratio*100,
                                               )))
                        sys.stdout.flush()

        counter.value += (word_count % step)
        ratio = float(counter.value)/node_vocab.node_count
