import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os
import pandas as pd

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)

        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        # s = min(test_idx_reorder)
        # t = max(test_idx_reorder)
        # tx_zero = np.zeros(tx.shape[1], dtype=np.float).reshape(1, -1)
        # print(tx.shape[0])
        # print("1")
        #
        # print(tx.shape[1])
        # print("2")
        # print(tx_zero.shape[0])
        # print("3")
        # print(tx_zero.shape)
        # print("4")
        # print(tx[0])
        # tx = np.array(tx)
        # print(tx.shape)
        # print("5")
        # for i in range(s, t + 1):
        #     if i not in test_idx_reorder:
        #         arr_i = np.array(i).reshape(1, )
        #         print(arr_i.shape)
        #         print("6")
        #         print(tx.shape)
        #         print("7")
        #         test_idx_reorder = np.concatenate((test_idx_reorder, arr_i), axis=0)
        #         print(tx_zero.shape)
        #         print("8")
        #         tx = np.concatenate((tx, tx_zero), axis=0)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features

def load_data222(dataset,embedding_file):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)

        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        # s = min(test_idx_reorder)
        # t = max(test_idx_reorder)
        # tx_zero = np.zeros(tx.shape[1], dtype=np.float).reshape(1, -1)
        # print(tx.shape[0])
        # print("1")
        #
        # print(tx.shape[1])
        # print("2")
        # print(tx_zero.shape[0])
        # print("3")
        # print(tx_zero.shape)
        # print("4")
        # print(tx[0])
        # tx = np.array(tx)
        # print(tx.shape)
        # print("5")
        # for i in range(s, t + 1):
        #     if i not in test_idx_reorder:
        #         arr_i = np.array(i).reshape(1, )
        #         print(arr_i.shape)
        #         print("6")
        #         print(tx.shape)
        #         print("7")
        #         test_idx_reorder = np.concatenate((test_idx_reorder, arr_i), axis=0)
        #         print(tx_zero.shape)
        #         print("8")
        #         tx = np.concatenate((tx, tx_zero), axis=0)

    data = pd.read_csv(embedding_file)
    data = data.values
    node_vec = []  # 存储格式为[节点向量]
    node_id = []  # 存储节点id
    for line in data:
        trans = [str(item) for item in line[1:]]  # temp[0]为节点编号
        node_vec.append(trans)
        node_id.append(str(line[0]))
    arr_vec = np.asarray(node_vec, dtype=np.float32)
    features = sp.csr_matrix(arr_vec)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    print(features.shape)
    return adj, features