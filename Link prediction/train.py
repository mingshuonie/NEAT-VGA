from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""


import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

from input_data import load_data, load_data222
from cora多分类 import read_data



adj, features = load_data222('cora', 'G:\\GAT\\实验一节点分类\\其他模型引文网络的嵌入\\cora_embedding.csv')

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train



# Some preprocessing
adj_norm = preprocess_graph(adj)



num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())




# features2 = sparse_to_tuple(features2.tocoo())
# features_nonzero2 = features2[1].shape[0]

num_features = features[2][1]
features_nonzero = features[1].shape[0]

# print(features_nonzero)



cost_val = []
acc_val = []


def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        print(1)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

emb=read_data('G:\\GAT\\实验一节点分类\\其他模型引文网络的嵌入\\cora_embedding.csv')

roc_score, ap_score = get_roc_score(test_edges, test_edges_false, emb)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))
