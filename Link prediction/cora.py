


from karateclub.dataset import GraphReader
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

import networkx as nx

import io

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



def get_graph() -> nx.classes.graph.Graph:
    r"""Getting the graph.

    Return types:
        * **graph** *(NetworkX graph)* - Graph of interest.
    """
    data = pd.read_csv("data/facebook/cora_edges.csv")
    graph = nx.convert_matrix.from_pandas_edgelist(data, "id_1", "id_2")
    return graph



def get_target() -> np.array:
    r"""Getting the class membership of nodes.

    Return types:
        * **target** *(Numpy array)* - Class membership vector.
    """
    data = pd.read_csv("data/cora/cora_target.csv")
    target = np.array(data["target"])
    return target

def read_data(embedding_file):
    data = pd.read_csv(embedding_file)
    data = data.values
    node_vec = []  # 存储格式为[节点向量]
    node_id = []  # 存储节点id
    for line in data:
        trans = [str(item) for item in line[1:]]  # temp[0]为节点编号
        node_vec.append(trans)
        node_id.append(str(line[0]))

    arr_vec = np.asarray(node_vec, dtype=np.float32)
    # print(node_vecF
    # print(node_id)
    return arr_vec

embedding_file = read_data('G:\\GAT\\实验一节点分类\\其他模型引文网络的嵌入\\cora_embedding3_400.csv')
# embedding_file = read_data('G:\\GAT\\实验一FGT数据集节点分类\\cora_xindeMHRWAE323.csv')


