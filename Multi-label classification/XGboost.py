import numpy as np
import pandas as pd

import networkx as nx

from sklearn.model_selection import train_test_split


def get_graph() -> nx.classes.graph.Graph:
    r"""Getting the graph.

    Return types:
        * **graph** *(NetworkX graph)* - Graph of interest.
    """
    data = pd.read_csv("data/twitch/PTBR_edges.csv")
    graph = nx.convert_matrix.from_pandas_edgelist(data, "from", "to")
    return graph


def get_target() -> np.array:
    r"""Getting the class membership of nodes.

    Return types:
        * **target** *(Numpy array)* - Class membership vector.
    """
    data = pd.read_csv("data/twitch/PTBR_target-副本.csv")
    target = np.array(data["mature"])
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
    return arr_vec


embedding_file = read_data('G:\\GAT\\实验一FGT数据集节点分类\\data\\New Folder\\PTBR_xindeMHRWAE.csv')

y = get_target()

X = embedding_file

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred, labels=[0, 1],average='micro'))
print(y_test.tolist())
print(y_pred.tolist())

train_y = model.predict(X_train)
print(f1_score(y_train, train_y, labels=[0, 1]))
print(y_train.tolist())
print(train_y.tolist())
