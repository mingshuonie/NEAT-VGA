

from karateclub.node_embedding.attributed.tadw import TADW
from karateclub.node_embedding.attributed.ae import AE
from karateclub.node_embedding.attributed.sine import SINE
from karateclub.node_embedding.attributed.asne import ASNE

from karateclub.dataset import GraphReader
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

import networkx as nx

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



def get_graph() -> nx.classes.graph.Graph:
    r"""Getting the graph.

    Return types:
        * **graph** *(NetworkX graph)* - Graph of interest.
    """
    data = pd.read_csv("data/pubmed/edges.csv")
    graph = nx.convert_matrix.from_pandas_edgelist(data, "id1", "id2")
    return graph



def get_target() -> np.array:
    r"""Getting the class membership of nodes.

    Return types:
        * **target** *(Numpy array)* - Class membership vector.
    """
    data = pd.read_csv("data/pubmed/target.csv")
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

embedding_file = read_data('G:\\GAT\\实验一节点分类\\其他模型引文网络的嵌入\\PubMed_ae_100.csv')
# embedding_file = read_data('G:\\GAT\\实验一FGT数据集节点分类\\cora_xindeMHRWAE323.csv')


# graph = get_graph()
# x = get_features()
y = get_target()
# print(x)
# model1 = TADW()
# model1.fit(g, x)

X = embedding_file
# X=X.reshape(-1,1)
y=y.reshape(-1,1)
print(X)
print(y)


# from sklearn.model_selection import KFold
#
from sklearn.ensemble import RandomForestClassifier
# kf = KFold(n_splits=10,shuffle=False,random_state=100)
# for X_train, y_train in kf.split(X):
#     print('train_index', X_train, 'test_index', y_train)
#     train_X, train_y = X[X_train], y[X_train]
#     X_test, y_test = X[y_train], y[y_train]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# from sklearn.metrics import accuracy_score
# p=accuracy_score(y_pred,y_test)
#
# print(p)
# from sklearn.metrics import recall_score
#
# r=recall_score(y_pred,y_test)
# print(r)
# f1=(2*p*r)/(p+r)
# print(f1)

from sklearn.metrics import f1_score

print(f1_score(y_test,y_pred,labels=[0,1,2],average='micro'))
# downstream_model = LogisticRegression(random_state=0).fit(X_train, y_train)
#
# y_hat = downstream_model.predict_proba(X_test)[:, 1]
#
# print(y_test)
# print(y_hat)

# f1=metrics.f1_score(y_test, y_hat)
# print('AUC: {:.4f}'.format(f1))


# auc = roc_auc_score(y_test, y_hat)
# print('AUC: {:.4f}'.format(auc))
