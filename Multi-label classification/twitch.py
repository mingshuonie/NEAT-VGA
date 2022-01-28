

from karateclub.node_embedding.attributed.tadw import TADW
from karateclub.node_embedding.attributed.ae import AE
from karateclub.node_embedding.attributed.sine import SINE
from karateclub.node_embedding.attributed.asne import ASNE

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
from xgboost.sklearn import XGBClassifier


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
    data = pd.read_csv("data/twitch/PTBR_target3.csv")
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
    # print(node_vec
    # print(node_id)
    return arr_vec

embedding_file = read_data('G:\\GAT\\实验一FGT数据集节点分类\\data\\New Folder\\PTBR_xindeMHRWAE.csv')
# embedding_file = read_data('C:\\Users\\Dell\\Desktop\\embedding2\\github_tadw.csv')


y = get_target()


X = embedding_file
# X=X.reshape(1,-1)
# y=y.reshape(1,-1)
print(X)
print(y)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


model = LogisticRegression(penalty='l2',C=1,)
model.fit(X_train, y_train)
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

print(f1_score(y_test, y_pred, labels=[0, 1], average='micro'))

train_y = model.predict(X_train)
print(y_train.tolist())
print(train_y.tolist())
print(f1_score(y_train, train_y, labels=[0, 1],average='micro'))

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

X = pd.DataFrame(X)

from sklearn.cluster import KMeans

model = KMeans(n_clusters=4)
model.fit(X)

r = pd.concat([X, pd.Series(model.labels_, X.index)], axis=1)
r.columns = list(X.columns) + [u'聚类类别']

from sklearn.manifold import TSNE
tsne = TSNE()
tsne.fit_transform(X) #进行数据降维,并返回结果
tsne = pd.DataFrame(tsne.embedding_, index=X.index) #转换数据格式

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

#不同类别用不同颜色和样式绘图
d =  tsne[r[u'聚类类别'] == 0]
plt.plot(d[0], d[1], 'r.')
d =  tsne[r[u'聚类类别'] == 1]
plt.plot(d[0], d[1], 'go')
d =  tsne[r[u'聚类类别'] == 2]
plt.plot(d[0], d[1], 'b*')
d =  tsne[r[u'聚类类别'] == 3]
plt.plot(d[0], d[1], 'k>')
plt.show()