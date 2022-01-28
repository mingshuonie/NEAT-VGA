import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 初始化参数
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

data = read_data('G:\\GAT\\实验一FGT数据集节点分类\\data\\facebook_xindeMHRWAE.csv')
data = pd.DataFrame(data)
data_zs = 1.0 * (data - data.mean()) / data.std()  # 数据标准化


model = KMeans(n_clusters=4,)  # 分为K类, 并发4
model.fit(data_zs)  # 开始聚类

# 标准化数据及其类别
## 每个样本对应得类别
r = pd.concat([data_zs, pd.Series(model.labels_, data.index)], axis=1)
r.columns = list(data.columns) + [u'聚类类别']

tsne = TSNE()
tsne.fit_transform(data_zs)   # 进行数据降维
tsne = pd.DataFrame(tsne.embedding_, index=data_zs.index) # 转换数据格式
plt.figure(figsize=(16, 8))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


#不同类别用不同颜色和样式绘图
d =  tsne[r[u'聚类类别'] == 0]
plt.plot(d[0], d[1], 'r.',)
d =  tsne[r[u'聚类类别'] == 1]
plt.plot(d[0], d[1], 'go')
d =  tsne[r[u'聚类类别'] == 2]
plt.plot(d[0], d[1], 'b*')
d =  tsne[r[u'聚类类别'] == 3]
plt.plot(d[0], d[1], 'k>')
plt.show()