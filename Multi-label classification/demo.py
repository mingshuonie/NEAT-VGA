


iris = pd.read_csv('iris.csv')

X = iris.iloc[:,1:5]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,random_state=0)
kmeans.fit(X)

labels = kmeans.labels_
labels = pd.DataFrame(labels,columns=['labels'])
X.insert(4,'labels',labels)

from sklearn.manifold import TSNE
tsne = TSNE()
a = tsne.fit_transform(X)
liris = pd.DataFrame(a,index=X.index)

d1 = liris[X['labels']==0]
#plt.plot(d1[0],d1[1],'r.')
d2 = liris[X['labels']==1]
#plt.plot(d2[0],d2[1],'go')
d3 = liris[X['labels']==2]
#plt.plot(d3[0],d3[1],'b*')
plt.plot(d1[0],d1[1],'r.',d2[0],d2[1],'go',d3[0],d3[1],'b*')