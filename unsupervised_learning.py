import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from scipy import stats


iris = load_iris()
X = iris.data
y = iris.target

y_pred = GaussianMixture(n_components=3).fit(X).predict(X)

mapping = {}
for class_id in np.unique(y):
    mode, _ = stats.mode(y_pred[y==class_id])
    mapping[mode[0]] = class_id

print(mapping)

y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])

print(np.sum(y_pred==y))
print(np.sum(y_pred==y) / len(y_pred))


k = 5
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X)

print(y_pred)
print(y_pred is kmeans.labels_)
print(kmeans.cluster_centers_)