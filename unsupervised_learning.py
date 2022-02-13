import os
from matplotlib import pyplot as plt
import numpy as np

from matplotlib.image import imread
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import load_digits, load_iris, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from scipy import stats
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline


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


blob_centers = np.array(
    [[0.2, 2.3],
    [-1.5, 2.3],
    [-2.8, 1.8],
    [-2.8, 2.8],
    [-2.8, 1.3]]
)
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std)

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)

print(y_pred)
print(y_pred is kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.labels_)

X_new = np.array([[0, 2], [3, 2], [ -3, 3], [-3, 2.5]])
print(kmeans.predict(X_new))
print(kmeans.transform(X_new))


print(kmeans.inertia_)

good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1, random_state=42)
kmeans.fit(X)
print(kmeans.inertia_)
print(kmeans.score(X_new))

kmeans_rnd_10_init = KMeans(n_clusters=5, init="random", n_init=10, algorithm="full")
kmeans_rnd_10_init.fit(X)
print(kmeans_rnd_10_init.score(X))
print(kmeans_rnd_10_init.score(X_new))


minibatch_kmeans = MiniBatchKMeans(n_clusters=5)
minibatch_kmeans.fit(X)
print(minibatch_kmeans.score(X))

kmeans_per_k = [KMeans(n_clusters=k).fit(X) for k in range(1, 10)]
print(silhouette_score(X, kmeans.labels_))
silhouette_scores = [silhouette_score(X, model.labels_) for model in kmeans_per_k[1:]]

X1, y1 = make_blobs(n_samples=1000, centers=((-4, -4), ((0, 0))))
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1)
X2 = X2 + [6, -8]
X = np.r_[X1, X2]
y = np.r_[y1, y2]

kmeans_good = KMeans(n_clusters=3, init=np.array([[-1.5, 2.5], [0.5, 0], [4, 0]]), n_init=1)
kmeans_bad = KMeans(n_clusters=3)
kmeans_good.fit(X)
kmeans_bad.fit(X)


img = imread(os.path.join("images", "ladybug.png"))
print(img.shape)

X = img.reshape(-1, 3)
kmeans = KMeans(n_clusters=8).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(img.shape)

segmented_imgs = []
n_colors = (10, 8, 6, 4, 2)
for n_clusters in n_colors:
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(img.shape))


X_digits, y_digits = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000)
log_reg.fit(X_train, y_train)
print(log_reg.score(X_test, y_test))

pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50)),
    ("log_reg", LogisticRegression()),
])
pipeline.fit(X_train, y_train)
print(pipeline.score(X_test, y_test))

param = dict(kmeans__n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param, cv=3)
grid_clf.fit(X_train, y_train)
print(grid_clf.best_params_)
grid_clf.score(X_test, y_test)

n_labeled = 50
log_reg = LogisticRegression()
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
print(log_reg.score(X_test, y_test))

k = 50
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]
y_representative_digits = np.array([
    4, 8, 0, 6, 8, 3, 7, 7, 9, 2,
    5, 5, 8, 5, 2, 1, 2, 9, 6, 2,
    1, 6, 9, 0, 8, 3, 0, 7, 4, 1,
    6, 5, 2, 4, 1, 8, 6, 3, 9, 2, 
    4, 2, 9, 4, 7, 6, 2, 3, 1, 1
])

log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=10000, n_jobs=-1, random_state=42)
log_reg.fit(X_representative_digits, y_representative_digits)
print(log_reg.score(X_test, y_test))

y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]

log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=10000, n_jobs=-1, random_state=42)
log_reg.fit(X_train, y_train_propagated)
print(log_reg.score(X_test, y_test))

percentile_closet = 75
X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closet)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1

partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=10000, n_jobs=-1, random_state=42)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
print(log_reg.score(X_test, y_test))
print(np.mean(y_train_partially_propagated == y_train[partially_propagated]))