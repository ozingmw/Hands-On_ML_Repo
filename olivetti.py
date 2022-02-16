from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline


olivetti = fetch_olivetti_faces()
print(olivetti.DESCR)

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=42)
train_valid_idx, test_idx = next(strat_split.split(olivetti.data, olivetti.target))
X_train_valid = olivetti.data[train_valid_idx]
y_train_valid = olivetti.target[train_valid_idx]
X_test = olivetti.data[test_idx]
y_test = olivetti.target[test_idx]

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=43)
train_idx, valid_idx = next(strat_split.split(X_train_valid, y_train_valid))
X_train = X_train_valid[train_idx]
y_train = y_train_valid[train_idx]
X_valid = X_train_valid[valid_idx]
y_valid = y_train_valid[valid_idx]

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)

pca = PCA(0.99)
X_train_pca = pca.fit_transform(X_train)
X_valid_pca = pca.transform(X_valid)
X_test_pca = pca.transform(X_test)
print(pca.n_components_)


range_per_k = range(100, 140, 2)
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_train_pca) for k in range_per_k]

# 이니셔로 엘보 지점 확인 불가능
inertias = [model.inertia_ for model in kmeans_per_k]
plt.figure(figsize=(16,12))
plt.plot(range_per_k, inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.show()

silhouette_scores = [silhouette_score(X_train, model.labels_) for model in kmeans_per_k]
plt.figure(figsize=(16, 12))
plt.plot(range_per_k, silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
# plt.show()

best_k = range_per_k[np.argmax(silhouette_scores)]
best_model = kmeans_per_k[np.argmax(silhouette_scores)]
print(f"k : {best_k}")

def plot_faces(faces, labels, n_cols=5):
    faces = faces.reshape(-1, 64, 64)
    n_rows = (len(faces) - 1) // n_cols + 1
    plt.figure(figsize=(n_cols, n_rows * 1.1))
    for index, (face, label) in enumerate(zip(faces, labels)):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(face, cmap="gray")
        plt.axis("off")
        plt.title(label)
    plt.show()

for cluster_id in np.unique(best_model.labels_):
    print("Cluster", cluster_id)
    in_cluster = best_model.labels_==cluster_id
    faces = X_train[in_cluster]
    labels = y_train[in_cluster]
    plot_faces(faces, labels)


rnd_clf = RandomForestClassifier(n_estimators=130, random_state=42)
rnd_clf.fit(X_train_pca, y_train)
print(rnd_clf.score(X_valid_pca, y_valid))

X_train_reduced = best_model.transform(X_train_pca)
X_valid_reduced = best_model.transform(X_valid_pca)
X_test_reduced = best_model.transform(X_test_pca)

rnd_clf = RandomForestClassifier(n_estimators=130, random_state=42)
rnd_clf.fit(X_train_reduced, y_train)
print(rnd_clf.score(X_valid_reduced, y_valid))


pipeline = Pipeline([
    ("kmeans", KMeans(random_state=42)),
    ("rnd_clf", RandomForestClassifier(random_state=42))
])

params = {"kmeans__n_clusters": range(50, 200, 5), "rnd_clf__n_estimators": range(50, 200, 5)}

grid_search = GridSearchCV(pipeline, params, n_jobs=-1)
grid_search.fit(X_train_pca, y_train)
print(grid_search.best_estimator_)
print(grid_search.best_estimator_.score(X_valid_pca, y_valid))

X_train_extended = np.c_[X_train_pca, X_train_reduced]
X_valid_extended = np.c_[X_valid_pca, X_valid_reduced]
X_test_extended = np.c_[X_test_pca, X_test_reduced]

grid_search = GridSearchCV(pipeline, params, n_jobs=-1)
grid_search.fit(X_train_extended, y_train)
print(grid_search.best_estimator_.score(X_valid_extended, y_valid))

rnd_clf = RandomForestClassifier(n_estimators=130, random_state=42)
rnd_clf.fit(X_train_extended, y_train)
print(rnd_clf.score(X_valid_extended, y_valid))


gmm = GaussianMixture(n_components=40, random_state=42)
y_pred = gmm.fit_predict(X_train_pca)

n_gen_faces = 20
gen_faces_reduced, y_gen_faces = gmm.sample(n_samples=n_gen_faces)
gen_faces = pca.inverse_transform(gen_faces_reduced)
plot_faces(gen_faces, y_gen_faces)

n_rotated = 4
rotated = np.transpose(X_train[:n_rotated].reshape(-1, 64, 64), axes=[0, 2, 1])
rotated = rotated.reshape(-1, 64*64)
y_rotated = y_train[:n_rotated]

n_flipped = 3
flipped = X_train[:n_flipped].reshape(-1, 64, 64)[:, ::-1]
flipped = flipped.reshape(-1, 64*64)
y_flipped = y_train[:n_flipped]

n_darkened = 3
darkened = X_train[:n_darkened].copy()
darkened[:, 1:-1] *= 0.3
y_darkened = y_train[:n_darkened]

X_bad_faces = np.r_[rotated, flipped, darkened]
y_bad = np.concatenate([y_rotated, y_flipped, y_darkened])

plot_faces(X_bad_faces, y_bad)

X_bad_faces_pca = pca.transform(X_bad_faces)
print(gmm.score_samples(X_bad_faces_pca))
print(gmm.score_samples(X_train_pca[:10]))


def reconstruction_errors(pca, X):
    X_pca = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)
    mse = np.square(X_reconstructed - X).mean(axis=-1)
    return mse

print(reconstruction_errors(pca, X_train).mean())
print(reconstruction_errors(pca, X_bad_faces).mean())