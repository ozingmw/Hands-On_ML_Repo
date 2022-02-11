import numpy as np
from sklearn.datasets import fetch_openml, make_swiss_roll
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]

m,n = X.shape
S = np.zeros(X_centered.shape)
S[:n, :n] = np.diag(s)
print(np.allclose(X_centered, U.dot(S).dot(Vt)))    # X_centered와 U.dot(S).dot(Vt)가 모든 요소가 허용 오차안에 있으면 True

W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)
X2D_using_svd = X2D

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)

print(X2D[:5])
print(X2D_using_svd[:5])
print(np.allclose(X2D, -X2D_using_svd))

X3D_inv = pca.inverse_transform(X2D)
print(np.allclose(X3D_inv, X))      # 복원 / 원본하고 완전 같진 않음
print(np.mean(np.sum(np.square(X3D_inv - X), axis=1)))      # 재구성 오차


print(pca.explained_variance_ratio_)


mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist["data"]
y = mnist["target"].astype(np.uint8)
X_train, X_test, y_train, y_test = train_test_split(X, y)

pca = PCA()
pca.fit(X, y)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) * 1
print(d)


pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
print(pca.n_components_)
print(np.sum(pca.explained_variance_ratio_))

pca = PCA(n_components=154)
X_reduced = pca.fit_transform(X_train)
X_recoverd = pca.inverse_transform(X_reduced)
X_reduced_pca = X_reduced

rnd_pca = PCA(n_components=154, svd_solver="randomized")
X_reduced = rnd_pca.fit_transform(X_train)


n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)
X_recoverd_inc_pca = inc_pca.inverse_transform(X_reduced)
X_reduced_inc_pca = X_reduced

print(np.allclose(pca.mean_, inc_pca.mean_))    #평균은 같다.
print(np.allclose(X_reduced_pca, X_reduced_inc_pca))    #완전히 동일하진 않다.


X, t = make_swiss_roll(n_samples=1000, noise=0.2)

rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)

lin_pca = KernelPCA(n_components=2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.443, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components=2, kernel="sigmoid", gamma=0.001, fit_inverse_transform=True)

y = t > 6.9

clf = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression()),
])

param_grid = [{
    "kpca__gamma": np.linspace(0.03, 0.05, 10),
    "kpca__kernel": ["rbf", "sigmoid"],
}]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)
print(grid_search.best_params_)
print(grid_search.best_estimator_)

rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0443, fit_inverse_transform=True)     #kPCA로 2D로 만든것과 특성맵을 사용하여 무한차원으로 매핑한 뒤 2D로 투영한것은 같다.
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)

print(mean_squared_error(X, X_preimage))


lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)