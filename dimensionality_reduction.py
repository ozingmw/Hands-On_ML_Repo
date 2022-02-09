import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
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