import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE


mnist = fetch_openml('mnist_784', version=1)
m = 10000
idx = np.random.permutation(60000)[:m]

X = mnist.data[idx]
y = mnist.target[idx].astype(np.uint8)  # <= CONVERT STRINGS TO INTEGERS

tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X)

plt.figure(figsize=(13,10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()