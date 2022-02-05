import numpy as np

from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC, SVC

iris = datasets.load_iris()
X = iris["data"][:, (2,3)]
y = (iris["target"] == 2).astype(np.float64)

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
])
svm_clf.fit(X, y)
print(svm_clf.predict([[5.5, 1.7]]))

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("SVC", SVC(kernel="linear", C=1)),
])
svm_clf.fit(X, y)
print(svm_clf.predict([[5.5, 1.7]]))

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("SGD_clf", SGDClassifier(loss="hinge", alpha=1/(len(X)*1))),
])
svm_clf.fit(X, y)
print(svm_clf.predict([[5.5, 1.7]]))


X, y = datasets.make_moons(n_samples=100, noise=0.15)
polynomial_svm_clf = Pipeline([
    ("poly_feature", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge")),
])
polynomial_svm_clf.fit(X, y)
print(polynomial_svm_clf.predict([[1.0, -0.5]]))

poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5)),
])
poly_kernel_svm_clf.fit(X, y)


rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001)),
])
rbf_kernel_svm_clf.fit(X, y)