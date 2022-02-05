import numpy as np

from scipy.stats import reciprocal, uniform

from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

mnist = fetch_openml('mnist_784', version=1, cache=True)

X = mnist["data"]
y = mnist["target"].astype(np.uint8)

X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_clf = SVC(gamma="scale")
svm_clf.fit(X_train_scaled[:6000], y_train[:6000])
svm_clf_predict = svm_clf.predict(X_train_scaled)
print(accuracy_score(y_train, svm_clf_predict))

params = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
rand_search = RandomizedSearchCV(svm_clf, params, n_iter=100, verbose=2, cv=3, n_jobs=-1)
rand_search.fit(X_train_scaled[:6000], y_train[:6000])
rand_search.best_estimator_.fit(X_train_scaled, y_train)
svm_clf_predict = rand_search.best_estimator_.predict(X_train_scaled)
print(accuracy_score(y_train, svm_clf_predict))

svm_clf_predict = rand_search.best_estimator_.predict(X_test_scaled)
print(accuracy_score(y_test, svm_clf_predict))