from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
print(mnist.keys())

X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)

import numpy as np
import matplotlib.pyplot as plt
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)

from sklearn.model_selection import cross_val_predict, cross_val_score
# print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))

# y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)

# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y_train, y_train_pred))

# from sklearn.metrics import f1_score, plot_precision_recall_curve
# print(f1_score(y_train, y_train_pred, average="weighted"))

# y_scores = cross_val_predict(sgd_clf, X_train, y_train, cv=3, method="decision_function")
# plot_precision_recall_curve()

from sklearn.svm import SVC
svc_clf = SVC(gamma="auto", random_state=42)
svc_clf.fit(X_train[:1000], y_train[:1000])
print(svc_clf.predict([X[0]]))
X0_scores = svc_clf.decision_function([X[0]])
print(X0_scores)
print(svc_clf.classes_)
print(svc_clf.classes_[5])

from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC(gamma="auto", random_state=42))
ovr_clf.fit(X_train[:1000], y_train[:1000])
print(ovr_clf.predict([X[0]]))
print(len(ovr_clf.estimators_))

sgd_clf.fit(X_train, y_train)
sgd_clf.predict([X[0]])
print(sgd_clf.decision_function([X[0]]))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy", n_jobs=-1))