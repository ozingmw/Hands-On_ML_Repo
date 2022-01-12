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

# from sklearn.linear_model import SGDClassifier
# sgd_clf = SGDClassifier(random_state=42)

# from sklearn.model_selection import cross_val_predict, cross_val_score
# # print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))

# # y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)

# from sklearn.metrics import confusion_matrix
# # print(confusion_matrix(y_train, y_train_pred))

# from sklearn.metrics import f1_score, plot_precision_recall_curve
# # print(f1_score(y_train, y_train_pred, average="weighted"))

# # y_scores = cross_val_predict(sgd_clf, X_train, y_train, cv=3, method="decision_function")
# # plot_precision_recall_curve()

# from sklearn.svm import SVC
# svc_clf = SVC(gamma="auto", random_state=42)
# svc_clf.fit(X_train[:1000], y_train[:1000])
# print(svc_clf.predict([X[0]]))
# X0_scores = svc_clf.decision_function([X[0]])
# print(X0_scores)
# print(svc_clf.classes_)
# print(svc_clf.classes_[5])

# from sklearn.multiclass import OneVsRestClassifier
# ovr_clf = OneVsRestClassifier(SVC(gamma="auto", random_state=42))
# ovr_clf.fit(X_train[:1000], y_train[:1000])
# print(ovr_clf.predict([X[0]]))
# print(len(ovr_clf.estimators_))

# sgd_clf.fit(X_train, y_train)
# sgd_clf.predict([X[0]])
# print(sgd_clf.decision_function([X[0]]))

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# # print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy", n_jobs=-1))

# y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3, n_jobs=-1)
# conf_mx = confusion_matrix(y_train, y_train_pred)
# print(conf_mx)

# # plt.matshow(conf_mx, cmap=plt.cm.gray)
# # plt.show()

# row_sums = conf_mx.sum(axis=1, keepdims=True)
# norm_conf_mx = conf_mx / row_sums

# np.fill_diagonal(norm_conf_mx, 0)
# # plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# # plt.show()

# cl_a, cl_b = 3, 5
# X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
# X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
# X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
# X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

# from sklearn.neighbors import KNeighborsClassifier

# y_train_large = (y_train >= 7)
# y_train_odd = (y_train % 2 == 1)
# y_multilabel = np.c_[y_train_large, y_train_odd]

# knn_clf = KNeighborsClassifier()
# knn_clf.fit(X_train, y_multilabel)

# print(knn_clf.predict([X[0]]))

# y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
# print(f1_score(y_multilabel, y_train_knn_pred, average="weighted"))

# noise = np.random.randint(p, 100, (len(X_train), 784))
# X_train_mod = X_train + noise
# noise = np.random.randint(0, 100, (len(X_test), 784))
# X_test_mod = X_test + noise
# y_train_mod = X_train
# y_test_mod = X_test

# knn_clf.fit(X_train_mod, y_train_mod)
# clean_digit = knn_clf.predict([X_test_mod[X[0]]])


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
param = [
    {"weights": ["uniform", "distance"], "n_neighbors": [3,4,5]}
]
knc_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knc_clf, param_grid=param, scoring="accuracy", cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
y_pred = grid_search.predict(X_test)
print(accuracy_score(y_test, y_pred))