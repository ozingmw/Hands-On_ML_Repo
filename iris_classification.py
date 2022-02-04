from sklearn import datasets

import numpy as np

iris = datasets.load_iris()
print(list(iris.keys()))

X = iris["data"][:, 3:]     #iris데이터 [[1,2,3,4],[1,2,3,4],[1,2,3,4], ... ] 중 각각 행에서 [3:] 행만 추출
y = (iris["target"] == 2).astype(np.int64)


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]
print(decision_boundary)
print(log_reg.predict_proba([[1.7], [1.67], [1.65]]))
print(log_reg.predict([[1.7], [1.67], [1.65]]))

X = iris["data"][:, (2,3)]
y = iris["target"]
softmax_reg = LogisticRegression(multi_class="multinomial", C=10)
softmax_reg.fit(X, y)

print(softmax_reg.predict_proba([[5, 2]]))
print(softmax_reg.predict([[5, 2]]))