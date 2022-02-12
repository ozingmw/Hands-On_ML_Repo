import time

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA       # PCA import 시 intellisense 안됨  ->  PCA파일이 커서 그런가?
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



mnist = fetch_openml('mnist_784', version=1, cache=True)
X_train, X_test, y_train, y_test = mnist["data"][:60000], mnist["data"][60000:], mnist["target"][:60000], mnist["target"][60000:]

rnd_clf = RandomForestClassifier(random_state=42)

start_time = time.time()
rnd_clf.fit(X_train, y_train)
end_time = time.time()
print(f"훈련시간 : {end_time - start_time:.2f}")

y_pred = rnd_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))


pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train)

rnd_clf_with_pca = RandomForestClassifier(random_state=42)
start_time = time.time()
rnd_clf_with_pca.fit(X_train_reduced, y_train)
end_time = time.time()
print(f"훈련시간 : {end_time - start_time:.2f}")

X_test_reduced = pca.transform(X_test)
y_pred = rnd_clf_with_pca.predict(X_test_reduced)
print(accuracy_score(y_test, y_pred))


log_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
start_time = time.time()
log_reg.fit(X_train, y_train)
end_time = time.time()
print(f"훈련시간 : {end_time - start_time:.2f}")

y_pred = log_reg.predict(X_test)
print(accuracy_score(y_test, y_pred))

log_reg_with_pca = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
start_time = time.time()
log_reg_with_pca.fit(X_train_reduced, y_train)
end_time = time.time()
print(f"훈련시간 : {end_time - start_time:.2f}")

y_pred = log_reg_with_pca.predict(X_test_reduced)
print(accuracy_score(y_test, y_pred))