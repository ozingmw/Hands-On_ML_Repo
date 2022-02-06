import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from scipy.stats import reciprocal, uniform

housing = fetch_california_housing()
X = housing["data"]
y = housing["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

params = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
rand_search = RandomizedSearchCV(SVR(), params, n_iter=30, n_jobs=-1, cv=3, verbose=2)
rand_search.fit(X_train_scaled, y_train)

rand_search_predict = rand_search.best_estimator_.predict(X_train_scaled)
mse = mean_squared_error(y_train, rand_search_predict)
print(np.sqrt(mse))

rand_search_predict = rand_search.best_estimator_.predict(X_test_scaled)
mse = mean_squared_error(y_test, rand_search_predict)
print(np.sqrt(mse))