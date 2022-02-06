import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_iris, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode

iris = load_iris()
X = iris.data[:, (2,3)]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

print(tree_clf.predict_proba([[5, 1.5]]))
print(tree_clf.predict([[5, 1.5]]))


X, y = make_moons(n_samples=1000, noise=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params = {"max_leaf_nodes": list(range(2, 50)), "min_samples_split": list(range(2, 10)), "min_samples_leaf": list(range(2,10))}
rand_search = RandomizedSearchCV(DecisionTreeClassifier(), params, n_iter=500, n_jobs=-1, cv=3)
rand_search.fit(X_train, y_train)
print(rand_search.best_estimator_)

rand_search_predict = rand_search.best_estimator_.predict(X_train)
print(accuracy_score(y_train, rand_search_predict))

rand_search_predict = rand_search.best_estimator_.predict(X_test)
print(accuracy_score(y_test, rand_search_predict))

grid_search = GridSearchCV(DecisionTreeClassifier(), params, n_jobs=-1, cv=3)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_)

grid_search_predict = grid_search.predict(X_train)
print(accuracy_score(y_train, grid_search_predict))

grid_search_predict = grid_search.predict(X_test)
print(accuracy_score(y_test, grid_search_predict))


n_trees = 1000
n_instances = 100
mini_sets = []

shuffle_split = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances)
for train_index, test_index in shuffle_split.split(X_train):
    X_train_mini = X_train[train_index]
    y_train_mini = y_train[train_index]
    mini_sets.append((X_train_mini, y_train_mini))
    
forest = [clone(rand_search.best_estimator_) for _ in range(n_trees)]
accuracy_scores = []
for tree, (X_train_mini, y_train_mini) in zip(forest, mini_sets):
    tree.fit(X_train_mini, y_train_mini)
    
    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

print(np.mean(accuracy_scores))

Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)
    
y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)

print(accuracy_score(y_test, y_pred_majority_votes.reshape([-1])))