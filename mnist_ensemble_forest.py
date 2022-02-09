import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

mnist = fetch_openml('mnist_784', version=1, cache=True)
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target.astype(np.uint8), test_size=10000)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=10000)

rnd_clf = RandomForestClassifier(random_state=42)
extree_clf = ExtraTreesClassifier(random_state=42)
svm_clf = LinearSVC(max_iter=100, tol=20 , random_state=42)
mlp_clf = MLPClassifier(random_state=42)

estimators = [rnd_clf, extree_clf, svm_clf, mlp_clf]
for estimator in estimators:
    estimator.fit(X_train, y_train)
    
print([estimator.score(X_val, y_val) for estimator in estimators])

voting_clf = VotingClassifier(estimators=[("rnd", rnd_clf), ("extree", extree_clf), ("svm", svm_clf), ("mlp", mlp_clf)])
voting_clf.fit(X_train, y_train)
print(voting_clf.score(X_val, y_val))
print([estimator.score(X_val, y_val) for estimator in voting_clf.estimators_])

del voting_clf.estimators_[2]
print(voting_clf.score(X_val, y_val))

voting_clf.voting = "soft"
print(voting_clf.score(X_val, y_val))

voting_clf.voting = "hard"
print(voting_clf.score(X_test, y_test))
print([estimator.score(X_test,y_test) for estimator in voting_clf.estimators_])

