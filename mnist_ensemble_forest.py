from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier

mnist = fetch_openml('mnist_784', version=1, cache=True)
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=10000)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=10000)

rnd_clf = RandomForestClassifier(random_state=42)
extree_clf = ExtraTreeClassifier(random_state=42)
svm_clf = SVC(random_state=42)
svm_clf_per = SVC(probability=True, random_state=42)
sgd_clf = SGDClassifier(random_state=42)

vote_clf = VotingClassifier(
    estimators=[("rnd", rnd_clf), ("extree", extree_clf), ("svm", svm_clf), ("sgd", sgd_clf)],
    voting="hard",
)

for clf in (rnd_clf, extree_clf, svm_clf, sgd_clf, vote_clf):
    clf.fit(X_train, y_train)
    print(clf.__class__.__name__, clf.score(X_val, y_val))
    

vote_clf = VotingClassifier(
    estimators=[("rnd", rnd_clf), ("extree", extree_clf), ("svm", svm_clf_per), ("sgd", sgd_clf)],
    voting="soft",
)

vote_clf.fit(X_train, y_train)
print(vote_clf.score(X_val, y_val))