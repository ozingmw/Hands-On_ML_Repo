from operator import index
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin

train_data = pd.read_csv("./datasets/titanic/train.csv")
test_data = pd.read_csv("./datasets/titanic/test.csv")

print(train_data.info())
print(train_data.describe())

# PassengerId: 각 승객의 고유 식별자.
# Survived: 타깃입니다. 0은 생존하지 못한 것이고 1은 생존을 의미합니다. / Pclass: 승객 등급. 1, 2, 3등석. / Name, Sex, Age: 이름 그대로 의미입니다. / SibSp: 함께 탑승한 형제, 배우자의 수.
# Parch: 함께 탑승한 자녀, 부모의 수. / Ticket: 티켓 아이디 / Fare: 티켓 요금 (파운드) / Cabin: 객실 번호 / Embarked: 승객이 탑승한 곳. C(Cherbourg), Q(Queenstown), S(Southampton)

train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")

train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
test_data["RelativesOnboard"] = test_data["SibSp"] + test_data["Parch"]

num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

print(train_data.head(10))
print(test_data.head())

print(test_data.info())

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        col_names = "SibSp", "Parch"
        self.sibsp_ix, self.parch_ix = [num_attribs.index(c) for c in col_names]
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        RelativesOnborad = X[:, self.sibsp_ix] + X[:, self.parch_ix]
        return np.c_[X, RelativesOnborad]

num_pipeline = Pipeline([
    ("imputer", KNNImputer()),
    ("attrib_adder", CombinedAttributesAdder()),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler", OneHotEncoder())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

X_train = full_pipeline.fit_transform(train_data[num_attribs+cat_attribs])
y_train = train_data["Survived"]

rf_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf_clf.fit(X_train, y_train)

X_test = full_pipeline.transform(test_data[num_attribs+cat_attribs])
y_pred = rf_clf.predict(X_test)
rf_score = cross_val_score(rf_clf, X_test, y_pred, cv=10)
print(rf_score.mean())
result = pd.DataFrame(np.c_[test_data.index, y_pred], columns=["PassengerId", "Survived"]).to_csv("./datasets/titanic/titanic_result.csv", index=False)