import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from titanic_pipeline import CabinTransform, FamilyAddAttrib


# 1차 목표 : 완성
# 2차 목표 : 80%
# 3차 목표 : 90%

# 현재 : 

train_data = pd.read_csv("./datasets/titanic/train.csv")
test_data = pd.read_csv("./datasets/titanic/test.csv")

print(train_data.info())
print(train_data.describe())

# PassengerId: 각 승객의 고유 식별자.
# Survived: 타깃입니다. 0은 생존하지 못한 것이고 1은 생존을 의미합니다. / Pclass: 승객 등급. 1, 2, 3등석.
# Name, Sex, Age: 이름 그대로 의미입니다. / SibSp: 함께 탑승한 형제, 배우자의 수.
# Parch: 함께 탑승한 자녀, 부모의 수. / Ticket: 티켓 아이디
# Fare: 티켓 요금 (파운드) / Cabin: 객실 번호 / Embarked: 승객이 탑승한 곳. C(Cherbourg), Q(Queenstown), S(Southampton)

# 사용 할 특성 : Pclass, Sex, Age, SibSp + Parch, Fare, Cabin, Embarked
# 결측치 있는 특성 : Age, Cabin, Embarked

train_data["Family"] = train_data["SibSp"] + train_data["Parch"]
train_data = train_data[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Family", "Fare", "Cabin", "Embarked"]]

# Age 결측치 제거
imputer = KNNImputer()
train_data["Age"] = imputer.fit_transform(train_data[["Age", "Pclass"]])

# Embarked 결측치 제거, 변경
imputer = SimpleImputer(strategy="most_frequent")
train_data["Embarked"] = imputer.fit_transform(train_data[["Embarked"]])
ohe = OneHotEncoder(sparse=False)
Embarked = ohe.fit_transform(train_data[["Embarked"]])
Embarked_df = pd.DataFrame(Embarked, columns=("Embarked_" + ohe.categories_[0]))
train_data = pd.concat([train_data, Embarked_df], axis=1)
train_data.drop("Embarked", axis=1, inplace=True)

# Sex 숫자로 변경
ohe = OneHotEncoder(sparse=False)
train_data["Sex"] = ohe.fit_transform(train_data[["Sex"]])

# Cabin 분류
'''
Na 채워넣어야함
(현재 nan값 Na로 변경후 그냥돌림)
'''
train_data["Cabin"] = train_data["Cabin"].str[:1]
train_data["Cabin"].fillna("Na", inplace=True)
ohe = OneHotEncoder(sparse=False)
cabin = ohe.fit_transform(train_data[["Cabin"]])
cabin_df = pd.DataFrame(cabin, columns=("Cabin_" + ohe.categories_[0]))
train_data = pd.concat([train_data, cabin_df], axis=1)
train_data.drop("Cabin", axis=1, inplace=True)

# 데이터 준비
X_train = train_data.drop("Survived", axis=1)
y_train = train_data["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 모든 특성에 스케일링?
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.fit_transform(X_val)

# 모델
# gridsearch 최적 파라미터
rnd_clf = RandomForestClassifier(max_depth=8, min_samples_leaf=5, n_estimators=350, random_state=42, oob_score=True)
rnd_clf.fit(X_train, y_train)
print(rnd_clf.oob_score_)
print(sorted(zip(rnd_clf.feature_importances_, X_train.columns), reverse=True))

print(accuracy_score(y_val, rnd_clf.predict(X_val)))


# 파이프라인
# age 결측치 제거, embarked 결측치 제거, 변경, sex숫자로 변경, cabin 변경


embarked_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(sparse=False)),
])

full_pipeline = ColumnTransformer([
    ("family", FamilyAddAttrib(), ["SibSp", "Parch"]),
    ("age", KNNImputer(), ["Age", "Pclass"]),
    # ("embarked", embarked_pipeline, ["Embarked"]),
    ("sex", OneHotEncoder(sparse=False), ["Sex"]),
    # ("cabin", CabinTransform(), ["Cabin"]),
])

train_data = pd.read_csv("./datasets/titanic/train.csv")
train_data = train_data[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
X_train = train_data.drop("Survived", axis=1)
y_train = train_data["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train = full_pipeline.fit_transform(X_train)

rnd_clf = RandomForestClassifier(max_depth=8, min_samples_leaf=5, n_estimators=350, random_state=42, oob_score=True)
rnd_clf.fit(X_train, y_train)

print(rnd_clf.oob_score_)
print(sorted(zip(rnd_clf.feature_importances_, train_data.columns[1:]), reverse=True))

# print(accuracy_score(y_val, rnd_clf.predict(X_val)))




# 마무리
# y_pred = rnd_clf.predict(train_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Family", "Fare", "Cabin", "Embarked"]])
# result = pd.DataFrame(np.c_[test_data.index, y_pred], columns=["PassengerId", "Survived"]).to_csv("./datasets/titanic/titanic_result.csv", index=False)