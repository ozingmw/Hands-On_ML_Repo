import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


train_data = pd.read_csv("./datasets/titanic/train.csv")
test_data = pd.read_csv("./datasets/titanic/test.csv")

print(train_data.info())
print(train_data.describe())

# PassengerId: 각 승객의 고유 식별자.
# Survived: 타깃입니다. 0은 생존하지 못한 것이고 1은 생존을 의미합니다. / Pclass: 승객 등급. 1, 2, 3등석.
# Name, Sex, Age: 이름 그대로 의미입니다. / SibSp: 함께 탑승한 형제, 배우자의 수.
# Parch: 함께 탑승한 자녀, 부모의 수. / Ticket: 티켓 아이디
# Fare: 티켓 요금 (파운드) / Cabin: 객실 번호 / Embarked: 승객이 탑승한 곳. C(Cherbourg), Q(Queenstown), S(Southampton)

# 사용 할 특성 : Pclass, Sex, Age, SibSp + Parch, Fare, Embarked
# 결측치 있는 특성 : Age, Embarked

train_data["Family"] = train_data["SibSp"] + train_data["Parch"]
train_data = train_data[["Pclass", "Sex", "Age", "Family", "Fare", "Embarked"]]

# Age 결측치 제거
imputer = KNNImputer()
train_data["Age"] = imputer.fit_transform(train_data[["Age", "Pclass"]])

# Embarked 결측치 제거
imputer = SimpleImputer(strategy="most_frequent")
train_data["Embarked"] = imputer.fit_transform(train_data[["Embarked"]])
ohe = OneHotEncoder()
temp = ohe.fit_transform(train_data[["Embarked"]])