import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import KNNImputer

train_data = pd.read_csv("./datasets/titanic/train.csv")
test_data = pd.read_csv("./datasets/titanic/test.csv")

print(train_data.info())

train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")

train_data["Sex"].replace("male", 1, True)
train_data["Sex"].replace("female", 0, True)

print(train_data.head(10))

print(train_data["Sex"].value_counts())

imputer_1 = KNNImputer(weights="distance")
age_by_class_1 = train_data.loc[train_data["Pclass"] == 1]["Age"]
train_data["Age"] = imputer_1.fit_transform(age_by_class_1)
print(train_data["Age"].isnull())

num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])