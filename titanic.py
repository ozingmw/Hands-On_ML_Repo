import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector

train_data = pd.read_csv("./datasets/titanic/train.csv")
test_data = pd.read_csv("./datasets/titanic/test.csv")

print(train_data.info())
print(train_data.describe())

train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")

train_data["Sex"].replace("male", 1, True)
train_data["Sex"].replace("female", 0, True)

print(train_data.head(10))

print(train_data["Sex"].value_counts())

num_pipeline = Pipeline([
    ("imputer", KNNImputer()),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler", OneHotEncoder())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, make_column_selector(dtype_exclude=object)),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object))
])

train_data_fp = full_pipeline.fit_transform(train_data)

print(train_data_fp.info())