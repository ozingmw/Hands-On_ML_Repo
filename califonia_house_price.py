# Hand-On Machine Learning with Scikit-Learn, Keras & Tensorflow
# 21.12.28 ~

import os

from file_download import github
from file_load import csv

HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
HOUSING_PATH = os.path.join("datasets", "housing")

github.download(url=HOUSING_URL, path=HOUSING_PATH)
housing = csv.load(os.path.join(HOUSING_PATH, "housing.csv"))

print(housing.head(10))
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())


import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


import numpy as np

#initialize every run
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(train_set)
print(test_set)


from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_ : test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


import pandas as pd

housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
housing["income_cat"].hist()
plt.show()


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]