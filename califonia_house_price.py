# Hand-On Machine Learning with Scikit-Learn, Keras & Tensorflow
# 21.12.28 ~

from ctypes import set_last_error
import os

from scipy.sparse import dia
from scipy.stats.mstats_basic import linregress
from sklearn import tree
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

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
#plt.show()


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

housing["income_cat"] = pd.cut(housing["median_income"], 
							   bins=[0., 1.5, 3.0, 4.5, 6., np.inf], 
                               labels=[1, 2, 3, 4, 5]
                               )
housing["income_cat"].hist()
#plt.show()


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude")
#plt.show()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
#plt.show()

#alpha = 투명도

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.3,
	s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False
)
plt.legend()
#plt.show()

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))


from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8), diagonal="kde")
#plt.show()

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)


housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)

from sklearn.impute import SimpleImputer, KNNImputer
#simple -> strategy에 따라 평균, 중앙값
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.median().values)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
print(housing_tr.info())

#KNN에 따라 최근접 이웃 방식
imputer_K = KNNImputer()
housing_num_K = housing.drop("ocean_proximity", axis=1)
X_K = imputer_K.fit_transform(housing_num_K)
housing_tr_K = pd.DataFrame(X_K, columns=housing_num_K.columns, index=housing_num_K.index)
print(housing_tr_K.info())

print(housing["ocean_proximity"])
print(housing[["ocean_proximity"]])
housing_cat = housing[["ocean_proximity"]]

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])

print(ordinal_encoder.categories_)


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_onehot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_onehot.toarray())


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    				("imputer", KNNImputer()),
                    ("attribs_adder", CombinedAttributesAdder()),
                    ("std_scaler", StandardScaler()),
			   ])

housing_num_tr = num_pipeline.fit_transform(housing_num_K)


from sklearn.compose import ColumnTransformer, make_column_selector

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    				("num", num_pipeline, num_attribs),
                    ("cat", OneHotEncoder(), cat_attribs),
				])

housing_prepared = full_pipeline.fit_transform(housing)


#make_colum_selector사용시 리스트 설정 필요 없이 알아서 추출해줌
#오류 발생시 확인(p.109)
full_pipeline_mcs = ColumnTransformer([
    					("num", num_pipeline, make_column_selector(dtype_include=np.float64)),
                        ("cat", OneHotEncoder(), make_column_selector(dtype_include=object)),
					])

housing_prepared_mcs = full_pipeline_mcs.fit_transform(housing)


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared_mcs, housing_labels)

sdata = housing.iloc[:5]
slabel = housing_labels.iloc[:5]
sdata_prepared = full_pipeline.transform(sdata)
print("predict : ", lin_reg.predict(sdata_prepared))
print("label : ", list(slabel))

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared_mcs)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared_mcs, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared_mcs)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared_mcs, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
	print("score = ", scores)
	print("avg = ", scores.mean())
	print("std = ", scores.std())

display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared_mcs, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared_mcs, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared_mcs)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)
forest_score = cross_val_score(forest_reg, housing_prepared_mcs, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_score = np.sqrt(-forest_score)
display_scores(forest_rmse_score)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(housing_prepared_mcs, housing_labels)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
for mean_score, params in zip(grid_search.cv_results_["mean_test_score"], grid_search.cv_results_["params"]):
    print(np.sqrt(-mean_score), params)


param_dists = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
random_search = RandomizedSearchCV(forest_reg, param_dists, n_iter=500, cv=5, scoring="neg_mean_squared_error", random_state=42, return_train_score=True)
random_search.fit(housing_prepared_mcs, housing_labels)
print(random_search.best_params_)
print(random_search.best_estimator_)
for mean_score, params in zip(random_search.cv_results_["mean_test_score"], random_search.cv_results_["params"]):
    print(np.sqrt(-mean_score), params)