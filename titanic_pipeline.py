import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer


class FamilyAddAttrib(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X["Family"] = X["SibSp"] + X["Parch"]
        return X

class CabinTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X["Cabin"] = X["Cabin"].str[:1]
        X["Cabin"].fillna("Na", inplace=True)
        return X