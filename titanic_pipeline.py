from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer


class FamilyAddAttrib(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X["Family"] = X["SibSp"] + X["Parch"]

class AgeTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        self = self
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        imputer = KNNImputer()
        imputer.fit_transform(X[["Age", "Pclass"]])
        
        return self

class CabinTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X["Cabin"] = X["Cabin"].str[:1]
        X["Cabin"].fillna("Na", inplace=True)
        return X
2