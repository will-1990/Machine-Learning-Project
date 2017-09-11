# encoding=utf-8
'''
----------------
date:2017-04-22
----------------
author: will
----------------
reference:
[1] Wei Xun,etc. An Ensemble Model for Diabetes Diagnosis in Large-scale and Imbalanced Dataset.
----------------
'''

from imblearn.ensemble import EasyEnsemble
import numpy as np
from sklearn.base import ClassifierMixin, clone


class XEnsemble(ClassifierMixin):
    def __init__(self, base_estimator=None, n_subsets=10):
        self.base_estimator = base_estimator
        self.n_subsets = n_subsets

    def fit(self, X, y):
        self.estimator_ = []
        ee = EasyEnsemble(n_subsets=self.n_subsets)
        X_new, y_new = ee.fit_sample(X, y)
        for i in range(self.n_subsets):
            estimator = clone(self.base_estimator)
            estimator.fit(X_new[i], y_new[i])
            self.estimator_.append(estimator)

        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        if len(proba.shape) > 1:
            indexes = np.argmax(proba, axis=1)
        else:
            indexes = np.repeat(0, proba.shape[0])
            indexes[proba > 0.5] = 1

        return np.mat(indexes).transpose()

    def predict_proba(self, X):
        total_proba = 0.0
        for i in range(self.n_subsets):
            proba = self.estimator_[i].predict_proba(X)
            total_proba += proba

        return total_proba / self.n_subsets

