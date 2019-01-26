from __future__ import print_function

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


class BootstrapRegression(LinearRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _get_random_sample(self, X, y):
        idx = np.random.choice(
            np.arange(0, len(X)), size=len(X))
        return X[idx], y[idx]
    
    def bootstrap(self, X, y, n=1000, seed=2323, **kwargs):
        np.random.seed(seed)
        self.boot_coef = np.zeros(shape=n)
        self.boot_intercept = np.zeros(shape=n)
        self.boot_score = np.zeros(shape=n)
        for i in range(n):
            X_r, y_r = self._get_random_sample(X, y)
            self.fit(X_r, y_r, **kwargs)
            self.boot_coef[i] = self.coef_
            self.boot_intercept[i] = self.intercept_
            self.boot_score[i] = self.score(X_r, y_r)
        return self
    
    def get_confint(self, q=(2.5, 97.5)):
        intercept = np.percentile(
            self.boot_intercept, q=q)
        coef = np.percentile(
            self.boot_coef, q=q)
        score = np.percentile(
            self.boot_score, q=q)
        return intercept, coef, score
    
    def get_mean_param(self):
        intercept = self.boot_intercept.mean()
        coef = self.boot_coef.mean()
        score = self.boot_score.mean()
        return intercept, coef, score


