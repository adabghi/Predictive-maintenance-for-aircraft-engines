from __future__ import division

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

class Classifier(BaseEstimator):
    def __init__(self):
        self.model = LogisticRegression(solver='lbfgs',multi_class='multinomial')
		
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
