"""
    Base class for any clustering algorithm
"""

class BaseCluster:

    def __init__(self):
        pass

    def hyperparameter_dictionary(self):
        raise NotImplementedError

    def fit(self, X, y=None):
        return self.model.fit(X, y)

    def predict(self, X, y=None):
        return self.model.predict(X, y)
