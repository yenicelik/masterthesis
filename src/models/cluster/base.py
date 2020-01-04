"""
    Base class for any clustering algorithm
"""

class BaseCluster:

    @property
    def max_optimization_iterations(self):
        """
            Maximum number of optimization steps to run
        :return:
        """
        return 100

    def __init__(self):
        self.model = None

    @classmethod
    def hyperparameter_dictionary(cls):
        raise NotImplementedError

    @property
    def _min_cluster_size(self):
        return 5

    def fit(self, X, y=None):
        return self.model.fit(X, y)

    def predict(self, X, y=None):
        return self.model.predict(X, y)
