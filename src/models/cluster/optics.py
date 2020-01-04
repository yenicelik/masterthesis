"""

"""
from ax import ParameterType, RangeParameter
import numpy as np
from sklearn.cluster import OPTICS

from src.models.cluster.base import BaseCluster


class MTOptics(BaseCluster):
    """
        Taking out any trace of DBSCAN,
        because we have a designated class for that.

        No meaningful open parameters
    """

    def _possible_metrics(self):
        # TODO: Perhaps turn this into an integer optimization,
        # (through dictionary translation)
        # although even neighborhood cannot really be destroyed...

        # Anything commented out is covered through other metrics,
        # or is clearly not part of this space ...
        return [
            'cosine',
            'braycurtis',
            'canberra',
            'chebyshev',
            'correlation',
            'mahalanobis',
            'minkowski',
        ]

    def _possible_cluster_methods(self):
        return [
            'xi'
        ]

    def __init__(self, metric='minkowski'):
        super(MTOptics, self).__init__()
        # metric is one of:

    @classmethod
    def hyperparameter_dictionary(cls):
        return [
            RangeParameter(
                name="min_samples",
                parameter_type=ParameterType.INT,
                lower=1, upper=50
            ),
            RangeParameter(
                name="max_eps",
                parameter_type=ParameterType.FLOAT,
                lower=50, upper=np.inf
            ),
            RangeParameter(
                name="p",
                parameter_type=ParameterType.FLOAT,
                lower=0.1, upper=10
            ),
            RangeParameter(
                name="xi",
                parameter_type=ParameterType.FLOAT,
                lower=0.001, upper=10
            ),
        ]

    def fit(self, X, y=None):
        # Run hyperparameter optimizeration inside of this...
        for i in range(self.max_optimization_iterations):
            # Sample
            self.model = OPTICS()

