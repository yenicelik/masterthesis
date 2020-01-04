"""

"""
from ax import ParameterType, RangeParameter, SearchSpace
import numpy as np
from sklearn.cluster import OPTICS, MeanShift

from src.models.cluster.base import BaseCluster


class MTMeanShift(BaseCluster):
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
        super(MTMeanShift, self).__init__()
        # metric is one of:

    @classmethod
    def hyperparameter_dictionary(cls):
        return [
            # RangeParameter(
            #     name="bandwidth",
            #     parameter_type=ParameterType.INT,
            #     lower=0.001, upper=100
            # ),
            RangeParameter(
                name="min_bin_freq",
                parameter_type=ParameterType.INT,
                lower=1, upper=10
            ),
            RangeParameter(
                name="mean_iter",
                parameter_type=ParameterType.INT,
                lower=300, upper=1000
            ),
        ]

    def fit(self, X, y=None):
        # Run hyperparameter optimizeration inside of this...
        for i in range(self.max_optimization_iterations):
            # Sample
            self.model = MeanShift()

