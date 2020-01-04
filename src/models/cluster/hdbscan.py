"""

"""
import hdbscan
from ax import ParameterType, RangeParameter, SearchSpace
import numpy as np
from sklearn.cluster import OPTICS

from src.models.cluster.base import BaseCluster


class MTHdbScan(BaseCluster):
    """
        No open parameters really
    """

    def __init__(self):
        super(MTHdbScan, self).__init__()
        # metric is one of:

    @classmethod
    def hyperparameter_dictionary(cls):
        return [
            RangeParameter(
                name="min_cluster_size",
                parameter_type=ParameterType.INT,
                lower=1, upper=50
            ),
            RangeParameter(
                name="min_samples",
                parameter_type=ParameterType.FLOAT,
                lower=1, upper=50
            ),
            RangeParameter(
                name="cluster_selection_epsilon",
                parameter_type=ParameterType.FLOAT,
                lower=0.001, upper=10
            ),
            RangeParameter(
                name="alpha",
                parameter_type=ParameterType.FLOAT,
                lower=0.5, upper=2.0
            )

        ]

    def fit(self, X, y=None):
        # Run hyperparameter optimizeration inside of this...
        for i in range(self.max_optimization_iterations):
            # Sample
            self.model = hdbscan.HDBSCAN(

            )
