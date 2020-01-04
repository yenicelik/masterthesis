"""

"""
from ax import ParameterType, RangeParameter, SearchSpace
import numpy as np
from sklearn.cluster import OPTICS, AffinityPropagation

from src.models.cluster.base import BaseCluster


class OurAffinityPropagation(BaseCluster):
    """
        Taking out any trace of DBSCAN,
        because we have a designated class for that.

        No meaningful open parameters

        Open parameters: affinity="euclidean"
        is the only real option
    """

    def __init__(self, metric='minkowski'):
        super(OurAffinityPropagation, self).__init__()
        # metric is one of:

    def hyperparameter_dictionary(self):
        return [
            RangeParameter(
                name="damping",
                parameter_type=ParameterType.INT,
                lower=0.01, upper=10
            ),
            RangeParameter(
                name="preference",
                parameter_type=ParameterType.FLOAT,
                lower=-200, upper=10
            ),
            RangeParameter(
                name="max_iter",
                parameter_type=ParameterType.FLOAT,
                lower=100, upper=500
            ),
        ]

    def fit(self, X, y=None):
        # Run hyperparameter optimizeration inside of this...
        for i in range(self.max_optimization_iterations):
            # Sample
            self.model = AffinityPropagation()

