"""
    Implements the chinese whispers clustering algorithm
"""
import numpy as np

from ax import ParameterType, RangeParameter, SearchSpace

from src.graph_clustering.vectorspace_to_graph import ChineseWhispersClustering
from src.models.cluster.base import BaseCluster

# TODO: Defer this to a later point in time

class ChineseWhispers(BaseCluster):
    """
        Open parameters are:
    """

    def __init__(self, metric='minkowski'):
        super(ChineseWhispers, self).__init__()
        # metric is one of:

    @classmethod
    def hyperparameter_dictionary(cls):
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
            self.model = ChineseWhispersClustering()

