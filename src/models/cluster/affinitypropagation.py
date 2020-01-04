"""

"""
from ax import ParameterType, RangeParameter, SearchSpace
import numpy as np
from sklearn.cluster import OPTICS, AffinityPropagation

from src.models.cluster.base import BaseCluster


class MTAffinityPropagation(BaseCluster):
    """
        Taking out any trace of DBSCAN,
        because we have a designated class for that.

        No meaningful open parameters

        Open parameters: affinity="euclidean"
        is the only real option
    """

    def __init__(self, metric='minkowski'):
        super(MTAffinityPropagation, self).__init__()
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
