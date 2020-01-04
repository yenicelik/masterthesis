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
            {
                "name": "damping",
                "type": "range",
                "bounds": [0.01, 10.]
            },
            {
                "name": "preference",
                "type": "range",
                "bounds": [-200, 10]
            },
            {
                "name": "max_iter",
                "type": "range",
                "bounds": [100, 500]
            }
        ]
