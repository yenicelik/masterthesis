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

    def __init__(self, kargs):
        super(MTMeanShift, self).__init__()
        # metric is one of:
        self.model = MeanShift(**kargs)

    @classmethod
    def hyperparameter_dictionary(cls):
        # removed mahalanobis
        return [
            {
                "name": "bandwidth",
                "type": "choice",
                "values": [(x**2) for x in range(1, 5)]
            },
            {
                "name": "min_bin_freq",
                "type": "choice",
                "values": [x for x in range(1, 10)]
            },
            {
                "name": "max_iter",
                "type": "choice",
                "values": [x for x in range(400, 1000, 50)]
            }
        ]
