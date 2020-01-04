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

    def __init__(self, metric='minkowski'):
        super(MTMeanShift, self).__init__()
        # metric is one of:

    @classmethod
    def hyperparameter_dictionary(cls):
        return [
            {
                "name": "metric",
                "type": "choice",
                "values": ['cosine', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'mahalanobis', 'minkowski', ]
            },
            {
                "name": "bandwidth",
                "type": "choice",
                "values": [x**2 for x in range(5)]
            },
            {
                "name": "min_bin_freq",
                "type": "choice",
                "values": [x for x in range(10)]
            },
            {
                "name": "mean_iter",
                "type": "choice",
                "values": [x for x in range(400, 1000, 50)]
            }
        ]
