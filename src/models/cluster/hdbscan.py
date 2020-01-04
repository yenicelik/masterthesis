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
            {
                "name": "min_samples",
                "type": "choice",
                "values": [2 ** x for x in range(5)],
            },
            {
                "name": "min_cluster_size",
                "type": "choice",
                "values": [2 ** x for x in range(6)],
            },
            {
                "name": "cluster_selection_epsilon",
                "type": "range",
                "bounds": [0.001, 10]
            },
            {
                "name": "cluster_selection_epsilon",
                "type": "range",
                "bounds": [0.5, 2.0]
            }
        ]