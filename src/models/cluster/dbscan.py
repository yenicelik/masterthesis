"""

"""
from ax import ParameterType, RangeParameter, SearchSpace
import numpy as np
from sklearn.cluster import OPTICS

from src.models.cluster.base import BaseCluster


class MTDbScan(BaseCluster):
    """
        No open parameters
    """

    def _possible_metrics(self):
        # TODO: Perhaps turn this into an integer optimization,
        # (through dictionary translation)
        # although even neighborhood cannot really be destroyed...

        # Anything commented out is covered through other metrics,
        # or is clearly not part of this space ...

        # set as `metric=`
        return [
            'cosine',
            'braycurtis',
            'canberra',
            'chebyshev',
            'correlation',
            'mahalanobis',
            'minkowski',
        ]

    def __init__(self, metric='minkowski'):
        super(MTDbScan, self).__init__()
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
                "name": "eps",
                "type": "range",
                "bounds": [0.01, 5.]
            },
            {
                "name": "metric_params",
                "type": "range",
                "bounds": [0.01, 10.]
            },
            {
                "name": "metric",
                "type": "choice",
                "values": ['cosine', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'mahalanobis', 'minkowski', ]
            },

        ]
