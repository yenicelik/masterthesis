"""

"""
from math import inf

from ax import ParameterType, RangeParameter
import numpy as np
from sklearn.cluster import OPTICS

from src.models.cluster.base import BaseCluster


class MTOptics(BaseCluster):
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

    def __init__(self, kargs):
        super(MTOptics, self).__init__()
        # metric is one of:
        self.model = OPTICS(**kargs)

    @classmethod
    def hyperparameter_dictionary(cls):
        return [
            {
                "name": "min_samples",
                "type": "choice",
                "values": [2**x for x in range(5)],
            },
            {
                "name": "cluster_method",
                "type": "fixed",
                "value": "xi",
            },
            {
                "name": "max_eps",
                "type": "range",
                "bounds": [50., float(inf)]
            },
            # RangeParameter(
            #     name="p",
            #     parameter_type=ParameterType.FLOAT,
            #     lower=0.1, upper=10
            # ),
            # RangeParameter(
            #     name="xi",
            #     parameter_type=ParameterType.FLOAT,
            #     lower=0.001, upper=10
            # ),
        ]

    def fit(self, X, y=None):
        # Run hyperparameter optimizeration inside of this...
        return self.model.fit(X, y)

