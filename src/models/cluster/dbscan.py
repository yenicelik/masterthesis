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
            RangeParameter(
                name="min_samples",
                parameter_type=ParameterType.INT,
                lower=1, upper=50
            ),
            RangeParameter(
                name="eps",
                parameter_type=ParameterType.FLOAT,
                lower=0.01, upper=5
            ),
            RangeParameter(
                name="metric_params",
                parameter_type=ParameterType.FLOAT,
                lower=0.01, upper=10
            )
        ]
