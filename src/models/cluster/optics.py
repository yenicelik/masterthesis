"""

"""
from ax import ParameterType, RangeParameter, SearchSpace
import numpy as np

from src.models.cluster.base import BaseCluster


class Optics(BaseCluster):

    def _possible_metrics(self):
        # TODO: Perhaps turn this into an integer optimization,
        # (through dictionary translation)
        # although even neighborhood cannot really be destroyed...

        # Anything commented out is covered through other metrics,
        # or is clearly not part of this space ...
        return [
        # 'cityblock',
        'cosine',
        # 'euclidean',
        # 'l1',
        # 'l2',
        # 'manhattan',
        'braycurtis',
        'canberra',
        'chebyshev',
        'correlation',
        # 'dice', categorical
        # 'hamming', binary/integer/categorical
        # 'jaccard', categorical
        # 'kulsinski', categorical
        'mahalanobis',
        'minkowski',
        # 'rogerstanimoto', categorical
        # 'russellrao', categorical
        # 'seuclidean', redundant with eucliedean
        # 'sokalmichener', categorical
        # 'sokalsneath', categorical
        # 'sqeuclidean', no idea
        # 'yule' categorical
        ]

    def __init__(self, metric='minkowski'):
        super(Optics, self).__init__()
        # metric is one of:
        possible_metrics = []

    def hyperparameter_dictionary(self):
        return [
            RangeParameter(
                name="min_samples",
                parameter_type=ParameterType.INT,
                lower=1, upper=50
            ),
            RangeParameter(
                name="max_eps",
                parameter_type=ParameterType.FLOAT,
                lower=50, upper=np.inf
            ),
            RangeParameter(
                name="max_eps",
                parameter_type=ParameterType.FLOAT,
                lower=50, upper=np.inf
            ),
        ]

    def a(self,  metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None,
          xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30):
