"""
    Implements the chinese whispers clustering algorithm

        [1] L 2 F/INESC-ID at SemEval-2019 Task 2: Unsupervised Lexical Semantic Frame Induction using Contextualized Word Representations

"""
import numpy as np

from ax import ParameterType, RangeParameter, SearchSpace
import networkx as nx
from chinese_whispers import chinese_whispers, aggregate_clusters
from sklearn.metrics.pairwise import cosine_similarity

from src.graph_clustering.vectorspace_to_graph import ChineseWhispersClustering
from src.models.cluster.base import BaseCluster

# TODO: Defer this to a later point in time

class ChineseWhispers(BaseCluster):
    """
        Open parameters are:
    """

    def _create_adjacency_matrix(self, X):
        """
            From a given feature matrix X (n_samples, n_features),
            generates a graph adjacency matrix
        :param X:
        :return:
        """

        def _cutoff_function_1(matr):
            return np.mean(matr) - self.std_multiplier_ * np.std(X)

        def _cutoff_function_2(matr):
            return (np.mean(matr) + np.std(X)) / 2.

        # TODO: Use a hyperparameter which selects the cutoff function ...
        _cutoff_function = _cutoff_function_1 if self.std_cutoff else _cutoff_function_2

        cos = cosine_similarity(X, X)

        # TODO: Use a hyperparameter to define whether we should take out identity elements ...
        # if self.remove_identity:
        #     cor[np.nonzero(np.identity(cor.shape[0]))] = 0.  # Was previously below the cutoff calculation..

        cos[cos > _cutoff_function(cos)] = 0.

        # TODO: Use a hyperparameter to define wheter we should take out hubs or not
        summed_weights = np.sum(cos, axis=1)
        print("Summed weights are", summed_weights)
        self.hubs_ = np.argsort(summed_weights)[-self.remove_hub_number_:]
        print("Summed weights are", summed_weights[self.hubs_])

        self.hubs_ = set(self.hubs_)
        self.hub_mask_ = [x for x in np.arange(cos.shape[0]) if x not in self.hubs_]
        cos = cos[self.hub_mask_, :]
        cos = cos[:, self.hub_mask_]

        # Shall I plot the chinese whispers just for the lulzz?
        cos[np.nonzero(np.identity(cos.shape[0]))] = 0.

        # Now run the chinese whispers algorithm
        graph = nx.to_networkx_graph(cos, create_using=nx.DiGraph)

        chinese_whispers(graph, iterations=30) # iterations might depend on the number of clusters...

        print("Clustered items are: ")
        out = list(sorted(aggregate_clusters(graph).items(), key=lambda e: len(e[1]), reverse=True))
        self.cluster_ = [x[0] for x in out]


    def __init__(self, std_multiplier=1.9, remove_hub_number=50, std_cutoff=False):
        super(ChineseWhispers, self).__init__()
        # metric is one of:

        # hyperparameters
        self.std_multiplier_ = std_multiplier
        self.remove_hub_number_ = remove_hub_number
        self.std_cutoff = std_cutoff

    @classmethod
    def hyperparameter_dictionary(cls):
        return [
            {
                "name": "std_multiplier",
                "type": "range",
                "values": [(2 ** x) for x in range(1, 5)],
            },
            {
                "name": "divided_cutoff",
                "type": "choice",
                "bounds": [False, True]
            },
            {
                "name": "remove_hub_number",
                "type": "range",
                "values": [1, 100]
            }
        ]

    def fit_predict(self, X, y=None):
        self._create_adjacency_matrix(X)
        return self.cluster_

