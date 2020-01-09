"""
    Implements the chinese whispers clustering algorithm

        [1] L 2 F/INESC-ID at SemEval-2019 Task 2: Unsupervised Lexical Semantic Frame Induction using Contextualized Word Representations

"""
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from chinese_whispers import chinese_whispers, aggregate_clusters
from sklearn.metrics.pairwise import cosine_similarity

from src.config import args
from src.models.cluster.base import BaseCluster


# TODO: Defer this to a later point in time

class MTChineseWhispers(BaseCluster):
    """
        Open parameters are:

        For testing purposes, we use random embeddings.
        We assume that BERT generate also random-like indecies
        (as they cannot be clustered properly..)
    """

    def insert_hubs_back(self, hubs_ids, original_cos, cluster_labels):

        for idx in sorted(hubs_ids):
            closest_elements = np.argsort(original_cos[idx, :])[::-1]

            for element in closest_elements:
                if not (element in hubs_ids):
                    elements_index_within_cluster_Labels = element - len([x for x in hubs_ids if x < element])
                    cluster_labels.insert(idx, cluster_labels[elements_index_within_cluster_Labels])
                    break

        return cluster_labels

    def merge_unclustered_points_to_closest_cluster(self, cos, cluster_labels):
        counter = Counter(cluster_labels)
        # remove these things
        # print("Counter of cluster labels is", counter)
        free_clusters = set(
            [int(x[0]) for x in
             Counter(int(el) for el in counter.elements() if int(counter[el]) < self.min_cluster_size).items()]
        )

        if len(free_clusters) >= len(np.unique(cluster_labels)):
            if self.verbose:
                print(
                    len(free_clusters), len(np.unique(cluster_labels)),
                    "as many free clusters as there are clusters!"
                )
            # Return all -1! (because clustering failed
            cluster_labels = [-1 for _ in cluster_labels]
            return cluster_labels

        # Check for closest item, which is in an acceptable cluster ...
        for idx in range(len(cluster_labels)):
            if cluster_labels[idx] in free_clusters:
                # Find closest cluster
                closest_elements = np.argsort(cos[idx, :])[::-1]

                # must also check that the two points are not in the same cluster ...
                for element in closest_elements:
                    element = int(element)

                    if cluster_labels[element] in free_clusters:
                        # If included within a cluster label
                        continue

                    cluster_labels[idx] = cluster_labels[element]
                    break

                # if no closest element is found, quit with an error emssage ...

        return cluster_labels

    def _create_adjacency_matrix(self, X):
        """
            From a given feature matrix X (n_samples, n_features),
            generates a graph adjacency matrix
        :param X:
        :return:
        """

        cos = cosine_similarity(X, X)

        lower_cutoff_value = np.mean(cos) + self.std_multiplier_ * np.std(cos)
        # if self.remove_identity:
        #     cor[np.nonzero(np.identity(cor.shape[0]))] = 0.  # Was previously below the cutoff calculation..
        cos[cos < lower_cutoff_value] = 0.

        if self.remove_hub_number_ > 0:
            summed_weights = np.sum(cos, axis=1)
            # print("Summed weights are", summed_weights)
            self.hubs_ = np.argsort(summed_weights)[-self.remove_hub_number_:]
            # print("Summed weights are", summed_weights[self.hubs_])

            self.hubs_ = set(self.hubs_)
            self.hub_mask_ = [x for x in np.arange(cos.shape[0]) if not (x in self.hubs_)]
            cos_hat = cos[self.hub_mask_, :]
            cos_hat = cos_hat[:, self.hub_mask_]

        else:
            cos_hat = cos

        # Shall I plot the chinese whispers just for the lulzz?
        cos_hat[np.nonzero(np.identity(cos_hat.shape[0]))] = 0.
        # print("Final cos is: ", cos)

        # Now run the chinese whispers algorithm
        graph = nx.to_networkx_graph(cos_hat, create_using=nx.DiGraph)
        # graph = graph.convert_to_undirected()
        nx.draw(graph, node_size=10)
        plt.show()

        chinese_whispers(graph, seed=1337)  # iterations might depend on the number of clusters...

        self.cluster_ = np.ones((cos_hat.shape[0],), dtype=int) * -1
        for cluster in aggregate_clusters(graph).items():
            for idx in cluster[1]:
                self.cluster_[idx] = cluster[0]

        assert not (np.any(self.cluster_ == -1)), (self.cluster_)

        self.cluster_ = self.cluster_.tolist()

        self.cluster_ = self.merge_unclustered_points_to_closest_cluster(
            cos=cos_hat,
            cluster_labels=self.cluster_
        )

        if self.verbose:
            colors = [self.cluster_[idx] for idx, node in enumerate(graph.nodes())]  # graph.nodes[node]['label']
            nx.draw_networkx(
                graph,
                node_color=colors,
                node_size=10,
                with_labels=False
            )  # font_color='white', # cmap=plt.get_cmap('jet'),
            plt.show()

        if self.remove_hub_number_ > 0:
            self.cluster_ = self.insert_hubs_back(
                hubs_ids=self.hubs_,
                original_cos=cos,
                cluster_labels=self.cluster_
            )

        self.cluster_ = np.asarray(self.cluster_).astype(int)

    def __init__(self, kwargs):
        super(MTChineseWhispers, self).__init__()
        # metric is one of:

        # A little hakish, because we don't have a wrapper as we do with other cluster-classes

        std_multiplier = kwargs['std_multiplier'] if 'std_multiplier' in kwargs else 2.
        remove_hub_number = kwargs['remove_hub_number'] if 'remove_hub_number' in kwargs else 100
        min_cluster_size = kwargs['min_cluster_size'] if 'min_cluster_size' in kwargs else 5
        verbose= kwargs['verbose'] if 'verbose' in kwargs else False

        self.verbose = verbose

        # hyperparameters
        self.std_multiplier_ = std_multiplier
        self.remove_hub_number_ = remove_hub_number
        self.min_cluster_size = min_cluster_size

    @classmethod
    def hyperparameter_dictionary(cls):
        return [
            {
                "name": "std_multiplier",
                "type": "range",
                "bounds": [-3., 3.],
            },
            {
                "name": "remove_hub_number",
                "type": "range",
                "bounds": [0, min(200, int(args.max_samples * 0.6))]
            },
            {
                "name": "min_cluster_size",
                "type": "range",
                "bounds": [1, 50]
            }
        ]

    def fit_predict(self, X, y=None):
        self._create_adjacency_matrix(X)
        return self.cluster_


if __name__ == "__main__":
    print("Check chinese whispers incl. plotting ..")
    X = np.random.random((500, 150))

    model = MTChineseWhispers(verbose=True)

    model.fit_predict(X)
