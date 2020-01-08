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

from src.models.cluster.base import BaseCluster


# TODO: Defer this to a later point in time

class ChineseWhispers(BaseCluster):
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
        print("Counter of cluster labels is", counter)
        free_clusters = set(
            [int(x[0]) for x in Counter(int(el) for el in counter.elements() if int(counter[el]) < self.min_cluster_size).items()]
        )

        print("Free clusters are: ", free_clusters)

        assert len(free_clusters) < len(np.unique(cluster_labels))

        print("Cluster labels at the beginning..", len(cluster_labels))

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

                    print("Assigning ", cluster_labels[idx], " to ", cluster_labels[element])
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
        print("Cos shape is: ", cos.shape)

        lower_cutoff_value = np.mean(cos) + self.std_multiplier_ * np.std(cos)
        # if self.remove_identity:
        #     cor[np.nonzero(np.identity(cor.shape[0]))] = 0.  # Was previously below the cutoff calculation..
        cos[cos < lower_cutoff_value] = 0.

        # # TODO: Use a hyperparameter to define wheter we should take out hubs or not
        summed_weights = np.sum(cos, axis=1)
        # print("Summed weights are", summed_weights)
        self.hubs_ = np.argsort(summed_weights)[-self.remove_hub_number_:]
        # print("Summed weights are", summed_weights[self.hubs_])

        self.hubs_ = set(self.hubs_)
        self.hub_mask_ = [x for x in np.arange(cos.shape[0]) if not (x in self.hubs_)]
        cos_hat = cos[self.hub_mask_, :]
        cos_hat = cos_hat[:, self.hub_mask_]

        # Shall I plot the chinese whispers just for the lulzz?
        cos_hat[np.nonzero(np.identity(cos_hat.shape[0]))] = 0.
        # print("Final cos is: ", cos)

        # Now run the chinese whispers algorithm
        graph = nx.to_networkx_graph(cos_hat, create_using=nx.DiGraph)
        # graph = graph.convert_to_undirected()
        nx.draw(graph, node_size=10)
        plt.show()

        chinese_whispers(graph, seed=1337)  # iterations might depend on the number of clusters...

        print("Clustered items are: ")
        self.cluster_ = np.ones((cos_hat.shape[0],), dtype=int) * -1
        print("aggregated clusters are", aggregate_clusters(graph).items())
        for cluster in aggregate_clusters(graph).items():
            print("Printing aggregated cluster")
            for idx in cluster[1]:
                self.cluster_[idx] = cluster[0]

        print("aggregated clusters are", self.cluster_)
        # self.cluster_ = [int(x) for x in self.cluster_)

        assert not (np.any(self.cluster_ == -1)), (self.cluster_)

        self.cluster_ = self.cluster_.tolist()

        self.cluster_ = self.merge_unclustered_points_to_closest_cluster(
            cos_hat,
            self.cluster_
        )
        print("Self cluster is: ")
        print(self.cluster_)
        print(len(self.cluster_))
        print(Counter(self.cluster_))
        print(len(np.unique(self.cluster_)))

        if self.verbose:
            colors = [1. / graph.nodes[node]['label'] for node in graph.nodes()]
            nx.draw_networkx(
                graph,
                node_color=colors,
                node_size=10,
                with_labels=False
            )  # font_color='white', # cmap=plt.get_cmap('jet'),
            plt.show()

        self.cluster_ = self.insert_hubs_back(
            hubs_ids=self.hubs_,
            original_cos=cos,
            cluster_labels=self.cluster_
        )

        self.cluster_ = np.asarray(self.cluster_).astype(int)

        print("Self cluster is: ")
        print(self.cluster_)
        print(len(self.cluster_))
        print(Counter(self.cluster_))
        print(len(np.unique(self.cluster_)))

    def __init__(self, std_multiplier=2., remove_hub_number=100, min_cluster_size=5, verbose=False):
        super(ChineseWhispers, self).__init__()
        # metric is one of:

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
                "values": [(2 ** x) for x in range(1, 5)],
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


if __name__ == "__main__":
    print("Check chinese whispers incl. plotting ..")
    X = np.random.random((500, 150))

    model = ChineseWhispers(verbose=True)

    model.fit_predict(X)
