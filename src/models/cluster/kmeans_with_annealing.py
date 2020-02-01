"""
    Implements k-means with annealing.
    Clusters that are
        (1) not big enough and
        (2) have centers too close to each other
    will be merged
"""
from collections import Counter

from ax import ParameterType, RangeParameter, SearchSpace
import numpy as np
from sklearn.cluster import OPTICS, DBSCAN, KMeans

from src.models.cluster.base import BaseCluster


class MTKMeansAnnealing(BaseCluster):
    """
        No open parameters
    """

    def merge_clusters_too_close_to_each_other(self, X_centers, cluster_labels):
        """
            Based on some heuristics, and hyperparameters,
            merges two clusters together if they are too close together
        :param X_centers:
        :param cluster_labels:
        :return:
        """
        pass

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

    def _possible_metrics(self):
        # TODO: Perhaps turn this into an integer optimization,
        # (through dictionary translation)
        # although even neighborhood cannot really be destroyed...

        # Anything commented out is covered through other metrics,
        # or is clearly not part of this space ...

        # set as `metric=`
        return [
            'k-means++',
            'random'
        ]

    def __init__(self, kwargs):
        super(MTKMeansAnnealing, self).__init__()
        # metric is one of:
        self.model = KMeans(**kwargs)

        min_cluster_size = kwargs['min_cluster_size'] if 'min_cluster_size' in kwargs else 5
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False

        self.verbose = verbose
        self.min_cluster_size = min_cluster_size

    @classmethod
    def hyperparameter_dictionary(cls):
        # removed mahalanobis
        return [
            {
                "name": "n_clusters",
                "type": "choice",
                "values": [10, 20, 40, 55, 70, 90],
            },
            {
                "name": "init",
                "type": "choice",
                "values": ['k-means++', 'random']
            },
        ]

    def fit_predict(self, X, y=None):

        labels = self.model.fit_predict(X)

        # (1) Merge all clusters that are too small with their closest neighbor
        labels = self.merge_unclustered_points_to_closest_cluster(
            cos=np.dot(X, X.T),
            cluster_labels=labels
        )

        # (2) Merge all clusters that are too close to another cluster with their closest cluster

        return labels

if __name__ == "__main__":
    print("Testing kmeans with annealing")

    kwargs = {
        'n_clusters': 20,
        'init': 'random'
    }
    model = MTKMeansAnnealing(kwargs)

    X = np.random.random(100, 20)
    labels = model.fit_predict(X)
    print("Labels are: ", labels)
