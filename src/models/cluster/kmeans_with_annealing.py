"""
    Implements k-means with annealing.
    Clusters that are
        (1) not big enough and
        (2) have centers too close to each other
    will be merged
"""
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from src.models.cluster.base import BaseCluster


class MTKMeansAnnealing(BaseCluster):
    """
        No open parameters
    """

    def merge_clusters_too_close_to_each_other(
            self,
            X_centers,
            cluster_labels
    ):
        """
            Based on some heuristics, and hyperparameters,
            merges two clusters together if they are too close together
        :param X_centers:
        :param cluster_labels:
        :return:
        """
        distances = pairwise_distances(X_centers)

        cluster_merge_value = np.mean(distances) + self.std_multiplier_ * np.std(distances)


        merge_indecies = np.argwhere(distances < cluster_merge_value)
        # print("Merge indecies: ", merge_indecies)
        # merge_indecies = list(set(set(tuple(merge_indecies[i, :])) for i in range(len(merge_indecies))))
        # print("New merge indecies: ", merge_indecies)

        labels = np.asarray(cluster_labels)

        for i in range(merge_indecies.shape[0]):
            replace_by = merge_indecies[i, 0]
            original = merge_indecies[i, 1]

            labels[labels == original] = replace_by

        # Annealing log-loss loss function possible using an elbow techniques ...
        # We need a function for this elbow-technique, where we find the maximal point of inflexion

        # Find all items that are below a given number
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

        min_cluster_size = kwargs['min_cluster_size'] if 'min_cluster_size' in kwargs else 5
        self.std_multiplier_ = kwargs['std_multiplier'] if 'std_multiplier' in kwargs else -0.5
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False

        # for kwargs.items():
        kwargs = {your_key: kwargs[your_key] for your_key in kwargs.keys() if your_key in
                  ['n_clusters', 'init', 'n_init', 'max_iter', 'tol', 'precompute_distances', 'verbose', 'random_state', 'copy_x', 'n_jobs, algorithm']
                  }

        self.model = KMeans(**kwargs)

        self.verbose = verbose
        self.min_cluster_size = min_cluster_size

    @classmethod
    def hyperparameter_dictionary(cls):
        # removed mahalanobis
        return [
            {
                "name": "n_clusters",
                "type": "choice",
                "values": [47, 55, 63, 70, 80, 90],
            },
            {
                "name": "init",
                "type": "choice",
                "values": ['k-means++', 'random'] # , 'random'
            },
            {
                "name": "std_multiplier",
                "type": "range",
                "bounds": [-2.5, 2.5]
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
        labels = self.merge_clusters_too_close_to_each_other(
            X_centers=self.model.cluster_centers_,
            cluster_labels=labels
        )

        return labels

if __name__ == "__main__":
    print("Testing kmeans with annealing")

    kwargs = {
        'n_clusters': 20,
        'init': 'random',
        'std_multiplier': -0.5,
    }
    model = MTKMeansAnnealing(kwargs)

    X = np.random.random((500, 20))
    labels = model.fit_predict(X)
    print("Labels are: ", labels)
