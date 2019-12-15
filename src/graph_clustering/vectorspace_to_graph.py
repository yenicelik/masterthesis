"""
    Converts a vectorspace into a graph

    Implemented as described at
        `Retrofitting Word Representations for Unsupervised Sense Aware Word Similarities`

    # Adopt similar kind of analysis, as was done in the paper for different words
    (look at individual words and their contexts, not at different words instead)

    # There is no

    -> Thresholds taken from `L 2 F/INESC-ID at SemEval-2019 Task 2: Unsupervised Lexical Semantic Frame Induction using Contextualized Word Representations`

    [1] L 2 F/INESC-ID at SemEval-2019 Task 2: Unsupervised Lexical Semantic Frame Induction using Contextualized Word Representations
"""
import numpy as np
import matplotlib.pyplot as plt
from networkx import draw, draw_networkx_edges

from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx
from chinese_whispers import chinese_whispers, aggregate_clusters

# TODO: Remove the items with highest degree (degree which is more than median
# Then run the chinese whispers...
# The authors in the other peaper do this with EGO clustering.

# -> Now we don't have ego networks. that means we would need to find some other kind of ego.
# this could be a hub, for example..., and we could apply this iteratively repeatedly.

def create_adjacency_matrix(X):
    """
        From the full embeddings matrix X, creates a graph adjacency matrix:
        - using cosine similarity
        - taking out all values below a certain threhsold, as defined by the weight-distribution
    :param X:
    :return:
    """

    def _cutoff_function(matr):
        """
            This was most effective for argument-clustering [1]
        """
        return np.mean(matr) + 1.5 * np.std(matr)

    cor = cosine_similarity(X, X)
    cor[cor < _cutoff_function(cor)] = 0.
    # This line makes a tremendous difference!
    cor[np.nonzero(np.identity(cor.shape[0]))] = 0. # Was previously below the cutoff calculation..

    # if self.top_nearest_neighbors_:
    #     nearest_neighbors = nearest_neighbors[:, :-self.top_nearest_neighbors_]

    return cor

def identify_hubs(cor):
    """
        Within the correlation matrix, we identify certain hubs.
    :param cor: correlation matrix
    :return:
    """
    node_degrees = np.sum(cor > 0, axis=1) # Previously, this was weighted

    # TODO: Remove less hubs perhaps..?
    def _nodedegree_cutoff_function(matr):
        """
            This was most effective for argument-clustering [1]
        """
        return np.mean(matr) + 1.5 * np.std(matr)

    # Mark all nodes whose degree is 2 standard deviations outside
    hubs = node_degrees > _nodedegree_cutoff_function(node_degrees)

    # Identify whatever items are hubs.
    return hubs

def _identify_hubs_nearest_neighbors(X, hubs, hub_indecies):
    """
        Identifies the nodes which are closest to the hubs.
        The idea is that
            - hubs are taken out before clustering
            - chinese whispers clustering is applied
            - hubs are assigned the same cluster, which their closest point has.
        This makes the chinese whispers algorithm more stable.
        This is because the hubs otherwise accumulate all clusters,
        which merely resluts in one big cluster
    :return:
    """
    overwrite_hub_by_dictionary = dict()

    local_correlation = cosine_similarity(X[hubs, :], X)
    # Want to take the most similar items, i.e. biggest cosine similarity, so ::-1
    nearest_neighbors = np.argsort(local_correlation, axis=1)[:, ::-1]
    for idx, hub in enumerate(hub_indecies):
        print("Looking how we can replace hub ", hub)
        for neighbor in nearest_neighbors[idx]:
            print("Neighbor is: ", neighbor, hub_indecies)
            if neighbor not in hub_indecies:
                print("Picking neighbor: ", neighbor)
                overwrite_hub_by_dictionary[hub] = neighbor
                break

    return overwrite_hub_by_dictionary

def run_chinese_whispers(cor):
    """
        The core part of this algorithm, which runs the chinese whispers algorithm#
    :param cor: the adjacency matrix
    :return:
    """
    print("So many edges in the graph: ", np.count_nonzero(cor))

    graph = nx.to_networkx_graph(cor, create_using=nx.DiGraph)
    print("Graph is: ", graph)
    draw(graph, node_size=10)
    plt.show()

    # We could run this a few times, until the silhouette score is best

    # Now run the chinese whispers algorithm
    chinese_whispers(graph, iterations=30, seed=1337)  # iterations might depend on the number of clusters...

    # Exctracting the individual clusters..
    cluster_assignments = dict()
    for label, cluster in sorted(aggregate_clusters(graph).items(), key=lambda e: len(e[1]), reverse=True):
        for nodeid in cluster:
            cluster_assignments[nodeid] = label
        print('{}\t{}\n'.format(label, cluster))

    # Print out the graph:
    colors = [1. / graph.nodes[node]['label'] for node in graph.nodes()]
    nx.draw_networkx(graph, node_color=colors, node_size=10)  # font_color='white', # cmap=plt.get_cmap('jet'),
    plt.show()

    return cluster_assignments


class ChineseWhispersClustering:

    def __init__(self, top_nearest_neighbors=40, remove_hub_number=50):
        """
        :param top_nearest_neighbors: The number of nearest neighbors to keep
        """
        self.top_nearest_neighbors_ = top_nearest_neighbors
        self.remove_hub_number_ = remove_hub_number

    def fit(self, X, y=None):
        """
            Assume that X is centered, and normalized
            X : [n_samples, n_features]
            y : ignored
        :param X:
        :return:
        """

        cos = create_adjacency_matrix(X)

        hubs = identify_hubs(cos)
        hub_indecies = np.nonzero(hubs)[0].tolist()
        common_indecies = np.nonzero(~hubs)[0].tolist()
        prehub2posthub = dict((idx, node) for idx, node in enumerate(common_indecies))

        overwrite_hub_by_dictionary = dict()
        if any(hub_indecies):
            overwrite_hub_by_dictionary = _identify_hubs_nearest_neighbors(
                X=X,
                hubs=hubs,
                hub_indecies=hub_indecies
            )

        # Replace cos with hub-masked cos
        cos_hat = cos[~hubs, :]
        cos_hat = cos_hat[:, ~hubs]

        clusters = run_chinese_whispers(cos_hat)

        print("Clusters are: ", clusters)
        backproject_cluster = np.zeros((X.shape[0], ))
        for idx, node_cluster in enumerate(clusters):
            backproject_cluster[prehub2posthub[idx]] = node_cluster

        # Now add all neighbors
        for hub_node in overwrite_hub_by_dictionary.keys():
            backproject_cluster[hub_node] = overwrite_hub_by_dictionary[hub_node]

        # After all the clustering is done, now we need to re-insert the hubs...
        assert X.shape[0] == len(backproject_cluster), ("Dont conform!", backproject_cluster, X.shape, len(backproject_cluster))

        self.cluster_ = backproject_cluster

        return self.cluster_

    def predict(self, X=None, y=None):
        """
            Inputs are ignored!
        :param X: ignored
        :param y: ignored
        :return:
        """
        return self.cluster_

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X, y)

if __name__ == "__main__":
    print("Emulating the chinese whispers algorithm")

    a = np.random.random((100, 50))

    # Generate a different kind of matrix...

    # Now apply the chinese whistering algorith..

    model = ChineseWhispersClustering(top_nearest_neighbors=50, remove_hub_number=50)

    model.fit(a)

    clusters = model.predict()
    print("Final clusters are: ")
    print(clusters)
    print(clusters.shape)
