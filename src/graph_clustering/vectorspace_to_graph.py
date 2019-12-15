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
from operator import itemgetter

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

    cos = cosine_similarity(X, X)
    cos[cos < _cutoff_function(cos)] = 0.
    # This line makes a tremendous difference!
    cos[np.nonzero(np.identity(cos.shape[0]))] = 0. # Was previously below the cutoff calculation..

    # if self.top_nearest_neighbors_:
    #     nearest_neighbors = nearest_neighbors[:, :-self.top_nearest_neighbors_]

    return cos

def identify_hubs(cor):
    """
        Within the correlation matrix, we identify certain hubs.
    :param cor: correlation matrix
    :return:
    """
    node_degrees = np.sum(cor > 0, axis=1) # Previously, this was weighted

    def _nodedegree_cutoff_function(matr):
        """
            This was most effective for argument-clustering [1]
        """
        return np.mean(matr) + 2 * np.std(matr)

    # Mark all nodes whose degree is 2 standard deviations outside
    hubs = node_degrees > _nodedegree_cutoff_function(node_degrees)

    print("Hubs shape is: ", hubs)
    print("Hubs shape is: ", hubs.shape)


    # Identify whatever items are hubs.
    return hubs

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

        # Zero out all hubs within the correlation matrix.
        # Overwrite the hubs with their closest rows

        if any(hub_indecies):

            local_correlation = cosine_similarity(X[hubs, :], X)
            # Want to take the most similar items, i.e. biggest cosine similarity, so ::-1
            nearest_neighbors = np.argsort(local_correlation, axis=1)[:, ::-1]
            for idx, hub in enumerate(hub_indecies):
                print("Looking how we can replace hub ", hub)
                for neighbor in nearest_neighbors[idx]:
                    print("Neighbor is: ", neighbor, hub_indecies)
                    if neighbor not in hub_indecies:
                        print("Picking neighbor: ", neighbor)
                        cos[hub] = cos[neighbor]
                        # TODO: Is this correct, actually...?
                        # Shouldn't we re-assign at X, and re-calculate the correlation matrix..?
                        # I think we should really remove the hubs, and put them back in at the very end..
                        break

        # We could also take out these hubs
        # And then add these hubs at the very end.
        # This will require a translation dictionary with
        # (idx of matrix incl. hubs -> idx of matrix ecl. hubs)
        # And then auxiliarly adding hubs back in

        print("Replaced hubs with their closest neighbor...")
        print("Will now apply the clustering algorithm")

        print(cos.shape)

        clusters = run_chinese_whispers(cos)

        print("Clusters are: ", clusters)

        assert X.shape[0] == len(clusters), ("Dont conform!", clusters, X.shape, len(clusters))

        # TODO: Now add the nodes which were previously removed

        exit(0)

        # TODO: Find the easiest mechanism which extracts these the indecies for the hubs
        # Assign the hubs to the closest point
        print("Hubs mask is: ", self.hub_mask_)
        # correlation_hub_rest = cosine_similarity(X[self.hubs_], X)
        cos[:, self.hubs_] = -1 * np.inf # Make all hubs infinitely away from everything else, s.t. these will not be chosen as neighbors
        # Assuming this returns the cosine similarity!!!
        hub_nearest_neighbor = np.argmax(cos, axis=1)
        # hub_nearest_neighbor_without_hubs = np.zeros()
        # for i in range(hub_nearest_neighbor.shape[0]):

        # Calculate the items which are closest to the hubs...
        hub_nearest_neighbor = np.argmax(correlation_hub_rest, axis=1)[self.hub_mask_]
        print("Hub nearest neighbors are: ")
        print(hub_nearest_neighbor)

        print("Calculated the hub nearest neighbors...")
        print(cos)
        print(cos.shape)

        # Randomly sample a subgraph 100 times
        # Put weight of this, and take the cliques of this graph..

        # Must mark these items as "hubs", and remove these from classification

        print("Cos shape is: ", cos.shape)

        nearest_neighbors = np.argmax(correlation_hub_rest, axis=1)
        print("Total number of nearest neighbors: ", nearest_neighbors.shape)
        # Must make nearest neighbor AFTER they are removed!!!!
        # Put whatever index is a top-nearest-neighbor to be this
        print("Nearest neighbors")
        # out[nearest_neighbors] = cos_hat[nearest_neighbors]

        # TODO: Make sure neighborhoods are propery taken out...



        # hub_nearest_neighbor = np.argmax(cos, axis=1)[self.hub_mask_]
        print("Self hubs are")
        print(self.hubs_set)

        for hub in self.hubs_set:
            print("Hub is: ", hub)
            print(cluster_assignments[hub])
            print("Initial hubkey is: ", cluster_assignments[hub] if hub in cluster_assignments else None)
            print("Hub number is: ", hub)
            print("Hubs nearest neighbor is: ")
            print(hub_nearest_neighbor[hub])
            cluster_assignments[hub] = cluster_assignments[hub_nearest_neighbor[hub]]

        # Take out the hubs ...
        # hubs correspond to items with ambigious meanings.. (i.e. between two contexts..!)
        cluster_assignments = list(sorted(cluster_assignments))
        print("Out length is: ", cluster_assignments)
        cluster_assignments = np.asarray(cluster_assignments)

        # Return a list [n_samples, ]
        # which returns which cluster each datapoint belongs to
        # Removing the hubs is like temporarily creating an ego-network

        return cluster_assignments

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
