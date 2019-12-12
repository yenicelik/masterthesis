"""
    Converts a vectorspace into a graph

    Implemented as described at
        `Retrofitting Word Representations for Unsupervised Sense Aware Word Similarities`

    # Adopt similar kind of analysis, as was done in the paper for different words
    (look at individual words and their contexts, not at different words instead)
"""
import numpy as np
import matplotlib.pyplot as plt
from networkx import draw, draw_networkx_edges

from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx
from chinese_whispers import chinese_whispers, aggregate_clusters

class ChineseWhispersClustering:

    def __init__(self, top_nearest_neighbors=50):
        """
        :param top_nearest_neighbors: The number of nearest neighbors to keep
        """
        # TODO: Not sure if we're supposed to prune the fully connected graph, and set all to zeroo
        # TODO: Currently, its a bipartite graph i believe. make this to a adjacency-matrix graph (to be interpreted by networkx)
        self.top_nearest_neighbors_ = top_nearest_neighbors

    def fit(self, X, y=None):
        """
            Assume that X is centered, and normalized
            X : [n_samples, n_features]
            y : ignored
        :param X:
        :return:
        """
        # calculate cosine similarity
        cos = cosine_similarity(X, X)

        #####
        # 1. compute v’s top n nearest neighbors (by some word- similarity notion)
        ######

        # For each sample
        # Take closest items, and put rest to 0!
        # Closest items are (closest to 1!)
        nearest_neighbors = np.argsort(cos, axis=1)[:, -self.top_nearest_neighbors_:]
        out = np.zeros_like(cos)

        print("Total number of nearest neighbors: ", nearest_neighbors.shape)

        ######
        # 2. compute a similarity score between every pairwise combination of nearest neighbors, which renders a fully connected similarity graph
        ######

        # Put whatever index is a top-nearest-neighbor to be this
        out[nearest_neighbors] = cos[nearest_neighbors]

        print(out)
        print(cos)
        print(cos.shape)

        # TODO: Think again how many entries we need to have matching

        # assert np.count_nonzero(out) // 2 == self.top_nearest_neighbors_ * X.shape[0], (
        #     "More nearest neighbors were recorded than desired",
        #     np.count_nonzero(out) // 2, self.top_nearest_neighbors_ * X.shape[0]
        # )

        # Now turn into a graph!
        graph = nx.to_networkx_graph(out, create_using=nx.DiGraph)

        print("Graph is: ")
        print(graph)

        draw(graph, node_size=10)

        plt.show()

        # Apply the chinese whispers algorithm...
        # TODO: Figure out if this works well with the bert embeddings..
        chinese_whispers(graph, weighting='top', seed=1337)
        print('Cluster ID\tCluster Elements\n')

        print("Clustered items are: ")
        out = list(sorted(aggregate_clusters(graph).items(), key=lambda e: len(e[1]), reverse=True))
        self.cluster_ = [x[0] for x in out]

        print("Self cluster is: ", self.cluster_)

        # TODO: Check size of this!!
        self.cluster_ = np.asarray(self.cluster_)

        print("Clusters are: ")
        print(self.cluster_)
        print(self.cluster_.shape)


        for label, cluster in sorted(aggregate_clusters(graph).items(), key=lambda e: len(e[1]), reverse=True):
            print('{}\t{}\n'.format(label, cluster))

        colors = [1. / graph.nodes[node]['label'] for node in graph.nodes()]

        nx.draw_networkx(graph, cmap=plt.get_cmap('jet'), node_color=colors, font_color='white', node_size=10)

        plt.show()

        # Return a list [n_samples, ]
        # which returns which cluster each datapoint belongs to

        return out

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

    a = np.random.random((100, 100))

    # Now apply the chinese whistering algorith..

    model = ChineseWhispersClustering(top_nearest_neighbors=20)

    model.fit(a)

    clusters = model.predict()
    print("Final clusters are: ")
    print(clusters)
    print(clusters.shape)
