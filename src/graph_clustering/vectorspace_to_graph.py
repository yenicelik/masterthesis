"""
    Converts a vectorspace into a graph

    Implemented as described at
        `Retrofitting Word Representations for Unsupervised Sense Aware Word Similarities`

    # Adopt similar kind of analysis, as was done in the paper for different words
    (look at individual words and their contexts, not at different words instead)

    # There is no

    -> Thresholds taken from `L 2 F/INESC-ID at SemEval-2019 Task 2: Unsupervised Lexical Semantic Frame Induction using Contextualized Word Representations`
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


class ChineseWhispersClustering:

    def __init__(self, top_nearest_neighbors=40, remove_hub_number=50):
        """
        :param top_nearest_neighbors: The number of nearest neighbors to keep
        """
        # TODO: Not sure if we're supposed to prune the fully connected graph, and set all to zeroo
        # TODO: Currently, its a bipartite graph i believe. make this to a adjacency-matrix graph (to be interpreted by networkx)
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
        # calculate cosine similarity
        cos = cosine_similarity(X, X)

        #####
        # 1. compute v’s top n nearest neighbors (by some word- similarity notion)
        ######

        # For each sample
        # Take closest items, and put rest to 0!
        # Closest items are (closest to 1!)

        # Remove all edges which have more than 1/4 of the median weight
        # Take the 75th percentile, and put all edges that are below this to 0
        # 0.75 is a hyperparameter..

        # The cutoff is defined by the mean, plus some standard deviation divided by two
        # This was most effective for verb-clustering
        # cutoff_value = ( np.mean(cos) + np.std(cos) ) / 2.

        # This was most effective for argument clustering
        # Because we don't calculate by distance, but rather by similar, so we make "plus"
        cutoff_value = np.mean(cos) - 1.5 * np.std(cos)

        # percentile = 0.75
        # cutoff_value = np.sort(cos.flatten())[int(len(cos.flatten()) * percentile)]

        print("Cutoff value is: ")
        print(np.median(cos), cutoff_value)

        # Put all non-medians to zero
        cos[cos < cutoff_value] = 0.


        # if self.top_nearest_neighbors_:
        #     nearest_neighbors = nearest_neighbors[:, :-self.top_nearest_neighbors_]

        # Remove all hubs!
        # This means removing all nodes whose edge-weights are too high!
        summed_weights = np.sum(cos, axis=1)
        print("Summed weights are", summed_weights)
        self.hubs_ = np.argsort(summed_weights)[-self.remove_hub_number_:]
        print("Summed weights are", summed_weights[self.hubs_])

        # include_index = set(sorted_summed_weights)
        # mask = np.array([])

        # Cannot just take them out, no..? These needs clusters as well!
        # Completely remove these from the hubs from the matrix

        self.hubs_ = set(self.hubs_)
        self.hub_mask_ = [x for x in np.arange(cos.shape[0]) if x not in self.hubs_]

        # hub_mask = np.empty_like(cos, dtype=np.bool)
        # hub_mask[:] = True
        # hub_mask[self.hubs_, :] = False
        # hub_mask[:, self.hubs_] = False
        # print("Hub mask is: ")

        # print("Self. hubs are: ")
        # print(hub_mask)
        # print(hub_mask.shape)
        # print(cos.shape)

        cos = cos[self.hub_mask_, :]
        cos = cos[:, self.hub_mask_]

        print(cos)
        print(cos.shape)

        # Randomly sample a subgraph 100 times
        # Put weight of this, and take the cliques of this graph..

        # Must mark these items as "hubs", and remove these from classification

        print("Cos shape is: ", cos.shape)

        out = np.zeros_like(cos)


        ######
        # 2. compute a similarity score between every pairwise combination of nearest neighbors, which renders a fully connected similarity graph
        ######

        # Not sure if I shold remove all other ones..

        # Put all diagonal elements to zero ...
        cos[np.nonzero(np.identity(cos.shape[0]))] = 0.

        nearest_neighbors = np.argsort(cos, axis=1)
        print("Total number of nearest neighbors: ", nearest_neighbors.shape)
        # Put whatever index is a top-nearest-neighbor to be this
        out[nearest_neighbors] = cos[nearest_neighbors]

        out = cos

        print(out)
        print(cos)
        print(cos.shape)
        print(np.count_nonzero(out))

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

        # Creating the EGO network...

        # node_and_degree = graph.degree()
        # (largest_hub, degree) = sorted(node_and_degree.items(), key=itemgetter(1))[-1]
        # # Create ego graph of main hub
        # hub_ego = nx.ego_graph(graph, largest_hub)
        # # Draw graph
        # pos = nx.spring_layout(hub_ego)
        # nx.draw(hub_ego, pos, node_color='b', node_size=50, with_labels=False)
        # # Draw ego as large and red
        # nx.draw_networkx_nodes(hub_ego, pos, nodelist=[largest_hub], node_size=300, node_color='r')
        # # plt.savefig('ego_graph.png')
        # plt.show()
        #
        # exit(0)

        plt.show()

        # Perhaps take only the hubs..? i.e. representative items..
        # Make all the weight "flow" into the hubs, and drop the rest...

        # Apply the chinese whispers algorithm...
        # TODO: Figure out if this works well with the bert embeddings..
        chinese_whispers(graph, iterations=30) # iterations might depend on the number of clusters...
        print('Cluster ID\tCluster Elements\n')

        print("Clustered items are: ")
        out = list(sorted(aggregate_clusters(graph).items(), key=lambda e: len(e[1]), reverse=True))
        self.cluster_ = [x[0] for x in out]

        print("Self cluster is: ", self.cluster_)

        # TODO: This should be the same size as the number of samples

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

    a = np.random.random((100, 50))

    # Generate a different kind of matrix...

    # Now apply the chinese whistering algorith..

    model = ChineseWhispersClustering(top_nearest_neighbors=50, remove_hub_number=50)

    model.fit(a)

    clusters = model.predict()
    print("Final clusters are: ")
    print(clusters)
    print(clusters.shape)
