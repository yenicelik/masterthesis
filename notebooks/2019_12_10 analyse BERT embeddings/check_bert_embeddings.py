"""
    Check some properties for some polysemous words in BERT embeddings.
    Specifically, make analysis based on correlaiton analysis.

    Figure out perhaps ways to determine if we have a multimodal distribution,
    or if we have a wide-stretched distribution

    -> newspaper dataset is biased w.r.t. financial news (bank..)

    -> Could perhaps do the orthogonal matchin pursuit to sample different meanings indeed (or find different such metrics..)
"""
import os
import random
import string
import time

import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation, MeanShift, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from src.config import args
from src.embedding_generators.bert_embeddings import BertEmbedding
from src.graph_clustering.vectorspace_to_graph import ChineseWhispersClustering
from src.knowledge_graphs.wordnet import WordNetDataset

# Try following:
# Consider only top 200 hubs
# Do chinese whispers amongst hubs
# go back, find whatever the hubs are closest to
# and consider these items

# what about affinity clustering? Is it not the same principle in the end..?
# Should we test out affinity cluster with a much higher sample-size?

# Try out graph partitioning methods

# Instead of graph-based clustering, do we apply probabilistic-based-clustering perhaps?

# LDA extract the number of topics

# Use random walk?

# TODO: Should probably implement logging instead of this, and just rewrite logging to write to stdout...
from src.models.cluster.chinesewhispers import MTChineseWhispers
from src.resources.corpus_semcor import CorpusSemCor
from src.sampler.sample_embedding_and_sentences import get_bert_embeddings_and_sentences


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def save_embedding_to_tsv(tuples, identifier, true_cluster_labels, predicted_cluster_labels=None):
    """
        Saving the embeddings and sampled sentence into a format that we can easily upload to tensorboard
    :param tuples: is a list of tuples (sentence, embeddings),
        where the embeddings are of size 768 (BERT)
    :return:
    """
    sentences = [x[0] for x in tuples]
    embeddings = [x[1] for x in tuples]
    true_cluster_labels = [x[2] for x in tuples]

    embeddings = [x.reshape(1, -1) for x in embeddings]

    embeddings_matrix = np.concatenate(embeddings, axis=0)
    print("Embeddings matrix has shape: ", embeddings_matrix.shape)

    if predicted_cluster_labels is not None:
        print("Length of true labels and clusters are: ")
        print(true_cluster_labels)
        print(predicted_cluster_labels)
        assert len(predicted_cluster_labels) == len(true_cluster_labels), (len(predicted_cluster_labels), len(true_cluster_labels))
        df = pd.DataFrame(data={
            "sentences": sentences,
            "clusters": predicted_cluster_labels,
            "true_labels": true_cluster_labels
        })
    else:
        df = pd.DataFrame(data={
            "sentences": sentences,
            "true_labels": true_cluster_labels
        })

    print(df.head())

    # TODO: Handle an experiment-creator for this, which reads and writes to a opened directory..

    assert len(df) == embeddings_matrix.shape[0], ("Shapes do not conform somehow!", len(df), embeddings_matrix.shape)

    df.to_csv(identifier + "{}_labels.tsv".format(len(sentences)), header=True, sep="\t")
    np.savetxt(fname=identifier + "{}_values.tsv".format(len(sentences)), X=embeddings_matrix, delimiter="\t")


def cluster_embeddings(tuples, method="chinese_whispers", pca=True):
    """
        Taking the embeddings, we cluster them (using non-parameteric algortihms!)
        using different clustering algorithms.

        We then return whatever we have
    :return:
    """

    assert method in (
        "affinity_propagation",
        "mean_shift",
        "dbscan",
        "chinese_whispers"
    )

    # The first clustering algorithm will consists of simple
    # TODO: Perhaps best to use the silhouette plot for choosing the optimal numebr of clusters...
    embedding_matrix = np.concatenate([x[1].reshape(1, -1) for x in tuples], axis=0)
    print("Embeddings matrix is: ", embedding_matrix.shape)

    # TODO: Find a good way to evaluate how many clusters one meaning is in
    # To make better sense in high-dimensional space with euclidean distance, perhaps project to a lower-dimension
    # keeping 200 out of 800 dimensions sounds like a good-enough capturing
    if pca:
        # Apply UMAP dimensionality reduction or so?
        # Do I need to normalize vectors first..?
        # Using a non-parametric clustering and manifold projection!
        # embedding_matrix = umap.UMAP(n_components=100).fit_transform(embedding_matrix)

        # Center the data
        embedding_matrix = embedding_matrix - np.mean(embedding_matrix, axis=0).reshape(1, -1)
        # Whiten data
        # TODO: Uncomment following and try again!
        # embedding_matrix = embedding_matrix / np.std(embedding_matrix, axis=0).reshape(1, -1)

        pca_model = PCA(n_components=min(20, embedding_matrix.shape[0]), whiten=True)

        embedding_matrix = pca_model.fit_transform(embedding_matrix)
        captured_variance = pca_model.explained_variance_ratio_
        print("Explained variance ratio is: ", np.sum(captured_variance))
        print("Embeddings matrix after PCA is: ", embedding_matrix.shape)
        # Normalize the vectors afterwarsd, s.t. we can more easily apply clustering...
        embedding_matrix = normalize(embedding_matrix)

    # Project to another manifold, such as UMAP?

    # -> perhaps it makes more sense to go directly to implement the chinese whispers algorithm...

    # Use one of the following, which output number of clusters

    # -> Many use the chinese whispers algorthm, but should be no difference..
    # AffinityPropagation, MeanShift, DBSCAN

    # TODO: Install sklearn
    # set the preference to pretty low
    # Is this similar enough to the chinese whispers algorithm

    # Cluster by the different principal components... that is also a possibility...
    # Pick principal components until more than 50% of the variance is explained.
    # Pick those principal components as centers, for all the clusters....
    #

    start_time = time.time()
    print(f"Starting to cluster using {method}")
    if method == "mean_shift":
        print("mean_shift")
        cluster_model = MeanShift()

    elif method == "affinity_propagation":
        print("affinity_propagation")
        # max_iter=500
        # preference=-100
        # Create correlation for non-euclidean, pre-computed, cosine logic

        # Implement the chinese whisper algorithms..

        # TODO: implement visualization with optimal parameters

        # embedding_matrix = np.dot(embedding_matrix, embedding_matrix.T)
        cluster_model = AffinityPropagation(preference=-3, max_iter=2000)  # Was manually chosen using the word " set " and wordnet number of synsets as reference...

    elif method == "dbscan":
        print("dbscan")
        cluster_model = DBSCAN(metric='cosine')

    elif method == "chinese_whispers":
        print("chinese whispers")
        arguments = {'std_multiplier': -3.0, 'remove_hub_number': 55, 'min_cluster_size': 1} # ({'objective': 0.40074227773260607}
        # arguments = {'std_multiplier': 1.3971661365029329, 'remove_hub_number': 0, 'min_cluster_size': 31} # ( {'objective': 0.4569029268755458}
        # This is the case for 500 items!!!
        cluster_model = MTChineseWhispers(arguments) # ChineseWhispersClustering(**arguments)

    else:
        assert False, ("This is not supposed to happen", method)

    predicted_labels = cluster_model.fit_predict(embedding_matrix)

    print(np.unique(predicted_labels))
    n_clusters_ = len(np.unique(predicted_labels))

    print("Took so many seconds: ", time.time() - start_time)

    # Cluster and then merge
    # This would be similar to affinity propagation, no?

    # Apply some sort of hyperparameter selection to match the wordnet number of definition

    # Replace by the array which assigns clusters to all items
    # Then calculate the silhouette score possibly
    return n_clusters_, predicted_labels


if __name__ == "__main__":
    print("Sampling random sentences from the corpus, and their respective BERT embeddings")

    # Make sure that the respective word does not get tokenized into more tokens!
    # corpus = Corpus()
    corpus = CorpusSemCor()
    lang_model = BertEmbedding(corpus=corpus)
    wordnet_model = WordNetDataset()

    # Check out different types of polysemy?

    # The word to be analysed
    # polysemous_words = [" set ", " bank ", " table ", " subject ", " key ", " book ", " mouse ", " pupil "]
    # polysemous_words = [" have ", " test ", " limit ", " concern ", " central ", " pizza "]
    # polysemous_words = [" live ", " report ", " use ", " know ", " write ", " tell ", " state ", " allow ", " enter ", " learn ",
    #                     " seek ", " final ", " critic ", " topic ", " obvious ", " kitchen "]
    polysemous_words = [
        # ' made ', # ' was ', # ' thought ',
        ' only ', ' central ', ' pizza '
    ]

    method = "chinese_whispers"

    rnd_string = randomString()
    if not os.path.exists(rnd_string):
        os.makedirs(rnd_string)

    for tgt_word in polysemous_words:
        print("Looking at word ", tgt_word)
        # tgt_word = " bank " # we add the space before and after for the sake of

        number_of_senses = wordnet_model.get_number_of_senses("".join(tgt_word.split()))

        print("Getting embeddings from BERT")
        tuples, true_cluster_labels, _ = get_bert_embeddings_and_sentences(model=lang_model, corpus=corpus, tgt_word=tgt_word)

        print("Clustering embeddings...")
        # cluster_labels = None
        n_clusters, cluster_labels = cluster_embeddings(tuples, method=method)
        print("Number of clusters, wordnet senses, sentences: ", tgt_word, n_clusters, number_of_senses, len(tuples))

        save_embedding_to_tsv(tuples, true_cluster_labels=true_cluster_labels, predicted_cluster_labels=cluster_labels, identifier=rnd_string + "/" + tgt_word + "_" + method + "_")

        # Now use this clustering to sample contexts
        # Ignore clusters which have less than 1% samples...
        #


    # TODO: Figure out how well one clustering matches another
    # (i.e. find a sklearn function or so!)

