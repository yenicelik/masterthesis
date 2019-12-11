"""
    Check some properties for some polysemous words in BERT embeddings.
    Specifically, make analysis based on correlaiton analysis.

    Figure out perhaps ways to determine if we have a multimodal distribution,
    or if we have a wide-stretched distribution
"""
import time

import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation, MeanShift, DBSCAN
from sklearn.decomposition import PCA

from src.config import args
from src.embedding_generators.bert_embeddings import BertEmbedding
from src.knowledge_graphs.wordnet import WordNetDataset

# TODO: Should probably implement logging instead of this, and just rewrite logging to write to stdout...
def get_bert_embeddings_and_sentences(model, tgt_word):
    """
    :param model: A language model, which implements both the
        `_sample_sentence_including_word_from_corpus` and the
        `get_embedding`
    function
    :return:
    """

    out = []

    if args.verbose >= 1:
        print("Retrieving example sentences from corpus")
    sampled_sentences = model._sample_sentence_including_word_from_corpus(word=tgt_word)

    if args.verbose >= 1:
        print("Retrieving sampled embeddings from BERT")
    sampled_embeddings = model.get_embedding(
        word=tgt_word,
        sample_sentences=sampled_sentences
    )

    if args.verbose >= 2:
        print("\nSampled sentences are: \n")
    for sentence, embedding in zip(sampled_sentences, sampled_embeddings):
        if args.verbose >= 2:
            print(sentence)
        embedding = embedding.flatten()
        if args.verbose >= 2:
            print(embedding.shape)
        out.append(
            (sentence, embedding)
        )

    return out

def save_embedding_to_tsv(tuples, identifier, cluster_labels=None, ):
    """
        Saving the embeddings and sampled sentence into a format that we can easily upload to tensorboard
    :param tuples: is a list of tuples (sentence, embeddings),
        where the embeddings are of size 768 (BERT)
    :return:
    """
    sentences = [x[0] for x in tuples]
    embeddings = [x[1] for x in tuples]

    embeddings = [x.reshape(1, -1) for x in embeddings]

    embeddings_matrix = np.concatenate(embeddings, axis=0)
    print("Embeddings matrix has shape: ", embeddings_matrix.shape)

    if cluster_labels:
        df = pd.DataFrame(data={
            "sentences": sentences,
            "clusters": cluster_labels

        })
    else:
        df = pd.DataFrame(data={
            "sentences": sentences
        })

    print(df.head())

    # TODO: Handle an experiment-creator for this, which reads and writes to a opened directory..

    assert len(df) == embeddings_matrix.shape[0], ("Shapes do not conform somehow!", len(df), embeddings_matrix.shape)

    df.to_csv(identifier + "{}_labels.tsv".format(len(sentences)), header=True, sep="\t")
    np.savetxt(fname=identifier + "{}_values.tsv".format(len(sentences)), X=embeddings_matrix, delimiter="\t")



def cluster_embeddings(tuples, method="affinity_propagation", pca=True):
    """
        Taking the embeddings, we cluster them (using non-parameteric algortihms!)
        using different clustering algorithms.

        We then return whatever we have
    :return:
    """

    assert method in ("affinity_propagation", "mean_shift", "dbscan")

    # The first clustering algorithm will consists of simple
    # TODO: Perhaps best to use the silhouette plot for choosing the optimal numebr of clusters...
    embedding_matrix = np.concatenate([x[1].reshape(1, -1) for x in tuples], axis=0)
    print("Embeddings matrix is: ", embedding_matrix.shape)

    # TODO: Find a good way to evaluate how many clusters one meaning is in
    # To make better sense in high-dimensional space with euclidean distance, perhaps project to a lower-dimension
    # keeping 200 out of 800 dimensions sounds like a good-enough capturing
    if pca:
        pca_model = PCA(n_components=100, whiten=True)
        embedding_matrix = pca_model.fit_transform(embedding_matrix)
        captured_variance = pca_model.explained_variance_ratio_
        print("Explained variance ratio is: ", np.sum(captured_variance))
        print("Embeddings matrix after PCA is: ", embedding_matrix.shape)

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
        cluster_model = AffinityPropagation() # Was manually chosen using the word " set " and wordnet number of synsets as reference...

    elif method == "dbscan":
        print("dbscan")
        cluster_model = DBSCAN()

    else:
        assert False, ("This is not supposed to happen", method)

    labels = cluster_model.fit_predict(embedding_matrix)
    print(np.unique(labels))
    n_clusters_ = len(np.unique(labels))

    print("Took so many seconds: ", time.time() - start_time)

    # Apply some sort of hyperparameter selection to match the wordnet number of definition

    # Replace by the array which assigns clusters to all items
    # Then calculate the silhouette score possibly
    return n_clusters_, labels



if __name__ == "__main__":
    print("Sampling random sentences from the corpus, and their respective BERT embeddings")

    # Make sure that the respective word does not get tokenized into more tokens!
    lang_model = BertEmbedding()
    wordnet_model = WordNetDataset()

    # Check out different types of polysemy?

    # The word to be analysed
    polysemous_words = [" set ", " bank ", " table ", " subject ", " key ", " pupil ", " book ", " mouse "]

    for tgt_word in polysemous_words:
        # tgt_word = " bank " # we add the space before and after for the sake of

        print("Getting number of senses...")
        number_of_senses = wordnet_model.get_number_of_senses("".join(tgt_word.split()))

        print("Getting embeddings from BERT")
        tuples = get_bert_embeddings_and_sentences(model=lang_model, tgt_word=tgt_word)

        print("Clustering embeddings...")
        n_clusters, cluster_labels = cluster_embeddings(tuples)
        print("Number of clusters, wordnet senses, sentences: ", tgt_word, n_clusters, number_of_senses, len(tuples))

        exit(0)


        save_embedding_to_tsv(tuples, cluster_labels=cluster_labels, identifier=tgt_word + "_")
