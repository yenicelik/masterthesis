"""
    Calculates the clustering based on
"""
import os
import random

import numpy as np
from scipy.signal import normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

from src.embedding_generators.bert_embeddings import BertEmbedding
from src.knowledge_graphs.wordnet import WordNetDataset
from src.models.cluster.chinesewhispers import MTChineseWhispers
from src.resources.corpus import Corpus
from src.resources.corpus_semcor import CorpusSemCor
from src.sampler.sample_embedding_and_sentences import get_bert_embeddings_and_sentences
from src.utils.create_experiments_folder import randomString


# All the sample data logic now ....


def predict_clustering(embedding_matrix):

    print("X shape")
    print(embedding_matrix.shape)

    # Normalize matrix
    X = StandardScaler().fit_transform(embedding_matrix)
    pca_model = PCA(n_components=min(20, X.shape[0]), whiten=False)
    X = pca_model.fit_transform(X)

    # These are the best parameters we had determined
    # arguments = {
    #     'std_multiplier': -3.0,
    #     'remove_hub_number': 55,
    #     'min_cluster_size': 1
    # }  # ({'objective': 0.40074227773260607}
    arguments = {
        'std_multiplier': 1.3971661365029329,
        'remove_hub_number': 0,
        'min_cluster_size': 31
    }  # ( {'objective': 0.4569029268755458}

    # {
    #     'std_multiplier': 2.0614460712833473,
    #     'remove_hub_number': 0,
    #     'min_cluster_size': 42
    # } # ({'objective': 0.4336888890925356}, {'objective': {'objective': 1.0887275582334232e-09}}

    cluster_model = MTChineseWhispers(arguments)  # ChineseWhispersClustering(**arguments)

    predicted_labels = cluster_model.fit_predict(X)
    return predicted_labels

def print_thesaurus(sentences, clusters, word, savepath=None, n=5):
    """
        Prints possible different use-cases of a word by taking
        :param : a set of tuples (sentence, cluster_label)
        :param n : number of examples to show per meaning clustered ...
    :return:
    """

    # for cluster, sentence in zip(clusters, sentences):
    data = []
    for cluster, sentence in zip(clusters, sentences):
        print(cluster, sentence)
        data.append((cluster, sentence))

    # Shuffle, and keep five (as determined by counter)
    random.shuffle(data)
    counter = dict()

    out = []
    for cluster, sentence in data:
        if cluster not in counter:
            counter[cluster] = 0
            out.append(
                (cluster, sentence)
            )
        elif counter[cluster] < 5:
            counter[cluster] += 1
            out.append(
                (cluster, sentence)
            )
        else:
            continue

    print("out", out)

    df = pd.DataFrame.from_records(out, columns =['cluster_id', 'sentence'])
    df.to_csv(savepath + f"/thesaurus_{word}.csv")

    df = pd.DataFrame.from_records(data, columns =['cluster_id', 'sentence'])
    df.to_csv(savepath + f"/thesaurus_{word}_full.csv")

    print("Sampled meanings through thesaurus are: ")
    print(df.head())


if __name__ == "__main__":
    print("Comparing our clusters with other clusters ...")
    print("This time we take into account the true clusters (take it from the other files ..")

    polysemous_words = [
        # ' thought ', ' made ',  # ' was ',
        # ' only ', ' central ', ' pizza '
        ' table ',
        ' bank ',
        ' cold ',
        ' table ',
        ' good ',
        ' mouse ',
        ' was ',
        ' key ',
        ' arms ',
        ' was ',
        ' thought ',
        ' pizza ',
        ' made ',
        ' book '
    ]

    corpus = Corpus()
    corpus_semcor = CorpusSemCor()
    # ALso take the second corpus to check if th
    lang_model = BertEmbedding(corpus=corpus_semcor)
    wordnet_model = WordNetDataset()

    savepath = randomString()

    for tgt_word in polysemous_words:
        print("Looking at word", tgt_word)

        number_of_senses = wordnet_model.get_number_of_senses("".join(tgt_word.split()))

        print("Getting embeddings from BERT")
        tuples_semcor, true_cluster_labels_semcor = get_bert_embeddings_and_sentences(model=lang_model, corpus=corpus_semcor, tgt_word=tgt_word)
        tuples, _ = get_bert_embeddings_and_sentences(model=lang_model, corpus=corpus, tgt_word=tgt_word)

        print("semcor tuples and normal tuples are")

        print(tuples_semcor)
        print(len(tuples_semcor))

        print(tuples)
        print(len(tuples))

        # Predict the clustering for the combined corpus ...
        X = np.concatenate(
            [x[1].reshape(1, -1) for x in (tuples_semcor + tuples)], axis=0
        )
        sentences = [
            x[0] for x in (tuples_semcor + tuples)
        ]

        # Labels also should be a python list
        labels = predict_clustering(
            X
        ).tolist()

        print("Printing items ...")

        # print(
        #     tuples_semcor,
        #     true_cluster_labels_semcor,
        #     tuples
        # )

        print(len(X), len(labels), len(sentences))

        assert len(X) == len(labels), (
            len(X), len(labels)
        )
        assert len(sentences) == len(labels), (
            len(sentences), len(labels)
        )

        print_thesaurus(
            sentences=sentences,
            clusters=labels,
            word=tgt_word,
            savepath=savepath
        )

    #     print("Clustering embeddings...")
    #     # cluster_labels = None
    #     n_clusters, cluster_labels = cluster_embeddings(tuples, method=method)
    #     print("Number of clusters, wordnet senses, sentences: ", tgt_word, n_clusters, number_of_senses, len(tuples))
    #
    #     save_embedding_to_tsv(tuples, true_cluster_labels=true_cluster_labels, predicted_cluster_labels=cluster_labels,
    #                           identifier=rnd_string + "/" + tgt_word + "_" + method + "_")
    #
    #     # Now use this clustering to sample contexts
    #     # Ignore clusters which have less than 1% samples...
    #     #
    #
    # # TODO: Figure out how well one clustering matches another
    # # (i.e. find a sklearn function or so!)



