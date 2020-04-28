"""
    Anything related to sampling different kinds of data
"""
from collections import Counter

import numpy as np
import umap
from sklearn import preprocessing
from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler

from src.config import args
from src.embedding_generators.bert_embeddings import BertEmbedding
from src.knowledge_graphs.wordnet import WordNetDataset
from src.resources.corpus import Corpus
from src.resources.corpus_semcor import CorpusSemCor
from src.sampler.sample_embedding_and_sentences import get_bert_embeddings_and_sentences


def load_target_word_embedding_with_oversampling(tgt_word):
    # Use items with more words ...
    X, y, _ = sample_semcor_data(tgt_word=tgt_word)
    # Just do more epochs I guess..?
    print(Counter(y))

    # TODO: Oversample here because highly unbalanced dataset
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X, y = ros.fit_resample(X, y)

    print(Counter(y))
    print(X.shape, y.shape)

    y = y.tolist()
    y = np.asarray([int(x) for x in y])

    return X, y

def sample_naive_data(tgt_word, n=None):
    corpus = Corpus()
    lang_model = BertEmbedding(corpus=corpus)

    tuples, true_cluster_labels, sentences = get_bert_embeddings_and_sentences(model=lang_model, corpus=corpus, tgt_word=tgt_word, n=n)

    # Just concat all to one big matrix
    if args.cuda:
        X = np.concatenate(
            [x[1].cpu().reshape(1, -1) for x in tuples],
            axis=0
        )
    else:
        X = np.concatenate(
            [x[1].reshape(1, -1) for x in tuples],
            axis=0
        )

    assert X.shape[0] == len(sentences), ("Shapes don't conform, ", X.shape, len(sentences))

    return X, sentences


def sample_semcor_data(tgt_word, n=None):
    corpus = CorpusSemCor()
    lang_model = BertEmbedding(corpus=corpus)

    tuples, true_cluster_labels, sentences = get_bert_embeddings_and_sentences(model=lang_model, corpus=corpus, tgt_word=tgt_word, n=n)

    if args.cuda:
        # Just concat all to one big matrix
        X = np.concatenate(
            [x[1].cpu().reshape(1, -1) for x in tuples],
            axis=0
        )

    else:
        X = np.concatenate(
            [x[1].reshape(1, -1) for x in tuples],
            axis=0
        )

    assert X.shape[0] == len(sentences), ("Shapes don't conform, ", X.shape, len(sentences))

    return X, true_cluster_labels, sentences

def sample_word_matrix(tgt_word):
    number_of_senses, X, true_cluster_labels, known_indices, _ = sample_embeddings_for_target_word(tgt_word)
    return number_of_senses, X, true_cluster_labels, known_indices

def sample_embeddings_for_target_word(tgt_word, semcor_only=False):
    print("Looking at word", tgt_word)
    wordnet_model = WordNetDataset()
    number_of_senses = wordnet_model.get_number_of_senses("".join(tgt_word.split()))

    X1, true_cluster_labels, sentences_semcor = sample_semcor_data(tgt_word)
    n = max(2, (args.max_samples - X1.shape[0]))
    if not semcor_only:
        X2, sentences_naive = sample_naive_data(tgt_word, n=n)
        X = np.concatenate([X1, X2], axis=0)
        sentences = sentences_semcor + sentences_naive
        print(X.shape, X1.shape, X2.shape)
    else:
        X = X1
        sentences = sentences_semcor
        print(X.shape, X1.shape)

    known_indices = list(np.arange(X1.shape[0], dtype=int).tolist())

    assert X.shape[0] == len(sentences), ("Shapes don't conform", X.shape[0], len(sentences))
    # print("Collected data is: ")

    # TODO: Figure out whether to do this or as in the other script..

    # Apply PCA
    if args.standardize:
        print("Standardizing!")
        X = StandardScaler().fit_transform(X)
    else:
        print("Not standardizing!")

    print("Args args.dimred is: ", args.dimred, type(args.dimred))

    if args.dimred == "pca":
        print("PCA")
        dimred_model = PCA(n_components=min(args.dimred_dimensions, X.shape[0]), whiten=False)
        X = dimred_model.fit_transform(X)

    elif args.dimred == "nmf":
        print("NMF")
        # Now make the X positive!
        if np.any(X < 0):
            X = X - np.min(X)  # Should we perhaps do this feature-wise?

        # Instead of PCA do NMF?
        dimred_model = NMF(n_components=min(args.dimred_dimensions, X.shape[0]))
        X = dimred_model.fit_transform(X)

    elif args.dimred == "lda":
        print("LDA")
        if np.any(X < 0):
            X = X - np.min(X)  # Should we perhaps do this feature-wise?

        dimred_model = LatentDirichletAllocation(n_components=min(args.dimred_dimensions, X.shape[0]))
        X = dimred_model.fit_transform(X)

    elif args.dimred == "umap":
        print("UMAP")
        dimred_model = umap.UMAP(n_components=min(args.dimred_dimensions, X.shape[0]))
        X = dimred_model.fit_transform(X)

    else:
        # assert False, ("Must specify method of dimensionality reduction")
        print("No dimred applied!")

    # Shall we normalize the vectors..
    if args.normalization_norm in ("l1", "l2"):
        X = preprocessing.normalize(X, norm=args.normalization_norm)

    return number_of_senses, X, true_cluster_labels, known_indices, sentences

def sample_pos_embeddings_for_target_word(tgt_word, n=None):

    if n is None:
        n = args.max_samples

    X, sentences = sample_naive_data(tgt_word, n=n)

    assert X.shape[0] == len(sentences), ("Shapes don't conform", X.shape[0], len(sentences))
    print("Collected data is: ")
    print(X.shape)

    # TODO: Figure out whether to do this or as in the other script..

    # Apply PCA
    if args.standardize:
        X = StandardScaler().fit_transform(X)
    else:
        print("Not standardizing!")

    print("Args args.dimred is: ", args.dimred, type(args.dimred))

    if args.dimred == "pca":
        print("PCA")
        # TODO: Whiten set to true, perhaps revert ...
        dimred_model = PCA(n_components=min(args.dimred_dimensions, X.shape[0]), whiten=args.pca_whiten)

    elif args.dimred == "nmf":
        print("NMF")
        # Now make the X positive!
        if np.any(X < 0):
            X = X - np.min(X)  # Should we perhaps do this feature-wise?

        # Instead of PCA do NMF?
        dimred_model = NMF(n_components=min(args.dimred_dimensions, X.shape[0]))

    elif args.dimred == "lda":
        print("LDA")
        if np.any(X < 0):
            X = X - np.min(X)  # Should we perhaps do this feature-wise?

        dimred_model = LatentDirichletAllocation(n_components=min(args.dimred_dimensions, X.shape[0]))

    elif args.dimred == "umap":
        print("UMAP")
        dimred_model = umap.UMAP(n_components=min(args.dimred_dimensions, X.shape[0]))

    elif args.dimred == "none":
        print("No dimred")

    else:
        assert False, ("Must specify method of dimensionality reduction")

    if args.dimred != "none":
        X = dimred_model.fit_transform(X)

    # Shall we normalize the vectors..
    if args.normalization_norm in ("l1", "l2"):
        X = preprocessing.normalize(X, norm=args.normalization_norm)

    return X, sentences


def get_pos_for_word(nlp, sentence, word):
    doc = nlp(sentence)

    for token in doc:
        # return first occurrence of the word which has a POS tag
        if word == token.text:
            return token.text, token.pos_ # token.tag_

    assert None, ("Sentence should be required to include the given token: ", sentence, word)


def retrieve_data_pos(nlp, tgt_word, pos_tag=False):

    if pos_tag:
        # In this case, this is supposed to retrieve the POS TAG instead of only the POS
        raise NotImplementedError

    X, sentences = sample_pos_embeddings_for_target_word(tgt_word)

    labels = []

    for sentence in sentences:
        token, pos = get_pos_for_word(nlp, sentence, tgt_word.strip())
        # print("POS is: ", token, pos)
        labels.append(pos)

    assert len(labels) == len(sentences), ("Dimensions don't conform!", len(labels), len(sentences))
    assert len(labels) == X.shape[0], ("Dimensions don't conform!", len(labels), X.shape)

    return X, sentences, labels


if __name__ == "__main__":
    print("Testing retrieval of sentences")
    sample_embeddings_for_target_word(" was ")
