"""
    We will find a few words which
      have a high number of samples for the 2 most common meanings.

    (1) Reduce dimensionality to a a number below the number of features.
       (make sure variance kept is not too low ...)

    (2) Check if the two meanings are separable.

    Do this for multiple sense-pairs (100 or so?), so you have statistically significant results ..)

"""
import numpy as np

from imblearn.over_sampling import RandomOverSampler

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split

from src.config import args
from src.embedding_generators.bert_embeddings import BertEmbedding
from src.knowledge_graphs.wordnet import WordNetDataset
from src.resources.corpus_semcor import CorpusSemCor
from src.sampler.sample_embedding_and_sentences import get_bert_embeddings_and_sentences


def sample_semcor_data(tgt_word):
    corpus = CorpusSemCor()
    lang_model = BertEmbedding(corpus=corpus)

    print("Lang model is: ", corpus, lang_model, tgt_word)

    tuples, true_cluster_labels, _ = get_bert_embeddings_and_sentences(model=lang_model, corpus=corpus, tgt_word=tgt_word)

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

    return X, true_cluster_labels

def sample_embeddings_for_target_word(tgt_word):
    print("Looking at word", tgt_word)
    wordnet_model = WordNetDataset()
    number_of_senses = wordnet_model.get_number_of_senses("".join(tgt_word.split()))

    X, true_cluster_labels = sample_semcor_data(tgt_word)

    return number_of_senses, X, true_cluster_labels

def sample_words_with_id(word, wordnet_ids):
    assert isinstance(wordnet_ids, list), (wordnet_ids, list)
    assert all([isinstance(x, int) for x in wordnet_ids]), wordnet_ids
    assert len(wordnet_ids) >= 2, len(wordnet_ids)
    wordnet_ids = [str(x) for x in wordnet_ids]

    number_of_senses, X, true_cluster_labels = sample_embeddings_for_target_word(word)
    # true_cluster_labels = [int(x) if x is not None else -1 for x in true_cluster_labels]

    print("Found following items")
    print(X.shape)
    print(number_of_senses)
    print(true_cluster_labels)

    # Only keep the wordnet_id's that you want
    # Identify arguments that contain wordnet ids
    fitting_args = [idx for idx, x in enumerate(true_cluster_labels) if x in wordnet_ids]
    print("Fitting args", fitting_args)
    print("Fitting args", sum(fitting_args))

    number_of_senses = len(np.unique(wordnet_ids))
    X = X[fitting_args]
    true_cluster_labels = np.asarray(true_cluster_labels)[fitting_args].tolist()

    print("Found following items")
    print(X.shape)
    print(number_of_senses)
    print(true_cluster_labels)

    return number_of_senses, X, true_cluster_labels

if __name__ == "__main__":
    print("Checking if hyperplane separable")

    # ('is', '1'), , ('is', '2')

    word = ' thought '
    word_ids = [1, 2, 3]

    # corpus = CorpusSemCor()
    # corpus.sample_sentence_including_word_from_corpus(word)

    number_of_senses, X, Y = sample_words_with_id(
        word=word,
        wordnet_ids=word_ids
    )

    assert X.shape[0] == len(Y), (X.shape[0], len(Y))

    print("Lets see if works..")
    print(number_of_senses, X.shape, Y)

    # oversample dataset
    ros = RandomOverSampler(random_state=0)
    X, Y = ros.fit_resample(X, Y)

    print("Resampled items are: ", X.shape, len(Y))

    unique = np.unique(Y)
    print("unique and counts are: ")
    print(unique)

    # shuffle data
    # Probably not needed anymore beecause it's random sampling anyways ...

    # Apply PCA
    X = StandardScaler().fit_transform(X)

    for dim in [2, 3, 10, 20, 30, 50, 75, 100]:

        # PCA should in the general rule be done after, but I think in this case it's ok
        pca_model = PCA(n_components=min(dim, X.shape[0]), whiten=False)
        _X = pca_model.fit_transform(X)

        print("\n\n\nDim is: ", dim)
        print("Variance kept through pca is: ", np.sum(pca_model.explained_variance_ratio_))

        # Sample some more sentences using the other corpus to fulfill this ...
        model = LogisticRegression()

        # Accuracy is:
        scores = cross_val_score(model, _X, Y, cv=5)
        print("Scores are (mean std): ", scores.mean(), scores.std())

        # Calculate the confusion matrix for a part of the full data ...
        X_train, X_test, Y_train, Y_test =  train_test_split(_X, Y, test_size=0.5)

        model.fit(X_train, Y_train)
        Y_hat = model.predict(X_test)
        conf = confusion_matrix(Y_test, Y_hat)
        print("confusion matrix is")
        print(conf)

