"""
    Exhaustively looks through all possible models, and uses the CV-loss to determine if a clustering is available.
    We aggregate a dataset by taking both from the SemCor dataset, and enriching these with BERT embeddings.
    The validation loss then consists of how well the items in same clusters are actually put into the same buckets

    To determine to what extent
     our predicted clustering conforms to the
     real clustering,
    we use the rand_score metric
    (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
    this is really as good as any other metric
"""

from ax import optimize, SearchSpace
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from src.config import args
from src.embedding_generators.bert_embeddings import BertEmbedding
from src.knowledge_graphs.wordnet import WordNetDataset
from src.models.cluster.affinitypropagation import MTAffinityPropagation
from src.models.cluster.dbscan import MTDbScan
from src.models.cluster.hdbscan import MTHdbScan
from src.models.cluster.meanshift import MTMeanShift
from src.models.cluster.optics import MTOptics
from src.resources.corpus import Corpus
from src.resources.corpus_semcor import CorpusSemCor
from src.sampler.sample_embedding_and_sentences import get_bert_embeddings_and_sentences


def _evaluate_model(model_class, arg, X, known_indices, true_clustering):
    """
        Evaluates a model class with the parameters that is was provided with.
        We assume that the optimization-hyperparameters
        exactly match the names of the model-hyperparamters
    :param model_class:
    :param arg:
    :param known_indices:
    :param true_clustering:
    :return:
    """
    assert len(known_indices) == len(true_clustering), (
        "Length of true clustering and known indices don't match up!",
        len(known_indices),
        len(true_clustering)
    )
    pred_clustering = model_class(arg).fit_predict(X)
    # Drop all indices that are unkown
    pred_clustering = pred_clustering[known_indices]

    if len(np.unique(pred_clustering)):
        print("Couldn't find cluster!", np.unique(pred_clustering))

    return adjusted_rand_score(true_clustering, pred_clustering)

def sample_semcor_data(tgt_word):
    corpus = CorpusSemCor()
    lang_model = BertEmbedding(corpus=corpus)

    tuples, true_cluster_labels = get_bert_embeddings_and_sentences(model=lang_model, corpus=corpus, tgt_word=tgt_word)

    # Just concat all to one big matrix
    X = np.concatenate(
        [x[1].reshape(1, -1) for x in tuples],
        axis=0
    )
    return X, true_cluster_labels

def sample_naive_data(tgt_word, n=None):
    corpus = Corpus()
    lang_model = BertEmbedding(corpus=corpus)

    tuples, true_cluster_labels = get_bert_embeddings_and_sentences(model=lang_model, corpus=corpus, tgt_word=tgt_word, n=n)

    # Just concat all to one big matrix
    X = np.concatenate(
        [x[1].reshape(1, -1) for x in tuples],
        axis=0
    )
    return X

if __name__ == "__main__":
    print("Starting hyper-parameter search of the model")

    # For each individual word, apply this clustering ...
    tgt_word = " use "

    # print("ADJ is: ", adjusted_rand_score([1, 2, 3, 4, 5], [0, 1, 2, 3, 4]))

    model_classes = [
        ("MTOptics", MTOptics),
        ("MTMeanShift", MTMeanShift),
        ("MTHdbScan", MTHdbScan),
        ("MTDbScan", MTDbScan),
        ("MTAffinityPropagation", MTAffinityPropagation)
    ]

    # TODO: bootstrap a dataset ...
    print("Looking at word", tgt_word)
    wordnet_model = WordNetDataset()
    number_of_senses = wordnet_model.get_number_of_senses("".join(tgt_word.split()))

    X1, true_cluster_labels = sample_semcor_data(tgt_word)
    n = max(0, (args.max_samples - X1.shape[0]))
    X2 = sample_naive_data(tgt_word, n=n)

    known_indices = list(np.arange(X1.shape[0], dtype=int).tolist())

    X = np.concatenate([X1, X2], axis=0)
    print("Collected data is: ")
    print(X.shape)

    # Apply PCA
    X = StandardScaler().fit_transform(X)
    pca_model = PCA(n_components=min(50, X.shape[0]), whiten=False)
    X = pca_model.fit_transform(X)
    print("Variance kept through pca is: ", np.sum(pca_model.explained_variance_ratio_))

    print("ADJ is: ", adjusted_rand_score([1, 2, 3, 4, 0], [0, 1, 2, 3, 4]))

    # Sample some more sentences using the other corpus to fulfill this ...

    print("Dataset shape is: ", X.shape)
    print("True cluster labels are", len(true_cluster_labels))

    # Apply PCA on X

    for model_name, model_class in model_classes:
        print(f"Running {model_name} {model_class}")

        params = model_class.hyperparameter_dictionary()

        def _current_eval_fun(p):
            return _evaluate_model(
                model_class=model_class,
                arg=p,
                X=X,
                known_indices=known_indices,
                true_clustering=true_cluster_labels
            )

        best_parameters, best_values, experiment, model = optimize(
            parameters=params,
            evaluation_function=_current_eval_fun,
            minimize=False,
            total_trials=len([x for x in params if x['type'] != "fixed"]) # * 10 * 2
        )

        print("Best parameters etc.")
        print(best_parameters, best_values, experiment, model)

