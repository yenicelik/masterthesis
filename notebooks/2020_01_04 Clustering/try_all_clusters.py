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

import traceback
import numpy as np
from sklearn.metrics import adjusted_rand_score
from ax import optimize

from src.models.cluster.affinitypropagation import MTAffinityPropagation
from src.models.cluster.chinesewhispers import MTChineseWhispers
from src.models.cluster.dbscan import MTDbScan
from src.models.cluster.hdbscan import MTHdbScan
from src.models.cluster.kmeans_with_annealing import MTKMeansAnnealing
from src.models.cluster.meanshift import MTMeanShift
from src.models.cluster.optics import MTOptics
from src.resources.samplers import sample_embeddings_for_target_word


def _evaluate_model(model_class, arg, crossvalidation_data):
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

    out = 0.

    for tgt_word, tpl in crossvalidation_data.items():
        # Unpack tuple
        # print("Optimizing over target word ...", tgt_word)
        number_of_senses, X, true_clustering, known_indices = tpl

        assert len(known_indices) == len(true_clustering), (
            "Length of true clustering and known indices don't match up!",
            len(known_indices),
            len(true_clustering)
        )
        pred_clustering = model_class(arg).fit_predict(X)
        # Drop all indices that are unknown
        try:
            pred_clustering = pred_clustering[known_indices]
        except Exception as e:
            # Apparently, here is the point of most error.
            # Will just try-catch heree until we find a better solution ...
            print("Logging all variables")
            print("Pred clustering is: ", pred_clustering)
            print("Known indices are: ", known_indices)
            print("Number of senses", number_of_senses, X.shape)
            print("True clustering is: ", true_clustering)
            print("Arguments are: ", arg)
            # Finally fail again for the stacktrace to appear in the outer loop
            pred_clustering = pred_clustering[known_indices]

        if len(np.unique(pred_clustering)) == 1:
            print("Couldn't find cluster!", np.unique(pred_clustering))

        # print("Current score is: ", adjusted_rand_score(true_clustering, pred_clustering))

        assert len(true_clustering) == len(pred_clustering), (len(true_clustering), len(pred_clustering))

        score = adjusted_rand_score(true_clustering, pred_clustering)

        # print("Input to adjusted random score is: ")
        # print("Content is 1: ", true_clustering)
        # print("Content is 2: ", pred_clustering)
        print("Score is: ", score)

        out += score

    # Return the score as the mean of all items
    return float(out) / len(crossvalidation_data)

def sample_all_clusterable_items(prepare_testset=False):
    """
        Prepares a dictionary of clusterable word-embeddings,
        for each of the polysemous words that we will be using for cross-validation ...
    :return:
    """
    # devset_polysemous_words = [
    #     ' use ', ' test ', ' limit ',
    #     ' concern ', ' central ', ' pizza '
    # ]

    devset_polysemous_words = [
        ' was ', ' thought ', ' made ',
        ' only ', ' central ', ' pizza '
    ]

    # TODO: Is a separate choice if we actually want to see performance on these...
    # But can definitely do this as a "test set" after we have the optimal parameters

    # polysemous_words = [
    #     " live ", " report ", " use ",
    #     " know ", " write ", " tell ",
    #     " state ", " allow ", " enter ",
    #     " learn ", " seek ", " final ",
    #     " critic ", " topic ", " obvious ",
    #     " kitchen "
    # ]

    devset = dict()

    # Create the devset
    for tgt_word in devset_polysemous_words:
        number_of_senses, X, true_cluster_labels, known_indices, _ = sample_embeddings_for_target_word(tgt_word)
        devset[tgt_word] = (number_of_senses, X, true_cluster_labels, known_indices)

    assert len(devset) == len(devset_polysemous_words), (len(devset), len(devset_polysemous_words))
    if not prepare_testset:
        return devset, dict()

    # Implement fetching of the testset as well
    raise NotImplementedError


if __name__ == "__main__":
    print("Starting hyper-parameter search of the model")

    # For each individual word, apply this clustering ...

    # TODO: Implement this for more than one word.
    # We want to find the best clustering algorithm applicable on a multitude of target words

    model_classes = [
        ("MTOptics", MTOptics),
        ("MTMeanShift", MTMeanShift),
        ("MTHdbScan", MTHdbScan),
        ("MTDbScan", MTDbScan),
        ("MTAffinityPropagation", MTAffinityPropagation),
        ("MTChineseWhispers", MTChineseWhispers),
        ("MTKMeansAnnealing", MTKMeansAnnealing)
    ]

    devset, _ = sample_all_clusterable_items(prepare_testset=False)

    for model_name, model_class in model_classes:
        print(f"Running {model_name} {model_class}")

        params = model_class.hyperparameter_dictionary()


        def _current_eval_fun(p):
            try:
                return _evaluate_model(
                    model_class=model_class,
                    arg=p,
                    crossvalidation_data=devset
                )
            except Exception as e:
                print("Error occurred!")
                print(e)
                traceback.print_tb(e.__traceback__)
                return 0.

        try:
            best_parameters, best_values, experiment, model = optimize(
                parameters=params,
                evaluation_function=_current_eval_fun,
                minimize=False,
                total_trials=len([x for x in params if x['type'] != "fixed"]) * 10 * 5
            )

            print("Best parameters etc.")
            print(best_parameters, best_values, experiment, model)

        except Exception as e:
            print("AGHHH")
            traceback.print_tb(e.__traceback__)
            print(e)
            print("\n\n\n\n")
