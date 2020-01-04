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

from ax import optimize
from sklearn.metrics import adjusted_rand_score

from src.models.cluster.affinitypropagation import MTAffinityPropagation
from src.models.cluster.dbscan import MTDbScan
from src.models.cluster.hdbscan import MTHdbScan
from src.models.cluster.meanshift import MTMeanShift
from src.models.cluster.optics import MTOptics


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
    pred_clustering = model_class(**arg).fit(X)
    # Drop all indices that are unkown
    pred_clustering = pred_clustering[known_indices]
    return adjusted_rand_score(true_clustering, pred_clustering)

if __name__ == "__main__":
    print("Starting hyper-parameter search of the model")

    params = [
          {
            "name": "x1",
            "type": "range",
            "bounds": [-10.0, 10.0],
          },
          {
            "name": "x2",
            "type": "range",
            "bounds": [-10.0, 10.0],
          },
        ]

    model_classes = [
        ("MTOptics", MTOptics),
        ("MTMeanShift", MTMeanShift),
        ("MTHdbScan", MTHdbScan),
        ("MTDbScan", MTDbScan),
        ("MTAffinityPropagation", MTAffinityPropagation)
    ]

    # TODO: bootstrap a dataset ...
    X = None

    for model_name, model_class in model_classes:
        print(f"Running {model_name}")

        lambda p: _evaluate_model(
            model_class=model_class,
            arg=p,
            X=X,
            known_indices=None,
            true_clustering=None
        )

        best_parameters, best_values, experiment, model = optimize(
            parameters=params,
            evaluation_function=_evaluate_model,
            minimize=True,
            total_trials=len(params) * 10
        )

        print("Best parameters etc.")
        print(best_parameters, best_values, experiment, model)

