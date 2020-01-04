"""
    Mostly includes common sklearn models where the output includes number of clusters (this is what we want to cross-validate agains)
    Models include:
    - hdbscan (https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html)
    - OPTICS (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html#sklearn.cluster.OPTICS)
    - DBSCAN (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN)
    - MeanShift (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift)
    - AffinityPropagation (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation)
    -

    We will apply hyperparameter optimization using the pytroch framework (BoTorch)[https://botorch.org/].
    We could also use our own (alphabox), but this will add another layer of complexity most likely

    - Outside hyperparameters include "enrich-data"
"""
