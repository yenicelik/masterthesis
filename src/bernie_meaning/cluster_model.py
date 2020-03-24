"""
    This cluster-model takes a clustering model in the style of sklearn
        (transform, fit, predict),
    and outputs the label of which meaning-cluster is contained within
"""
import os
import pickle
from collections import Counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.models.cluster.chinesewhispers import MTChineseWhispers
from src.resources.samplers import sample_naive_data

def predict_meaning_cluster(word, embedding, clustermodel_savedir, knn_n=10):
    """

        Given a word, we want to predict the cluster.
        This prediction requires a model to be trained.

        If the model does not exist yet, train it and save it.
        Otherwise load it and use it for prediction.

        The embedding should be given with shape (786, ),
        where 786 is the vanilla embedding dimension

    :param word:
    :return:
    """
    assert word[0] == " ", ("First char must be empty", word)
    assert word[-1] == " ", ("Last char must be empty", word)

    # TODO: Generate folder if not existent
    savedir = os.path.join(clustermodel_savedir)
    if not os.path.exists(savedir):
        print("Creating the path")
        os.mkdir(savedir)

    savepath = os.path.join(savedir, word)
    if os.path.isfile(savepath + "X.pkl"):
        print("Loading....")
        # Loads the pickle file here
        with open(savepath + "_std.pkl", 'rb') as file:
            standardize_model = pickle.load(file)
        with open(savepath + "_pca.pkl", 'rb') as file:
            pca_model = pickle.load(file)
        with open(savepath + "X.pkl", 'rb') as file:
            X_hat = np.load(file)
        with open(savepath + "Y.pkl", 'rb') as file:
            y_hat = np.load(file)

    else:
        # Sample the data to train on ...
        X, _ = sample_naive_data(word, n=500)

        # Train the model on a sampled set from the news corpus
        params = {
            'std_multiplier': 1.3971661365029329,
            'remove_hub_number': 0,
            'min_cluster_size': 31
        }
        cluster_model = MTChineseWhispers(params)
        standardize_model = StandardScaler()
        pca_model = PCA(n_components=min(20, X.shape[0]))

        print("PCA + Cluster model")
        X_hat = standardize_model.fit_transform(X)
        X_hat = pca_model.fit_transform(X_hat)
        y_hat = cluster_model.fit_predict(X_hat)

        print("Models are: ", pca_model, cluster_model)

        # Save PCA and cluster model as a pickle file s.t. can be loaded next time!
        with open(savepath + "_std.pkl", 'wb') as file:
            pickle.dump(standardize_model, file)

        with open(savepath + "_pca.pkl", 'wb') as file:
            pickle.dump(pca_model, file)

        with open(savepath + "X.pkl", 'wb') as file:
            np.save(file, X_hat)

        with open(savepath + "Y.pkl", 'wb') as file:
            np.save(file, y_hat)

        print("Models are: ", pca_model, cluster_model)

    # Transform to PCA
    # also do standardization
    embedding_hat = standardize_model.transform(embedding.reshape(1, -1))
    embedding_hat = pca_model.transform(embedding_hat)

    # Predict using this corpus
    # Do a transform first, especially if it is a vanilla BERT vector that is inputted ...
    # Apply kNN to decide which cluster to assign to.
    # If this is not clear, then take whatever first cluster is there

    # TODO: Do we apply euclidean distance or cosine distance? -> cosine because high-dimensional ...

    # Should also do this as an argument ....
    # lol too many hyperparameters to tune lol
    print("X_hat and emb are: ", X_hat.shape, embedding_hat.shape)
    # Cosine similarity, or sth else?
    cos = 1. - cosine_similarity(X=X_hat, Y=embedding_hat)
    print("Similarity matrix is: ", cos.shape)
    knn_idx = np.argsort(cos.flatten()).tolist()[:knn_n] # Take most similar items
    assert len(set(knn_idx)) == len(knn_idx)
    print("knn indecies are: ", knn_idx)
    print("Y hat is: ", y_hat, np.unique(y_hat))
    y_hat[y_hat == -1] = 1000
    labels = y_hat[knn_idx].tolist()
    print("Labels are: ", labels)
    out = Counter(labels).most_common(1)[0][0]

    # Return an arbitrary number if not enough vocabulary items are found!
    # (i.e. if output is -1)
    print("Out is: ", out)

    return out

if __name__ == "__main__":
    word = " was "
    print("Predicting the meaning for this word")

    embedding = np.random.random((768, ))

    # args.output_meaning_dir
    savedir = "/Users/david/GoogleDrive/_MasterThesis/savedir/cluster_model_caches"
    out = predict_meaning_cluster(word, embedding, clustermodel_savedir=savedir)
    print("Labels for clusters are: ", out)

    # For now, gonna use HdbScan because chinese whispers does not necessarily adapt to new samples
    # Or it must re-calculated each time again...

    print("Should now load the model")
    out = predict_meaning_cluster(word, embedding, clustermodel_savedir=savedir)
    print("Labels for clusters are: ", out)

