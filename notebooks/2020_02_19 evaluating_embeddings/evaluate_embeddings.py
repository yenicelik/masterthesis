"""
    Evaluate the embeddings
"""
import spacy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import args
from src.embedding_generators.bert_embeddings import BertEmbedding
from src.functional.pdf import pdf_gaussian, pdf_gmm_diagional_covariance
from src.models.cluster.chinesewhispers import MTChineseWhispers
from src.resources.corpus import Corpus
from src.resources.samplers import retrieve_data
from src.resources.similarity.get_corpus_words import get_words_in_benchmarks
from src.utils.create_experiments_folder import randomString

def static_principal_component(matr):
    # Calculate a static principle component from the vector
    pca_model = PCA()
    pca_model.fit(matr)
    principal_component = pca_model.components_[0, :]
    print(principal_component.shape)
    print("Explained variance is: ", pca_model.explained_variance_ratio_[0])
    principal_variance = pca_model.explained_variance_
    return principal_component, principal_variance

def preprocess_matrix(matr):

    X_hat = StandardScaler().fit_transform(matr)
    X_hat = PCA(n_components=20).fit_transform(X_hat)

    return X_hat

def cluster_then_get_static_principal_component(X):
    # Calculate a static principle component from the vector

    # Do all the clustering steps
    arguments = {
        'std_multiplier': 1.3971661365029329,
        'remove_hub_number': 0,
        'min_cluster_size': 31
    }
    cluster_model = MTChineseWhispers(arguments)  # ChineseWhispersClustering(**arguments)

    predicted_labels = cluster_model.fit_predict(X)
    return predicted_labels

def sample_principal_components_from_cluster_centers(X, cluster_labels, n=10):
    """
        Sample evenly from cluster labels
    :param X:
    :param cluster_labels:
    :return:
    """
    representative_set = []
    for label in np.unique(cluster_labels):
        idx = np.argwhere(cluster_labels == label)
        # Create a principal component from this ...
        pvec, pvar = static_principal_component(X[idx, :])
        representative_set.append(
            (pvec, pvar)
        )

    # Now do something with this representative set
    return representative_set

def return_gaussian_mixture_from_principal_components(cluster_center_variance_pairs):
    """
        From these principal components, perhaps now create.
        There should be duplicate code from the normalizing flow items ...
    :param X:
    :param cluster_labels:
    :return:
    """

    mus = [x[0] for x in cluster_center_variance_pairs]
    covs = [x[1] for x in cluster_center_variance_pairs]

    # Assume that we don't necessarily have a matrix, but usually just a scalar ...
    pdf_gmm = pdf_gmm_diagional_covariance(mus, covs)

    # Sample a few times to check here if this actually worked perhaps
    return pdf_gmm

def return_gaussian_process_from_datasamples(matr):
    """
        From these principal components, perhaps now create.
        There should be duplicate code from the normalizing flow items ...
    :param X:
    :param cluster_labels:
    :return:
    """

    pdf_gmm_diagional_covariance

    # Now do something with this representative set

def predict_rank():
    pass

if __name__ == "__main__":

    print("Creating our own contextualized embeddings, and then evaluating these using the gold-standard datasets")

    rnd_str = randomString(additonal_label=f"_wordembeddings_{args.dimred}_{args.dimred_dimensions}_whiten{args.pca_whiten}_norm{args.normalization_norm}")
    loadpath = "/Users/david/GoogleDrive/_MasterThesis/notebooks/2020_02_19 evaluating_embeddings/_wordembeddings_none_768_whitenFalse_normcedtqifitj/"

    nlp = spacy.load("en_core_web_sm")

    for word in ['state']: #get_words_in_benchmarks("simlex"):

        # For each item in the ordering-dictionary, get the items
        print(f"Sampling word {word}")
        matr = np.loadtxt(loadpath + f' {word} _matr.tsv', delimiter="\t")

        static_principal_component(matr)

        # Question is do we want to preprocess these
        X_hat = X_hat # preprocess_matrix(matr)

        print(X_hat.shape)

        clusters_hat = cluster_then_get_static_principal_component(X_hat)
        centers_and_variances = sample_principal_components_from_cluster_centers(X_hat)

        gmm_pdf = return_gaussian_mixture_from_principal_components(centers_and_variances)

        gp_pdf = return_gaussian_process_from_datasamples(centers_and_variances)
        # Get representative vector from this ..


        # Grab vectors, and do whatever you wanna do to get the vectors ...

        # For now, let us evaluate the GloVe embeddings

        # First of all do some clustering ...

        # Do some clustering, and select out some samples ... (perhaps do some preprocessing first)

        # for tgt_word in polypos_words:
        #     X, sentences, labels = retrieve_data(nlp, tgt_word=tgt_word)
        #
        #     # Save as a numpy array
        #     np.savetxt(rnd_str + f"/{tgt_word}_matr.tsv", X, delimiter="\t")
        #     pd.DataFrame(
        #         {
        #             "sentece": sentences,
        #             "labels": labels
        #         }
        #     ).to_csv(rnd_str + f"/{tgt_word}_labels.tsv", sep="\t")

        # First of all, get the words in the benchmark
