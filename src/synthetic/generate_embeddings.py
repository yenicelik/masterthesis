"""
    Different functions to (synthetically) generate the Gaussian embeddings

    This file contains the code to generate a set of basis vectors that are each individually orthogonal
    to each vector within a given matrix A

"""

# Can translate all this into tensorflow later, now should get numpy up and running

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from src.config import args
from src.functional.linalg import covariance_multiplication, mean_multiplication
from src.synthetic.generate_rotation_matrix import generate_rotation_matrix


def generate_synthetic_embedding(d, components, spherical=True, maximum_variance=None, center=None, noise=None, lam=1.e-2):
    """
        Generates some synthetic Gaussian Embeddings. For each embedding, it generates
        - mean of the Gaussian
        - (diagonal) covariance matrix of the Gaussian embeddings

        In the end, this should sample a Gaussian Mixture Model in the end
    :param d: Number of dimensions the embedding vectors should have
    :param components: Number of embedding vectors to generate
    :param center:
    :param spherical: If spherical, all embeddings will be uniformly distributed around the hypersphere
    :param noise:
    :return:
    """
    assert d > 1, ("Dimensionality must be positive and bigger than 1!", d)
    print("Generating embedding of size: ", d)

    if maximum_variance is None:
        maximum_variance = np.sqrt(d)

    if args.random_seed:
        tf.random.set_seed(args.random_seed)

    emb_mu = tf.random.normal((components, d)) * d * 5
    emb_sigma = tf.random.normal(
        (components, d)) * d  # Can be unfolded column-wise because diagonal covariance matrices
    emb_sigma = tf.math.abs(
        emb_sigma)  # Making the covariance matrix psd! (cov matrix cannot define a negative eigenvalues)

    if spherical:
        # TODO:, how to make this spherical!
        elementswise_norm = tf.norm(emb_mu, axis=1, ord=2, keepdims=True)
        # print("elementwise norm: ", elementswise_norm)
        # emb_mu = tf.math.divide(emb_mu, elementswise_norm)
        # emb_sigma = tf.math.divide(emb_sigma, maximum_variance)

    # Create a list from this..

    # Finally, make the covariance matrix numerically stable for psd operations....

    emb_mus = [tf.reshape(emb_mu[i, :], (1, -1)) for i in range(components)]
    emb_sigmas = [emb_sigma[i, :] * tf.eye(d) + (lam * tf.eye(d)) for i in range(components)]

    return emb_mus, emb_sigmas

def generate_synthetic_src_tgt_embedding(d, components, orthogonal_rotation_matrix=True):
    """
        Generates synthetic source and target embeddings, where the target embeddings
        is perturbated by a function f (right now only a Rotation matrix W) is implemented!
    :return:
    """
    tf.compat.v1.enable_eager_execution()

    mus_src, cov_src = generate_synthetic_embedding(d, components)

    M_rotation = generate_rotation_matrix(src_dim=d, tgt_dim=d, orthogonal=orthogonal_rotation_matrix)

    print("mus cov src are: ")
    print(mus_src)
    print(cov_src)

    mus_tgt = [mean_multiplication(mus_src[i], M_rotation) for i in range(components)]
    cov_tgt = [covariance_multiplication(cov_src[i], M_rotation) for i in range(components)]

    tf.compat.v1.disable_eager_execution()

    return mus_src, cov_src, mus_tgt, cov_tgt

if __name__ == "__main__":
    print("Generating the embedding")

    dimensions = 2
    src_components = 10

    mus_src, cov_src, mus_tgt, cov_tgt = generate_synthetic_src_tgt_embedding(d=dimensions, components=src_components)

    print(mus_src)
    print(cov_src)
    print(mus_tgt)
    print(cov_tgt)
