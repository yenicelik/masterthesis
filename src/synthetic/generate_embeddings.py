"""
    Different functions to (synthetically) generate the Gaussian embeddings

    This file contains the code to generate a set of basis vectors that are each individually orthogonal
    to each vector within a given matrix A

"""

# Can translate all this into tensorflow later, now should get numpy up and running

import numpy as np
import tensorflow as tf


def generate_synthetic_embedding(d, components, spherical=True, maximum_variance=None, center=None, noise=None):
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

    emb_mu = tf.random.normal((components, d)) * d * 5
    emb_sigma = tf.random.normal((components, d)) * d # Can be unfolded column-wise because diagonal covariance matrices
    emb_sigma = tf.math.abs(emb_sigma) # Making the covariance matrix psd! (cov matrix cannot define a negative eigenvalues)

    if spherical:
        # TODO:, how to make this spherical!
        elementswise_norm = tf.norm(emb_mu, axis=1, ord=2, keepdims=True)
        #print("elementwise norm: ", elementswise_norm)
        # emb_mu = tf.math.divide(emb_mu, elementswise_norm)
        # emb_sigma = tf.math.divide(emb_sigma, maximum_variance)

    return emb_mu, emb_sigma

if __name__ == "__main__":
    print("Generating the embedding")

    emb_src = generate_synthetic_embedding(
        d=5,
        components=10
    )

    # Define the embedding targets

    # TODO: Define the target embedding at a rotation of the source embedding
    emb_tgt = generate_synthetic_embedding(
        d=5,
        components=10
    )

    np.random.seed()
    A = np.random.rand(5, 4)
    # generate_orthogonal_matrix_to_A(A, 1)


    # mus = [tf.expand_dims(emb_src[0][i, :], axis=0) for i in range(src_components)]
    mus_src = [emb_src[0][i, :] for i in range(src_components)]
    covs_src = [emb_src[1][i, :] * tf.eye(dimensions) for i in range(src_components)]

    mus_tgt = [emb_tgt[0][i, :] for i in range(src_components)]
    covs_tgt = [emb_tgt[1][i, :] * tf.eye(dimensions) for i in range(src_components)]




    print(emb_src[0].shape, emb_src[1].shape)
    print(emb_tgt[0].shape, emb_tgt[1].shape)
