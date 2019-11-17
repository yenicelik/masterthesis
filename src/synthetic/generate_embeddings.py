"""
    Different functions to (synthetically) generate the Gaussian embeddings
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
    emb_tgt = generate_synthetic_embedding(
        d=5,
        components=10
    )

    print(emb_src[0].shape, emb_src[1].shape)
    print(emb_tgt[0].shape, emb_tgt[1].shape)
