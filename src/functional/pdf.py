"""
    Common probability density functions for different distributions,
    including
        - Unimodal Gaussian (Normal) Distribution
        - Gaussian Mixture Model
"""
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# import tensorflow_probability.python.distributions as tfd

def pdf_gaussian(X, mu, cov):
    """
        A discrete function which returns the
        probability density (pdf) at a given point X.
        Should be able to visualize a full covariance matrix,
        because the rotation will make the diagonal matrix a full matrix
    :param X: (samples, dimensions)
    :param mu:
    :param cov:
    :return:
    """
    print("Inputs mean and cov are: ")

    # TODO: must assert that the covariance matrix is psd

    print(mu)
    print(cov)

    gaussian = tfd.MultivariateNormalFullCovariance(
        loc=mu,
        covariance_matrix=cov
    )

    return gaussian.prob(X)

def pdf_gmm(X, mus, covs):
    """
        A discrete function which returns the
        probability density (pdf) at a given point X
    :param X: (samples, dimensions)
    :param mus:
    :param covs:
    :return:
    """
    # Sum over all components
    pass

