"""
    Common probability density functions for different distributions,
    including
        - Unimodal Gaussian (Normal) Distribution
        - Gaussian Mixture Model
"""
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

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

def pdf_gmm_diagional_covariance(X, mus, covs, mixture_weights=None):
    """
        A discrete function which returns the
        probability density (pdf) at a given point X
    :param X: (samples, dimensions)
    :param mus: (components, dimensoins)
    :param covs: (components, dimensions
    :return:
    """
    assert isinstance(mus, list), ("Mean of distributions must be of type list", type(mus))
    assert isinstance(covs, list), ("Mean of distributions must be of type list", type(mus))

    if mixture_weights is None:
        # If mixture weights are None, then we can generate uniform mixture weights (uniform initialization)
        mixture_weights = tf.ones((len(mus),)) / len(mus)
        # TODO: What is the initializtion with EM? Is is all have same weight, or is it uniform sampling

    assert len(mus) == len(covs), ("Length of means and covs", len(mus), len(covs))
    assert tf.math.reduce_sum(mixture_weights) == 1.0, ("Mixture weights do not sum up to 1!", tf.math.reduce_sum(mixture_weights))

    # Assert matching batch types
    for i in range(len(mus)):
        assert mus[0].shape == mus[i].shape, (mus[0].shape, mus[i].shape)
        assert covs[0].shape == covs[i].shape, (covs[0].shape, covs[i].shape)

    # Generate all individual components first
    all_gaussians = []
    for mu, cov in zip(mus, covs):
        single_gaussian = tfd.MultivariateNormalFullCovariance(
            loc=mu,
            covariance_matrix=cov
        )
        all_gaussians.append(single_gaussian)

    assert len(mus) == len(all_gaussians), (len(mus), len(all_gaussians))
    assert len(all_gaussians) == len(mixture_weights), (len(all_gaussians), len(mixture_weights))

    print(all_gaussians)

    gmm = tfd.Mixture(
      cat=tfd.Categorical(probs=mixture_weights),
      components=all_gaussians
    )

    return gmm.prob(X)

