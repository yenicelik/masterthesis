"""
    Visualizing a 2D (and possibly 3D)
    Gaussian Mixture Model in matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from src.functional.pdf import pdf_gaussian
from src.synthetic.generate_embeddings import generate_synthetic_embedding


def plot_contour3d_input2d(
        pdf,
        x0_min=None,
        x0_max=None,
        x1_min=None,
        x1_max=None
):
    """
        Visualizing Gaussian Mixture Models.
        Somehow assert that the input to the function `pdf` is 2D!
        Does a 3D visualizations
    :param mu: array of mean vectors
    :param cov: array of covariance vectors
    :return:
    """

    assert pdf is not None, (pdf, "PDF must be a pdf function!")

    # This one visualizes a 2-d input!
    # assert len(mu.shape) == 2, ("Tensorf must be 2-dimensional", mu.shape)
    # assert mu.shape[1] == 2, ("Input is not 2D!", mu.shape, 2)

    _x0 = np.linspace(x0_min, x0_max, 50)
    _x1 = np.linspace(x1_min, x1_max, 50)

    _X, _Y = np.meshgrid(_x0, _x1)

    _XX = np.array([_X.ravel(), _Y.ravel()]).T

    # Now evaluate the point at each gaussian mixture model...
    # Should refactor this function into visualizing any probability density!!
    _Z = tf.math.log(pdf(_XX) + 1.e-3)

    print("Input Z is: ")
    print(_Z)
    print(_Z.shape)
    _Z = tf.reshape(_Z, _X.shape)

    assert np.count_nonzero(_Z) > 0, _Z

    plt.contour(_X, _Y, _Z,
                norm=LogNorm(vmin=1.0, vmax=1000.0),
                levels=np.logspace(0, 3, 10)
                )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Make data
    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi, 100)
    # x = 10 * np.outer(np.cos(u), np.sin(v))
    # y = 10 * np.outer(np.sin(u), np.sin(v))
    # z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(_X, _Y, _Z,
                    cmap=cm.coolwarm,
                    linewidth=0,
                    antialiased=False
                    )

    # print("Contour done")
    # plt.scatter(_X[:, 0], _X[:, 1], .8)
    print("Scatter done")
    plt.show()


def plot_contour2d_input2d(
        pdf,
        x0_min=None,
        x0_max=None,
        x1_min=None,
        x1_max=None
):
    """
        Visualizing Gaussian Mixture Models.
        Somehow assert that the input to the function `pdf` is 2D!
    :param mu: array of mean vectors
    :param cov: array of covariance vectors
    :return:
    """

    assert pdf is not None, (pdf, "PDF must be a pdf function!")

    # This one visualizes a 2-d input!
    # assert len(mu.shape) == 2, ("Tensorf must be 2-dimensional", mu.shape)
    # assert mu.shape[1] == 2, ("Input is not 2D!", mu.shape, 2)

    _x0 = np.linspace(x0_min, x0_max, 50)
    _x1 = np.linspace(x1_min, x1_max, 50)

    _X, _Y = np.meshgrid(_x0, _x1)

    _XX = np.array([_X.ravel(), _Y.ravel()]).T

    # Now evaluate the point at each gaussian mixture model...
    # Should refactor this function into visualizing any probability density!!
    _Z = -1. * tf.math.log(pdf(_XX))

    print("Input Z is: ")
    print(_Z)
    print(_Z.shape)
    _Z = tf.reshape(_Z, _X.shape)

    assert np.count_nonzero(_Z) > 0, _Z

    plt.contour(_X, _Y, _Z,
                norm=LogNorm(vmin=1.0, vmax=1000.0),
                levels=np.logspace(0, 3, 10)
                )

    print("Contour done")
    plt.scatter(_X[:, 0], _X[:, 1], .8)
    print("Scatter done")
    plt.show()


if __name__ == "__main__":
    print("Plotting Gaussian Mixutre Models GMMs using matplotlib")

    import tensorflow as tf

    dimensions = 2

    emb_src = generate_synthetic_embedding(
        d=dimensions,
        components=10
    )
    emb_tgt = generate_synthetic_embedding(
        d=dimensions,
        components=10
    )

    #########################################
    # Visualize the first sampled Gaussian
    #########################################

    mu = tf.expand_dims(emb_src[0][0, :], 0)
    cov = emb_src[1][0, :] * tf.eye(dimensions)

    print("Mus and cov are: ", mu.shape, cov.shape)

    # We will ignore the covariance for the
    # calculate of the plotting boundaries!
    quadratic_scale = 20

    plot_contour3d_input2d(
        pdf=lambda X: pdf_gaussian(X, mu, cov),
        x0_min=-1. * quadratic_scale,
        x0_max=1. * quadratic_scale,
        x1_min=-1. * quadratic_scale,
        x1_max=1. * quadratic_scale
    )
