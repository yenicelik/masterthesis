"""
    Visualizing a 2D (and possibly 3D)
    Gaussian Mixture Model in matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from src.functional.pdf import pdf_gaussian, pdf_gmm_diagional_covariance
from src.synthetic.generate_embeddings import generate_synthetic_embedding

def plot_two_contour3d_input2d(
        pdf1,
        pdf2,
        x0_min=None,
        x0_max=None,
        x1_min=None,
        x1_max=None,
        ringcontour=True
):
    """
        Visualizing Gaussian Mixture Models.
        Somehow assert that the input to the function `pdf` is 2D!
        Does a 3D visualizations
    :param mu: array of mean vectors
    :param cov: array of covariance vectors
    :return:
    """

    assert pdf1 is not None, (pdf1, "PDF1 must be a pdf function!")
    assert pdf2 is not None, (pdf2, "PDF2 must be a pdf function!")

    _x0 = np.linspace(x0_min, x0_max, 50)
    _x1 = np.linspace(x1_min, x1_max, 50)

    _X, _Y = np.meshgrid(_x0, _x1)

    _XX = np.array([_X.ravel(), _Y.ravel()]).T

    # Now evaluate the point at each gaussian mixture model...
    # Should refactor this function into visualizing any probability density!!
    _Z1 = -tf.math.log(pdf1(_XX) + 1.e-3)
    _Z2 = -tf.math.log(pdf2(_XX) + 1.e-3)

    _Z1 = tf.reshape(_Z1, _X.shape)
    _Z2 = tf.reshape(_Z2, _X.shape)

    assert np.count_nonzero(_Z1) > 0, _Z1
    assert np.count_nonzero(_Z2) > 0, _Z2

    fig = plt.figure()
    if ringcontour:
        ax = fig.gca(projection='3d')
    else:
        ax = fig.add_subplot(111, projection='3d')

    _Z1 = _Z1.numpy()
    _Z2 = _Z2.numpy()
    # Let's not plot places that are nan!
    mode = stats.mode(_Z1)
    print("Mode is: ", mode)
    # TODO: Pick it up from here!
    _Z1[_Z1 == mode] = np.nan
    mode = stats.mode(_Z2)
    _Z2[_Z2 == mode] = np.nan

    print("Z is: ")
    print(_Z1)
    print(_Z2)

    # Plot the surface
    if ringcontour:
        ax.contourf(_X, _Y, _Z1,
                    title="PDF1"
                    )
        ax.contourf(_X, _Y, _Z2,
                    title="PDF2"
                    )
    else:
        ax.plot_surface(_X, _Y, _Z1,
                        cmap=cm.coolwarm,
                        linewidth=0,
                        antialiased=False,
                        label="PDF1"
                        )
        ax.plot_surface(_X, _Y, _Z2,
                        # cmap=cm.coolwarm,
                        linewidth=0,
                        antialiased=False,
                        label="PDF2"
                        )

    # ax.legend()
    print("Scatter done")
    plt.show()


def plot_contour3d_input2d(
        pdf,
        x0_min=None,
        x0_max=None,
        x1_min=None,
        x1_max=None,
        ringcontour=True
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
    _Z = -tf.math.log(pdf(_XX) + 1.e-3)

    print("Input Z is: ")
    print(_Z)
    print(_Z.shape)
    _Z = tf.reshape(_Z, _X.shape)

    assert np.count_nonzero(_Z) > 0, _Z

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    if ringcontour:
        ax.contourf(_X, _Y, _Z)
    else:
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

    src_components = 10
    tgt_components = 5

    emb_src = generate_synthetic_embedding(
        d=dimensions,
        components=src_components
    )
    emb_tgt = generate_synthetic_embedding(
        d=dimensions,
        components=tgt_components
    )

    #########################################
    # Visualize the first sampled Gaussian
    #########################################

    # We will ignore the covariance for the
    # calculate of the plotting boundaries!
    quadratic_scale = 20

    # mus = [tf.expand_dims(emb_src[0][i, :], axis=0) for i in range(src_components)]
    mus_src = [emb_src[0][i, :] for i in range(src_components)]
    covs_src = [emb_src[1][i, :] * tf.eye(dimensions) for i in range(src_components)]

    mus_tgt = [emb_tgt[0][i, :] for i in range(tgt_components)]
    covs_tgt = [emb_tgt[1][i, :] * tf.eye(dimensions) for i in range(tgt_components)]

    plot_two_contour3d_input2d(
        pdf1=lambda X: pdf_gmm_diagional_covariance(X, mus_src, covs_src),
        pdf2=lambda X: pdf_gmm_diagional_covariance(X, mus_tgt, covs_tgt),
        x0_min=-1. * quadratic_scale,
        x0_max=1. * quadratic_scale,
        x1_min=-1. * quadratic_scale,
        x1_max=1. * quadratic_scale,
        ringcontour=False
    )
