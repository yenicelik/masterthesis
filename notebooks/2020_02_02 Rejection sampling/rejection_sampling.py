"""
    Instead of calculating clusters (which is a generative modelling task),
    we deviate to a sampling task (which can be considered a "verification" task).

    We will implement rejection sampling,
    with samples being accepted when the density in a region is high enough,
    and rejected when the density in the region is not high enough.

    This requires very strong dimensionality reduction techniques, s.t. we don't sample from an infinitely high number of possible points
"""

import numpy as np
import matplotlib

from src.resources.samplers import sample_embeddings_for_target_word

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from src.config import args
from src.utils.create_experiments_folder import randomString


def linearly_spaced_combinations(bounds, num_samples):
    """
    Return 2-D array with all linearly spaced combinations with the bounds.
    Parameters
    ----------
    bounds: sequence of tuples
        The bounds for the variables, [(x1_min, x1_max), (x2_min, x2_max), ...]
    num_samples: integer or array_likem
        Number of samples to use for every dimension. Can be a constant if
        the same number should be used for all, or an array to fine-tune
        precision. Total number of data points is num_samples ** len(bounds).
    Returns
    -------
    combinations: 2-d array
        A 2-d arrray. If d = len(bounds) and l = prod(num_samples) then it
        is of size l x d, that is, every row contains one combination of
        inputs.
    """
    num_vars = len(bounds)

    if not isinstance(num_samples, list):
        num_samples = [num_samples] * num_vars

    if len(bounds) == 1:
        return np.linspace(bounds[0][0], bounds[0][1], num_samples[0])[:, None]

    # Create linearly spaced test inputs
    inputs = [np.linspace(b[0], b[1], n) for b, n in zip(bounds, num_samples)]

    # Convert to 2-D array
    return np.array([x.ravel() for x in np.meshgrid(*inputs)]).T

def create_grid(X, sample_per_dim=4):
    """
    :param X:
    :param sample_per_dim: TODO: Could be a hyperparameter!
    :return:
    """

    # Create a uniform grid within X's boundaries.
    boundary_min = np.min(X, axis=0)
    boundary_max = np.max(X, axis=0)

    assert X.shape[1] <= 4, ("X will be likely too big to compute a grid for!", X.shape)
    assert sample_per_dim <= 10, ("This will be too slow!")

    arrays = []

    # Create a n-dimensional grid with the given boundaries
    for i in range(X.shape[1]):
        # Subtract a few percentages
        _min = boundary_min[i]
        _max = boundary_max[i]
        rangelength = 0. # _max - _min
        # Do the bounds always make sense ..
        _min = _min + rangelength
        _max = _max - rangelength
        _range = (_min, _max)
        print("_range is: ", _range)
        arrays.append(_range)

    print("arrays are: ")
    print(arrays)

    out = linearly_spaced_combinations(arrays, sample_per_dim)

    # calculate the volume of one of these (approximate) boxes ...

    print("Out shape is: ", out.shape)

    # Take out the very last grid items ...
    # maximal_arguments = np.argwhere(out == np.amax(out, axis=0))

    # print("Maximal arguments are: ", maximal_arguments)

    # exit(0)

    return out

def calculate_cubes(grid):
    """
        From the grid, create a set of cubes.
        You create these cubes by taking the distance between items for each dimension,
        and taking the grid-coordinate as a starting point
    :param X:
    :param grid:
    :return:
    """
    # 1. From the grid, create a set of cubes
    cube = np.max(np.diff(grid, axis=0), axis=0)
    print("cube dimensions are: ", cube)

    return cube


def rejection_sampling(X, cube, grid, rejection_threshold):
    """
        Perhaps a better way to reject-sample is to calculate the threshold with a similar mechanism as to
            normal + stddev (after sampling the entire space)
    :param rejection_threshold:
    :return:
    """

    densities = []

    for row_idx in range(grid.shape[0]):
        anchor_point = grid[row_idx]
        end_point = anchor_point + cube

        def inside_cube(vec):
            tmp = np.all(anchor_point <= vec) and np.all(vec <= end_point)
            tmp = tmp.all()
            return 1 if tmp else 0

        density = 0.

        for i in range(X.shape[0]):

            count_inside_cube = inside_cube(X[i, :])
            density += count_inside_cube

            if count_inside_cube:
                # print("Count inside cube are. ", count_inside_cube)
                # print("Anchor point and end point are: ")
                # print(anchor_point, X[i, :], end_point)
                pass

        # print("Density here is: ", density)

        densities.append(density)

    # Now apply some very basic modality detection ...
    return densities

def sample_thesaurus_by_density_distribution(X, sentences, grid, distribution, std_parameter=-1.0):
    """
        We take out certain items if they are below a certain threshold
    :param X:
    :param grid:
    :param distribution:
    :param std_parameter:
    :return:
    """

    mu = np.mean(distribution)
    std = np.std(distribution)

    threshold = mu + std_parameter * std

    out = []

    # This is very important! We have to traverse the grid in the same way
    # we didcalculate the individual grid items
    for idx, row_idx in enumerate(grid.shape[0]):
        anchor_point = grid[row_idx]
        end_point = anchor_point + cube

        def inside_cube(vec):
            tmp = np.all(anchor_point <= vec) and np.all(vec <= end_point)
            tmp = tmp.all()
            return 1 if tmp else 0

        density = 0.

        for i in range(X.shape[0]):

            count_inside_cube = inside_cube(X[i, :])
            density += count_inside_cube

        if density > threshold:
            # Because the density if above a certain threshold,
            # we can now sample one of the words here
            # Take the sentence corresponding to "X"
            out.append(
                (idx, sentences[i], X[i, :])
            )












if __name__ == "__main__":
    print("Rejection sampling")

    # ' was ',
    # ' made '
    devset_polysemous_words = [
        # ' was ',
        ' thought ',
        # ' table ',
        # ' only ',
        ' central ',
        ' pizza ',
        ' bank ',
        ' cold ',
        ' mouse ',
        ' good ',
        ' key ',
        ' arms '
    ]

    rnd_str = randomString(additonal_label=f"{args.dimred}_{args.dimred_dimensions}")

    for tgt_word in devset_polysemous_words:

        # Perhaps also try out a parzen window estimate ..?
        # tgt_word = ' bank '
        number_of_senses, X, true_cluster_labels, known_indices = sample_embeddings_for_target_word(tgt_word)
        print("Number of senses: ", number_of_senses)

        # First, we start with a random matrix
        # X = np.random.random((1000, 3))
        grid = create_grid(X, sample_per_dim=6)
        print("Grid is: ", grid)
        cube = calculate_cubes(grid)

        densities = rejection_sampling(X, cube, grid, rejection_threshold=None)

        print("Densities are: ", densities)

        # Print a histogram plot of how many density distributions we have for each square ...
        plt.hist(densities, bins=100, log=True)
        plt.title("log" + tgt_word)

        plt.savefig(rnd_str + f"/log_{tgt_word}_samples{args.max_samples}_senses{len(number_of_senses)}.png")
        # plt.show()
        plt.clf()
