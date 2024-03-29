"""
    Any functions and operations that act on the
    - mean of gaussian embeddings
    - covariance of gaussian embeddings

    that may generally be useful to numerically support, or where small mistakes can have big consequences
"""

import tensorflow as tf

def mean_multiplication(mean_matrix, rotation_matrix):
    """
        Just a wrapper around multiplying mean vectors.
        This retains the shape of the original vector
    :param mean_matrix:
    :param roation_matrix:
    :return:
    """
    # tf.enable_eager_execution()
    return tf.reshape(tf.matmul(mean_matrix, rotation_matrix), (1, -1))

def covariance_multiplication(covariance_matrix, rotation_matrix):
    """
    According to https://robotics.stackexchange.com/questions/2556/how-to-rotate-covariance,
    the rotation of the covariance matrix is defined as
        C' = R C R^T,
    where R is a rotation matrix.
    :param covariance_matrix:
    :param rotation_matrix:
    :return:
    """
    # tf.enable_eager_execution()
    return tf.matmul(rotation_matrix, tf.matmul(covariance_matrix, tf.transpose(rotation_matrix)))