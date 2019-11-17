
import tensorflow as tf

def mean_multiplication(mean_matrix, rotation_matrix):
    """
        Just a wrapper around multiplying mean vectors.
        This retains the shape of the original vector
    :param mean_matrix:
    :param roation_matrix:
    :return:
    """
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
    return tf.matmul(rotation_matrix, tf.matmul(covariance_matrix, tf.transpose(rotation_matrix)))