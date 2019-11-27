"""
    A file which generates an orthogonal rotation matrix which is to be found by one of the algorithms.
    This serves as a sanit check for any algorithm.

    The functions here are taken from the Bachelortehsis code.
    The output of the tensor is translated to tensorflow (that's easier right now..).
"""
import tensorflow as tf

from src.config import args


def generate_rotation_matrix(src_dim, tgt_dim, orthogonal=True):

    if args.random_seed:
        tf.random.set_seed(args.random_seed)

    matr = tf.random.normal((src_dim, tgt_dim))

    shape = matr.shape

    if orthogonal:
        matr, _ = tf.linalg.qr(matr) # full_matrices=True

    assert shape == matr.shape, (matr.shape, shape)

    return matr

if __name__ == "__main__":
    print("Generating rotation matrix")
    matr = generate_rotation_matrix(10, 5, orthogonal=True)
    print(matr.shape)
    print(tf.matmul(tf.transpose(matr), matr))