"""
    A file which generates an orthogonal rotation matrix which is to be found by one of the algorithms.
    This serves as a sanit check for any algorithm.
"""
import tensorflow as tf

def generate_rotation_matrix(src_dim, tgt_dim, orthogonal=False):
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