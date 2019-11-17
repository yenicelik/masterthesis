"""
    Tests whether a given matrix was generated orthogonally
"""
import tensorflow as tf

from src.synthetic.generate_rotation_matrix import generate_rotation_matrix

if __name__ == "__main__":
    print("Testing if a matrix was generated orthogonally...")
    matr = generate_rotation_matrix(10, 5, orthogonal=True)
    assert matr.shape == (10, 5), ("Shape is not as indicated!", matr.shape, (10, 5))
    I_hat = tf.matmul(tf.transpose(matr), matr)
    assert tf.reduce_sum(I_hat - tf.eye(5)) < 1.e-6, ("Matrix does not seem orthogonal!", I_hat)
    print("Success")