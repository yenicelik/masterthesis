"""
    Initializers are defined here
"""
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

_glorot_uniform = tf.glorot_uniform_initializer()

def glorot_uniform():
    return _glorot_uniform
