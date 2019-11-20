"""
    Initializers are defined here
"""
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

_glorot_uniform = tf.keras.initializers.GlorotUniform()

def glorot_uniform():
    return _glorot_uniform
