"""
    Initializers are defined here
"""
import tensorflow as tf

_glorot_uniform = tf.keras.initializers.GlorotUniform()

def glorot_uniform():
    return _glorot_uniform
