"""
    Implements the argparser file, including default configs that are used for the process right now.
    If any new experiment is created, these values will be flushed to a json,
    which will be included in the experiment folder
"""

import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description='Experiment Configuration Parser')
parser.add_argument('--random_seed', default=0, type=int, help='an integer which determines the random seed. if no seed shall be provided, set this to 0')


parser.add_argument('--dtype', default='tf.float32', type=int, help='an integer which determines the random seed. if no seed shall be provided, set this to 0')

args = parser.parse_args()

if args.dtype == "tf.float32":
    args.dtype = tf.float32
print(args)