"""
    Implements the argparser file, including default configs that are used for the process right now.
    If any new experiment is created, these values will be flushed to a json,
    which will be included in the experiment folder
"""

import argparse
import tensorflow as tf
import torch

parser = argparse.ArgumentParser(description='Experiment Configuration Parser')
parser.add_argument('--random_seed', default=0, type=int, help='an integer which determines the random seed. if no seed shall be provided, set this to 0')
parser.add_argument('--dtype', default='tf.float32', help='the floating point type that is going to be used globally')

parser.add_argument('--max_samples', default=1000, type=int, help='the number of sentences to sample for BERT embeddings')
parser.add_argument('--cuda', default='False', help='Whether or not CUDA will be used. This argument will be ignored if CUDA is available')

parser.add_argument('--verbose', default=1, type=int, help='verbosity level. higher means more verbose')
parser.add_argument('--stemsearch', default=0, type=int, help='whether or not to stem the sentences to look for')

args = parser.parse_args()

if args.dtype == "tf.float32":
    args.dtype = tf.float32

# Automatically use cuda if cuda is available
if torch.cuda.is_available():
    args.cuda = True
    print("CUDA active!")
else:
    args.cuda = False
    print("CUDA is not available!")

assert args.max_samples > 32, ("If args is less than 32, it will not function properly!", args.max_samples)

print(args)
