"""
    Implements the argparser file, including default configs that are used for the process right now.
    If any new experiment is created, these values will be flushed to a json,
    which will be included in the experiment folder
"""

import argparse
import tensorflow as tf
import torch

parser = argparse.ArgumentParser(description='Experiment Configuration Parser')
parser.add_argument('--random_seed', default=0, type=int,
                    help='an integer which determines the random seed. if no seed shall be provided, set this to 0')
parser.add_argument('--dtype', default='tf.float32', help='the floating point type that is going to be used globally')

parser.add_argument('--max_samples', default=1000, type=int,
                    help='the number of sentences to sample for BERT embeddings')
parser.add_argument('--cuda', default='False',
                    help='Whether or not CUDA will be used. This argument will be ignored if CUDA is available')

parser.add_argument('--verbose', default=1, type=int, help='verbosity level. higher means more verbose')
parser.add_argument('--stemsearch', default=0, type=int, help='whether or not to stem the sentences to look for')

parser.add_argument('--dimred', default="none", help='which dimensionality reduction algorithm to use. If none specified, falling back to PCA. One of "nmf", "pca", "umap", "lda" ')
parser.add_argument('--dimred_dimensions', default=768, help='which dimensionality to reduce to during the dimred phase. Falling back to 4 if not specified')
parser.add_argument('--pca_whiten', default=False, help='which dimensionality to reduce to during the dimred phase. Falling back to 4 if not specified')
parser.add_argument('--normalization_norm', default="", help='What norm to normalize the vectors by before applying clustering. set to an invalid value if you dont want any normalization')
parser.add_argument('--standardize', default=False, help='What norm to normalize the vectors by before applying clustering. set to an invalid value if you dont want any normalization')


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
