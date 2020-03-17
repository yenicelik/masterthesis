"""
    Implements the argparser file, including default configs that are used for the process right now.
    If any new experiment is created, these values will be flushed to a json,
    which will be included in the experiment folder
"""

import argparse
import tensorflow as tf
import torch

from transformers import glue_processors as processors

model_parser = argparse.ArgumentParser(description='Experiment Configuration Parser')
model_parser.add_argument('--random_seed', default=0, type=int,
                          help='an integer which determines the random seed. if no seed shall be provided, set this to 0')
model_parser.add_argument('--dtype', default='tf.float32', help='the floating point type that is going to be used globally')

model_parser.add_argument('--max_samples', default=500, type=int,
                          help='the number of sentences to sample for BERT embeddings')
model_parser.add_argument('--cuda', default='False',
                          help='Whether or not CUDA will be used. This argument will be ignored if CUDA is available')

model_parser.add_argument('--verbose', default=0, type=int, help='verbosity level. higher means more verbose')
model_parser.add_argument('--stemsearch', default=0, type=int, help='whether or not to stem the sentences to look for')

model_parser.add_argument('--dimred', default="none", help='which dimensionality reduction algorithm to use. If none specified, falling back to PCA. One of "nmf", "pca", "umap", "lda" ')
model_parser.add_argument('--dimred_dimensions', default=768, help='which dimensionality to reduce to during the dimred phase. Falling back to 4 if not specified')
model_parser.add_argument('--pca_whiten', default=False, help='which dimensionality to reduce to during the dimred phase. Falling back to 4 if not specified')
model_parser.add_argument('--normalization_norm', default="", help='What norm to normalize the vectors by before applying clustering. set to an invalid value if you dont want any normalization')
model_parser.add_argument('--standardize', default=False, help='What norm to normalize the vectors by before applying clustering. set to an invalid value if you dont want any normalization')


# Now add GLUE args

# Required parameters
model_parser.add_argument(
    "--data_dir", default=None, type=str, required=True,
    help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
)
model_parser.add_argument(
    "--model_type", default=None, type=str, required=True,
    help="Model type selected in the list: " + ", ",
)
model_parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list: ",
)
model_parser.add_argument("--task_name", default=None, type=str, required=True,
                    help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
                    )
model_parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help="The output directory where the model predictions and checkpoints will be written.",
)

# Other parameters
model_parser.add_argument(
    "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
)
model_parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
)
model_parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
)
model_parser.add_argument(
    "--max_seq_length",
    default=128,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
         "than this will be truncated, sequences shorter will be padded.",
)
model_parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
model_parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
model_parser.add_argument(
    "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
)
model_parser.add_argument(
    "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
)

model_parser.add_argument(
    "--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.",
)
model_parser.add_argument(
    "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
)
model_parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
model_parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
model_parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
model_parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
model_parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
model_parser.add_argument(
    "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
)
model_parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
)
model_parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

model_parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
model_parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
model_parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
)
model_parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
model_parser.add_argument(
    "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
)
model_parser.add_argument(
    "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
)

model_parser.add_argument(
    "--output_meaning_dir",
    default="/Users/david/GoogleDrive/_MasterThesis/savedir/cluster_model_caches",
    type=str,
    required=True,
    help="The output directory where the meaning-cluster model trains and caches the cluster-models for each word individually.",
)

model_parser.add_argument("--seed", type=int, default=101, help="random seed for initialization")

model_parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
model_parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
         "See details at https://nvidia.github.io/apex/amp.html",
)
model_parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")


args = model_parser.parse_args()

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
