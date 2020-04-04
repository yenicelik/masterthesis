"""
    The main runner to run the GLUE tasks. We will start with a simple one.

    Depending on the model, we will need a different "augmented" class for the corpus

    This has a guide on how to do additional fine-tuning on an existing model

    https://mccormickml.com/2019/09/19/XLNet-fine-tuning/

"""
import glob
import logging

# I de-activated caching, because we are playing around with the tokenizer..
from src.bernie_meaning.bernie_meaning_sequence_model import BernieMeaningForSequenceClassification, \
    BernieMeaningForMaskedLM
from src.config import args
from src.functional.saveload import save_model, load_model
from src.glue.additional_pretrainer import LineByLineTextDataset, pretrain
from src.glue.evaluate import load_and_cache_examples, evaluate
from src.glue.logger import logger
import os
import torch

from transformers import WEIGHTS_NAME
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from src.glue.trainer import set_seed, train
from src.glue_args import MODEL_CLASSES

from src.resources.split_words import get_polysemous_splitup_words

def print_model_tokenizer_stats(tokenizer, model, title):
    # Do a bunch of asserts
    print(title)
    print("Re-loaded splitwords are: ")
    print(tokenizer.split_tokens)
    print(tokenizer.replace_dict)
    print(tokenizer.added_tokens)
    print("Embedding sizes")
    print(model.bert.embeddings.word_embeddings.weight.shape)

    # Return if loading was successful

def prepare_runs():
    ##########################################################
    #                                                        #
    # Create the output directory if this does not exist yet #
    #                                                        #
    ##########################################################
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    ##########################################################
    #                                                        #
    # Setup device, CUDA, GPU & distributed training         #
    #                                                        #
    ##########################################################
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    ##########################################################
    #                                                        #
    # Setup basic logging                                    #
    # Setup logging                                          #
    #                                                        #
    ##########################################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    ##########################################################
    #                                                        #
    # Set seed                                               #
    #                                                        #
    ##########################################################
    set_seed(args)


def load_model_and_tokenizer(num_labels, finetuning_task, model_classes=None):
    ##########################################################
    #                                                        #
    # Load the model and tokenizer                           #
    #                                                        #
    ##########################################################
    if model_classes is None:
        model_classes = MODEL_CLASSES

    args.model_type = args.model_type.lower()
    print("args model type is: ", args.model_type)
    print(MODEL_CLASSES)
    # TODO: Replace by our own BERT (alternatively, allow import through this as well...
    config_class, model_class, tokenizer_class = model_classes[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=finetuning_task,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # TODO: Could also just augment the tokenizer here, and replace it with your own ...

    # TODO: Double check if bernie_pos was actually loaded, and not something else ...
    print("Tokenizer, config and model are")
    print(tokenizer, model, config)

    return tokenizer, model, config, model_class, tokenizer_class

def run_pretrain_on_dataset(model, tokenizer, train_dataset):
    """
        Pre-Train the BERT model based on this.
    :param model:
    :param tokenizer:
    :param train_dataset:
    :return:
    """

    if (args.additional_pretraining and args.model_type in ("bernie_meaning", "bernie_pos")):

        if args.local_rank == 0:
            torch.distributed.barrier()

        model.to(args.device)

        print("Train datasete is: (1) ", train_dataset)

        # Send the actual underlying BERT model, not the BERTforSequenceClassification model
        global_step, tr_loss = pretrain(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Dont add new tokens after pretraining!! (which makes sense...) but deactivate this functionality
    # This disables the "if" statement which generates new tokens ... (I think..)
    tokenizer.set_split_tokens(split_tokens={})
    return model

def inject_tokens_into_bert(tokenizer, model):
    # For all the split words, introduce the split token

    if args.model_type in ("bernie_pos", "bernie_meaning"):

        print("Using BERNIE model")

        # Inject the base model to the tokenizer
        tokenizer.inject_model(model.bert)

        # Inject the split tokens, s.t. new tokens are created for these over time
        polysemous_words = get_polysemous_splitup_words()
        tokenizer.set_split_tokens(polysemous_words)

        print("Polysemous words are!", polysemous_words)

        tokenizer.output_meaning_dir = args.output_meaning_dir

    else:
        print("Not using bernie_pos model!!!")
        print(args.model_type)

    return tokenizer, model

def prepare_glue_tasks():
    """
        Prepares the GLUE tasks
    :return:
    """
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    return processor, label_list, num_labels

def pretrain_bernie_meaning():
    """
        Pretrains the BERnie meaning embeddings after additional tokens were injectd
    :return:
    """
    exit(0)  # Safety line so we don't accidentaly overwrite the savemodel
    prepare_runs()
    processor, label_list, num_labels = prepare_glue_tasks()

    # TODO: Not suree what num_labels should be
    # Load model
    tokenizer, seq_model, config, model_class, tokenizer_class = load_model_and_tokenizer(
        num_labels=num_labels,
        finetuning_task=args.task_name,
        model_classes=None
    )

    # Inject tokens # Inject BERnie Tokenizer
    tokenizer, seq_model = inject_tokens_into_bert(tokenizer, seq_model)
    tokenizer.inject_model(seq_model.bert)

    print("DONE TRAINING LULULULU")
    # Try to load the tokenizer and model if this is possible
    # Create output directory if needed
    if os.path.exists(args.output_dir + "pretrained/") and args.local_rank in [-1, 0]:
        seq_model, tokenizer = load_model(
            args=args,
            path=args.output_dir + "pretrained/",
            model_class=model_class,
            tokenizer_class=tokenizer_class
        )
        seq_model.to(args.device)

        print_model_tokenizer_stats(tokenizer=tokenizer, model=seq_model, title="\n\n\n Loading the model...")

        assert tokenizer.vocab_size + len(tokenizer.added_tokens_decoder) == seq_model.bert.embeddings.word_embeddings.weight.shape[0], (
            tokenizer.vocab_size,
            len(tokenizer.replace_dict),
            len(tokenizer.added_tokens_decoder),
            seq_model.bert.embeddings.word_embeddings.weight.shape[0]
        )

        return seq_model, tokenizer

    seq_model.to(args.device)

    # Set only part of the embeddings as trainable
    print("Running the tokenizer through the Dataset to populate split-tokens")

    # 1. Generate the additional pre-training corpus
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        args=args,
        file_path=os.getenv("EN_CORPUS"),
        block_size=50  # This is the default BERT block size
    )

    if args.model_type in ("bernie_meaning", "bernie_pos"):
        # old_split_tokens = tokenizer.split_tokens
        old_replace_dict = tokenizer.replace_dict
        old_matr_shape = seq_model.bert.embeddings.word_embeddings.weight.shape
        old_added_tokens = tokenizer.added_tokens

        print_model_tokenizer_stats(tokenizer=tokenizer, model=seq_model, title="\n\n\n Before saving the model...")

    # Put it back to CPU for now
    seq_model.to('cpu')

    # Instead of the original configs, use the modified configs!
    # Modify the config by vocab-size
    # Now spawn BernieMeaningForSequenceClassification
    config.architectures = ["BernieForSequenceClassification"]
    config.finetuning_task = None
    config.bernie_meaning = None
    # TODO: Do I have to manually add these tokens? // Check this out ..
    config.vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens_decoder)

    # TODO: Check embeddings of underlying BERT model
    print("Tokens before passing data through tokenizer is: ")
    print(config)

    # TODO: Instantiate BertMaskedLMModel by this config
    # TODO: Or just create a new config..
    model = BernieMeaningForMaskedLM(config)
    model.bert = seq_model.bert
    # The rest stay the same

    # TODO: Save the trained BERT model
    # TODO: Load this saved BERT model (or pass it on ...)

    # Do the actual pre-training
    model = run_pretrain_on_dataset(model, tokenizer, train_dataset)

    # Now take it, and put it back into the original model
    seq_model.bert = model.bert

    # Skip the saving etc right now ..

    # Save for a second time after pre-training is done:
    # Create output directory if needed
    print("Inside lala")
    print_model_tokenizer_stats(tokenizer=tokenizer, model=seq_model, title="\n\n\n Saving the model...")
    save_model(args=args, path=args.output_dir + "pretrained/", model=model, tokenizer=tokenizer)
    # Load model to check if it was successful
    seq_model, tokenizer = load_model(
        args=args,
        path=args.output_dir + "pretrained/",
        model_class=model_class,
        tokenizer_class=tokenizer_class
    )
    seq_model.to(args.device)
    print_model_tokenizer_stats(tokenizer=tokenizer, model=seq_model, title="\n\n\n Loading the model...")
    assert tokenizer.vocab_size + len(tokenizer.added_tokens_decoder) == \
           seq_model.bert.embeddings.word_embeddings.weight.shape[0], (
        tokenizer.vocab_size,
        len(tokenizer.replace_dict),
        len(tokenizer.added_tokens_decoder),
        seq_model.bert.embeddings.word_embeddings.weight.shape[0]
    )
    model.to(args.device)

    if args.model_type in ("bernie_meaning"):
        # Do a bunch of asserts
        print_model_tokenizer_stats(tokenizer=tokenizer, model=seq_model, title="\n\n\n Re-loaded after splitwords are loaded")

        # Assert that model was loaded successfully
        assert old_matr_shape == model.bert.embeddings.word_embeddings.weight.shape, (
            old_matr_shape, model.bert.embeddings.word_embeddings.weight.shape)
        assert set(old_added_tokens) == set(tokenizer.added_tokens), (old_added_tokens, tokenizer.added_tokens)
        assert set(old_replace_dict.keys()) == set(tokenizer.replace_dict.keys()), (
            old_replace_dict, tokenizer.replace_dict)

    # This is the newly pretrained sequence model
    return seq_model

def main():
    print("Will now run the GLUE tasks")
    prepare_runs()

    processor, label_list, num_labels = prepare_glue_tasks()
    tokenizer, model, config, model_class, tokenizer_class = load_model_and_tokenizer(num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer, model = inject_tokens_into_bert(tokenizer, model)
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Load model ...
    model, tokenizer = load_model(args=args, path=args.output_dir + "_pretrained", model_class=model_class, tokenizer_class=tokenizer_class)
    print("Loaded config and model are: ", tokenizer)
    # modify the config accordingly ....
    config.vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens)
    print(config)

    print(tokenizer.vocab_size, len(tokenizer.added_tokens), model.bert.embeddings.word_embeddings.weight.shape)
    # Finally, print the number of predictor's output items ...

    exit()
    ##########################################################
    #                                                        #
    # Saving the model and re-loading it                     #
    #                                                        #
    ##########################################################
    # TODO: Do this saving (and loading ..) only right th pre-training!
    # if False and args.do_train and (args.local_rank == -1):
    #     # Create output directory if needed
    #
    #     print("Inside lala")
    #     if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #         save_model(args=args, path=args.output_dir, tokenizer=tokenizer, model=model)
    #     else:
    #         load_model(args=args, path=args.output_dir, model_class=model_class, tokenizer_class=tokenizer_class)
    #
    #     model.to(args.device)
    #
    # else:
    #     print("Outside lolo")

    print("Now going into training...")

    # TODO: I guess only save the underlying BERT model, and not together with the wrapper...?

    # 2. Fix any parameters you do not want to further train with BERT

    print("Successful training!")

    ##########################################################
    #                                                        #
    # Training                                               #
    #                                                        #
    ##########################################################
    # TODO: Uncomment!
    if args.do_train:
        # TODO: Do the dataset augmentation here!
        # TODO: Make sure cached is somehow turned off!!!

        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)

        # print("Added tokens for the tokenizer are (1) : ", tokenizer.added_tokens)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        # print("Added tokens for the tokenizer are (2): ", tokenizer.added_tokens)

    ##########################################################
    #                                                        #
    #        Final Evaluation                                #
    #                                                        #
    ##########################################################
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            # model = model_class.from_pretrained(checkpoint)
            # # TODO: Model class should check if embedding dimensions add up?
            # model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    print("Starting to train and evaluate the model")

    # Manually assign these variables ...?

    # if args.additional_pretraining and args.model_type in ("bernie_meaning", "bernie_pos"):
    #     pretrain_bernie_meaning()

    main()
