"""
    The main runner to run the GLUE tasks. We will start with a simple one.

    Depending on the model, we will need a different "augmented" class for the corpus

    This has a guide on how to do additional fine-tuning on an existing model

    https://mccormickml.com/2019/09/19/XLNet-fine-tuning/

"""
import glob
import logging

# I de-activated caching, because we are playing around with the tokenizer..
from src.config import args
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


def load_model_and_tokenizer(num_labels, finetuning_task):
    ##########################################################
    #                                                        #
    # Load the model and tokenizer                           #
    #                                                        #
    ##########################################################
    args.model_type = args.model_type.lower()
    print("args model type is: ", args.model_type)
    print(MODEL_CLASSES)
    # TODO: Replace by our own BERT (alternatively, allow import through this as well...
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
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

    if (args.additional_pretraining and args.model_type in ("bernie_meaning", "bernie_pos")):

        if args.local_rank == 0:
            torch.distributed.barrier()

        model.to(args.device)

        print("Train datasete is: (1) ", train_dataset)

        # Send the actual underlying BERT model, not the BERTforSequenceClassification model
        global_step, tr_loss = pretrain(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # TODO!!!! Dont add new tokens after pretraining!! (which makes sense...) but deactivate this functionality
    return model

def load_model(path, model_class, tokenizer_class):
    # If path exists, load the model
    # # Load a trained model and vocabulary that you have fine-tuned
    print("Loading a model!!!", path)
    model = model_class.from_pretrained(path)
    tokenizer = tokenizer_class.from_pretrained(path)
    if args.model_type in ("bernie_meaning"):
        tokenizer.load_bernie_specifics(path, bernie_model=model)

def save_model(path, model, tokenizer):
    print("Saving a model!!!", path)
    os.makedirs(path)
    logger.info("Saving model checkpoint to %s", path)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    print("We are saving: ", model_to_save)
    print(model_to_save.bert.embeddings.word_embeddings.weight.size)
    model_to_save.save_pretrained(path)
    tokenizer.save_pretrained(path)
    if args.model_type in ("bernie_meaning"):
        print("Saving special items ...")
        tokenizer.save_bernie_specifics(path)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(path, "training_args.bin"))

def pretrain_bernie_meaning():
    prepare_runs()

    # TODO: Not suree what num_labels should be
    # Load model
    tokenizer, model, config, model_class, tokenizer_class = load_model_and_tokenizer(num_labels=None, finetuning_task=args.task_name)

    # Inject tokens
    tokenizer, model = inject_tokens_into_bert(tokenizer, model)

    # Instead of the original configs, use the modified configs that include the language files!

    # Run the tokenizer through the dataset
    if args.additional_pretraining and args.model_type in ("bernie_meaning", "bernie_pos"):

        # TODO: Uncomment this at a later stage!
        # Try to load the tokenizer and model if this is possible
        if args.do_train and (args.local_rank == -1):
            # Create output directory if needed

            print("Inside lala")
            if os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                load_model(path=args.output_dir, model_class=model_class, tokenizer_class=tokenizer_class)

            model.to(args.device)

            # Do a bunch of asserts
            print("\n\n\n AFTER SAVE")
            print("Re-loaded splitwords are: ")
            print(tokenizer.split_tokens)
            print(tokenizer.replace_dict)
            print(tokenizer.added_tokens)
            print("Embedding sizes")
            print(model.bert.embeddings.word_embeddings.weight.shape)

            # Assert that model was loaded successfully

        else:
            print("Outside lolo")

        # Set only part of the embeddings as trainable

        print("Additional pre-training!!!")

        if args.model_type in ("bernie_meaning", "bernie_pos"):
            print("\n\n\n BEFORE TRAIN")
            print("Re-loaded splitwords are: ")
            print(tokenizer.split_tokens)
            print(tokenizer.replace_dict)
            print(tokenizer.added_tokens)
            print("Embedding sizes")
            print(model.bert.embeddings.word_embeddings.weight.shape)

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
            old_split_tokens = tokenizer.split_tokens
            old_replace_dict = tokenizer.replace_dict
            old_matr_shape = model.bert.embeddings.word_embeddings.weight.shape
            old_added_tokens = tokenizer.added_tokens

            print("\n\n\n BEFORE SAVE")
            print("Re-loaded splitwords are: ")
            print(tokenizer.split_tokens)
            print(tokenizer.replace_dict)
            print(tokenizer.added_tokens)
            print("Embedding sizes")
            print(model.bert.embeddings.word_embeddings.weight.shape)

        # Put it back to CPU for now
        model.to('cpu')

    else:
        print("Will not do additional pre-training")

    print("DONE TRAINING LULULULU")

    # TODO: Pass through dict

    # TODO: Modify the config by vocab-size

    # TODO: Instantiate BertMaskedLMModel by this config

    # TODO: Do the actual pre-training

    # TODO: Save the trained BERT model

    # TODO: Load this saved BERT model (or pass it on ...)


    # Pass through dataset, then pre-traing
    # TODO: Re-create the model
    if args.model_type in ("bernie_meaning"):
        # Do a bunch of asserts
        print("\n\n\n AFTER SAVE")
        print("Re-loaded splitwords are: ")
        print(tokenizer.split_tokens)
        print(tokenizer.replace_dict)
        print(tokenizer.added_tokens)
        print("Embedding sizes")
        print(model.bert.embeddings.word_embeddings.weight.shape)

        # Assert that model was loaded successfully
        assert set(old_split_tokens) == set(tokenizer.split_tokens), (old_split_tokens, tokenizer.split_tokens)
        assert old_matr_shape == model.bert.embeddings.word_embeddings.weight.shape, (
            old_matr_shape, model.bert.embeddings.word_embeddings.weight.shape)
        assert set(old_added_tokens) == set(tokenizer.added_tokens), (old_added_tokens, tokenizer.added_tokens)
        assert set(old_replace_dict.keys()) == set(tokenizer.replace_dict.keys()), (
            old_replace_dict, tokenizer.replace_dict)

    model = run_pretrain_on_dataset(model, tokenizer, train_dataset)


def inject_tokens_into_bert(tokenizer, model):
    # For all the split words, introduce the split token

    # TODO: Do all this here just-in-time ..

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
    ##########################################################
    #                                                        #
    # Prepare GLUE task                                      #
    #                                                        #
    ##########################################################
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    return processor, label_list, num_labels


def main():
    print("Will now run the GLUE tasks")
    prepare_runs()

    processor, label_list, num_labels = prepare_glue_tasks()
    tokenizer, model, config, model_class, tokenizer_class = load_model_and_tokenizer(num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer, model = inject_tokens_into_bert(tokenizer, model)
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    ##########################################################
    #                                                        #
    # Saving the model and re-loading it                     #
    #                                                        #
    ##########################################################
    # TODO: Do this saving (and loading ..) only right th pre-training!
    if False and args.do_train and (args.local_rank == -1):
        # Create output directory if needed

        print("Inside lala")
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            save_model(path=args.output_dir, tokenizer=tokenizer, model=model)
        else:
            load_model(path=args.output_dir, model_class=model_class, tokenizer_class=tokenizer_class)

        model.to(args.device)

    else:
        print("Outside lolo")

    print("Now going into training...")

    # TODO: I guess only save the underlying BERT model, and not together with the wrapper...?

    # Save for a second time after pre-training is done:
    if False and args.do_train and (args.local_rank == -1):
        # Create output directory if needed
        print("Inside lala")
        if not os.path.exists(args.output_dir + "pretrained/") and args.local_rank in [-1, 0]:
            save_model(path=args.output_dir + "pretrained/", model=model, tokenizer=tokenizer)
        else:
            load_model(path=args.output_dir + "pretrained/", model_class=model_class, tokenizer_class=tokenizer_class)

        model.to(args.device)

        # if args.model_type in ("bernie_meaning", "bernie_pos"):
        #     # Do a bunch of asserts
        #     print("\n\n\n AFTER SAVE")
        #     print("Re-loaded splitwords are: ")
        #     print(tokenizer.split_tokens)
        #     print(tokenizer.replace_dict)
        #     print(tokenizer.added_tokens)
        #     print("Embedding sizes")
        #     print(model.bert.embeddings.word_embeddings.weight.shape)
        #
        #     # Assert that model was loaded successfully
        #     assert old_split_tokens == tokenizer.split_tokens, (old_split_tokens, tokenizer.split_tokens)
        #     assert old_replace_dict == tokenizer.replace_dict, (old_replace_dict, tokenizer.replace_dict)
        #     assert old_matr_shape == model.bert.embeddings.word_embeddings.weight.shape, (
        #     old_matr_shape, model.bert.embeddings.word_embeddings.weight.shape)
        #     assert old_added_tokens == tokenizer.added_tokens, (old_added_tokens, tokenizer.added_tokens)

    else:
        print("Outside lolo")

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
    # pretrain_bernie_meaning()

    main()
