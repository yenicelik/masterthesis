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
from src.glue.additional_pretrainer import LineByLineTextDataset
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


def main():
    print("Will now run the GLUE tasks")

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
        finetuning_task=args.task_name,
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

    ##########################################################
    #                                                        #
    # and then modify the config file by that ...            #
    #                                                        #
    ##########################################################

    # For all the split words, introduce the split token

    # TODO: Do all this here just-in-time ..

    if args.model_type in ("bernie_pos", "bernie_meaning"):

        print("Using BERNIE model")

        # Inject model to the tokenizer
        tokenizer.inject_model(model)

        # Inject the split tokens, s.t. new tokens are created for these over time
        polysemous_words = get_polysemous_splitup_words()
        tokenizer.set_split_tokens(polysemous_words)

        print("Polysemous words are!", polysemous_words)

    else:
        print("Not using bernie_pos model!!!")
        print(args.model_type)

    if args.model_type in ("bernie_meaning"):
        tokenizer.output_meaning_dir = args.output_meaning_dir

    # TODO: Do some additional pre-training if BERnie PoS or Meaning!
    if args.additional_pretraining and args.model_type in ("bernie_meaning", "bernie_pos"):

        # TODO: Uncomment this at a later stage!
        # Try to load the tokenizer and model if this is possible
        # if args.do_train and (args.local_rank == -1):
        #     # Create output directory if needed
        #
        #     print("Inside lala")
        #     if os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        #         # If path exists, load the model
        #         # # Load a trained model and vocabulary that you have fine-tuned
        #         print("Loading a model!!!", args.output_dir)
        #         model = model_class.from_pretrained(args.output_dir)
        #         tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        #         if args.model_type in ("bernie_meaning"):
        #             tokenizer.load_bernie_specifics(args.output_dir, bernie_model=model)
        #
        #     model.to(args.device)
        #
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
        #
        # else:
        #     print("Outside lolo")


        # Set only part of the embeddings as trainable

        print("Additional pre-training!!!")

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

        # TODO: Fix all but the new vocabulary's embeddings

        # 3.

        # 1. Load the corpus dataset (
        # corpus =

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
        model.to(0)

    else:
        print("Will not do additional pre-training")

    print("DONE TRAINING LULULULU")

    ##########################################################
    #                                                        #
    # Start the actual training...                           #
    #                                                        #
    ##########################################################
    model.to(args.device)

    # TODO: BUG! Update within replace-dict and added-tokens does not happen simultanouelsy! Check this out..

    logger.info("Training/evaluation parameters %s", args)

    # TODO: Perhapds do a sample tokenization run with a sentence containing "book"?

    # Overwriting the tokenizer does not work ... need to manually write a `from_pretrained` function probably...?

    # TODO: Only one item is added to the hashmap at a time... fix this ...

    # example_sentence = "It costs the Open Content Alliance as much as $30 to scan each book, a cost shared by the group’s members and benefactors, so there are obvious financial benefits to libraries of Google’s wide-ranging offer, started in 2004."
    # print("Run through the tokenizer, and check if it successfully tokenizes 'book'")
    # new_example_sentence = tokenizer.tokenize(example_sentence)
    # print("New example sentence", new_example_sentence)

    ##########################################################
    #                                                        #
    # Saving the model and re-loading it                     #
    #                                                        #
    ##########################################################
    print("Condition is: ", args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0))
    print(args.do_train)
    print(args.local_rank)
    # print(args.torch.distributed.get_rank())
    #  or torch.distributed.get_rank() == 0
    # TODO: Do this saving (and loading ..) only right th pre-training!
    if args.do_train and (args.local_rank == -1):
        # Create output directory if needed
        print("Inside lala")
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            print("Saving a model!!!", args.output_dir)
            os.makedirs(args.output_dir)
            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            if args.model_type in ("bernie_meaning"):
                print("Saving special items ...")
                tokenizer.save_bernie_specifics(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        else:
            # If path exists, load the model
            # # Load a trained model and vocabulary that you have fine-tuned
            print("Loading a model!!!", args.output_dir)
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = tokenizer_class.from_pretrained(args.output_dir)
            if args.model_type in ("bernie_meaning"):
                tokenizer.load_bernie_specifics(args.output_dir, bernie_model=model)

        model.to(args.device)

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
            assert old_matr_shape == model.bert.embeddings.word_embeddings.weight.shape, (old_matr_shape, model.bert.embeddings.word_embeddings.weight.shape)
            assert set(old_added_tokens) == set(tokenizer.added_tokens), (old_added_tokens, tokenizer.added_tokens)
            assert set(old_replace_dict.keys()) == set(tokenizer.replace_dict.keys()), (old_replace_dict, tokenizer.replace_dict)

    else:
        print("Outside lolo")


    print("Now going into training...")
    if args.additional_pretraining and args.model_type in ("bernie_meaning", "bernie_pos"):

        if args.local_rank == 0:
            torch.distributed.barrier()

        model.to(args.device)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Save for a second time after pre-training is done:
    if args.do_train and (args.local_rank == -1):
        # Create output directory if needed
        print("Inside lala")
        if not os.path.exists(args.output_dir + "pretrained/") and args.local_rank in [-1, 0]:
            print("Saving a model!!!", args.output_dir + "pretrained/")
            os.makedirs(args.output_dir + "pretrained/")
            logger.info("Saving model checkpoint to %s", args.output_dir + "pretrained/")
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir + "pretrained/")
            tokenizer.save_pretrained(args.output_dir + "pretrained/")
            if args.model_type in ("bernie_meaning"):
                print("Saving special items ...")
                tokenizer.save_bernie_specifics(args.output_dir + "pretrained/")

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir + "pretrained/", "training_args.bin"))
        else:
            # If path exists, load the model
            # # Load a trained model and vocabulary that you have fine-tuned
            print("Loading a model!!!", args.output_dir + "pretrained/")
            model = model_class.from_pretrained(args.output_dir + "pretrained/")
            tokenizer = tokenizer_class.from_pretrained(args.output_dir + "pretrained/")
            if args.model_type in ("bernie_meaning"):
                tokenizer.load_bernie_specifics(args.output_dir + "pretrained/", bernie_model=model)

        model.to(args.device)

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
            assert old_split_tokens == tokenizer.split_tokens, (old_split_tokens, tokenizer.split_tokens)
            assert old_replace_dict == tokenizer.replace_dict, (old_replace_dict, tokenizer.replace_dict)
            assert old_matr_shape == model.bert.embeddings.word_embeddings.weight.shape, (old_matr_shape, model.bert.embeddings.word_embeddings.weight.shape)
            assert old_added_tokens == tokenizer.added_tokens, (old_added_tokens, tokenizer.added_tokens)

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

        print("Added tokens for the tokenizer are (1) : ", tokenizer.added_tokens)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        print("Added tokens for the tokenizer are (2): ", tokenizer.added_tokens)

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

    main()
