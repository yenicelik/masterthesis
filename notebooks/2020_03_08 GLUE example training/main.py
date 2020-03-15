"""
    The main runner to run the GLUE tasks. We will start with a simple one.

    Depending on the model, we will need a different "augmented" class for the corpus
"""
import glob
import logging

# I de-activated caching, because we are playing around with the tokenizer..
from src.glue.evaluate import load_and_cache_examples, evaluate
from src.glue.logger import logger
import os
import torch

from transformers import WEIGHTS_NAME
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from src.glue.args import args, MODEL_CLASSES
from src.glue.trainer import set_seed, train

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
    # Training                                               #
    #                                                        #
    ##########################################################
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
    # Saving the model and re-loading it                     #
    #                                                        #
    ##########################################################
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        # torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        #
        # # Load a trained model and vocabulary that you have fine-tuned
        # model = model_class.from_pretrained(args.output_dir)
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        # # TODO: Split words must be saved and re-loaded as well!!!
        # # TODO: Replace-dict needs to be saved and re-loaded as well!!!
        # model.to(args.device)

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
