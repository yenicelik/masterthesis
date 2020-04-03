"""
    Anything related to saving and loading a model
"""
import os

import torch

from src.glue.logger import logger

def load_model(args, path, model_class, tokenizer_class):
    # If path exists, load the model
    # # Load a trained model and vocabulary that you have fine-tuned
    print("Loading a model!!!", path)
    model = model_class.from_pretrained(path)
    tokenizer = tokenizer_class.from_pretrained(path)
    if args.model_type in ("bernie_meaning"):
        tokenizer.load_bernie_specifics(path, bernie_model=model)

def save_model(args, path, model, tokenizer):
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
