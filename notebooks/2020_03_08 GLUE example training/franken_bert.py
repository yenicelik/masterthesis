"""
    This is the test where we try to inject certain word-vectors into BERT.
    We will likely need to modify following items:
    - BERT vocabulary (dictionary probably)
    - BERT lowest-level vectors -> replace these by mean of cluster vectors, or similar

    Resources on how to add your custom tokens and vocabulary
    - https://github.com/huggingface/transformers/blob/master/README.md in Section "Serialization"
"""
from transformers import BertTokenizer, BertForSequenceClassification

from src.resources.augment import expand_bert_by_target_word, _get_bert_size_stats
from src.resources.split_words import get_polysemous_splitup_words
from src.utils.create_experiments_folder import randomString, create_path


def get_bert_model():
    # Take the uncased ones, as this make tokenization etc. much easier ..
    # Can then use cased if this proves to be success

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased')  # No, let's not use TFBert, let's use PyTorch BERT

    _get_bert_size_stats(model.bert, tokenizer)

    # for word in extra-words ...
    ######################################################
    #
    # Add words to the BERT tokenizer and embedding
    #
    ######################################################

    # Get fine-tune-words
    fine_tune_words = get_polysemous_splitup_words()
    for tgt_word in fine_tune_words:
        print(f"Expanding the word {tgt_word} within BERT")
        # tgt_word = "run"
        model, tokenizer = expand_bert_by_target_word(tgt_word, model, tokenizer)

    # get_bert_size_stats(model.bert, tokenizer)

    # now fine-tune the model on one of the GLUE language tasks
    rnd_str = randomString(additonal_label=f"_fine-tune-BERT/")

    _get_bert_size_stats(model.bert, tokenizer)

    # Save the model somewhere, s.t. you can run the standard trainer ...

    ######################################################
    # Save the model and re-load as a tensorflow model
    ######################################################
    print("Save dir is: ", f'{rnd_str}my_saved_model_directory/')
    create_path(f'{rnd_str}my_saved_model_directory/')

    model.save_pretrained(f'{rnd_str}my_saved_model_directory/')
    tokenizer.save_pretrained(f'{rnd_str}my_saved_model_directory/')

    ######################################################
    # Make sure loading works as well (perhaps also re-load an item ...)
    ######################################################
    tokenizer = BertTokenizer.from_pretrained(f'{rnd_str}my_saved_model_directory/')
    model = BertForSequenceClassification.from_pretrained(f'{rnd_str}my_saved_model_directory/')

    sentence_0 = "This research was consistent with his findings."
    sentence_1 = "His findings were compatible with this research."
    sentence_2 = "His findings were not compatible with this research."
    inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors='pt')
    inputs_2 = tokenizer.encode_plus(sentence_0, sentence_2, add_special_tokens=True, return_tensors='pt')

    pred_1 = model(inputs_1['input_ids'], token_type_ids=inputs_1['token_type_ids'])[0].argmax().item()
    pred_2 = model(inputs_2['input_ids'], token_type_ids=inputs_2['token_type_ids'])[0].argmax().item()

    print("sentence_1 is", "a paraphrase" if pred_1 else "not a paraphrase", "of sentence_0")
    print("sentence_2 is", "a paraphrase" if pred_2 else "not a paraphrase", "of sentence_0")


if __name__ == "__main__":
    print("starting to frankenstein bert")
    get_bert_model()
