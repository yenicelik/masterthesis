"""
    This is the test where we try to inject certain word-vectors into BERT.
    We will likely need to modify following items:
    - BERT vocabulary (dictionary probably)
    - BERT lowest-level vectors -> replace these by mean of cluster vectors, or similar

    Resources on how to add your custom tokens and vocabulary
    - https://github.com/huggingface/transformers/blob/master/README.md in Section "Serialization"
"""
import os

from transformers import BertTokenizer, BertForSequenceClassification

from src.resources.split_words import get_polysemous_splitup_words
from src.utils.create_experiments_folder import randomString

# TODO: Replace the vocabulary as well (...)


def expand_bert_by_target_word(tgt_word, model: BertForSequenceClassification, tokenizer: BertTokenizer, n=5):
    """
        Getting the BERT embedding
    :return:
    """
    # Number of new tokens to add
    old_vocab_size = len(tokenizer.added_tokens_decoder)

    # 0: Find the target word idx in the vocabulary
    print(tokenizer.added_tokens_decoder)
    # Return the target word if it is not within the vocabulary
    if tgt_word not in tokenizer.vocab:
        print(f"{tgt_word} not in vocabulary! Cannot add this!")
        return model, tokenizer
    word_idx = tokenizer.vocab[tgt_word]

    # 1: Retrieve the embedding in the vocabulary
    old_vector = model.bert.embeddings.word_embeddings.weight.data[word_idx, :]

    # 2: Add tokens into tokenizeer and make space for new vectors
    tokens_to_add = [(f'{tgt_word}_{i}') for i in range(n - 1)]
    number_new_tokens = len(tokens_to_add)
    print("Tokens before", tokenizer.vocab_size)
    added_tokens = tokenizer.add_tokens(tokens_to_add)  # TODO: add_special_tokens or add_tokens?
    assert added_tokens == n - 1, (added_tokens, n)
    print("added tokens", added_tokens)
    model.resize_token_embeddings(len(tokenizer))
    print("Tokens after", tokenizer.vocab_size)

    # 2.1: Test if tokenization worked out
    # print([tokenizer.convert_tokens_to_ids(x) for x in tokens_to_add])
    assert all([tokenizer.convert_tokens_to_ids(x) for x in tokens_to_add])

    # 2: Inject / overwrite new word-embeddings with old embedding
    model.bert.embeddings.word_embeddings.weight.data[-number_new_tokens:, :] = old_vector.reshape(1, -1).repeat(
        (n - 1, 1))

    assert (model.bert.embeddings.word_embeddings.weight.data[-number_new_tokens + 1, :] == \
            model.bert.embeddings.word_embeddings.weight.data[word_idx, :]).all()

    # Take the embeddings vector at position `word_idx` and add this to the embedding
    # Double check if these were successfully copied ...

    new_vocab_size = len(tokenizer.added_tokens_decoder)

    assert new_vocab_size == old_vocab_size + (n - 1), (new_vocab_size, old_vocab_size, (n - 1))

    return model, tokenizer


def get_bert_size_stats(model, tokenizer):
    print("Number of embeddings in BERT model and tokenizer")
    print("Tokenizer number of vocabs is: ", len(tokenizer))
    # print(model)
    # print("Encoder: ", model.encoder)
    print("Encoder number of attention mechanisms: ", model.encoder.output_attentions)
    print("Embeddings: ", model.embeddings)
    print("Word emb: ", model.embeddings.word_embeddings.num_embeddings)
    print("Position emb: ", model.embeddings.position_embeddings.num_embeddings)
    print("Token type emb: ", model.embeddings.token_type_embeddings.num_embeddings)


def create_path(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_bert_model():
    # Take the uncased ones, as this make tokenization etc. much easier ..
    # Can then use cased if this proves to be success

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased')  # No, let's not use TFBert, let's use PyTorch BERT

    get_bert_size_stats(model.bert, tokenizer)

    # for word in extra-words ...
    ######################################################
    #
    # Add words to the BERT tokenizer and embedding
    #
    ######################################################

    # # TODO: Extend by more than just this words
    # Get fine-tune-words
    fine_tune_words = get_polysemous_splitup_words()
    for tgt_word in fine_tune_words:
        print(f"Expanding the word {tgt_word} within BERT")
        # tgt_word = "run"
        model, tokenizer = expand_bert_by_target_word(tgt_word, model, tokenizer)

    # get_bert_size_stats(model.bert, tokenizer)

    # now fine-tune the model on one of the GLUE language tasks
    rnd_str = randomString(additonal_label=f"_fine-tune-BERT/")

    get_bert_size_stats(model.bert, tokenizer)

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
