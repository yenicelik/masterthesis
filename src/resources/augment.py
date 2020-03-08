"""
    Augments the respective sentence by replacing this with a more specific word-token.
    Example:
        run -> run_0, run_1, run_2
        where run_0, run_1 have different PoS, meanings, etc. (something to be "discriminated" against)

    Also includes functions to incorporate these changes into BERT
"""
from transformers import BertForSequenceClassification, BertTokenizer


def _get_bert_size_stats(model, tokenizer):
    print("Number of embeddings in BERT model and tokenizer")
    print("Tokenizer number of vocabs is: ", len(tokenizer))
    # print(model)
    # print("Encoder: ", model.encoder)
    print("Encoder number of attention mechanisms: ", model.encoder.output_attentions)
    print("Embeddings: ", model.embeddings)
    print("Word emb: ", model.embeddings.word_embeddings.num_embeddings)
    print("Position emb: ", model.embeddings.position_embeddings.num_embeddings)
    print("Token type emb: ", model.embeddings.token_type_embeddings.num_embeddings)


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


######################################################
#
#  Below anything related to augmenting sentences
#
######################################################

def augment_sentence_by_pos(sentence, nlp, tgt_words, replace_dict):
    """
        Apply this function before feeding a sentence in to BERET.
        Due to the nature of the `replace_dict`,
        this function must be applied to the entire corpus before the application.

        This function takes 0.01 (0.05 completely non-cached) seconds per run, which is considered fast enough for actual runs.

        replace_dict must be a reference!!!

    :param sentence: the sentence to be replaced
    :param nlp: the spacy nlp tokenizer
    :param tgt_words: The target words which shall all be replaced
    :param replace_dict: The dictionary which translates the "meaning" to the number # MUST BE A DICT-reference!!!
    :return:
    """

    new_sentence = []

    doc = nlp(sentence)

    # For all the above target words
    for token in doc:
        # Identify if it is in sentence

        # If it is not a target word, don't modify
        if token.text not in tgt_words:
            new_sentence.append(token.text)
        else:
            pos = token.pos_
            if token.text in replace_dict:
                # retrieve index of item
                idx = replace_dict[token.text].index(pos) if pos in replace_dict[token.text] else -1
                if idx == -1:
                    replace_dict[token.text].append(pos)
                    idx = replace_dict[token.text].index(pos)
                    assert idx >= 0
            else:
                replace_dict[token.text] = [pos, ]
                idx = 0

            # print("Replacing with ", token.text, token.pos)

            new_token = f"{token.text}_{idx}"

            # replace the token with the new token
            new_sentence.append(new_token)

    res_str = " ".join(new_sentence)
    new_sentence = res_str \
        .replace(" .", ".") \
        .replace(" ,", ",") \
        .replace(" ’", "’") \
        .replace(" - ", "-") \
        .replace("$ ", "$")

    return new_sentence
