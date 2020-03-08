"""
    Augments the respective sentence by replacing this with a more specific word-token.
    Example:
        run -> run_0, run_1, run_2
        where run_0, run_1 have different PoS, meanings, etc. (something to be "discriminated" against)
"""

def augment_sentence_by_pos(sentence, nlp, tgt_words, replace_dict):
    """
        Apply this function before feeding a sentence in to BERET.
        Due to the nature of the `replace_dict`,
        this function must be applied to the entire corpus before the application.

        This function takes 0.01 (0.05 completely non-cached) seconds per run, which is considered fast enough for actual runs.

    :param sentence: the sentence to be replaced
    :param nlp: the spacy nlp tokenizer
    :param tgt_words: The target words which shall all be replaced
    :param replace_dict: The dictionary which translates the "meaning" to the number
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
    new_sentence = res_str\
        .replace(" .", ".")\
        .replace(" ,", ",")\
        .replace(" ’", "’")\
        .replace(" - ", "-")\
        .replace("$ ", "$")

    return new_sentence
