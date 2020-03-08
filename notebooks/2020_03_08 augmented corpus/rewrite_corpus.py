"""
    We want to run two experiments.

    1) The new augmented corpus is clustered by our clustering approach
    (usually Chinese Whispers,
    hyperparameter optimization applied on the WordNet SemCor 3.0 dataseet)
    2) The new augmented corpus is clustered by a PoS tagger which is using the PoS nlp pipeline
"""
import time

import spacy

from src.resources.corpus import Corpus
from src.resources.split_words import get_polysemous_splitup_words


def create_pos_corpus(corpus, nlp, tgt_words, dev=False):
    """
        Apply this function before feeding a sentence in to BERET.
        Due to the nature of the `replace_dict`,
        this function must be applied to the entire corpus before the application
    :param tgt_words:
    :param dev:
    :return:
    """
    # The new corpus you will be saving should be a list of sentences,
    # each separated by the newline token "\n"

    # tgt word should occur as-such

    replace_dict = dict()
    out = []

    if dev:
        corpus.data = corpus.data[:1000]


    for sentence in corpus.sentences:
        start_time = time.time()

        new_sentence = []

        doc = nlp(sentence)

        # print("Doc is: ")
        # print(doc)
        # print([token.text for token in doc])

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


        # print("Replace dict is: ", replace_dict)
        # print("Out is: ", new_sentence)
        # print()

        out.append(new_sentence)

    return out


if __name__ == "__main__":
    print("We want to run two expe")
    tgt_words = get_polysemous_splitup_words()
    tgt_words = [x.strip() for x in tgt_words]

    print("Target words are: ")
    print(tgt_words)

    corpus = Corpus()
    nlp = spacy.load("en_core_web_sm")

    new_sentences = create_pos_corpus(corpus, nlp, tgt_words, dev=True)
    print(new_sentences)

    # Should also just save this in a new text file ...
