"""
    We want to run two experiments.

    1) The new augmented corpus is clustered by our clustering approach
    (usually Chinese Whispers,
    hyperparameter optimization applied on the WordNet SemCor 3.0 dataseet)
    2) The new augmented corpus is clustered by a PoS tagger which is using the PoS nlp pipeline
"""
import time

import spacy

from src.resources.augment import augment_sentence_by_pos
from src.resources.corpus import Corpus
from src.resources.split_words import get_polysemous_splitup_words


if __name__ == "__main__":
    print("We want to run two expe")
    tgt_words = get_polysemous_splitup_words()
    tgt_words = [x.strip() for x in tgt_words]

    print("Target words are: ")
    print(tgt_words)

    out = []

    corpus = Corpus()
    nlp = spacy.load("en_core_web_sm")

    replace_dict = dict()

    corpus.data = corpus.data[:1]

    start_time = time.time()
    for sentence in corpus.sentences:
        new_sentence = augment_sentence_by_pos(sentence, nlp, tgt_words, replace_dict)
        out.append(new_sentence)

    print("Took this many seconds for 1000 items:", time.time() - start_time)

    print(new_sentence)

    # This is fast enough to be deployed in a "just-in-time" fashion

    # Should also just save this in a new text file ...
