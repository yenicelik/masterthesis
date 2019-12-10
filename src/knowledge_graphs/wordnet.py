"""
    Import WordNet.
    We use the standard nltk package to get wordnet items
"""
import numpy as np

from nltk.corpus import wordnet as wn

class WordNetDataset:

    def __init__(self):
        pass

    def get_number_of_senses(self, word):
        """
            For a given word, get how many different synsets it is included in.
            We log this number, to achieve a better comparison score
            (more meanings implies that this can be more-difficultly estimated..)
        :param word:
        :return:
        """
        out = wn.synsets(word)
        return np.log(len(out)), out
