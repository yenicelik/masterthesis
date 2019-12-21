"""
    IMplements any logic which takes in and processes SemCor logic
"""
import math
import os
import time
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn

from dotenv import load_dotenv

load_dotenv()


class CorpusSemCor:

    @property
    def sentences(self):
        # Make two nested lists
        # One captures the words
        # One captures the WordNet ids
        return self.data

    def __init__(self):

        print("Starting corpus")

        self.folderpath = os.getenv("SEMCOR_CORPUS")

        self.data, self.data_wordnet_ids = self._load_corpus()

    def xml2arrays(self, filepath):
        print("Turning the filepath to ")

    def _load_corpus(self):

        start_time = time.time()

        xml_files = []
        data = []
        data_wordnet_ids = []

        for brown_number in ["brown/tagfiles/", "brown1/tagfiles/", "brown2/tagfiles/"]:

            # Enumerate all xml files
            for r, d, f in os.walk(self.folderpath + brown_number):
                for file in f:
                    if '.xml' in file:
                        filepath = os.path.join(r, file)
                        print("Adding...", filepath)
                        xml_files.append(filepath)

        max_number_senses = []
        mean_number_of_wordset_senses = []

        # 1 Parse HTML into a tree
        for file in xml_files:
            # Get root
            tree = ET.parse(file)
            root = tree.getroot()
            # Get one-below-root
            node = root[0]
            # print("root is: ", root, node)
            for _sentence in node:
                sentence = _sentence[0]
                # print(sentence)
                words = []
                synset_ids = []
                for word in sentence:
                    word_txt = word.text
                    word_synsetid = word.get('wnsn')
                    # Get number of meanings from word-net

                    wordnet_synsets = wn.synsets(word_txt)
                    print("wordnet synset: ", word_synsetid, wordnet_synsets)
                    mean_number_of_wordset_senses.append(len(wordnet_synsets))

                    # Get number of senses...
                    # print("Word synsetid is: ", word_synsetid, word_txt)
                    words.append(word_txt)
                    synset_ids.append((word_synsetid))

                    # Only for dev purposes
                    max_number_senses.append((word_txt, word_synsetid))

                assert len(words) == len(synset_ids), (
                    "Lengths between synset ids and words don't match up",
                    len(words),
                    len(synset_ids)
                )

                print("Sentence is: ", words)
                data.append(words)
                data_wordnet_ids.append(synset_ids)

        print("Max number senses are")
        print(list(
            sorted([x for x in max_number_senses if (x[1] is not None) and (len(x[1]) <= 2)], key=lambda x: int(x[1])))[
              ::-1])

        assert len(data) == len(data_wordnet_ids), (
        "Number of wordnet ids and data do not match up ", len(data), len(data_wordnet_ids))

        # TODO: The following prunes "9;2" and "9;1", whatever this means. look it up!

        # Create a distribution over the number of senses
        sense_id_distribution = [int(x[1]) for x in max_number_senses if (x[1] is not None) and (len(x[1]) <= 2)]  # Take logarithm because otherwise we see mostly 1s ...

        # calculate histogram and write it out

        # Distribution of synsets

        mean_number_of_wordset_senses = [x / 2. for x in mean_number_of_wordset_senses]
        plt.hist(mean_number_of_wordset_senses, bins=70, range=[0, 70], log=True,
                 label="log distribution of mean synset lengths")
        plt.hist(sense_id_distribution, bins=70, range=[0, 70], log=True, label="log distribution of corpus synset id")
        plt.legend()
        plt.xlabel("Log Frequency")
        plt.show()

        # There is a left-skew of the data
        # This is probably an effect that the first few meanings of wordnet cover the more common terms (i.e. distribution of language)

        # Now do a bunch of stuff
        # Now you can apply the tokenizer for the individual sentences...
        print("Number of sentences are: ", len(data), time.time() - start_time)

        return data, data_wordnet_ids


if __name__ == "__main__":
    print("Loading example corpus!")
    corpus = CorpusSemCor()
