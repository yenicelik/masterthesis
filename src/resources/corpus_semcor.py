"""
    Implements any logic which takes in and processes SemCor logic

    These are all the words for which we have ids

"""
import os
import time
from collections import Counter

from nltk.stem import PorterStemmer

import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn

from dotenv import load_dotenv

load_dotenv()

from src.config import args


def _get_all_top_words(n=5000):
    filepath = os.getenv('TOP_20000_EN')
    with open(filepath, 'r') as fp:
        out = fp.read()
    out = out.split()[:n]
    out = set(out)
    return out


class CorpusSemCor:

    def sample_sentence_including_word_from_corpus(self, word, n=None):
        """
        :param word: The word for which we want to find example words
        :return:
        """

        # TODO: The stemmed option should be a variable!
        # Behavior will change based on stemming ...

        # Should I stem retrieve from stemmed dictionary before feeding in?
        # for these ...?
        word = word.replace(" ", "") # because this time we have lists of lists of tokens
        # stemmed_word = self.stemmer.stem(word)
        # pad front and back by a " "
        # print("Stemmed word", stemmed_word)

        out = []  # Will cover a list of sentences which contain the respective word
        out_idx = []  # Will cover a list of which cluster the given sentence belongs to

        print("Looking for word ...", word)

        # These sentences are not word-delimited!!!
        for i in range(len(self.data)):
            # Iterate through all sentences
            # TODO: Decide when it should be stemmed and when not!
            if args.stemsearch:
                query_sentence = [self.stemmer.stem(x) for x in self.data[i]]
            else:
                query_sentence = [x for x in self.data[i]]
            # print("Gotta check if we should stem the word or not")
            # if 'was' in query_sentence or 'was' in query_sentence_:
            #     print("In sentence!")
            #     print(query_sentence)
            #     print(query_sentence_)
            try:
                idx = query_sentence.index(word)
            except:
                continue

            synset_id = self.data_wordnet_ids[i][idx]
            output_sentence = " ".join(self.data[i])

            if args.verbose == 2:
                print("Query sentence")
                print(query_sentence)
                print("Output sentence")
                print(output_sentence)

            # replace all "_" by " "
            output_sentence = output_sentence.replace("_", " ")

            # also remove all special characters
            output_sentence = output_sentence.replace("`", "").replace("'", "")

            out.append(
                "[CLS] " + output_sentence
            )
            out_idx.append(synset_id)

        # Keep only top samples
        out = out[:args.max_samples]
        out_idx = out_idx[:args.max_samples]

        print("Number of sample sentences found", len(out))

        # out = ["[CLS] " + x for x in self.corpus.sentences if word in x][:args.max_samples]
        # Must not allow any words that happen less than 5 times!
        assert len(out) >= 1, ("Not enough examples found for this word!", out, word)
        # Perhaps best not to simply change the function signature, but to make it an attribute
        return out, out_idx

    @property
    def sentences(self):
        return self.data

    @property
    def synset_ids(self):
        return self.data_wordnet_ids

    @property
    def word_sense_tuples(self):
        if self.words_ is None:
            out = []
            assert len(self.sentences) == len(self.synset_ids)
            for sentence, senses in zip(self.sentences, self.synset_ids):
                assert len(sentence) == len(senses)
                for word, sense in zip(sentence, senses):
                    # append these to word
                    if sense is None:
                        continue
                    if len(sense) > 2:
                        continue
                    out.append(
                        (word, sense)
                    )

            self.words_ = out

        return self.words_

    def __init__(self):

        print("Starting corpus")
        self.stemmer = PorterStemmer()

        self.folderpath = os.getenv("SEMCOR_CORPUS")

        self.data, self.data_wordnet_ids = self._load_corpus()

        print(type(self.data), type(self.data_wordnet_ids))

        self.words_ = None

    def xml2arrays(self, filepath):
        print("Turning the filepath to ")

    def _load_corpus(self, visualize_distributions=False, verbose=False):

        most_common_words = _get_all_top_words()

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
                        # print("Adding...", filepath)
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
                    # print("wordnet synset: ", word_synsetid, wordnet_synsets)
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

                if args.verbose == 2:
                    print("Sentence is: ", words)
                data.append(words)
                data_wordnet_ids.append(synset_ids)

        # print("Max number senses are")
        # print(list(
        #     sorted([x for x in max_number_senses if (x[1] is not None) and (len(x[1]) <= 2)], key=lambda x: int(x[1])))[
        #       ::-1])

        assert len(data) == len(data_wordnet_ids), (
            "Number of wordnet ids and data do not match up ", len(data), len(data_wordnet_ids))

        # TODO: The following prunes "9;2" and "9;1", whatever this means. look it up!
        # This will prune the set of all possible synsets

        # Create a distribution over the number of senses
        sense_id_distribution = [int(x[1]) for x in max_number_senses if (x[1] is not None) and (
                len(x[1]) <= 2)]  # Take logarithm because otherwise we see mostly 1s ...

        # calculate histogram and write it out

        # Finally, find all words for which we can query the corpus with multiple meanings ...
        # We discard and wordnet id's including semicolons for easier processing...

        # Filter most "varied" occurrenses...

        # Only keep top 1000 english words
        unique_word_pairs = set([(x[0].lower(), x[1]) for x in max_number_senses if
                                 (x[1] is not None) and (len(x[1]) <= 2) and (x[0] in most_common_words)])  # Take logarithm because otherwise we see mostly 1s ...

        if verbose:
            print("Queryable words in dictioanry are")
            print(unique_word_pairs)

        unique_words = set([x[0] for x in unique_word_pairs])
        # Include only items which have more than one item
        if verbose:
            print("Queryable concepts are")
            print(unique_words)

        global_meaning_occurence = [self.stemmer.stem(x[0]) for x in unique_word_pairs]

        # Look for the corresponding wordnet meanings ...

        # get a counter, and check which words are most varied in this corpus (w.r.t. semantic meaning)
        if verbose:
            print("Global meanings are")
            c = Counter(global_meaning_occurence)
            for word, semcor_senses in c.most_common():
                no_wordnet_meanings = len(wn.synsets(word))
                ratio = (semcor_senses + 1) / float(no_wordnet_meanings + 1)
                if ratio > 1.1:
                    continue
                if ratio < 0.9:
                    continue
                print("--- {:.2f}, {}, {}, {} ---".format(
                    ratio,
                    semcor_senses,
                    no_wordnet_meanings,
                    word
                ))

        # Filter our unique words which are not in the top 20'000 english words
        # Create a collections dict

        if visualize_distributions:
            # Distribution of synsets
            mean_number_of_wordset_senses = [x / 2. for x in mean_number_of_wordset_senses]
            plt.hist(mean_number_of_wordset_senses, bins=70, range=[0, 70], log=True,
                     label="log distribution of mean synset lengths")
            plt.hist(sense_id_distribution, bins=70, range=[0, 70], log=True,
                     label="log distribution of corpus synset id")
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
    # _get_all_top_words()
    corpus = CorpusSemCor()
    print("\n\n\n\n\n")
    corpus.sample_sentence_including_word_from_corpus('central')
    print("\n\n\n\n\n")
    corpus.sample_sentence_including_word_from_corpus('have')
