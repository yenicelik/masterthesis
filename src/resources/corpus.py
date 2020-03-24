
import os
import re
import time

from dotenv import load_dotenv

from src.config import args

load_dotenv()

class Corpus:

    def sample_sentence_including_word_from_corpus(self, word, n=None):
        """
            The Corpus is some corpus that
            Probably better ways to parse this
        :return:
        """
        if n is None:
            n = args.max_samples
        if n == 0:
            n = 1
        # Find all words, together with their respective synset id ...
        # -> Could best parallelize this ...
        # Strip word of all whitespaces
        print("FROM naive sampling following word...", word)
        assert n > 1
        print("Number of samples to keep is: ", n)
        out = []
        for x in self.data:
            if word in x:
                out.append(
                    "[CLS] " + x + " [SEP]"
                )
        out = out[:n]
        # print("Self data is. ")
        # print(out)
        # print(self.data)
        # Must not allow any words that happen less than 5 times!
        assert len(out) >= 1, ("Not enough examples found for this word!", out, word)
        # Perhaps best not to simply change the function signature, but to make it an attribute
        return out, [-1, ] * len(out)

    @property
    def sentences(self):
        # Tokenize sentences ...
        # need to be tokenized by the BERT tokenizer ...
        # no, shouldn't be tokenized by BERT tokenizer ...
        return self.data

    def __init__(self):

        print("Starting corpus")

        self.filepath = os.getenv("EN_CORPUS")
        self.stemmer = None

        self.data = self._load_corpus()
        # TODO: BERT does not allow for sentences which have more than 512 tokens!!!
        self.data = [x for x in self.data if len(x.split()) < (args.max_samples - 100)]

    def _load_corpus(self):
        """^
            Creates a python list.
            Each element of the list is a
        :return:
        """
        start_time = time.time()

        with open(self.filepath) as f:
            data = f.read()

        print("Reading file...", len(data), time.time() - start_time)


        # Is this processing enough? Do we need to add any additional for "David:" (the punctuation)
        # data = data.replace(';', '.').replace(' - ', '.').replace('?', '.').replace('!', '.')
        data = data.split('\n')
        print("Extracting words..", time.time() - start_time)
        # data = [sentence.split(" ") for sentence in data]
        # print("Extracting special characters...", time.time() - start_time)
        # data = [[x.replace("@!", "").replace("@", "").replace("#", "") for x in sentence if (not x.startswith("@@")) and x != ""] for sentence in data ]
        # print("Taking out empty values", time.time() - start_time)
        # data = [x for x in data if x != []]
        # print("Joining into one sentence again...", time.time() - start_time)
        # data = [" ".join(sentence) + "." for sentence in data]

        print("Data is: ", time.time() - start_time)
        # print(data[:100])

        # Still returning this sample: [CLS] Not without more information           of the person who hacked into the foundation 's bank account , but -- .
        # Which is strongly incorrect I believe!

        # Now you can apply the tokenizer for the individual sentences...
        print("Number of sentences are: ", len(data), time.time() - start_time)

        return data

if __name__ == "__main__":
    print("Loading example corpus!")
    corpus = Corpus()

