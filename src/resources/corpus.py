
import os
import re
import time

from dotenv import load_dotenv

from src.config import args

load_dotenv()

class Corpus:

    def sample_sentence_including_word_from_corpus(self, word):
        """
            The Corpus is some corpus that
            Probably better ways to parse this
        :return:
        """
        # Find all words, together with their respective synset id ...
        # -> Could best parallelize this ...
        # Strip word of all whitespaces
        out = []
        word = word.replace(" ", "")
        # We assume that self.corpus is a list of lists of words (not a list of sentences!)
        for i in range(len(self.sentences)):
            # Need to join the sentences ...
            idx = self.sentences[i].find(word)
            if idx == -1:
                continue
            out.append(
                "[CLS] " + self.sentences[i]
            )

        # Keep only top samples
        out = out[:args.max_samples]

        # out = ["[CLS] " + x for x in self.corpus.sentences if word in x][:args.max_samples]
        # Must not allow any words that happen less than 5 times!
        assert len(out) >= 1, ("Not enough examples found for this word!", out, word)
        # Perhaps best not to simply change the function signature, but to make it an attribute
        return out, [-1, ] * len(out)

    @property
    def sentences(self):
        return self.data

    def __init__(self):

        print("Starting corpus")

        self.filepath = os.getenv("EN_CORPUS")

        self.data = self._load_corpus()

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
        print(data[:100])

        # Still returning this sample: [CLS] Not without more information           of the person who hacked into the foundation 's bank account , but -- .
        # Which is strongly incorrect I believe!

        # Now you can apply the tokenizer for the individual sentences...
        print("Number of sentences are: ", len(data), time.time() - start_time)

        return data

if __name__ == "__main__":
    print("Loading example corpus!")
    corpus = Corpus()

