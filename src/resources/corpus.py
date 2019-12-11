
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

class Corpus:

    @property
    def sentences(self):
        return self.data

    def __init__(self):

        print("Starting corpus")

        self.filepath = os.getenv("EN_CORPUS")

        self.data = self._load_corpus()

    def _load_corpus(self):
        """
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
