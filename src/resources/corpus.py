
import os
import re
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
        with open(self.filepath) as f:
            data = f.read().replace('\n', '')

        # Is this processing enough? Do we need to add any additional for "David:" (the punctuation)
        data = data.replace(';', '.').replace(' - ', '.').replace('?', '.').replace('!', '.')
        data = data.split('.')
        # data = re.findall(r"[\w']+", data)
        # data = re.split('; |. |? |! |- ', data)
        data = [sentence.split(" ") for sentence in data]
        data = [[x.replace("@!", "").replace("@", "").replace("#", "") for x in sentence if (not x.startswith("@@")) and x != ""] for sentence in data ]
        data = [x for x in data if x != []]
        data = [" ".join(sentence) + "." for sentence in data]

        # Still returning this sample: [CLS] Not without more information           of the person who hacked into the foundation 's bank account , but -- .
        # Which is strongly incorrect I believe!

        # Now you can apply the tokenizer for the individual sentences...
        print("Number of sentences are: ", len(data))

        return data

if __name__ == "__main__":
    print("Loading example corpus!")
    corpus = Corpus()

