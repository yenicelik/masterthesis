"""
    For the similarity corpora (where we might want to pre-fetch the vectors),
    for this, we get the load the items from the dictionary ...
"""

import os
import pandas as pd
from os.path import expanduser

from dotenv import load_dotenv

load_dotenv()


def get_words_in_benchmarks(benchmark):
    assert benchmark is not None
    assert benchmark in ("simlex")

    if benchmark == "simlex":

        basepath = os.getenv("BASEHOME")

        wordlist_1 = pd.read_csv(f"{basepath}/_MasterThesis/data/SimLex/SimLex-999.txt", sep="\t")[
            'word1'].values.tolist()
        wordlist_2 = pd.read_csv(f"{basepath}/_MasterThesis/data/SimLex/SimLex-999.txt", sep="\t")[
            'word2'].values.tolist()
        return set(wordlist_1).union(set(wordlist_2))

    else:
        assert False


if __name__ == "__main__":
    print("Starting the benchmarks")

    words = get_words_in_benchmarks("simlex")
    print("words are")
    print(words)
    print(len(words))
