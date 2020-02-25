"""
    Importing simlex and creating an ordering from this dictionary ...
"""

import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def _get_simlex_ordering():
    basepath = os.getenv("BASEHOME")

    # Load the simlex
    wordlist = pd.read_csv(f"{basepath}/_MasterThesis/data/SimLex/SimLex-999.txt", sep="\t")

    orderings = dict()

    print(wordlist)
    print(wordlist.columns)

    # Do this by occurence of words (take as keys which occur often ...) start with them and go on

    for idx, word in wordlist.iterrows():
        tmp = dict(word)

        if tmp['word1'] not in orderings:
            orderings[tmp['word1']] = [
                (tmp['word2'], tmp['SimLex999'])
            ]
        else:
            orderings[tmp['word1']].append(
                (tmp['word2'], tmp['SimLex999'])
            )

        if tmp['word2'] not in orderings:
            orderings[tmp['word2']] = [
                (tmp['word1'], tmp['SimLex999'])
            ]
        else:
            orderings[tmp['word2']].append(
                (tmp['word1'], tmp['SimLex999'])
            )

    # Do perhaps add a second way to order these ... ?
    singular_keys = set()
    for key, val in orderings.items():
        print([x for x in orderings[key]])
        orderings[key] = sorted(orderings[key], key=lambda x: -1. * x[1])
        orderings[key] = [x[0] for x in orderings[key]]

        if len(orderings[key]) == 1:
            singular_keys.add(key)

    for key in singular_keys:
        del orderings[key]

    # Now drop all keys which only have a single item ...
    print(orderings)

    return orderings

def get_ordering(corpus):
    assert corpus in ("simlex")

    if corpus == "simlex":
        return _get_simlex_ordering()


if __name__ == "__main__":
    print("Hello")
    get_ordering("simlex")
