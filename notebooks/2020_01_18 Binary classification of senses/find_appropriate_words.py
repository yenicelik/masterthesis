"""
    We will find a few words which
      have a high number of samples for the 2 most common meanings.

    check which words and which wordnet ids have appropriate
    can be used for this experiment
    - we require a high number of samples (defined by a cutoff)
    - and at least 2 distinct wordnet senses
"""

from collections import Counter

from src.resources.corpus_semcor import CorpusSemCor


def collect_high_occuring_senses(semcor_corpus: CorpusSemCor):
    """
        Retrieves all the senses whose number
        of samples is above a certain threshold in the SemCor dataset
    :return:
    """
    flattened_wordnet_ids = [j for sub in semcor_corpus.synset_ids for j in sub if ((j is not None) and len(j) <= 2)]
    counter = Counter(flattened_wordnet_ids)
    print(counter)

def get_words_and_classes(corpus, cutoff):
    # print("\n\n\n")
    word_sense_clusters = corpus.word_sense_tuples
    counter = Counter(word_sense_clusters)
    counter = Counter(el for el in counter.elements() if counter[el] >= cutoff)

    # print("List of candidates (not pair-filtered are", counter)

    # Now remove all word-meanings which have no corresponding pair ...
    pair_list_counter = Counter([x[0][0] for x in counter.most_common()])
    # Removing all items where not enough pairs are present
    pair_list_counter = [el for el in pair_list_counter.elements() if pair_list_counter[el] >= 2]

    counter = Counter(el for el in counter.elements() if el[0] in pair_list_counter)
    # print(counter)
    # print(list(counter.keys()))

    out = list(counter.keys())

    return out

if __name__ == "__main__":
    print("Getting semcor samples to be separated")
    corpus = CorpusSemCor()

    word_pairs = get_words_and_classes(
        corpus=corpus,
        cutoff=10
    )
    print(sorted(list(set(word_pairs)), key=lambda x: x[0]))

    # Check a 4-way clustering perhaps ... for the words that work here ...
