"""
    We will find a few words which
      have a high number of samples for the 2 most common meanings.

    (1) Reduce dimensionality to a a number below the number of features.
       (make sure variance kept is not too low ...)

    (2) Check if the two meanings are separable.

    Do this for multiple sense-pairs (100 or so?), so you have statistically significant results ..)

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

if __name__ == "__main__":
    print("Getting semcor samples to be separated")
    corpus = CorpusSemCor()

    print("Rocpus data is. ")
    # print(corpus.data)
    word_sense_clusters = corpus.word_sense_tuples
    counter = Counter(word_sense_clusters)
    counter = Counter(el for el in counter.elements() if counter[el] >= 80)
    print(counter.most_common())
    print(len(counter.most_common()))

    # collect_high_occuring_senses(corpus)

    # Now remove all word-meanings which have no corresponding pair ...
    pair_list_counter = Counter([x[0][0] for x in counter.most_common()])
    print("Pair list is: ")
    print(pair_list_counter)
    # Removing all items where not enough pairs are present
    pair_list_counter = Counter(el for el in pair_list_counter.elements() if pair_list_counter[el] >= 2)
    print(pair_list_counter)
    print(len(pair_list_counter))
