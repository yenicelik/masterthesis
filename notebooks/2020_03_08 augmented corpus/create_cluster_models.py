"""
    This file creates all the cluster models for the designated words

    -> Result, about 60% of all samples include one of the words.
        This is high enough to proceed with the embedding split as suggested
        -> another splitup may be done by PoS!

    - 30% of all sentences include these words without pre-processing

"""
from collections import Counter

from src.resources.corpus import Corpus
from src.resources.split_words import get_polysemous_splitup_words

if __name__ == "__main__":
    print(""
          "We will now count the number of sentences "
          "which include one of the split-words "
          "within the corpus that we intend to use"
          )

    tgt_words = get_polysemous_splitup_words()
    tgt_words = set(tgt_words)
    print(tgt_words)

    # We will first apply this on the local development set to have an idea of the statistical sample

    # Load this within naive corpus ...

    corpus = Corpus()
    print("Total number of sentences is: ", len(corpus.sentences))

    print(corpus.sentences[0])

    # Count the number of occurences that any of these words occur
    def word_occurs_in_sentence(sentence):
        for word in tgt_words:
            if word in sentence:
                return True
        return False

    def negative_samples(sentences):
        out = Counter()
        for sentence in sentences:
            for word in tgt_words:
                if word in sentence:
                    continue
                else:
                    c = Counter(sentence.split())
                    out = out + c
                    # out.append(sentence.split())
        return out.most_common(100)

    positive_occurence = map(lambda x: word_occurs_in_sentence(x), corpus.sentences)
    positive_samples = len([x for x in positive_occurence if x])
    print("Percentage is: ", positive_samples / float(len(corpus.sentences)))

    print("Negative samples are")
    print(negative_samples(corpus.sentences))
