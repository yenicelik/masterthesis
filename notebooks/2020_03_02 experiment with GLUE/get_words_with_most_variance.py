"""
    Small experiment to check which words from the BERT tokenizer have most variance.
    We then want to use these to check the most variance
"""
import re
import nltk
from nltk.corpus import stopwords

from transformers import BertTokenizer

from src.embedding_generators.bert_embeddings import BertEmbedding
from src.resources.corpus import Corpus
from src.resources.samplers import retrieve_data_pos


def has_number(x):
    return bool(re.search(r'\d', x))

if __name__ == "__main__":
    print("Getting the variance")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Getting tokenizer words..")

    stopwords = set(stopwords.words('english'))

    bert_words = [
        x for x in tokenizer.vocab.keys()
        if ('unused' not in x) and
           (len(x) > 1) and  # should not be a special symbol
           (x not in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']) and  # should not be a pre-reserved token
           (x not in stopwords) and  # should not be a stopword
           ('##' not in x) and  # should not be splittable tokens
           (x.isalnum()) and  # should be alphanumerical
           (not has_number(x))  # should not contain numbers
    ]

    print(bert_words)
    print(len(bert_words))
    # sample from BERT, and analyse which ones have highest variance
    # save mean and variance in to vector
    print("Creating corpus ...")
    corpus = Corpus()
    lang_model = BertEmbedding(corpus=corpus)


    # Sample from BERT
    for word in bert_words:
        X, sentences, labels = retrieve_data_pos(nlp, tgt_word=tgt_word)

    # Calculate mean and stddev vectors per dimension

    # Save the to CSV


