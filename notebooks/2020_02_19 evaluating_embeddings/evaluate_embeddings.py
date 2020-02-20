"""
    Evaluate the embeddings
"""
import spacy
import numpy as np
import pandas as pd

from src.config import args
from src.embedding_generators.bert_embeddings import BertEmbedding
from src.resources.corpus import Corpus
from src.resources.samplers import retrieve_data
from src.resources.similarity.get_corpus_words import get_words_in_benchmarks
from src.utils.create_experiments_folder import randomString

def static_principal_component(matr):
    # Calculate a static principle component from the vector



def predict_rank():
    pass

if __name__ == "__main__":

    print("Creating our own contextualized embeddings, and then evaluating these using the gold-standard datasets")

    rnd_str = randomString(additonal_label=f"_wordembeddings_{args.dimred}_{args.dimred_dimensions}_whiten{args.pca_whiten}_norm{args.normalization_norm}")
    loadpath = "/Users/david/GoogleDrive/_MasterThesis/notebooks/2020_02_19 evaluating_embeddings/_wordembeddings_none_768_whitenFalse_normcedtqifitj/"

    nlp = spacy.load("en_core_web_sm")

    for word in get_words_in_benchmarks("simlex"):

        # For each item in the ordering-dictionary, get the items
        print(f"Sampling word {word}")
        matr = np.load(loadpath)

        # Grab vectors, and do whatever you wanna do to get the vectors ...

        # First of all do some clustering ...

        # Do some clustering, and select out some samples ... (perhaps do some preprocessing first)

        # for tgt_word in polypos_words:
        #     X, sentences, labels = retrieve_data(nlp, tgt_word=tgt_word)
        #
        #     # Save as a numpy array
        #     np.savetxt(rnd_str + f"/{tgt_word}_matr.tsv", X, delimiter="\t")
        #     pd.DataFrame(
        #         {
        #             "sentece": sentences,
        #             "labels": labels
        #         }
        #     ).to_csv(rnd_str + f"/{tgt_word}_labels.tsv", sep="\t")

        # First of all, get the words in the benchmark
