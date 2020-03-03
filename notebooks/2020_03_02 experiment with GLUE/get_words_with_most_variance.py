"""
    Small experiment to check which words from the BERT tokenizer have most variance.
    We then want to use these to check the most variance
"""
import re
import nltk
import numpy as np
import pandas as pd

from nltk.corpus import stopwords

from transformers import BertTokenizer

from src.config import args
from src.embedding_generators.bert_embeddings import BertEmbedding
from src.resources.corpus import Corpus
from src.resources.samplers import sample_embeddings_for_target_word
from src.utils.create_experiments_folder import randomString


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

    # Save intermediate csv
    csv_splitoff = 2000

    # Generate foldere to save this in
    rnd_str = randomString(additonal_label=f"_mean_std_vector_{args.dimred}_{args.dimred_dimensions}_whiten{args.pca_whiten}_norm{args.normalization_norm}/")

    out = []
    df = None
    # Sample from BERT
    for word in bert_words:

        if len(out) % csv_splitoff == 0:
            # Write to csv and reset next csv part
            if df is not None:
                df.to_csv(rnd_str + "mean_std_vectors.csv")
            df = pd.DataFrame(
                out, columns=['word', 'wordnet_senses', 'semcor_senses', 'mean_vec', 'std_vec']
            )
            out = []

        tgt_word = f' {word} '  # So we only take words that are not part of any other word ...
        number_of_senses, X, true_cluster_labels, known_indices, sentences = sample_embeddings_for_target_word(
            tgt_word=tgt_word
        )

        # Prune if X is not enough ..
        if X.shape[0] < 500:  # Keep only top matches
            continue

        semcor_senses = np.unique(true_cluster_labels) - 1

        # Assert that the number
        X_mean = np.mean(X, axis=0).flatten()
        X_std = np.stddev(X, axis=0).flatten()

        assert len(X_mean.shape) == 1, (X_mean.shape)
        assert len(X_std.shape) == 1, (X_std.shape)

        tpl = (
            tgt_word,
            number_of_senses,
            semcor_senses,
            X_mean.tolist(),
            X_std.tolist()
        )

        print("tuple is: ", tpl)

        out.append(tpl)

    # Calculate mean and stddev vectors per dimension

    # Save the to CSV
