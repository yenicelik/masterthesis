"""
    Making prediction with best model
"""
import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.config import args
from src.embedding_generators.bert_embeddings import BertEmbedding
from src.models.cluster.chinesewhispers import MTChineseWhispers
from src.resources.corpus import Corpus
from src.resources.samplers import retrieve_data
from src.utils.create_experiments_folder import randomString
from src.utils.thesaurus_io import print_thesaurus


def visualize_true_cluster_embeddings(savepath, tgt_word, cluster_labels):

    dist = pd.DataFrame(
        data=X, columns=[f'lat{i}' for i in range(X.shape[1])]
    )
    colors = pd.DataFrame(
        data=cluster_labels, columns=['PoS']
    )
    dist = pd.concat([dist, colors], axis=1)

    # Apply some visualization based on seaborn distribution analysis
    sns.pairplot(dist, hue='PoS')

    plt.savefig(savepath + f'/matplotlib_pairplot_{tgt_word}_true.png')
    plt.show()

def visualize_predicted_cluster_embeddings(savepath, tgt_word, cluster_labels_pred):
    dist = pd.DataFrame(
        data=X, columns=[f'lat{i}' for i in range(X.shape[1])]
    )
    # Append to dataframe another dimension which is the
    colors = pd.DataFrame(
        data=pred_cluster_labels, columns=['PoS']
    )
    dist = pd.concat([dist, colors], axis=1)

    # Apply some visualization based on seaborn distribution analysis
    sns.pairplot(dist, hue='PoS')

    plt.savefig(savepath + f'/matplotlib_pairplot_{tgt_word}_pred.png')
    plt.show()

def create_thesaurus():
    """
        create the full thesaurus given the labels,
        and the
    :return:
    """
    pass

if __name__ == "__main__":
    print("Writing to thesaurus, and printing best model")

    # We first sample a few sentences which include a word
    polypos_words = [
        ' run ',
        ' well ',
        ' round ',
        # ' damn ',
        ' down ',
        ' table ',
        ' bank ',
        ' cold ',
        ' good ',
        ' mouse ',
        ' was ',
        ' key ',
        ' arms ',
        ' thought ',
        ' pizza ',
        # ' made ',
        # ' book '
    ]

    print("Creating corpus ...")
    corpus = Corpus()
    lang_model = BertEmbedding(corpus=corpus)

    rnd_str = randomString(additonal_label=f"PoS_{args.dimred}{args.dimred_dimensions}")

    print("Loading spacy")
    # Not sure if we need anything more from this
    nlp = spacy.load("en_core_web_sm")

    # TODO: Do this same visualization for the meaning-clusters!

    for tgt_word in polypos_words:
        X, sentences, labels = retrieve_data(nlp, tgt_word=tgt_word)

        kwargs = {'std_multiplier': 0.05224261597234525, 'remove_hub_number': 0,
                  'min_cluster_size': 18}  # {'objective': 0.21264054073371708}
        pred_cluster_labels = MTChineseWhispers(kwargs).fit_predict(X)

        assert len(labels) == len(pred_cluster_labels), (len(labels), len(pred_cluster_labels))
        assert len(sentences) == len(labels), (len(sentences), len(labels))

        # Pairplots!
        # Visualize the predictions
        print("Visualizing predictions ...")
        visualize_true_cluster_embeddings(rnd_str, tgt_word, labels)
        visualize_predicted_cluster_embeddings(rnd_str, tgt_word, pred_cluster_labels)
        # Visualize the true labels

        # Write to full thesaurus
        print_thesaurus(
            sentences=sentences,
            clusters=labels,
            word=tgt_word,
            true_clusters=pred_cluster_labels,
            savepath=rnd_str
        )

