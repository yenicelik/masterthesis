"""
    Making prediction with best model
"""
import spacy
from sklearn.decomposition import PCA

from src.config import args
from src.embedding_generators.bert_embeddings import BertEmbedding
from src.models.cluster.chinesewhispers import MTChineseWhispers
from src.models.cluster.dbscan import MTDbScan
from src.resources.corpus import Corpus
from src.resources.samplers import retrieve_data
from src.utils.create_experiments_folder import randomString
from src.utils.thesaurus_io import print_thesaurus
from src.visualize.pairplots import visualize_true_cluster_embeddings, visualize_predicted_cluster_embeddings

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

        if args.dimred == "pca" and args.dimred_dimensions == 20:
            kwargs = {'std_multiplier': 0.05224261597234525, 'remove_hub_number': 0,
                      'min_cluster_size': 18}  # {'objective': 0.21264054073371708}
            pred_cluster_labels = MTChineseWhispers(kwargs).fit_predict(X)
        elif args.dimred == "umap" and args.dimred_dimensions in (2, 4):
            kwargs = {'eps': 1.8823876589536668, 'min_samples': 16, 'metric': 'chebyshev'}
            pred_cluster_labels = MTDbScan(kwargs).fit_predict(X)
        else:
            assert False, ("You should run the model selection experiment before doing this first!")

        assert len(labels) == len(pred_cluster_labels), (len(labels), len(pred_cluster_labels))
        assert len(sentences) == len(labels), (len(sentences), len(labels))

        # TODO: Project again onto a lower dimension, so we can actually visualize this ...
        pca_model = PCA(n_components=min(5, X.shape[1]), whiten=False)
        X = pca_model.fit_transform(X)

        # Pairplots!
        # Visualize the predictions
        print("Visualizing predictions ...")
        visualize_true_cluster_embeddings(X, rnd_str, tgt_word, labels, title="PoS")
        visualize_predicted_cluster_embeddings(X, rnd_str, tgt_word, pred_cluster_labels, title="PoS")
        # Visualize the true labels

        # Write to full thesaurus
        print_thesaurus(
            sentences=sentences,
            clusters=labels,
            word=tgt_word,
            true_clusters=pred_cluster_labels,
            savepath=rnd_str
        )

