"""
    We will do pairplots based of predicted vs real semantic labels.
    It seems more accurate to predic this than the PoS in the end
    (which is quite weird..)
"""
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import args
from src.embedding_generators.bert_embeddings import BertEmbedding
from src.knowledge_graphs.wordnet import WordNetDataset
from src.models.cluster.chinesewhispers import MTChineseWhispers
from src.resources.corpus import Corpus
from src.resources.corpus_semcor import CorpusSemCor
from src.sampler.sample_embedding_and_sentences import get_bert_embeddings_and_sentences
from src.utils.create_experiments_folder import randomString
from src.utils.thesaurus_io import print_thesaurus
from src.visualize.pairplots import visualize_true_cluster_embeddings, visualize_predicted_cluster_embeddings

if __name__ == "__main__":
    print("Visualizing the predicted vs real wordnet meanings")

    polysemous_words = [
        # ' thought ', ' made ',  # ' was ',
        # ' only ', ' central ', ' pizza '
        # ' table ',
        # ' bank ',
        # ' cold ',
        # ' table ',
        # ' good ',
        # ' mouse ',
        # ' was ',
        # ' key ',
        # ' arms ',
        # ' was ',
        # ' thought ',
        # ' pizza ',
        # ' made ',

        ' act ',
        ' address ',
        ' back ',
        ' bear ',
        ' block ',
        ' catch ',

        ' crack ',

        ' face ',
        ' head ',
        ' march ',

        ' play ',
        ' roll ',
        ' saw ',
        ' tie ',
        ' train ',
        ' treat ',
        ' value ',
        ' visit ',
        ' wake ',
        ' work ',
        ' zone ',

    ]

    corpus = Corpus()
    corpus_semcor = CorpusSemCor()
    # ALso take the second corpus to check if th
    lang_model = BertEmbedding(corpus=corpus_semcor)
    wordnet_model = WordNetDataset()

    rnd_str = randomString(additonal_label=f"meaning_{args.dimred}{args.dimred_dimensions}")

    for tgt_word in polysemous_words:
        print("Looking at word", tgt_word)

        number_of_senses = wordnet_model.get_number_of_senses("".join(tgt_word.split()))

        # TODO: Could actually replace this with the other sample function ...

        print("Getting embeddings from BERT")
        tuples_semcor, true_cluster_labels_semcor, _ = get_bert_embeddings_and_sentences(model=lang_model, corpus=corpus_semcor, tgt_word=tgt_word)
        tuples, _, _ = get_bert_embeddings_and_sentences(model=lang_model, corpus=corpus, tgt_word=tgt_word)

        semcor_length = len(tuples_semcor)

        # Predict the clustering for the combined corpus ...
        X = np.concatenate(
            [x[1].reshape(1, -1) for x in (tuples_semcor + tuples)], axis=0
        )
        sentences = [
            x[0] for x in (tuples_semcor + tuples)
        ]

        X = StandardScaler().fit_transform(X)
        dimred_model = umap.UMAP(n_epochs=500, n_components=min(2, X.shape[0]))
        X = dimred_model.fit_transform(X)

        arguments = {
            'std_multiplier': 0.7303137869824812,
            'remove_hub_number': 106,
            'min_cluster_size': 7
        }
        # arguments = {
        #     'std_multiplier': 1.3971661365029329,
        #     'remove_hub_number': 0,
        #     'min_cluster_size': 31
        # }  # ( {'objective': 0.4569029268755458}
        # with PCA20
        cluster_model = MTChineseWhispers(arguments)  # ChineseWhispersClustering(**arguments)

        labels = cluster_model.fit_predict(X)

        assert len(X) == len(labels), (
            len(X), len(labels)
        )
        assert len(sentences) == len(labels), (
            len(sentences), len(labels)
        )
        assert semcor_length == len(true_cluster_labels_semcor), (
            semcor_length, len(true_cluster_labels_semcor)
        )

        # # TODO: Project again onto a lower dimension, so we can actually visualize this ...
        # pca_model = PCA(n_components=min(5, X.shape[1]), whiten=False)
        # X = pca_model.fit_transform(X)

        # Visualize only the first n items which correspond to the SemCor dataset...
        # Print the pairplots
        print("Visualizing predictions ...")
        visualize_true_cluster_embeddings(X[:semcor_length], rnd_str, tgt_word, true_cluster_labels_semcor, title="meaning")
        visualize_predicted_cluster_embeddings(X[:semcor_length], rnd_str, tgt_word, labels[:semcor_length], title="meaning")

        # Print the thesaurus
        print_thesaurus(
            sentences=sentences,
            clusters=labels,
            true_clusters=true_cluster_labels_semcor + ( (len(sentences) - semcor_length) * [-2, ]),
            word=tgt_word,
            savepath=rnd_str
        )


