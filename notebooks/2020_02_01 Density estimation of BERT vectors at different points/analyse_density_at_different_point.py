"""
    Does some analysis including density estimation at different points
"""
import numpy as np
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.embedding_generators.bert_embeddings import BertEmbedding
from src.knowledge_graphs.wordnet import WordNetDataset
from src.resources.corpus import Corpus
from src.resources.corpus_semcor import CorpusSemCor
from src.sampler.sample_embedding_and_sentences import get_bert_embeddings_and_sentences
from src.utils.create_experiments_folder import randomString

if __name__ == "__main__":
    print("Doing density calculation and estimation")

    # Sample the semword corpus
    corpus = Corpus()
    corpus_semcor = CorpusSemCor()
    # ALso take the second corpus to check if th
    lang_model = BertEmbedding(corpus=corpus_semcor)
    wordnet_model = WordNetDataset()

    savepath = randomString()

    # TODO: Install the yaml library
    with open("../../test_words.yaml", 'r') as stream:
        try:
            # print(yaml.safe_load(stream))
            polysemous_words = yaml.safe_load(stream)
            polysemous_words = polysemous_words['words']
        except yaml.YAMLError as exc:
            print(exc)
            exit(-1)

    print("Polysemous words are: ", polysemous_words)

    for tgt_word in polysemous_words:
        print("Looking at word", tgt_word)

        number_of_senses = wordnet_model.get_number_of_senses("".join(tgt_word.split()))

        print("Getting embeddings from BERT")
        tuples_semcor, true_cluster_labels_semcor = get_bert_embeddings_and_sentences(model=lang_model, corpus=corpus_semcor, tgt_word=tgt_word)
        tuples, _ = get_bert_embeddings_and_sentences(model=lang_model, corpus=corpus, tgt_word=tgt_word)

        print("semcor tuples and normal tuples are")

        print(tuples_semcor)
        print(len(tuples_semcor))

        print(tuples)
        print(len(tuples))

        # Predict the clustering for the combined corpus ...
        X = np.concatenate(
            [x[1].reshape(1, -1) for x in (tuples_semcor + tuples)], axis=0
        )
        sentences = [
            x[0] for x in (tuples_semcor + tuples)
        ]

        # Labels also should be a python list
        # Do some analysis on X

        X = StandardScaler().fit_transform(X)

        # # Now make the X positive!
        # if np.any(X < 0):
        #     X = X - np.min(X) # Should we perhaps do this feature-wise?
        #     print("adding negative values to X")
        #     print(np.min(X))

        # Instead of PCA do NMF?
        dimred_model = TSNE(n_components=min(20, X.shape[0]))
        # dimred_model = PCA(n_components=min(20, X.shape[0]), whiten=False)
        X = dimred_model.fit_transform(X)

        dist = pd.DataFrame(
            data=X, columns=[f'lat{i}' for i in range(X.shape[1])]
        )

        # Apply some visualization based on seaborn distribution analysis
        sns.pairplot(dist)
        plt.savefig(savepath + f'/matplotlib_pairplot_{tgt_word}.png')
        plt.show()

        # Do some calculation how much volume the space consumes (euclidean min/max), or also cosine distances
        
        # TODO: We can perhaps sample across a grid, and then reject when the density at that pont is too little?


        # You can also calculate this per wordnet meaning (there was a cone-paper, which was calculating this...)

        # Figure out how to proceed (i.e. with kNN with annealing, check if there's possible librares for this.

        # For each "obvious" cluster, you can perhaps sample a certain number



