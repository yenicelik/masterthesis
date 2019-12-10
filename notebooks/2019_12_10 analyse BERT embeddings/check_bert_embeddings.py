"""
    Check some properties for some polysemous words in BERT embeddings.
    Specifically, make analysis based on correlaiton analysis.

    Figure out perhaps ways to determine if we have a multimodal distribution,
    or if we have a wide-stretched distribution
"""
import pandas as pd
import numpy as np

from src.config import args
from src.embedding_generators.bert_embeddings import BertEmbedding
from src.knowledge_graphs.wordnet import WordNetDataset


def get_bert_embeddings_and_sentences(model):
    """
    :param model: A language model, which implements both the
        `_sample_sentence_including_word_from_corpus` and the
        `get_embedding`
    function
    :return:
    """

    out = []

    sampled_sentences = model._sample_sentence_including_word_from_corpus(word=tgt_word)

    sampled_embeddings = model.get_embedding(
        word=tgt_word,
        sample_sentences=sampled_sentences
    )

    print("\nSampled sentences are: \n")
    for sentence, embedding in zip(sampled_sentences, sampled_embeddings):
        print(sentence)
        embedding = embedding.flatten()
        print(embedding.shape)
        out.append(
            (sentence, embedding)
        )

    return out

def save_embedding_to_tsv(tuples, identifier):
    """
        Saving the embeddings and sampled sentence into a format that we can easily upload to tensorboard
    :param tuples: is a list of tuples (sentence, embeddings),
        where the embeddings are of size 768 (BERT)
    :return:
    """
    sentences = [x[0] for x in tuples]
    embeddings = [x[1] for x in tuples]

    embeddings = [x.reshape(1, -1) for x in embeddings]

    embeddings_matrix = np.concatenate(embeddings, axis=0)
    print("Embeddings matrix has shape: ", embeddings_matrix.shape)

    df = pd.DataFrame(data={
        "sentences": sentences
    })

    print(df.head())

    # TODO: Handle an experiment-creator for this, which reads and writes to a opened directory..

    assert len(df) == embeddings_matrix.shape[0], ("Shapes do not conform somehow!", len(df), embeddings_matrix.shape)

    df.to_csv(identifier + "{}_labels.tsv".format(len(sentences)), header=True, sep="\t")
    np.savetxt(fname=identifier + "{}_values.tsv".format(len(sentences)), X=embeddings_matrix, delimiter="\t")

def cluster_embeddings(tuples):
    """
        Taking the embeddings, we cluster them (using non-parameteric algortihms!)
        using different clustering algorithms.

        We then return whatever we have
    :return:
    """
    # The first clustering algorithm will consists of simple
    # TODO: Perhaps best to use the silhouette plot for choosing the optimal numebr of clusters...
    embedding_matrix = np.concatenate([x.reshape(1, -1) for x in tuples], axis=0)
    print("Embeddings matrix is: ", embedding_matrix.shape)

    # TODO: Find a good way to evaluate how many clusters one meaning is in

    # perhaps run k-means a few times, and just output the total error? mean and stddev...
    for k in [1, 2, 3, 5, 7, 9, 11, 13]: # , 40, 100, 200 # Do ablation study later on, which items have the highest loss..
        # Let's go up until 10 clusters maximum..

        print("Testing clusters... ", k)

        for i in range(100):
            # Cluster the matrix into different items
            # output the prediction error
            pass


if __name__ == "__main__":
    print("Sampling random sentences from the corpus, and their respective BERT embeddings")

    # Make sure that the respective word does not get tokenized into more tokens!
    lang_model = BertEmbedding()
    wordnet_model = WordNetDataset()

    # Check out different types of polysemy?

    # The word to be analysed
    polysemous_words = [" bank ", " table ", " subject ", " key ", " pupil ", " book ", " mouse "]

    for tgt_word in polysemous_words:
        # tgt_word = " bank " # we add the space before and after for the sake of

        number_of_senses = wordnet_model.get_number_of_senses(tgt_word)

        tuples = get_bert_embeddings_and_sentences(model=lang_model)


        print("Number of senses are: ", np.exp(number_of_senses))

        save_embedding_to_tsv(tuples, identifier=tgt_word + "_")
