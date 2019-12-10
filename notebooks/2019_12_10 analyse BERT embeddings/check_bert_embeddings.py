"""
    Check some properties for some polysemous words in BERT embeddings.
    Specifically, make analysis based on correlaiton analysis.

    Figure out perhaps ways to determine if we have a multimodal distribution,
    or if we have a wide-stretched distribution
"""
import pandas as pd
from src.embedding_generators.bert_embeddings import BertEmbedding


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

def save_embedding_to_tsv(tuples):
    """
        Saving the embeddings and sampled sentence into a format that we can easily upload to tensorboard
    :param tuples: is a list of tuples (sentence, embeddings),
        where the embeddings are of size 768 (BERT)
    :return:
    """
    sentences = [x[0] for x in tuples]
    embeddings = [x[1] for x in tuples]

    df = pd.DataFrame(data={
        "sentences": sentences,
        "embeddings": embeddings
    })

    print(df.head())



if __name__ == "__main__":
    print("Sampling random sentences from the corpus, and their respective BERT embeddings")

    # The word to be analysed
    tgt_word = " bank " # we add the space before and after for the sake of
    # other target words could include " table ", " subject ", " key ", " pupil "

    # Make sure that the respective word does not get tokenized into more tokens!
    lang_model = BertEmbedding()

    tuples = get_bert_embeddings_and_sentences(model=lang_model)
    save_embedding_to_tsv(tuples)

