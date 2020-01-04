from src.config import args


def get_bert_embeddings_and_sentences(model, corpus, tgt_word):
    """
    :param model: A language model, which implements both the
        `_sample_sentence_including_word_from_corpus` and the
        `get_embedding`
    function
    :return:
    """

    out = []

    if args.verbose >= 1:
        print("Retrieving example sentences from corpus")
    sampled_sentences, sampled_cluster_true_labels = corpus.sample_sentence_including_word_from_corpus(word=tgt_word)

    if args.verbose >= 1:
        print("Retrieving sampled embeddings from BERT")
    sampled_embeddings = model.get_embedding(
        word=tgt_word,
        sample_sentences=sampled_sentences
    )

    if args.verbose >= 2:
        print("\nSampled sentences are: \n")
    for sentence, embedding, cluster_label in zip(sampled_sentences, sampled_embeddings, sampled_cluster_true_labels):
        if args.verbose >= 2:
            print(sentence)
        embedding = embedding.flatten()
        if args.verbose >= 2:
            print(embedding.shape)
        out.append(
            (sentence, embedding, cluster_label)
        )

    return out, sampled_cluster_true_labels