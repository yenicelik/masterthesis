"""
    Implements word-embeddings that are generated by the BERT lanauge model.
    We will use the standard BERT language model (not distilled), as this is better captured by language

    # What is the difference between BertMaskedModel and BertModel

    Seems to be a good resource on how to extract word-embeddings
        https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#23-segment-id
"""
import time

import torch

from src.config import args
from src.functional.string_searchers import find_all_indecies_subarray
from src.language_models.model_wrappers.bert_wrapper import BertWrapper
from src.resources.corpus import Corpus


class BertEmbedding:

    @property
    def sentence_sample(self):
        """
            The number of sentences to sample for each word,
            s.t. we can sample a probability distribution from the context-word

            We do this for a few language only (i.e. German, English, Turkish??),
            and we use the Wikipedia corpus, as this is available in many language "for free"

            # TODO: ->
                Following Artetxe, can we sample a few sentences (using multinomial sampling),
                and then use this for the sample-embedding?

            NOTE: I will start with a small corpus first, and then slowly scale up.
            A larger corpus may be included on Leonhard or so (or on my private cluster lol)

            Will download the corpus from here:
                https://www.english-corpora.org/
                https://www.corpusdata.org/

            # Find out what the stupid paper used as a corpora..
        :return:
        """
        pass

    def __init__(self):
        self.corpus = Corpus()
        self.max_samples = args.max_samples

        self.wrapper = BertWrapper()
        self.bert_layer = 1  # Which BERT layer to take the embeddings from

    def _sample_sentence_including_word_from_corpus(self, word):
        """
            The Corpus is some corpus that
            Probably better ways to parse this
        :return:
        """
        #
        out = ["[CLS] " + x for x in self.corpus.sentences if word in x][:args.max_samples]
        # Must not allow any words that happen less than 5 times!
        assert len(out) >= 1, ("Not enough examples found for this word!", out, word)
        return out

    def get_embedding(self, word, sample_sentences=None):
        """
            For a given word (or concept), we want to generate an embedding.
            The question is also, do we generate probabilistic embeddings or static ones (by pooling..?)
            TODO: Perhaps the probabilistic embeddings can overcome the static ones?
        :param token:
        :return:
        """
        assert isinstance(word, str), ("Word is not of type string", word)

        word = word.lower()

        # 1. Sample k sentences that include this word w
        if sample_sentences is None:
            sample_sentences = self._sample_sentence_including_word_from_corpus(word)
        tokenized_word = self.wrapper.tokenizer.tokenize(word)
        tokenized_word_window = len(tokenized_word)

        # 2. Tokenize the sentences
        sample_sentences = [self.wrapper.tokenizer.tokenize(x) for x in sample_sentences]

        # Collecting all embeddings within one array
        # If the python list is nested, it means that the word was split up into multiple tokens,
        # and we can aggregate them somehow furthermore
        embeddings = []


        # 3. Run through language model, look at how the other paper reprocuded generating embeddings for word using BERT
        for sentence in sample_sentences:
            start_time = time.time()

            # We only sample per sentence, so it is always the same segment...
            segments_ids = [0, ] * len(sentence)
            indexed_tokens = self.wrapper.tokenizer.convert_tokens_to_ids(sentence)

            # Find all indecies of tokenized word within array
            # For simplicity, taking the first occurence...
            tokenized_word_idx = find_all_indecies_subarray(tokenized_word, sentence)[0]

            # Now convert to pytorch tensors..
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            # Retrieve the embeddings at layer no `self.bert_layer`
            # TODO: Somehow not all layers are returned... check this out lol
            # Take the outputs of the forward body lol
            outputs = self.wrapper.forward(
                tokens_tensor=tokens_tensor,
                segments_tensors=segments_tensors
            )
            word_embedding = outputs[0, tokenized_word_idx:tokenized_word_idx + tokenized_word_window, :]

            # Reshape amongst new dimension
            # And apply aggregation if desired

            assert word_embedding.shape == (1, 768), (word_embedding.shape, (1, 768))
            embeddings.append(word_embedding)

            if args.verbose >= 2:
                print("One sentence-embedding-etrieval from BERT takes: ", time.time() - start_time)

        return embeddings


if __name__ == "__main__":
    print("Starting to generate embeddings from the BERT model")
    embeddings_model = BertEmbedding()

    # I add spaces before and after, s.t. the word must occur within a sentence (and not at the beginning!)
    # This is not fully unbiased, I gues...?
    example_words = [" bank "]
    for word in example_words:
        print(word)
        print([x.shape for x in embeddings_model.get_embedding(word)])

    # Now do with the embeddings whatever you want to do lol