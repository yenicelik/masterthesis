"""
    Rewrites the Bernie model.
    This adapts to the huggingface framework
"""
from transformers import BertModel


class BernieMeaningModel(BertModel):

    def __init__(self, config):
        super().__init__(config)
        print("Using BERnie meaning ..")

    def inject_split_token(self, new_total_vocabulary_size, token_idx, number_new_tokens):
        """
            Spawns new token idx's that are a copy of the token_idx' embedding vector.
            The token_idx needs to be retrieved from the tokenizeer that we are using
            (in this case, it is the BerniePoSTokenizer!)
        :param token_idx: Token idx to copy-over to the newly generated tokens
        :param n: How many new vectors to add. This corresponds to how many new tokens were spawned
        :return:
        """

        old_embedding_len = self.embeddings.word_embeddings.weight.size(0)

        # 1. Get the old vector
        old_vector = self.embeddings.word_embeddings.weight.data[token_idx, :]

        # 2. Increase the size of the embedding matrix
        self.resize_token_embeddings(new_total_vocabulary_size)

        # 3. Inject the mean vector to the new vectors
        self.embeddings.word_embeddings.weight.data[-number_new_tokens:, :] = old_vector.reshape(1, -1).repeat(
            (number_new_tokens, 1))

        # 4. Make sure that copy-over tokens are
        assert (self.embeddings.word_embeddings.weight.data[-number_new_tokens,
                :] == self.embeddings.word_embeddings.weight.data[token_idx, :]).all()

        new_embedding_len = self.embeddings.word_embeddings.weight.size(0)
        assert new_embedding_len == old_embedding_len + number_new_tokens, (
        new_embedding_len, old_embedding_len + number_new_tokens)

        # No need to return anything, because this model will be used as an "end-product"

if __name__ == "__maine__":
    print("Testing if the model resizing works well!")
