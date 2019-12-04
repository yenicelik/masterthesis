"""
    Includes the logic which samples a few embeddings for each word.
    This is similar to Monte-Carlo sampling to represent words.

    - Do 1000 times:
        - Find all occurences of "word" in BERT
        - Run BERT on the word (or ELMo if tokenization is easier)
    - Create a Probability distribution from the above samples
        (do we need a priior for this distribution...? or can we just generate a pointcloud with this)

    This assumes that the biLM outputs a single vector somehow which can then used to apply nearest-neighbor search
    But this is perhaps the idea behind this. Get individual word-embeddings through the initial one.
    -> Can then generate a thesaurus, or sample sentences where these words can be used within.

    Perhaps ELMo is a better choice. BERT seems a bit too "masked". However, ELMo does not have embeddings for all languages...
"""

# TODO: How about repetetive maximal marginal prediction to generate a thesaurus?
# i.e.
# 1. [MASK], [MASK], cat, [MASK], [MASK] -> generate argmax of probability
# 2. [MASK], the, cat, [MASK], [MASK]
# 3. [MASK], the, cat, ate, [MASK]
# 4. [MASK], the, cat, ate, fish
# 5. and, the, cat, ate, fish

import torch
from transformers import BertTokenizer, DistilBertModel, BertForMaskedLM, BertModel


class BertWrapper:

    def _load_model(self):
        """
            Evaluate this function at load
        :return:
        """
        # Let's not care about case right now..
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Run an example text through this:
        text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
        tokenized_text = tokenizer.tokenize(text)

        masked_index = 8
        tokenized_text[masked_index] = '[MASK]'
        predicted_tokenized_sentence = ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was',
                                        'a',
                                        'puppet', '##eer', '[SEP]']
        if tokenized_text != predicted_tokenized_sentence:
            for x, y in zip(tokenized_text, predicted_tokenized_sentence):
                print(x, y)
            assert False

        # Make it computer-readable
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # Define which item corresponds to which sentence..
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

        # Convert to pytorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        print("Loading bert model")

        # now get the bert pre-trained weights
        # Using Distilbert, as anything else will not really fit into memory lol
        # Use fp16!
        # model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()

        # Should make this code modular lol
        # # If you have a GPU, put everything on cuda
        # tokens_tensor = tokens_tensor.to('cuda')
        # segments_tensors = segments_tensors.to('cuda')
        # model.to('cuda')

        # Predict hidden states features for each layer
        with torch.no_grad():
            # See the models docstrings for the detail of the inputs
            outputs = model(tokens_tensor, token_type_ids=segments_tensors)
            # Transformers models always output tuples.
            # See the models docstrings for the detail of all the outputs
            # In our case, the first element is the hidden state of the last layer of the Bert model
            encoded_layers = outputs[0]

        # We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
        assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)

        # We now predict the next token lol
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        model.eval()

        # If you have a GPU, put everything on cuda
        # tokens_tensor = tokens_tensor.to('cuda')
        # segments_tensors = segments_tensors.to('cuda')
        # model.to('cuda')

        # Predict all tokens
        with torch.no_grad():
            outputs = model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

        # confirm we were able to predict 'henson'
        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        print("Predicted token is: ", predicted_token)
        assert predicted_token == 'henson'

    def __init__(self):
        self._load_model()


if __name__ == "__main__":
    print("Loadng the BERT model (distillbert)")

    model = BertWrapper()
