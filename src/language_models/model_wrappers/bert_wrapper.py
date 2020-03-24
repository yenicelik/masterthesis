"""
    Wraps around the base BERT model.
    Includes logic to predict masked words and some other stuff
"""

import torch

from transformers import BertTokenizer, BertModel, BertForMaskedLM

from src.config import args

class BertWrapper:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()  # We're not gonna train it any further

        # Masked predictor
        self.masekd_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.masekd_model.eval()

        if args.cuda:
            self.model.to(args.device)

    def forward(self, tokens_tensor, segments_tensors):
        """
            Predicts the hidden states features for each layer
        :return: torch Tensor with torch.Size([batch_size, sentence_length, hidden_size]),
            i.e. Encoded layers are:  torch.Size([1, 48, 768])
        """
        # Project to CUDA if not projected yet
        if args.cuda:
            # print("tokens tensor")
            # print(tokens_tensor)
            tokens_tensor = tokens_tensor.to(args.device)
            segments_tensors = segments_tensors.to(args.device)

        with torch.no_grad():
            # See the models docstrings for the detail of the inputs
            print("Inputs are: ", tokens_tensor)
            encoded_layers, _ = self.model.forward(
                input_ids=tokens_tensor,
                token_type_ids=segments_tensors
            )
            # print("Encoded layers are: ", encoded_layers.shape)

        return encoded_layers

        # Assert that all shapes conform with each other

    def predict_token(self, tokens_tensor, segments_tensors):
        """
            Predicts the token for any masked items
        """
        if args.cuda:
            tokens_tensor = tokens_tensor.to(args.device)
            segments_tensors = segments_tensors.to(args.device)

        with torch.no_grad():
            outputs = self.masekd_model(
                input_ids=tokens_tensor,
                token_type_ids=segments_tensors
            )
            # Checking what the output is:
            # print("Outputs are: ")
            # print(outputs)
            predictions = outputs[0]

        # Not sure what the outputs for this are
        return predictions


if __name__ == "__main__":
    print("Testing the BERT wrapper to startup")

    model = BertWrapper()

    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = model.tokenizer.tokenize(text)

    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    predicted_tokenized_sentence = ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was',
                                    'a',
                                    'puppet', '##eer', '[SEP]']

    if tokenized_text != predicted_tokenized_sentence:
        for x, y in zip(tokenized_text, predicted_tokenized_sentence):
            print(x, y)
        assert False

    # TODO: Put these into another function which increments id's whenever a SEP occurs

    # Make it computer-readable
    indexed_tokens = model.tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define which item corresponds to which sentence..
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # Convert to pytorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    output = model.forward(tokens_tensor=tokens_tensor, segments_tensors=segments_tensors)
    encoded_layers = output[0]
    assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), self.model.config.hidden_size)

    predictions = model.predict_token(tokens_tensor=tokens_tensor, segments_tensors=segments_tensors)

    print("Predictions are: ")
    print(predictions.shape)

    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = model.tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print("Predicted token is: ", predicted_token)

    assert predicted_token == 'henson'


