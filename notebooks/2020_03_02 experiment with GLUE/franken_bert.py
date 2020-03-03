"""
    This is the test where we try to inject certain word-vectors into BERT.
    We will likely need to modify following items:
    - BERT vocabulary (dictionary probably)
    - BERT lowest-level vectors -> replace these by mean of cluster vectors, or similar

    Resources on how to add your custom tokens and vocabulary
    - https://github.com/huggingface/transformers/blob/master/README.md in Section "Serialization"
"""
from transformers import BertTokenizer, TFBertForSequenceClassification, BertForSequenceClassification

def get_bert_size_stats(model, tokenizer):
    print("Number of embeddings in BERT model and tokenizer")
    print("Tokenizer number of vocabs is: ", len(tokenizer))
    # print(model)
    # print("Encoder: ", model.encoder)
    print("Encoder number of attention mechanisms: ", model.encoder.output_attentions)
    print("Embeddings: ", model.embeddings)
    print("Word emb: ", model.embeddings.word_embeddings.num_embeddings)
    print("Position emb: ", model.embeddings.position_embeddings.num_embeddings)
    print("Token type emb: ", model.embeddings.token_type_embeddings.num_embeddings)

def get_bert_model():
    # Take the uncased ones, as this make tokenization etc. much easier ..
    # Can then use cased if this proves to be success

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased') # No, let's not use TFBert, let's use PyTorch BERT

    old_vocab_size = len(tokenizer)

    get_bert_size_stats(model.bert, tokenizer)

    # Iteratively apply this for a set of tokens.
    # at least 1000 tokens that are polysemous
    # TODO: Perhaps pick tokens which have high variance..?


    # I guess this is how you add new tokens / vocabulary..?
    # Theoretically, here we inject two types of words ..
    # Add n-1 tokens. The nth token is the unmodified word (without any underscore ...)

    # In the lower-level function (for BERT), the old embeddings are copied over.
    # New embeddings are appended to axis 0
    tokenizer.add_tokens(['[run_1]', '[run_2]'])
    model.resize_token_embeddings(len(tokenizer))

    # Now initialize the new embeddings with copy of the intial embedding..
    idx = None  # TODO: replace this by the word that was split into multiple tokens
    word_embedding = model.bert.embeddings.word_embeddings[idx]
    # Overwrite embeddings
    word_embedding[old_vocab_size:, :] = word_embedding

    # Does this change the model somehow, or are weights properly kept?

    # Save this model somewhere ..
    # ### Now let's save our model and tokenizer to a directory
    # model.save_pretrained('./my_saved_model_directory/')
    # tokenizer.save_pretrained('./my_saved_model_directory/')
    #
    # # Fine-tune pretrained BERT ...
    #
    # ### Reload the model and the tokenizer
    # tokenizer = BertTokenizer.from_pretrained('./my_saved_model_directory/')
    # model = BertForSequenceClassification.from_pretrained('./my_saved_model_directory/')

    get_bert_size_stats(model.bert, tokenizer)

    print("tokenizer has the class", type(tokenizer))
    print("model has the class", type(model))

    # print(tokenizer)
    # print(model)


if __name__ == "__main__":
    print("starting to frankenstein bert")
    get_bert_model()
