"""
    This is the test where we try to inject certain word-vectors into BERT.
    We will likely need to modify following items:
    - BERT vocabulary (dictionary probably)
    - BERT lowest-level vectors -> replace these by mean of cluster vectors, or similar

    Resources on how to add your custom tokens and vocabulary
    - https://github.com/huggingface/transformers/blob/master/README.md in Section "Serialization"
"""
import tensorflow_datasets
from transformers import BertTokenizer, BertForSequenceClassification, glue_convert_examples_to_features

from src.utils.create_experiments_folder import randomString


def expand_bert_by_target_word(tgt_word, model: BertForSequenceClassification, tokenizer: BertTokenizer, n=5):
    """
        Getting the BERT embedding
    :return:
    """
    # Number of new tokens to add
    old_vocab_size = len(tokenizer.added_tokens_decoder)

    # 0: Find the target word idx in the vocabulary
    print(tokenizer.added_tokens_decoder)
    word_idx = tokenizer.vocab[tgt_word]

    # 1: Retrieve the embedding in the vocabulary
    old_vector = model.bert.embeddings.word_embeddings.weight.data[word_idx, :]

    # 2: Add tokens into tokenizeer and make space for new vectors
    tokens_to_add = [(f'{tgt_word}_{i}') for i in range(n - 1)]
    number_new_tokens = len(tokens_to_add)
    print("Tokens before", tokenizer.vocab_size)
    added_tokens = tokenizer.add_tokens(tokens_to_add)  # TODO: add_special_tokens or add_tokens?
    assert added_tokens == n - 1, (added_tokens, n)
    print("added tokens", added_tokens)
    model.resize_token_embeddings(len(tokenizer))
    print("Tokens after", tokenizer.vocab_size)

    # 2.1: Test if tokenization worked out
    # print([tokenizer.convert_tokens_to_ids(x) for x in tokens_to_add])
    assert all([tokenizer.convert_tokens_to_ids(x) for x in tokens_to_add])

    # 2: Inject / overwrite new word-embeddings with old embedding
    model.bert.embeddings.word_embeddings.weight.data[-number_new_tokens:, :] = old_vector.reshape(1, -1).repeat((n - 1, 1))

    assert (model.bert.embeddings.word_embeddings.weight.data[-number_new_tokens + 1, :] == \
           model.bert.embeddings.word_embeddings.weight.data[word_idx, :]).all()

    # Take the embeddings vector at position `word_idx` and add this to the embedding
    # Double check if these were successfully copied ...

    new_vocab_size = len(tokenizer.added_tokens_decoder)

    assert new_vocab_size == old_vocab_size + (n - 1), (new_vocab_size, old_vocab_size, (n - 1))

    return model, tokenizer

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

    get_bert_size_stats(model.bert, tokenizer)

    # for word in extra-words ...
    ######################################################
    #
    # Add words to the BERT tokenizer and embedding
    #
    ######################################################
    tgt_word = "run"
    model, tokenizer = expand_bert_by_target_word(tgt_word, model, tokenizer)

    get_bert_size_stats(model.bert, tokenizer)

    # now fine-tune the model on one of the GLUE language tasks
    rnd_str = randomString(additonal_label=f"_fine-tune-BERT")

    ######################################################
    #
    # Fine Tune BERT here
    #
    ######################################################

    # You could save it as a BERT models, and load for tensorflow ...

    # for i in language task

    data = tensorflow_datasets.load('glue/mrpc')
    train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')
    valid_dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=128, task='mrpc')
    train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
    valid_dataset = valid_dataset.batch(64)

    # Do it simple, and with ... keras?! Let's use pytorch instead ....
    # Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule

    # optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    # model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # Save this model somewhere ..
    # ### Now let's save our model and tokenizer to a directory
    model.save_pretrained(f'{rnd_str}/-my_saved_model_directory/')
    tokenizer.save_pretrained(f'{rnd_str}/my_saved_model_directory/')

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
