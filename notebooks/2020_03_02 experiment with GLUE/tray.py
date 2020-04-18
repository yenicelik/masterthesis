"""
    Some code snippets from other files
    which we may use in the imminent future
"""
import tensorflow as tf

import tensorflow_datasets
from transformers import glue_convert_examples_to_features


def train_bert_model():
    ######################################################
    # Load the Dataset
    ######################################################
    data = tensorflow_datasets.load('glue/mrpc')
    train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')
    valid_dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=128, task='mrpc')
    train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
    valid_dataset = valid_dataset.batch(64)

    model = None
    tokenizer = None

    ######################################################
    # Run the actual fine-tuning
    ######################################################
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    history = model.fit(train_dataset, epochs=2, steps_per_epoch=115,
                        validation_data=valid_dataset, validation_steps=7)

    ######################################################
    # Save the model and re-load as a pytorch model (because this is much nicer lol)
    ######################################################
    # Save this model somewhere ..
    # ### Now let's save our model and tokenizer to a directory
    create_path(f'{rnd_str}my_saved_model_directory_l{loss}/')

    model.save_pretrained(f'{rnd_str}my_saved_model_directory_l{loss}/')
    tokenizer.save_pretrained(f'{rnd_str}my_saved_model_directory_l{loss}/')

    # Fine-tune pretrained BERT ...
    # ### Reload the model and the tokenizer
    tokenizer = BertTokenizer.from_pretrained(f'{rnd_str}my_saved_model_directory_l{loss}/')
    model = BertForSequenceClassification.from_pretrained(f'{rnd_str}my_saved_model_directory_l{loss}/')

    get_bert_size_stats(model.bert, tokenizer)

    ######################################################
    # Do a simple sanity cheeck ...
    ######################################################
    sentence_0 = "This research was consistent with his findings."
    sentence_1 = "His findings were compatible with this research."
    sentence_2 = "His findings were not compatible with this research."
    inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors='pt')
    inputs_2 = tokenizer.encode_plus(sentence_0, sentence_2, add_special_tokens=True, return_tensors='pt')

    pred_1 = model(inputs_1['input_ids'], token_type_ids=inputs_1['token_type_ids'])[0].argmax().item()
    pred_2 = model(inputs_2['input_ids'], token_type_ids=inputs_2['token_type_ids'])[0].argmax().item()

    print("sentence_1 is", "a paraphrase" if pred_1 else "not a paraphrase", "of sentence_0")
    print("sentence_2 is", "a paraphrase" if pred_2 else "not a paraphrase", "of sentence_0")

    ######################################################
    #
    # Save this model somewhere, then load it and train it
    # using one of the training script provided by huggingface
    #
    ######################################################

