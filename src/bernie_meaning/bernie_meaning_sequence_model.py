"""
    Wrapper around the BERTForSequenceClassification, such that the BerniePoS model is User!
"""

from torch import nn
from transformers import BertForSequenceClassification, BertPreTrainedModel
from transformers.modeling_bert import BertPreTrainingHeads, BertForPreTraining, BertOnlyMLMHead, BertForMaskedLM

from src.bernie_meaning.bernie_meaning_model import BernieMeaningModel

class BernieMeaningForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        # Blatantly remove the config to check if this still works ...
        print("Config input is: ", config)
        self.bert = BernieMeaningModel(config)

        # Now extract anything that has changed in the above config,
        # and transfer it to the MLM head!
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()  # Does init weights overwrite if existent? ... (probably not..)

        # Steps look as follows!

        # 1. Pass ONLY bert through tokenizer
        # 2. Save the model
        # 3. Use the "load_pretrained" for BernieMeaningForMaskedLM to load the pretrained model

        # OR
        # 1. Create BERT file
        # 2.
        # 4. Create a config from this
        # 5. Initialiate a new BernieMeaningForMaskedLM using the savefile

        # OR
        # 1. Pass through Tokenizer
        # 2. Create a new object of this using modified config
        # 3. And then inject the injected (tokenization-passed) BERT into this

        # Overwrite the BERT model within this class ... or re-create the cls with the new config ...

    def spawn_new_LML_head(self, new_config):
        self.cls = BertOnlyMLMHead(new_config)

class BernieMeaningForSequenceClassification(BertForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BernieMeaningModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    # Here again, you can replace the bert model the config ...
    def spawn_new_bert_model(self, bert_model):
        self.bert = bert_model
