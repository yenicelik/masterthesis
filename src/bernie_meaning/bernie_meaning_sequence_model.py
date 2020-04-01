"""
    Wrapper around the BERTForSequenceClassification, such that the BerniePoS model is User!
"""

from torch import nn
from transformers import BertForSequenceClassification, BertPreTrainedModel
from transformers.modeling_bert import BertPreTrainingHeads, BertForPreTraining, BertOnlyMLMHead, BertForMaskedLM

from src.bernie_meaning.bernie_meaning_model import BernieMeaningModel

class BernieMeaningForPreTraining(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BernieMeaningModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

class BernieMeaningForSequenceClassification(BertForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BernieMeaningModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()
