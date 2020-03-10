"""
    Wrapper around the BERTForSequenceClassification, such that the BerniePoS model is User!
"""

from torch import nn
from transformers import BertForSequenceClassification

from src.bernie.bernie_model import BerniePoSModel


class BerniePoSForSequenceClassification(BertForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BerniePoSModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()
