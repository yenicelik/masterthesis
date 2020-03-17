"""
    Implements the arg file.
    This determined things like learning rate, weight decay etc. for each GLUE experiment
"""
import argparse

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, AlbertTokenizer, \
    AlbertForSequenceClassification, AlbertConfig

from src.bernie_meaning.bernie_meaning_configuration import BernieMeaningConfig
from src.bernie_meaning.bernie_meaning_sequence_model import BernieMeaningForSequenceClassification
from src.bernie_meaning.bernie_meaning_tokenizer import BernieMeaningTokenizer
from src.bernie_pos.bernie_pos_configuration import BerniePoSConfig
from src.bernie_pos.bernie_pos_sequence_model import BerniePoSForSequenceClassification
from src.bernie_pos.bernie_pos_tokenizer import BerniePoSTokenizer

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
        BertConfig,
        BerniePoSConfig,
        BernieMeaningConfig,
        AlbertConfig
    )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "bernie_pos": (BerniePoSConfig, BerniePoSForSequenceClassification, BerniePoSTokenizer),
    "bernie_meaning": (BernieMeaningConfig, BernieMeaningForSequenceClassification, BernieMeaningTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
}
