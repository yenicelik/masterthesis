"""
    Making prediction with best model
"""
import spacy
import numpy as np
import pandas as pd


from src.config import args
from src.embedding_generators.bert_embeddings import BertEmbedding
from src.resources.corpus import Corpus
from src.resources.samplers import retrieve_data
from src.utils.create_experiments_folder import randomString

if __name__ == "__main__":
    print("Writing to thesaurus, and printing best model")

    # We first sample a few sentences which include a word
    polypos_words = [

        # ' attack ',
        # ' act ',
        # ' die ',
        # ' kick ',
        # ' sand ',
        # ' address ',
        # ' aim ',
        # '  '

        # ' act ',
        # ' address ',
        # ' back ',
        # ' bear ',
        # ' block ',
        # ' catch ',
        # ' charge ',

        # ' crack ',
        # ' double ',

        # ' face ',
        # ' head ',
        # ' march ',
        # ' order ',

        ' play ',
        ' roll ',
        ' saw ',
        ' tie ',
        ' train ',
        ' treat ',
        ' value ',
        ' visit ',
        ' wake ',
        ' work ',
        ' zone ',

    # ' act ',
    # ' address ',
    # ' aim ',
    # ' answer ',
    # ' back ',
    # ' balloon ',
    # ' bank ',
    # ' battle ',
    # ' bear ',
    # ' bend ',
    # ' blast ',
    # ' block ',
    # ' break ',
    # ' brush ',
    # ' catch ',
    # ' challenge ',
    # ' charge ',
    # ' cheer ',
    # ' color ',
    # ' cook ',
    # ' crack ',
    # ' curl ',
    # ' cycle ',
    # ' dance ',
    # ' design ',
    # ' die ',
    # ' double ',
    # ' doubt ',
    # ' dust ',
    # ' echo ',
    # ' end ',
    # ' estimate ',
    # ' face ',
    # ' finish ',
    # ' fish ',
    # ' flood ',
    # ' fool ',
    # ' frown ',
    # ' garden ',
    # ' glue ',
    # ' guard ',
    # ' guess ',
    # ' hammer ',
    # ' hand ',
    # ' head ',
    # ' hug ',
    # ' insult ',
    # ' iron ',
    # ' kiss ',
    # ' laugh ',
    # ' loan ',
    # ' love ',
    # ' man ',
    # ' march ',
    # ' milk ',
    # ' object ',
    # ' order ',
    # ' paddle ',
    # ' peel ',
    # ' permit ',
    # ' play ',
    # ' pop ',
    # ' practice ',
    # ' produce ',
    # ' punch ',
    # ' question ',
    # ' quiz ',
    # ' rhyme ',
    # ' rock ',
    # ' roll ',
    # ' run ',
    # ' sand ',
    # ' saw ',
    # ' skate ',
    # ' smell ',
    # ' surprise ',
    # ' thunder ',
    # ' tie ',
    # ' time ',
    # ' toast ',
    # ' trace ',
    # ' train ',
    # ' treat ',
    # ' trick ',
    # ' use ',
    # ' vacuum ',
    # ' value ',
    # ' visit ',
    # ' wake ',
    # ' walk ',
    # ' water ',
    # ' wish ',
    # ' work ',
    # ' x - ray ',
    # ' yawn ',
    # ' zone ',


        # ' cut ',
        # ' break ',

        # ' well ',
        # ' down ',

        # ' run ',
        # ' round ',
        # ' table ',
        # ' bank ',
        # ' cold ',
        # ' good ',
        # ' mouse ',
        # ' was ',
        # ' key ',
        # ' arms ',
        # ' thought ',
        # ' pizza ',
        # ' made ',
        # ' book ',
        # ' damn ',
    ]

    # First of all, check which ones have more then one meaning
    # (taking items with only two meanings,
    # does not make sense ..)

    # for word in polypos_words:
    #     print()
    #     print(word, wn.synsets(word.strip()))
    #
    # exit(0)

    print("Creating corpus ...")
    corpus = Corpus()
    lang_model = BertEmbedding(corpus=corpus)

    rnd_str = randomString(additonal_label=f"_tensorboard_PoS_{args.dimred}_{args.dimred_dimensions}_whiten{args.pca_whiten}_norm{args.normalization_norm}")

    print("Loading spacy")
    # Not sure if we need anything more from this
    nlp = spacy.load("en_core_web_sm")

    # TODO: Do this same visualization for the meaning-clusters!

    # Export the retrieved data to tensorboard

    for tgt_word in polypos_words:
        X, sentences, labels = retrieve_data(nlp, tgt_word=tgt_word)

        # Save as a numpy array
        np.savetxt(rnd_str + f"/{tgt_word}_matr.tsv", X, delimiter="\t")
        pd.DataFrame(
            {
                "sentece": sentences,
                "labels": labels
            }
        ).to_csv(rnd_str + f"/{tgt_word}_labels.tsv", sep="\t")
