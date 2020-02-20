"""
    Instead of clustering over senses,
    we now simplify the problem and try to cluster over words
"""
import traceback

import spacy
import numpy as np
from ax import optimize
from sklearn.metrics import adjusted_rand_score

from src.config import args
from src.embedding_generators.bert_embeddings import BertEmbedding
from src.models.cluster.affinitypropagation import MTAffinityPropagation
from src.models.cluster.chinesewhispers import MTChineseWhispers
from src.models.cluster.dbscan import MTDbScan
from src.models.cluster.hdbscan import MTHdbScan
from src.models.cluster.kmeans_with_annealing import MTKMeansAnnealing
from src.models.cluster.meanshift import MTMeanShift
from src.resources.corpus import Corpus
from src.resources.samplers import retrieve_data
from src.utils.create_experiments_folder import randomString


def create_evaluation_sets(nlp, wordlist):

    devset = dict()

    for word in wordlist:
        X, sentences, labels = retrieve_data(nlp, word)
        devset[word] = X, sentences, labels

    assert len(devset) == len(wordlist), (len(devset), len(wordlist), wordlist, devset.keys())

    assert labels, ("Labels is empty", labels)

    if len(np.unique(labels)) == 0:
        print("Only one POS found ...")

    return devset


def evaluate_model(model_class, arg, devset):
    final_score = 0.

    for word, tpl in devset.items():
        X, sentences, labels = tpl

        pred_clustering = model_class(arg).fit_predict(X)

        # Nothing about "known indices" ... this is only the case for limited corpora, but this time we have spacy ...
        score = adjusted_rand_score(labels, pred_clustering)

        # print("Input to adjusted random score is: ")
        # print("Content is 1: ", labels)
        # print("Content is 2: ", pred_clustering)
        print("Score is: ", score)

        final_score += score

    return float(final_score) / len(devset)

if __name__ == "__main__":
    print("Sampling with a few words etc.")

    # We first sample a few sentences which include a word
    polypos_words = [
        ' run ',
        # ' well ',
        ' round ',
        ' down ',
        # ' table ',
        ' bank ',
        ' cold ',
        # ' good ',
        ' mouse ',
        ' was ',
        # ' key ',
        # ' arms ',
        # ' thought ',
        # ' pizza ',

        # ' made ',
        # ' book ',
        # ' damn ',

    ]

    print("Creating corpus ...")
    corpus = Corpus()
    lang_model = BertEmbedding(corpus=corpus)

    print("Loading spacy")
    # Not sure if we need anything more from this
    nlp = spacy.load("en_core_web_sm")

    model_classes = [
        # ("MTMeanShift", MTMeanShift),
        ("MTHdbScan", MTHdbScan),
        ("MTDbScan", MTDbScan),
        # ("MTAffinityPropagation", MTAffinityPropagation),
        ("MTChineseWhispers", MTChineseWhispers),
        ("MTKMeansAnnealing", MTKMeansAnnealing)
    ]

    devset = create_evaluation_sets(nlp=nlp, wordlist=polypos_words)

    for model_name, model_class in model_classes:
        print(f"Running {model_name} {model_class}")

        params = model_class.hyperparameter_dictionary()

        # Define the evaluation functions ...
        def _current_eval_fun(p):
            try:
                return evaluate_model(
                    model_class=model_class,
                    arg=p,
                    devset=devset
                )
            except Exception as e:
                print("Error occurred!")
                print(e)
                traceback.print_tb(e.__traceback__)
                return 0.

        try:
            best_parameters, best_values, experiment, model = optimize(
                parameters=params,
                evaluation_function=_current_eval_fun,
                minimize=False,
                total_trials=len([x for x in params if x['type'] != "fixed"]) * 10 * 3
            )

            print("Best parameters etc.")
            print(best_parameters, best_values, experiment, model)

        except Exception as e:
            print("AGHHH")
            traceback.print_tb(e.__traceback__)
            print(e)
            print("\n\n\n\n")
