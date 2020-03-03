"""
    Here we calculate to what extent the PoS correlates to the SemCor word meaning.

    For this we might need to write a new sampler which gets both PoS and SemCor
"""
from collections import Counter

import spacy
import numpy as np
import pandas as pd

from src.config import args
from src.embedding_generators.bert_embeddings import BertEmbedding
from src.resources.corpus import Corpus
from src.resources.samplers import sample_embeddings_for_target_word, get_pos_for_word
from src.utils.create_experiments_folder import randomString

if __name__ == "__main__":
    print("Calculate the PoS' correlation to word meaning")

    corpus = Corpus()
    lang_model = BertEmbedding(corpus=corpus)

    # Generate foldere to save this in
    rnd_str = randomString(additonal_label=f"_percentage_PoS_{args.dimred}_{args.dimred_dimensions}_whiten{args.pca_whiten}_norm{args.normalization_norm}_standardize{args.standardize}")

    # Run the actual experiment for each word ...
    # TODO: Get words from the BERT vocabulary list

    # Apply the clustering algorithm ..
    # words = ['run'] # , 'first', 'also', 'new', 'two', 'time', 'would', 'said'

    words = [

        ' attack ',
        ' act ',
        ' die ',
        ' kick ',
        ' sand ',
        ' address ',
        ' aim ',

        ' act ',
        ' address ',
        ' back ',
        ' bear ',
        ' block ',
        ' catch ',
        ' charge ',

        ' crack ',
        ' double ',

        ' face ',
        ' head ',
        ' march ',
        ' order ',

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

        ' act ',
        ' address ',
        ' aim ',
        ' answer ',
        ' back ',
        ' balloon ',
        ' bank ',
        ' battle ',
        ' bear ',
        ' bend ',
        ' blast ',
        ' block ',
        ' break ',
        ' brush ',
        ' catch ',
        ' challenge ',
        ' charge ',
        ' cheer ',
        ' color ',
        ' cook ',
        ' crack ',
        ' curl ',
        ' cycle ',
        ' dance ',
        ' design ',
        ' die ',
        ' double ',
        ' doubt ',
        ' dust ',
        ' echo ',
        ' end ',
        ' estimate ',
        ' face ',
        ' finish ',
        ' fish ',
        ' flood ',
        ' fool ',
        ' frown ',
        ' garden ',
        ' glue ',
        ' guard ',
        ' guess ',
        ' hammer ',
        ' hand ',
        ' head ',
        ' hug ',
        ' insult ',
        ' iron ',
        ' kiss ',
        ' laugh ',
        ' loan ',
        ' love ',
        ' man ',
        ' march ',
        ' milk ',
        ' object ',
        ' order ',
        ' paddle ',
        ' peel ',
        ' permit ',
        ' play ',
        ' pop ',
        ' practice ',
        ' produce ',
        ' punch ',
        ' question ',
        ' quiz ',
        ' rhyme ',
        ' rock ',
        ' roll ',
        ' run ',
        ' sand ',
        ' saw ',
        ' skate ',
        ' smell ',
        ' surprise ',
        ' thunder ',
        ' tie ',
        ' time ',
        ' toast ',
        ' trace ',
        ' train ',
        ' treat ',
        ' trick ',
        ' use ',
        ' vacuum ',
        ' value ',
        ' visit ',
        ' wake ',
        ' walk ',
        ' water ',
        ' wish ',
        ' work ',
        ' x - ray ',
        ' yawn ',
        ' zone ',

        ' cut ',
        ' break ',

        ' well ',
        ' down ',

        ' run ',
        ' round ',
        ' table ',
        ' bank ',
        ' cold ',
        ' good ',
        ' mouse ',
        ' was ',
        ' key ',
        ' arms ',
        ' thought ',
        ' pizza ',
        ' made ',
        ' book ',
        ' damn ',
    ]

    words = sorted(list(set(words)))

    # words = [f" {x} " for x in words]

    nlp = spacy.load("en_core_web_sm")

    out = []

    for tgt_word in words:

        try:

            # Get the respective PoS embeddings
            # Try catch if the respectiv word does not exist within the corpus
            number_of_senses, X, true_cluster_labels, known_indices, sentences = sample_embeddings_for_target_word(
                tgt_word=tgt_word,
                semcor_only=True
            )

            pos_lables = [get_pos_for_word(nlp, x, tgt_word.strip())[1] for x in sentences]

            assert len(pos_lables) == len(known_indices), ("Known indices does not match the pos labels", pos_lables, known_indices)

            # For each of the sentences get the respective PoS tag (NOUN, VERB)
            # Calculate the adjusted random score ...
            # Calculate the homogenity (percentage of dominant class) within

            # Perhaps calculate something you can later on analyse on a per-word-meaning
            all_pos = np.unique(pos_lables)

            true_cluster_labels = np.asarray(true_cluster_labels).astype(int)
            pos_lables = np.asarray(pos_lables)

            # Go over each cluster, and check if all pos are uniform ...
            unique_clusters = np.unique(true_cluster_labels)

            # If there is only a single SemCor meaning, skip this
            if len(unique_clusters) == 1:
                continue

            for cluster in unique_clusters:
                print("Comparands are: ", cluster, true_cluster_labels)

                # Get majority part of speech
                occurencens = np.argwhere(cluster == true_cluster_labels).flatten()
                print("Argwhere is: ", occurencens)
                if len(occurencens) == 1:
                    # if only one occurence, then skip
                    # print("Only one occurence found!", occurencens)
                    continue

                pos_within_cluster_subarray = pos_lables[occurencens]

                print("pos within cluster subarray is: ", pos_within_cluster_subarray)

                # dominant_label = np.mode(pos_within_cluster_subarray)

                # Could use counter instead
                dominant_label = Counter(pos_within_cluster_subarray.tolist()).most_common(1)[0][0]

                # Get all words that are within one cluster
                # Calculate the percentage of dominant labels
                print("Fed-in items are: ", dominant_label, pos_within_cluster_subarray)
                print("Argwhere comparison is: ", np.where(dominant_label == pos_within_cluster_subarray)[0])
                print("Absolute occurence is: ", len(np.where(dominant_label == pos_within_cluster_subarray)[0]))
                percentage = len(np.where(dominant_label == pos_within_cluster_subarray)[0]) / float(len(pos_within_cluster_subarray))
                print("Percentage is: ", percentage)
                tpl = (tgt_word, cluster, dominant_label, percentage, len(unique_clusters), pos_within_cluster_subarray)
                out.append(tpl)

        except Exception as e:
            print(e)
            print(f"Something went wrong with {tgt_word}")

    df = pd.DataFrame(out, columns=['word', 'semcor cluster', 'dominant label', 'percentage dominance', 'no_semcor_clusters', 'pos labels'])
    df.to_csv(rnd_str + "/_correlation_pos_to_semcor_cluster.csv")
