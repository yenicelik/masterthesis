"""
    From the CSVs, writes a more easily comprehensive thesaurus file
"""
import numpy as np
import pandas as pd

def csv_to_thesaurus(filepath, tgt_word):
    df = pd.read_csv(filepath)

    out = dict()
    # dictionary includes key : cluster_id and values : [sentence]

    # Find the first position of the target word
    cluster_ids = np.unique(df['cluster_id'])
    print("cluster_ids are", cluster_ids)

    max_padding = 0

    # First pass-through
    for cluster in cluster_ids:
        print("\n")
        current_df = df[df['cluster_id'] == cluster]
        sentences = current_df['sentence'].values.tolist()
        for sentence in sentences:
            idx = sentence.find(tgt_word)
            assert idx != -1, idx

            if idx > max_padding:
                max_padding = idx

    print("Maximum padding is ", max_padding)

    # Must now print, and add (max_padding - idx) * ' ' to print-string ...
    for cluster in cluster_ids:
        # max_padding = 0

        print("\n")
        print(max_padding * ' ' + len(tgt_word) * '-')
        current_df = df[df['cluster_id'] == cluster]
        sentences = current_df['sentence'].values.tolist()
        first_flag = True

        # for sentence in sentences:
        #     idx = sentence.find(tgt_word)
        #     assert idx != -1, idx
        #
        #     if idx > max_padding:
        #         max_padding = idx

        #############
        # Identify max-padding of clusters ...
        #############
        for sentence in sentences:
            idx = sentence.find(tgt_word)

            out_sentence = (max_padding - idx) * ' ' + sentence
            print(out_sentence)

            if first_flag:
                first_flag = False
                print(max_padding * ' ' + len(tgt_word) * '-')

            # print(type(sentence))
        print(max_padding * ' ' + len(tgt_word) * '-')


    # print(current_df.head())
    # print(df.head())

if __name__ == "__main__":
    print("Starting to print out the thesaurus")
    filepath = "/Users/david/GoogleDrive/_MasterThesis/notebooks/2020_01_22 our clustering vs true clustering/kucilnghrr/thesaurus_ arms .csv"
    csv_to_thesaurus(filepath, tgt_word=" arms ")
