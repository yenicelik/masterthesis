"""
    From the CSVs, writes a more easily comprehensive thesaurus file
"""
import numpy as np
import pandas as pd

def csv_to_thesaurus(filepath, savefile, tgt_word):
    df = pd.read_csv(filepath)

    out = dict()
    # dictionary includes key : cluster_id and values : [sentence]

    # Find the first position of the target word
    cluster_ids = np.unique(df['cluster_id'])
    print("cluster_ids are", cluster_ids)

    max_padding = 0

    # First pass-through
    # for cluster in cluster_ids:
    #     print("\n")
    #     current_df = df[df['cluster_id'] == cluster]
    #     sentences = current_df['sentence'].values.tolist()
    #     for sentence in sentences:
    #         idx = sentence.find(tgt_word)
    #         assert idx != -1, idx
    #
    #         if idx > max_padding:
    #             max_padding = idx

    print("Maximum padding is ", max_padding)

    # Open a file here, and write to file
    fptr = open(savefile, "w")

    # Must now print, and add (max_padding - idx) * ' ' to print-string ...
    for cluster in cluster_ids:
        max_padding = 0

        print("\n")
        fptr.write("\n")
        current_df = df[df['cluster_id'] == cluster]
        sentences = current_df['sentence'].values.tolist()
        first_flag = True

        for sentence in sentences:
            idx = sentence.find(tgt_word)
            assert idx != -1, idx

            if idx > max_padding:
                max_padding = idx

        print(max_padding * ' ' + len(tgt_word) * '-')
        fptr.write(max_padding * ' ' + len(tgt_word) * '-' + "\n")

        #############
        # Identify max-padding of clusters ...
        #############
        for sentence in sentences:
            idx = sentence.find(tgt_word)

            out_sentence = (max_padding - idx) * ' ' + sentence
            print(out_sentence)
            fptr.write(out_sentence + "\n")

            if first_flag:
                first_flag = False
                print(max_padding * ' ' + len(tgt_word) * '-')
                fptr.write(max_padding * ' ' + len(tgt_word) * '-' + "\n" )

            # print(type(sentence))
        print(max_padding * ' ' + len(tgt_word) * '-')
        fptr.write(max_padding * ' ' + len(tgt_word) * '-' + "\n")

    fptr.close()

    # print(current_df.head())
    # print(df.head())

if __name__ == "__main__":
    print("Starting to print out the thesaurus")

    for tgt_word in [
        ' table ',
        ' bank ',
        ' cold ',
        ' table ',
        ' good ',
        # ' book ',
        ' mouse ',
        ' was ',
        ' key ',
        ' arms ',
        ' was ',
        ' thought ',
        ' pizza ',
        ' made ']:
        filepath = f"/Users/david/GoogleDrive/_MasterThesis/notebooks/2020_01_22 our clustering vs true clustering/bxcavoyrgw/thesaurus_{tgt_word}.csv"
        savefile= f"/Users/david/GoogleDrive/_MasterThesis/notebooks/2020_01_22 our clustering vs true clustering/bxcavoyrgw/thesaurus_{tgt_word}.txt"
        csv_to_thesaurus(filepath, savefile=savefile, tgt_word=tgt_word)
