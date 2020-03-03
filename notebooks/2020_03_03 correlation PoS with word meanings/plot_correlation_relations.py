import seaborn as sns
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

if __name__ == "__main__":
    print("Calculates the correlation relationships")

    df = pd.read_csv("/Users/david/GoogleDrive/_MasterThesis/notebooks/2020_03_03 correlation PoS with word meanings/_percentage_PoS_none_768_whitenFalse_norm_standardizeFalsergnjnddggf/_correlation_pos_to_semcor_cluster.csv")
    print("df head is: ")
    print(df.head())
    print(df.columns)
    df['number_items_in_cluster'] = df['pos labels'].apply(lambda x: len(x))
    print(df.head())

    # Filter out clusters that don't have enough items inside..
    # df = df[df['number_items_in_cluster'] > 3]  # we should see at least 3 examples to consider this a valid observation

    plt.hist(df['percentage dominance'].values, bins=10, cumulative=True, density=True, stacked=True)
    plt.show()
    plt.clf()

    plt.hist(df['percentage dominance'].values, bins=10, cumulative=False, density=False, stacked=False)
    plt.show()
    plt.clf()

    # print(df['percentage dominance'].values)
    # print(np.sum(df['percentage dominance'].values))
    #
    # histogram = np.histogram(df['percentage dominance'].values / np.sum(df['percentage dominance'].values))
    # print(histogram)
    #
    # # # TODO: Something is funky here ..
    # # # print("Histogram is: ", histogram)
    # # sns.distplot(df['percentage dominance'])
    #
    # sns.lineplot(histogram[1].tolist(), histogram[0].tolist() + [0,])
    #
    # # sns.distplot()
    # plt.show()
    # plt.clf()
    #


