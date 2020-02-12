import random
import pandas as pd


def print_thesaurus(sentences, clusters, word, true_clusters=None, savepath=None, n=5):
    """
        Prints possible different use-cases of a word by taking
        :param : a set of tuples (sentence, cluster_label)
        :param n : number of examples to show per meaning clustered ...
    :return:
    """

    # TODO: Sample examples from the cluster-centers!!!! random sampling can introduce bad examples,
    # whereas we could fine-tune even more!

    if true_clusters is None:
        true_clusters = [None,] * len(clusters)

    # for cluster, sentence in zip(clusters, sentences):
    data = []
    for cluster, true_cluster, sentence in zip(clusters, true_clusters, sentences):
        print(cluster, true_cluster, sentence)
        data.append((cluster, true_cluster, sentence))

    # Shuffle, and keep five (as determined by counter)
    random.shuffle(data)
    counter = dict()

    out = []
    duplicates = set()
    for cluster, true_cluster, sentence in data:
        if cluster not in counter:
            counter[cluster] = 0
            out.append(
                (cluster, true_cluster, sentence)
            )
        elif counter[cluster] < 5:
            if sentence in duplicates:
                continue
            counter[cluster] += 1
            out.append(
                (cluster, true_cluster, sentence)
            )
            duplicates.add(sentence)
        else:
            continue

    print("out", out)

    df = pd.DataFrame.from_records(out, columns =['cluster_id', 'cluster_id_true', 'sentence'])
    df.to_csv(savepath + f"/thesaurus_{word}.csv")

    df = pd.DataFrame.from_records(data, columns =['cluster_id', 'cluster_id_true', 'sentence'])
    df.to_csv(savepath + f"/thesaurus_{word}_full.csv")

    print("Sampled meanings through thesaurus are: ")
    print(df.head())
