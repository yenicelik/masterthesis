"""
    We will do pairplots based of predicted vs real semantic labels.
    It seems more accurate to predic this than the PoS in the end
    (which is quite weird..)
"""
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

def visualize_true_cluster_embeddings(X, savepath, tgt_word, cluster_labels, title="PoS"):

    dist = pd.DataFrame(
        data=X, columns=[f'lat{i}' for i in range(X.shape[1])]
    )
    colors = pd.DataFrame(
        data=cluster_labels, columns=[title]
    )
    dist = pd.concat([dist, colors], axis=1)

    # Apply some visualization based on seaborn distribution analysis
    sns.pairplot(dist, hue=title)

    plt.savefig(savepath + f'/matplotlib_pairplot_{tgt_word}_true.png')
    plt.show()

def visualize_predicted_cluster_embeddings(X, savepath, tgt_word, cluster_labels_pred, title="PoS"):
    dist = pd.DataFrame(
        data=X, columns=[f'lat{i}' for i in range(X.shape[1])]
    )
    # Append to dataframe another dimension which is the
    colors = pd.DataFrame(
        data=cluster_labels_pred, columns=[title]
    )
    dist = pd.concat([dist, colors], axis=1)

    # Apply some visualization based on seaborn distribution analysis
    sns.pairplot(dist, hue=title)

    plt.savefig(savepath + f'/matplotlib_pairplot_{tgt_word}_pred.png')
    plt.show()

if __name__ == "__main__":
    print("Visualizing the predicted vs real wordnet meanings")
