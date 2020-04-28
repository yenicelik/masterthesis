"""
    Uses metric learning
    (using some common metric learning library)

"""
import numpy as np
from metric_learn import NCA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from src.resources.samplers import load_target_word_embedding_with_oversampling
from src.utils.create_experiments_folder import randomString

def load_bert_embeddings_with_labels():
    print("Starting siamese network")
    tgt_word = ' was '

    rnd_str = randomString(additonal_label=f"_siamase_bn_") + "/"

    # Concat multiple words to arrive at the X that we gun train
    Xs, ys = [], []
    for idx, tgt_word in enumerate([
        # ' know ',
        # ' one ',
        # ' have ',
        # ' live ',
        # ' report ',
        # ' use ',
        ' was '
    ]):
        X_tmp, y_tmp = load_target_word_embedding_with_oversampling(tgt_word)
        Xs.append(X_tmp)
        ys.append(y_tmp + idx * 100)

        # TODO: Push the paired dataset creation into here, so we can compare pairs within the same word, but across word no constraint is put.

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    n_samples = X.shape[0]

    print("Number of data samples are: ", X.shape)

    dim = X.shape[1]
    latent_dim = 2  # Let's project this on a 2-D plane for better visualization
    samples = n_samples
    n_classes = len(np.unique(y))
    print("Number of classes are: ", n_classes)

    return tgt_word, rnd_str, X, y, n_samples, dim, latent_dim, samples, n_classes

if __name__ == "__main__":
    print("Starting to use metric learning with independence assumption amongst words")

    tgt_word, rnd_str, X, y, n_samples, dim, latent_dim, samples, n_classes = load_bert_embeddings_with_labels()

    # Also, check out the test loss

    # Do a fit using pyDML
    nca = NCA(num_dims=latent_dim)
    X_hat = nca.fit_transform(X, y=y)
    X_pca = PCA(n_components=latent_dim).fit_transform(X)

    colors = [np.random.rand(3, ) for _ in np.unique(y)]
    plt_colors = [
        colors[int(label) % len(colors)]
        for label in y
    ]

    plt.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        c=plt_colors,
        # marker=plt_markers
    )
    plt.title(f"Using raw PCA")
    # plt.savefig(rnd_str + f"pca_{tgt_word}_{title}")

    plt.show()
    plt.clf()

    # Set a cmap perhaps
    plt.scatter(
        x=X_hat[:, 0],
        y=X_hat[:, 1],
        c=plt_colors,
        # marker=plt_markers
    )
    plt.title(f"Using metric learning NCA")
    # plt.savefig(rnd_str + f"trained_siamese_{tgt_word}_{title}")

    plt.show()
    plt.clf()
    
    



