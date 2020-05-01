"""
    Implementing siamese network for BERT embeddings
"""
import numpy as np

import torch

from collections import Counter

import matplotlib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from torch import optim

from sklearn import datasets
from sklearn.model_selection import train_test_split

from src.metric_learning.siamese.siamese_network import create_dataset, siamese_train, Siamese
from src.resources.samplers import load_target_word_embedding_with_oversampling
from src.utils.create_experiments_folder import randomString


# TODO: Make comparing pairs only within the same word; do not put in a constraint across words!

def load_bert_embeddings_with_labels():
    print("Starting siamese network")
    tgt_word = ' was '

    rnd_str = randomString(additonal_label=f"_pydml_bn_") + "/"

    # Concat multiple words to arrive at the X that we gun train
    X1s, X2s, ys, ygs = [], [], [], []
    word_idx = []
    for idx, tgt_word in enumerate([
        ' know ',
        ' one ',
        ' have ',
        ' live ',
        ' report ',
        ' use ',
        ' was '
    ]):
        X_tmp, y_tmp = load_target_word_embedding_with_oversampling(tgt_word)

        # Do some oversampling?
        X_tmp = torch.from_numpy(X_tmp).float()
        y_tmp = torch.from_numpy(y_tmp).float()

        X1s_tmp, X2s_tmp, ys_tmp, yg_tmp = create_dataset(X_tmp, y_tmp)
        X1s_tmp, X2s_tmp, ys_tmp = X1s_tmp.detach().numpy(), X2s_tmp.detach().numpy(), ys_tmp.detach().numpy()

        assert X1s_tmp.shape[0] == X2s_tmp.shape[0], (X1s_tmp.shape, X2s_tmp.shape)
        assert X1s_tmp.shape[0] == ys_tmp.shape[0], (X1s_tmp.shape, ys_tmp.shape)
        assert X1s_tmp.shape[0] == yg_tmp.shape[0], (X1s_tmp.shape, yg_tmp.shape)

        X1s.append(X1s_tmp)
        X2s.append(X2s_tmp)
        ys.append(ys_tmp)
        ygs.append(yg_tmp + (idx * 10))
        word_idx = word_idx + [idx,] * X1s_tmp.shape[0]

        # TODO: Push the paired dataset creation into here, so we can compare pairs within the same word, but across word no constraint is put.

    X1 = np.concatenate(X1s, axis=0)
    X2 = np.concatenate(X2s, axis=0)
    y = np.concatenate(ys, axis=0)
    ygs = np.concatenate(ygs)

    # Return the indecies for which the items we look are separated ...

    n_samples = X1.shape[0]
    assert X1.shape[0] == X2.shape[0], (X1.shape[0], X2.shape[0])

    print("Number of data samples are: ", X1.shape)

    dim = X1.shape[1]
    latent_dim = 2  # Let's project this on a 2-D plane for better visualization
    samples = n_samples
    n_classes = len(np.unique(y))
    print("Number of classes are: ", n_classes)

    return tgt_word, rnd_str, X1, X2, y, ygs, n_samples, dim, latent_dim, samples, n_classes, word_idx

if __name__ == "__main__":
    print("Doing independent metric learning")
    tgt_word, rnd_str, X1, X2, y, ygs, n_samples, dim, latent_dim, samples, n_classes, word_idx = load_bert_embeddings_with_labels()
    # Implement zero shot learning afterwards

    # Siamese network
    net = Siamese(dim, latent_dim)
    # TODO Instead of doing a random train test split,
    # splt it by words (i.e. by word_idx)
    X1_train, X1_test, X2_train, X2_test, y_train, y_test, ygs_train, ygs_test = train_test_split(X1, X2, y, ygs, test_size=0.33, random_state=42)

    X1_train = torch.from_numpy(X1_train).float()
    X2_train = torch.from_numpy(X2_train).float()
    y_train = torch.from_numpy(y_train).float()
    X1_test = torch.from_numpy(X1_test).float()
    X2_test = torch.from_numpy(X2_test).float()
    y_test = torch.from_numpy(y_test).float()

    net.train()

    print("Net parameters are: ")
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(10):
        # Increase number of epochs ..
        # print([x for x in net.parameters()])
        siamese_train(model=net, X1s=X1_train, X2s=X2_train, ys=y_train, optimizer=optimizer)

    # Make a pass-forward, and visualize that
    X_hat = net.fc1.forward(X1_train).detach().numpy()
    X_pca = PCA(n_components=latent_dim).fit_transform(X1_train.detach().numpy())
    colors = [np.random.rand(3, ) for _ in np.unique(ygs_train)]
    plt_colors = [
        colors[int(label) % len(colors)]
        for label in ygs_train
    ]
    # TODO: This train is separate!! It should be the actual class color, not the binary..
    # TODO we are printing the wrong labels!!
    plt.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        c=plt_colors,
        # marker=plt_markers
    )
    plt.title(f"Using raw PCA for")
    plt.show()
    plt.clf()

    plt.scatter(
        x=X_hat[:, 0],
        y=X_hat[:, 1],
        c=plt_colors,
        # marker=plt_markers
    )
    plt.title(f"Using metric learning Siamese Network for")
    plt.show()
    plt.clf()

    # Use the projected X to arrive at the actual dataset
    # visualize_siamese_pca(net, X1_train, y_train, "train")
    # visualize_siamese_pca(net, X1_test, y_test, "test")





