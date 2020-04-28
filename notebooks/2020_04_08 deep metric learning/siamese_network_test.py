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

    rnd_str = randomString(additonal_label=f"_siamase_bn_") + "/"

    # Concat multiple words to arrive at the X that we gun train
    Xs, ys = [], []
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

def visualize_siamese_pca(net, X, y, title):
    # visualize assuming a 2-dimensional latent space if this is going to work
    print("Passing through final")
    X_hat = net.fc1.forward(X).detach().numpy()
    X_pca = PCA(n_components=latent_dim).fit_transform(X)
    print(X_hat.shape)

    markers = [
        # MarkerStyle(marker=x)
        # for x in ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        x for x in range(10)
    ]

    # TODO: Do a per-word training ... (currently you mixx all words together ..)

    colors = [np.random.rand(3, ) for _ in np.unique(y)]

    # plt_markers = [
    #     markers[idx % len(markers)]
    #     for _, idx in enumerate(range(X.shape[0]))
    # ]

    plt_colors = [
        colors[int(label) % len(colors)]
        for label in y
    ]
    # print("Labels are")
    # tmp = [
    #     int(label) % len(colors)
    #     for label in y
    # ]
    # print(tmp)

    # print("y is: ", y)
    print("Colors are: ", plt_colors)

    # Set a cmap perhaps
    plt.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        c=plt_colors,
        # marker=plt_markers
    )
    plt.title(f"Using the raw PCA for '{tgt_word}' {title}")
    plt.savefig(rnd_str + f"pca_{tgt_word}_{title}")

    plt.show()
    plt.clf()

    # Set a cmap perhaps
    plt.scatter(
        x=X_hat[:, 0],
        y=X_hat[:, 1],
        c=plt_colors,
        # marker=plt_markers
    )
    plt.title(f"Using the metric learning Siamese Network for '{tgt_word}' {title}")
    plt.savefig(rnd_str + f"trained_siamese_{tgt_word}_{title}")

    plt.show()
    plt.clf()

    # Also check for other words if this embedding is generalizable
    for comparand in [
        # ' know ',
        # ' one ',
        # ' have ',
        # ' live '
        # ' report ',
        # ' use ',
        # ' concern ',
        # ' allow ',
        # ' state ',
        # ' key ',
        # ' arms ',
        ' thought ',
        ' pizza ',
        ' made ',
        ' thought ',
        ' only ',
        # ' central ',
        ' pizza ',
        # ' bank ',
        # ' cold ',
        # ' table ',
        ' good ',
        ' mouse ',
        # ' was ',

    ]:
        try:
            X_comparand, y_comparand = load_target_word_embedding_with_oversampling(comparand)
            y_comparand = y_comparand.tolist()
            y_comparand = np.asarray([int(x) for x in y_comparand])
            X_comparand = torch.from_numpy(X_comparand).float()
            X_comparand = net.fc1.forward(X_comparand).detach().numpy()

            plt_colors = [
                colors[int(label) % len(colors)]
                for label in y_comparand
            ]

            # Set a cmap perhaps
            plt.scatter(
                x=X_comparand[:, 0],
                y=X_comparand[:, 1],
                c=plt_colors,
                # marker=plt_markers
            )
            plt.title(f"Using the metric learning Siamese Network for '{comparand}' {title}")
            plt.savefig(rnd_str + f"domain_adapt_siamese_{comparand}_{title}")

            plt.show()
            plt.clf()
        except Exception as e:
            print("Error with: ", comparand)
            print(e)


if __name__ == "__main__":

    tgt_word, rnd_str, X, y, n_samples, dim, latent_dim, samples, n_classes = load_bert_embeddings_with_labels()
    # Use MNIST data perhaps

    net = Siamese(dim, latent_dim)
    matr1 = torch.rand((samples, dim))
    matr2 = torch.rand((samples, dim))

    # forward pass through network
    out = net.forward(matr1, matr2)[0]
    print("Output is: ", out)
    print("Output is: ", out.shape)

    # Data matr:
    # X = torch.rand((2*samples, dim))
    # y = torch.Tensor(2*samples).random_(0, n_classes)  # Up to five different classes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    # Use ADAM optimizer instead ..
    # optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.5)
    print("Net parameters are: ")
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(30):
        # Increase number of epochs ..
        # print([x for x in net.parameters()])
        X1s, X2s, ys, _ = create_dataset(X_train, y_train)
        siamese_train(model=net, X1s=X1s, X2s=X2s, ys=ys, optimizer=optimizer)

    # Use the projected X to arrive at the actual dataset
    visualize_siamese_pca(net, X_train, y_train, "train")
    visualize_siamese_pca(net, X_test, y_test, "test")

