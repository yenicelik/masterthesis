"""
    Implementing siamese network for BERT embeddings
"""
import random
import numpy as np

import torch

from collections import Counter

from sklearn.decomposition import PCA
from torch import optim

from sklearn import datasets
from sklearn.model_selection import train_test_split

from src.metric_learning.siamese.siamenese_network import create_dataset, siamese_train, Siamese
from src.resources.samplers import sample_semcor_data


def load_bert_embeddings_with_labels():
    pass

def load_target_word_embedding_with_oversampling(tgt_word):
    # Use items with more words ...
    X, y, _ = sample_semcor_data(tgt_word=tgt_word)
    # Just do more epochs I guess..?
    print(Counter(y))

    # TODO: Oversample here because highly unbalanced dataset
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X, y = ros.fit_resample(X, y)

    print(Counter(y))
    print(X.shape, y.shape)

    y = y.tolist()
    y = np.asarray([int(x) for x in y])

    return X, y


if __name__ == "__main__":

    print("Starting siamese network")
    tgt_word = ' was '

    X, y = load_target_word_embedding_with_oversampling(tgt_word)

    n_samples = X.shape[0]

    print("Number of data samples are: ", X.shape)

    dim = X.shape[1]
    latent_dim = 2  # Let's project this on a 2-D plane for better visualization
    samples = n_samples
    n_classes = len(np.unique(y))
    print("Number of classes are: ", n_classes)

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
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()


    # Use ADAM optimizer instead ..
    # optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.5)
    print("Net parameters are: ")
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(30):
        # Increase number of epochs ..
        # print([x for x in net.parameters()])
        X1s, X2s, ys = create_dataset(X, y)
        siamese_train(model=net, X1s=X1s, X2s=X2s, ys=ys, optimizer=optimizer)

    import matplotlib
    import matplotlib.pyplot as plt

    # Use the projected X to arrive at the actual dataset

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

    colors = [np.random.rand(3,) for _ in np.unique(y)]

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
    plt.title(f"Using the raw PCA for '{tgt_word}'")

    plt.show()
    plt.clf()

    # Set a cmap perhaps
    plt.scatter(
        x=X_hat[:, 0],
        y=X_hat[:, 1],
        c=plt_colors,
        # marker=plt_markers
    )
    plt.title(f"Using the metric learning Siamese Network for '{tgt_word}'")

    plt.show()
