"""
    Implements a siamese network for metric learning.

    We want to learn an interpretaion which is able to push common classes together,
    and distinct ones away from each other.

    Hopefully we can create a general representation using this.
"""
import random
import numpy as np

import torch
from matplotlib.markers import MarkerStyle
from sklearn.decomposition import PCA
from torch import nn, optim
from torch.distributions import Categorical


# TODO: we need to make sure approx 50% of images are in the same class
# I think this is the reason I get so much nan ..

class Siamese(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(Siamese, self).__init__()

        # Let's just use linear layers ..
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, latent_dim, bias=False),  # Should not require bias ...
            # nn.ReLU(inplace=True),
            # nn.Linear(latent_dim * 3, latent_dim*3, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Linear(latent_dim * 3, latent_dim*3, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Linear(latent_dim * 3, latent_dim*3, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Linear(latent_dim * 3, latent_dim*3, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Linear(latent_dim*3, latent_dim, bias=False)
        )

        # Probably should have a much bigger latent dimensionality

        # self.cnn2 = self.cnn1
        # self.fc2= self.fc1

        # print self.cnn1 is self.cnn2

    def forward(self, input1, input2):
        output1 = self.fc1(input1)
        output2 = self.fc1(input2)
        # print output1 - output2
        distance = torch.sqrt(torch.sum((output1 - output2) * (output1 - output2), 1))

        # returns a squared distance metric
        return distance, output1, output2

def create_dataset(X, y, batch_size):
    """
        Creates tuples of 'batch_size' over which you can iterate,
        which have balanced pairs of datasets
    :param X:
    :param y:
    :param batch_size:
    :return:
    """
    categorical = Categorical(
        torch.ones(size=(len(np.unique(y)),)) / float(len(np.unique(y)))
    )

    out = []

    def _sample_base_positive_negative(base_idx, X, y):
        # Just fckn use numpy
        base_class = y[base_idx]
        pos_idx = np.random.choice(np.arange(y.shape[0])[y == base_class])
        neg_idx = np.random.choice(np.arange(y.shape[0])[y != base_class])
        return X[base_idx], y[base_idx], X[pos_idx], y[pos_idx], X[neg_idx], y[neg_idx],

    # Prepare triplets
    X_triples = [
        _sample_base_positive_negative(x, X, y)
        for x in range(X.shape[0])
    ]

    print(X_triples)
    print("X triplets are")
    print(len(X_triples))


    exit()


    for i in range(0, X.shape[0], batch_size):

        # Do the full following mechanism twice, so we have contrastive loss!

        # Is it ok if we are always focusing on only one class?
        # I think sampling over multiple classes is also a good idea ...
        # (We could concatenate in this case ...)

        ################
        # Base samples
        ################
        majority_class = categorical.sample()
        print("Majority class is: ", majority_class)
        # Select one "majority" class
        pos_idx = torch.where(y == majority_class)[0]
        neg_idx = torch.where(y != majority_class)[0]

        # This will cause a variable batch size! (which is fine when we use pytorch..)
        k = min(len(pos_idx), batch_size)
        k = min(k, len(neg_idx))

        print("Positive and negative idx are: ")
        print(pos_idx)
        print(neg_idx)

        # Choose "base", i.e. items that we will be comparing to ..
        base_perm = torch.randperm(pos_idx.size(0))
        base_idx = pos_idx[pos_perm[:batch_size // 2]]

        # Choose positive-examples (low distance)
        pos_perm = torch.randperm(pos_idx.size(0))
        pos_idx = pos_idx[pos_perm[:batch_size // 2]]

        # Choose negative-examples (high distance)
        neg_perm = torch.randperm(neg_idx.size(0))
        neg_idx = neg_idx[neg_perm[:batch_size // 2]]

        # samples = tensor[idx]
        X_batch = torch.cat([X[pos_idx], X[neg_idx]], 0)
        y_batch = torch.cat([y[pos_idx], y[neg_idx]], 0)

        print("Select pos neg idx are")
        print(pos_idx)
        print(neg_idx)

        print("X and y shapes are")
        print(X_batch)
        print(y_batch)
        print(X_batch.shape)
        print(y_batch.shape)

        out.append(X_batch, y_batch)

    return out


# Write a short training loop
def siamese_train(model, X_tpl, y_tpl, epochs, optimizer):
    """
        X_tpl is a tuple.

        Each element in the array consists of samples,
        which are specific to one class.
        If samples are in the same class, they should be pushed toghether.
        If samples are not within the same class, they should be pushed away.

        In other words,
        inputs are X, y.
        We select pairs from X (and correspondingly y).
        If the class corresponds, these should be pushed closer together.
        If not, the classes should be pushed away from each other.

    :param model:
    :param X:
    :param epochs:
    :param optimizer:
    :return:
    """
    model.train()
    assert len(X_tpl) == len(y), (len(X), len(y))

    m = 10.

    # Prepare the batches here
    for e in range(epochs):

        # SemCor really does not have many labels lol
        # As such, the batch size is very small
        for X_batch, y_batch in zip(X_tpl, y_tpl):

            assert X.shape[0] == y[0].shape[0], (X.shape, y.shape)
            assert len(y.shape) == 1, ("Y should be 1-dimensional!", y.shape)

            # Zero our any previous gradients
            optimizer.zero_grad()

            # TODO: Must sample in such a way, that the y consists of uniform

            # A label of "1" means that the two elements are in the
            # same class of some sort
            label = torch.ones(batch_size)
            label[torch.eq(y[inp1_indices], y[inp2_indices])] = 0.

            print("Labels are: ", label)
            print(np.count_nonzero(label))

            # print("Inputs are: ")
            # print(inp1, inp2)

            distance, _, _ = model(inp1, inp2)
            # print("Distance is:", distance)
            loss = (1. - label) * 0.5 * torch.pow(distance, 2)
            # print("Loss is: ", loss)
            loss += label * 0.5 * torch.pow(torch.clamp(m - distance, 0., 12.), 2)

            # print("Loss is: ", loss)

            # take mean for the batch-wise loss
            # print("Loss shape is: ", loss.shape)
            loss = torch.mean(loss)

            # Create a zero-matrix whenever the above condition is fulfilled (else one!)
            # # Difference between the predicted and the real distance
            # Let's check if the loss function can be reverted back lol

            # need to run optimizer.backwards

            print("Loss is: {:.5f}".format(loss.item()))
            loss.backward()
            optimizer.step()

    print("Training the siamese network (3)")

    # We learn distances that are all within 0. and 1. as range

# Write a short evaluation loop

if __name__ == "__main__":
    print("Starting siamese network")

    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    X, _, y, _ = train_test_split(
        X, y, test_size=0.95, shuffle=True)

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

    X_tpls, y_tpls = create_dataset(X, y, batch_size=16)

    # Use ADAM optimizer instead ..
    # optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.5)
    optimizer = optim.Adam(net.parameters(), lr=0.0005, weight_decay=0.95)
    # Increase number of epochs ..
    siamese_train(model=net, X_tpl=X_tpls, y_tpl=y_tpls, epochs=1, optimizer=optimizer)

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

    print("y is: ", y)
    print("Colors are: ", plt_colors)

    # Set a cmap perhaps
    plt.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        c=plt_colors,
        # marker=plt_markers
    )
    plt.title("Using the raw PCA")

    plt.show()
    plt.clf()

    # Set a cmap perhaps
    plt.scatter(
        x=X_hat[:, 0],
        y=X_hat[:, 1],
        c=plt_colors,
        # marker=plt_markers
    )
    plt.title("Using the metric learning Siamese Network")

    plt.show()

