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
            # # nn.Linear(latent_dim, latent_dim, bias=False),
            # # nn.ReLU(inplace=True),
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
        # Remove the sqrt!
        distance = torch.sqrt(
            torch.sum((output1 - output2) * (output1 - output2), 1) + 0.01
        )

        # returns a squared distance metric
        return distance, output1, output2

def create_dataset(X, y, shuffle=True):
    """
        Creates tuples of 'batch_size' over which you can iterate,
        which have balanced pairs of datasets
    :param X:
    :param y:
    :param batch_size:
    :return:
    """

    def _sample_pos(base_idx, X, y):
        base_class = y[base_idx]
        pos_idx = np.random.choice(np.arange(y.shape[0])[y == base_class])
        return X[base_idx], X[pos_idx], torch.Tensor([1.])  # 0 siganlises that it is the same class!

    def _sample_neg(base_idx, X, y):
        base_class = y[base_idx]
        neg_idx = np.random.choice(np.arange(y.shape[0])[y != base_class])
        return X[base_idx], X[neg_idx], torch.Tensor([0.])  # 0 siganlises that it is the same class!

    # Prepare final dataset items
    X1s, X2s, ys = [], [], []

    # Prepare triplets
    idx = [x for x in range(X.shape[0])]
    if shuffle:
        random.shuffle(idx)
    for i in idx:
        # Add one positive example
        base_X1, pos_X, pos_y = _sample_pos(i, X, y)
        # Add one negative example
        base_X2, neg_X, neg_y = _sample_neg(i, X, y)

        assert torch.all(torch.eq(base_X1, base_X2))

        X1s.append(base_X1.reshape(1, -1))
        X2s.append(pos_X.reshape(1, -1))
        ys.append(pos_y.reshape(1, ))
        X1s.append(base_X2.reshape(1, -1))
        X2s.append(neg_X.reshape(1, -1))
        ys.append(neg_y.reshape(1, ))

    # Create the datasets, so you can slice them into batches in a bit
    X1s, X2s, ys = torch.cat(X1s, 0), torch.cat(X2s, 0), torch.cat(ys, 0)

    assert len(X1s) == len(X2s)
    assert len(X2s) == len(ys)

    return X1s, X2s, ys


# Write a short training loop
def siamese_train(model, X1s, X2s, ys, optimizer, batch_size=32):
    """
        X_tpl is a tuple.

        Each element in the array consists of samples,
        which are specific to one class.
        If samples are in the same class , they should be pushed toghether.
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
    assert len(X1s) == len(X2s)
    assert len(X2s) == len(ys)

    m = 10.0

    # SemCor really does not have many labels lol
    # As such, the batch size is very small
    for idx in range(0, X1s.shape[0], batch_size):
        inp1, inp2, labels = X1s[idx:idx+batch_size, :], X2s[idx:idx+batch_size, :], ys[idx:idx+batch_size]

        assert inp1.shape[0] == inp2.shape[0], (inp1.shape, inp2.shape)
        assert inp1.shape[0] == labels.shape[0], (inp1.shape, labels.shape)
        assert len(labels.shape) == 1, ("Y should be 1-dimensional!", labels.shape)

        # TODO: Must sample in such a way, that the y consists of uniform

        # A label of "1" means that the two elements are in the
        # same class of some sort
        # print("shapes are")
        # print(labels.shape)
        # print(inp1.shape)
        # print(inp2.shape)
        # print("Labels are: ", labels)
        # print(np.count_nonzero(labels))

        # print("Inputs are: ")
        # print(inp1, inp2)

        distance, _, _ = model(inp1, inp2)
        # print("Distance is:", distance)
        loss = labels * torch.pow(distance, 2)
        # print("Loss is: ", loss)
        loss += (1. - labels) * torch.pow(torch.max(torch.Tensor([0.]), m - distance), 2)

        # Weights are:
        # print(model.fc1[0].weight)

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
        # Zero our any previous gradients
        optimizer.zero_grad()

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


    # Use ADAM optimizer instead ..
    # optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.5)
    print("Net parameters are: ")
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(10):
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
