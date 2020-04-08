"""
    Implements a siamese network for metric learning.

    We want to learn an interpretaion which is able to push common classes together,
    and distinct ones away from each other.

    Hopefully we can create a general representation using this.
"""
import torch
from torch import nn, optim
from torch.distributions import Categorical


class Siamese(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(Siamese, self).__init__()

        # Let's just use linear layers ..
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(latent_dim, 10),
            # nn.Linear(10, 2)
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

# Write a short training loop
def siamese_train(model, X, y, epochs, optimizer):
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

    print("Training the siamese network (1)")

    batch_size = 4

    assert X.shape[0] == y.shape[0], (X.shape, y.shape)
    assert len(y.shape) == 1, ("Y should be 1-dimensional!", y.shape)

    print("Training the siamese network (1.1)")

    # indices = torch.arange(X.shape[0])
    print(torch.ones(size=(batch_size, X.shape[0],)) / float(X.shape[0]))
    categorical = Categorical(torch.ones(size=(batch_size, X.shape[0],)) / float(X.shape[0]))

    print("Training the siamese network (1.2)")

    criterion = nn.MSELoss()

    print("Training the siamese network (2)")

    # Prepare the batches here
    for e in range(epochs):

        # SemCor really does not have many labels lol
        # As such, the batch size is very small
        for batch_idx in range(0, X.shape[0], batch_size):

            # Zero our any previous gradients
            optimizer.zero_grad()

            # Select indices twice
            inp1_indices = categorical.sample()
            inp2_indices = categorical.sample()

            print("Input 1 and 2 indices are: ")
            print(inp1_indices)
            print(inp2_indices)

            inp1 = X[inp1_indices]
            inp2 = X[inp2_indices]

            print("Input 1 and 2 are: ")
            print(inp1)
            print(inp2)

            # the distance should be determined by whether or not the comparands
            # are in the same "cluster"
            # Create a log-likelihood loss from here indeed
            distance = torch.zeros(batch_size)
            print("A")
            print(y[inp1_indices], y[inp2_indices])
            print(torch.eq(y[inp1_indices], y[inp2_indices]))
            print("Distance size is: ", distance)
            distance[torch.eq(y[inp1_indices], y[inp2_indices])] = 1.

            distance_hat, _, _ = model(inp1, inp2)

            # Create a zero-matrix whenever the above condition is fulfilled (else one!)
            # # Difference between the predicted and the real distance
            loss = criterion(distance, distance_hat)

            print("Loss is: {:.5f}".format(loss.item()))

    print("Training the siamese network (3)")

    # We learn distances that are all within 0. and 1. as range

# Write a short evaluation loop

if __name__ == "__main__":
    print("Starting siamese network")

    dim = 5
    latent_dim = 2  # Let's project this on a 2-D plane for better visualization
    samples = 10

    net = Siamese(dim, latent_dim)
    matr1 = torch.rand((10, dim))
    matr2 = torch.rand((10, dim))

    # forward pass through network
    out = net.forward(matr1, matr2)[0]
    print("Output is: ", out)
    print("Output is: ", out.shape)

    # Data matr:
    X = torch.rand((2*samples, dim))
    y = torch.Tensor(2*samples).random_(0, 5)  # Up to five different classes

    # Use ADAM optimizer instead ..
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    siamese_train(model=net, X=X, y=y, epochs=1, optimizer=optimizer)
