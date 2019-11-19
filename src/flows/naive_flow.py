"""
    This implements a first try of a naive flow as described in this post:
    - https://github.com/ericjang/normalizing-flows-tutorial/blob/master/nf_part1_intro.ipynb
"""
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import matplotlib.pyplot as plt

from src.config import args
from src.flows.components.affine import AffineLayer
from src.flows.components.leakyrelu import LeakyReLULayer

tfd = tfp.distributions
tfb = tfp.bijectors

# TODO: Implement an "experiment" folder with different experiments (i.e. normal to GMM; GMM to GMM, etc.)

class NaiveFlow():

    def __init__(self):
        pass

if __name__ == "__main__":
    print("Checking if the above flow model works, and to what extent")

    batch_size = 512

    def _sample_from_distribution():
        x2_dist = tfd.Normal(loc=0., scale=4.)
        x2_samples = x2_dist.sample(batch_size)
        x1 = tfd.Normal(
            loc=.25 * tf.square(x2_samples),
            scale=tf.ones(batch_size, dtype=args.dtype)
                        )

        ###### This is the dataset definition
        x1_samples = x1.sample()
        x_samples = tf.stack([x1_samples, x2_samples], axis=1)
        return x_samples


    # for i in range(2):
    #     x_samples = _sample_from_distribution()
    #     np_samples = x_samples.numpy()
    #     plt.scatter(np_samples[:, 0], np_samples[:, 1], s=10, color='red')
    #     plt.xlim([-5, 30])
    #     plt.ylim([-10, 10])
    #     plt.show()

    ######

    DTYPE = tf.float32

    # 2d is the dimension, and r is the wutttt????TODO
    d, r = 2, 2
    bijectors = []
    num_layers = 6

    variables = dict()

    for i in range(num_layers):
        bijectors.append(
            AffineLayer(input_dim=d, r=r, name=f"affine_{i}")
        )
        # bijectors.append(
        #     LeakyReLULayer()
        # )

    # Leaves out the very last LeakyRelu...
    mlp_bijector = tfb.Chain(list(reversed(bijectors[:-1])), name='mlp_bijector_2D')

    # SANITY CHECK IF EVERYTHING WAS IMPLEMENTED PROPERLY (TO A CERTAIN EXTENT)
    print("The bijector is: ")
    print(mlp_bijector)
    for x in bijectors:
        print(x)

    # Last layer is affine. Note that tfb.Chain takes a list of bijectors in the *reverse* order
    # that they are applied..

    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([2]))

    # Generating a base-transformation
    dist = tfd.TransformedDistribution(
        distribution=base_dist,
        bijector=mlp_bijector
    )

    # print("Distribution is: ")
    # print(base_dist.sample(sample_shape=(512,)))

    # visualization
    # print("Base distribution is: ", base_dist)
    x = base_dist.sample(sample_shape=(512,))
    # Saving all intermediate items in an array for easier visualization later on
    samples = [x]
    names = [base_dist.name]
    for bijector in reversed(dist.bijector.bijectors):
        # print("Bijector is:", bijector)
        # print("Forward item is: ", x.shape if x is not None else None)
        x = bijector.forward(x)
        samples.append(x)
        names.append(bijector.name)

    print("Running the samples")
    print("Number of samples is: ", len(samples))

    f, arr = plt.subplots(1, len(samples), figsize=(4 * (len(samples)), 4))
    X0 = samples[0].numpy()
    print("Converting the samples to numpy arrays for easier slicing...")
    for i in range(len(samples)):
        X1 = samples[i].numpy()
        idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='red')
        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='green')
        idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue')
        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='black')
        arr[i].set_xlim([-2, 2])
        arr[i].set_ylim([-2, 2])
        arr[i].set_title(names[i])

    print("Should now show the plot right")
    plt.ylim([-10, 10])
    plt.xlim([-10, 10])
    plt.show()

    # Reduce in a for-loop, no?

    # Optimize the flow
    NUM_STEPS = int(1e5)
    global_step = []
    np_losses = []
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)

    # TODO: Figure out how to actively assign _trainable_ variables

    # TODO: Create a new loss-model which allows for modular executioin...
    class LossModel(tf.keras.models.Model):

        def __init__(self, bijector_layer):
            super().__init__()
            self.model = bijector_layer

        def call(self, *x):
            return dict(
                loss=tf.reduce_mean(
                    self.model.model.log_prob(x)
                )
            )

    # Just going to generate this ugly class...

    variables = []
    for x in bijectors:
        variables.extend(list(x.trainable_variables))
    # variables = [x.vari]
    print("Variables are: ", variables)
    print("Variables are: ", len(variables))

    # Using an "unsupervised as supervised" approach

    @tf.function
    def loss_objective(x, y):
        _loss = y.log_prob(x)
        return -tf.reduce_mean(_loss)

    @tf.function
    def train_step(inputs):
        with tf.GradientTape() as tape:
            # TODO: Do I have to watch the gradients...
            for var in variables:
                tape.watch(var)
            tape.watch(dist.trainable_variables)
            tape.watch(inputs)
            loss = loss_objective(inputs, dist)

        gradients = tape.gradient(loss, variables)
        print("Gradients are", gradients)
        optimizer.apply_gradients(zip(gradients, variables))

        return loss.numpy()

    for i in range(NUM_STEPS):
        x_samples = _sample_from_distribution()
        loss = train_step(x_samples)

        if i % 1000 == 0:
            global_step.append(i)
            np_losses.append(loss)
        if i % int(1e4) == 0:
            print(i, loss)

        # Sample a new amount of data
        # print("x_samples are: ", x_samples.shape)
        # print("after)")
        # print("y_hat samples are: ", y_hat)
        # print("y_hat samples are: ", y_hat.shape)

        # prediction = path1(x)
        # loss = loss_fn_head1(prediction, y)
        # Simultaneously optimize trunk and head1 weights.

        # print("Loss: ", loss)
        # train_op.minimize(loss, var_list=variables)

    print("Training...")

    start = 10
    plt.plot(np_losses[start:])

    results = samples
    f, arr = plt.subplots(1, len(results), figsize=(4 * (len(results)), 4))
    X0 = samples[0].numpy()
    for i in range(len(results)):
        X1 = samples[i].numpy()
        idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='red')
        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='green')
        idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue')
        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='black')
        arr[i].set_xlim([-2, 2])
        arr[i].set_ylim([-2, 2])
        arr[i].set_title(names[i])

    plt.ylim([-10, 10])
    plt.xlim([-10, 10])
    plt.savefig('toy2d_flow.png', dpi=300)
    plt.show()

    X1 = dist.sample((4000, ))
    plt.scatter(X1[:, 0], X1[:, 1], color='red', s=2)
    arr[i].set_xlim([-2.5, 2.5])
    arr[i].set_ylim([-.5, .5])
    plt.ylim([-10, 10])
    plt.xlim([-10, 10])
    plt.savefig('toy2d_out.png', dpi=300)
    plt.show()
