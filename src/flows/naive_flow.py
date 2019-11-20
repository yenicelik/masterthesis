"""
    This implements a first try of a naive flow as described in this post:
    - https://github.com/ericjang/normalizing-flows-tutorial/blob/master/nf_part1_intro.ipynb
"""
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import matplotlib.pyplot as plt

from src.config import args
from src.flows.components.LeakyReLU import LeakyReLU

tfd = tfp.distributions
tfb = tfp.bijectors

# TODO: Implement an "experiment" folder with different experiments (i.e. normal to GMM; GMM to GMM, etc.)

if __name__ == "__main__":
    print("Checking if the above flow model works, and to what extent")

    tf.set_random_seed(args.random_seed)
    sess = tf.InteractiveSession()
    batch_size=512
    DTYPE=tf.float32
    NP_DTYPE=np.float32

    x2_dist = tfd.Normal(loc=0., scale=4.)
    x2_samples = x2_dist.sample(batch_size)
    x1 = tfd.Normal(loc=.25 * tf.square(x2_samples),
                    scale=tf.ones(batch_size, dtype=DTYPE))
    x1_samples = x1.sample()
    x_samples = tf.stack([x1_samples, x2_samples], axis=1)
    np_samples = sess.run(x_samples)
    plt.scatter(np_samples[:, 0], np_samples[:, 1], s=10, color='red')
    plt.xlim([-5, 30])
    plt.ylim([-10, 10])
    plt.show()

    # This is the distribution we want to mimick
    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([2], DTYPE))

    print("Creating flow..")

    d, r = 2, 2
    bijectors = []
    num_layers = 6
    for i in range(num_layers):
        with tf.variable_scope('bijector_%d' % i):
            V = tf.get_variable('V', [d, r], dtype=DTYPE)  # factor loading
            shift = tf.get_variable('shift', [d], dtype=DTYPE)  # affine shift
            L = tf.get_variable('L', [d * (d + 1) / 2],
                                dtype=DTYPE)  # lower triangular
            bijectors.append(tfb.Affine(
                scale_tril=tfd.fill_triangular(L),
                scale_perturb_factor=V,
                shift=shift,
            ))
            alpha = tf.abs(tf.get_variable('alpha', [], dtype=DTYPE)) + .01
            bijectors.append(LeakyReLU(alpha=alpha))
    # Last layer is affine. Note that tfb.Chain takes a list of bijectors in the *reverse* order
    # that they are applied..
    mlp_bijector = tfb.Chain(
        list(reversed(bijectors[:-1])), name='2d_mlp_bijector')

    print("Creating the distribution based on the flow and simple distribution!")
    dist = tfd.TransformedDistribution(
        distribution=base_dist,
        bijector=mlp_bijector
    )

    # visualization
    x = base_dist.sample(512)
    samples = [x]
    names = [base_dist.name]
    for bijector in reversed(dist.bijector.bijectors):
        x = bijector.forward(x)
        samples.append(x)
        names.append(bijector.name)

    sess.run(tf.compat.v1.global_variables_initializer())

    results = sess.run(samples)
    f, arr = plt.subplots(1, len(results), figsize=(4 * (len(results)), 4))
    X0 = results[0]
    for i in range(len(results)):
        X1 = results[i]
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

    plt.show()

    loss = -tf.reduce_mean(dist.log_prob(x_samples))
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    sess.run(tf.compat.v1.global_variables_initializer())

    print("Optimizing weights...")

    NUM_STEPS = int(1e5)
    global_step = []
    np_losses = []
    for i in range(NUM_STEPS):
        _, np_loss = sess.run([train_op, loss])
        if i % 1000 == 0:
            global_step.append(i)
            np_losses.append(np_loss)
        if i % int(1e4) == 0:
            print(i, np_loss)
    start = 10
    plt.plot(np_losses[start:])

    results = sess.run(samples)
    f, arr = plt.subplots(1, len(results), figsize=(4 * (len(results)), 4))
    X0 = results[0]
    for i in range(len(results)):
        X1 = results[i]
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
    plt.savefig('toy2d_flow.png', dpi=300)

    plt.show()

    print("Printing the final show..")

    X1 = sess.run(dist.sample(4000))
    plt.scatter(X1[:, 0], X1[:, 1], color='red', s=2)
    arr[i].set_xlim([-2.5, 2.5])
    arr[i].set_ylim([-.5, .5])
    plt.savefig('toy2d_out.png', dpi=300)

    plt.show()

