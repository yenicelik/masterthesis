"""
    Replicating the normalising flow implemented by Eric Jang in
    https://blog.evjang.com/2018/01/nf2.html
    https://github.com/ericjang/normalizing-flows-tutorial
"""
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow_probability.python.bijectors import BatchNormalization

from src.config import args
from src.flows.custom_bijectors.NVPCoupling import NVPCoupling

tfd = tf.contrib.distributions
tfb = tfd.bijectors
layers = tf.contrib.layers

if __name__ == "__main__":
    print("Starting to generate a normalising flow for the SIGGRAPH items")
    # This can implement different flows including
    # Real-NVP, Inverse

    print(tf.VERSION)

    tf.set_random_seed(args.random_seed)

    sess = tf.InteractiveSession()

    NP_DTYPE = np.float32
    MODEL = 'MAF'  # Which Normalizing Flow to use. 'NVP' or 'MAF' or 'IAF'
    TARGET_DENSITY = 'SIGGRAPH'  # Which dataset to model. 'MOONS' or 'SIGGRAPH' or 'GAUSSIAN'
    USE_BATCHNORM = False

    # dataset-specific settings
    settings = {
        'SIGGRAPH': {
            'batch_size': 1500,
            'num_bijectors': 8,
            'train_iters': 2e5
        },
        'MOONS': {
            'batch_size': 100,
            'num_bijectors': 4,
            'train_iters': 2e4
        }
    }

    # TODO: Depending on debug-purposes, you can active the moon or normal distribution!
    if TARGET_DENSITY == 'SIGGRAPH':
        import pickle

        with open('../../resources/siggraph.pkl', 'rb') as f:
            X = np.array(pickle.load(f))
        X -= np.mean(X, axis=0)  # center
        xlim, ylim = [-4, 4], [-2, 2]
    elif TARGET_DENSITY == 'MOONS':
        from sklearn import cluster, datasets, mixture
        from sklearn.preprocessing import StandardScaler

        n_samples = 1000
        noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
        X, y = noisy_moons
        X = StandardScaler().fit_transform(X)
        xlim, ylim = [-2, 2], [-2, 2]
    elif TARGET_DENSITY == 'GAUSSIAN':
        mean = [0.4, 1]
        A = np.array([[2, .3], [-1., 4]])
        cov = A.T.dot(A)
        print(mean)
        print(cov)
        X = np.random.multivariate_normal(mean, cov, 2000)
        xlim, ylim = [-2, 2], [-2, 2]
    plt.scatter(X[:, 0], X[:, 1], s=10, color='red')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()

    print("Creating the generator for the randomly shuffled dataset..")
    dataset = tf.data.Dataset.from_tensor_slices(X.astype(NP_DTYPE))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=X.shape[0])
    dataset = dataset.prefetch(3 * settings[TARGET_DENSITY]['batch_size'])
    dataset = dataset.batch(settings[TARGET_DENSITY]['batch_size'])
    data_iterator = dataset.make_one_shot_iterator()
    x_samples = data_iterator.get_next()

    # Define the network that we are going to be using..

    # Now we construct the flow which goes from a simple distribution to a more complex distribution
    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([2], args.dtype))

    # Create the bijects with whatever is used as the flow-layers
    num_bijectors = settings[TARGET_DENSITY]['num_bijectors']
    bijectors = []

    for i in range(num_bijectors):
        if MODEL == 'NVP':
            bijectors.append(NVPCoupling(D=2, d=1, layer_id=i))
        elif MODEL == 'MAF':
            bijectors.append(tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                    hidden_layers=[512, 512])))
        elif MODEL == 'IAF':
            bijectors.append(tfb.Invert(tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                    hidden_layers=[512, 512]))))
        if USE_BATCHNORM and i % 2 == 0:
            # BatchNorm helps to stabilize deep normalizing flows, esp. Real-NVP
            bijectors.append(BatchNormalization(name='batch_norm%d' % i)) # TODO: Perhaps have to change this... but should be fine in theory, no?
        bijectors.append(tfb.Permute(permutation=[1, 0]))
    # Discard the last Permute layer.
    flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))

    print("Flow successfully created..")

    dist = tfd.TransformedDistribution(
        distribution=base_dist,
        bijector=flow_bijector)

    # visualization
    x = base_dist.sample(8000)
    samples = [x]
    names = [base_dist.name]
    for bijector in reversed(dist.bijector.bijectors):
        x = bijector.forward(x)
        samples.append(x)
        names.append(bijector.name)

    sess.run(tf.global_variables_initializer())

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
        arr[i].set_xlim([-10, 10])
        arr[i].set_ylim([-10, 10])
        arr[i].set_title(names[i])

    loss = -tf.reduce_mean(dist.log_prob(x_samples))
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

    sess.run(tf.global_variables_initializer())

    NUM_STEPS = int(settings[TARGET_DENSITY]['train_iters'])
    global_step = []
    np_losses = []
    for i in range(NUM_STEPS):
        _, np_loss = sess.run([train_op, loss])
        if i % 1000 == 0:
            global_step.append(i)
            np_losses.append(np_loss)
        if i % int(1e4) == 0:
            print(i, np_loss)
    start = 0
    plt.plot(np_losses[start:])

    results = sess.run(samples)
    X0 = results[0]
    rows = 2
    cols = int(len(results) / 2)
    f, arr = plt.subplots(2, cols, figsize=(4 * (cols), 4 * rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            X1 = results[i]
            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
            arr[r, c].scatter(X1[idx, 0], X1[idx, 1], s=10, color='red')
            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
            arr[r, c].scatter(X1[idx, 0], X1[idx, 1], s=10, color='green')
            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
            arr[r, c].scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue')
            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
            arr[r, c].scatter(X1[idx, 0], X1[idx, 1], s=10, color='black')
            arr[r, c].set_xlim([-5, 5])
            arr[r, c].set_ylim([-5, 5])
            arr[r, c].set_title(names[i])

            i += 1
    plt.savefig('siggraph_trained.png', dpi=300)

    # plot the last one, scaled up
    idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
    plt.scatter(X1[idx, 0], X1[idx, 1], s=10, color='red')
    idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
    plt.scatter(X1[idx, 0], X1[idx, 1], s=10, color='green')
    idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
    plt.scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue')
    idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
    plt.scatter(X1[idx, 0], X1[idx, 1], s=10, color='black')
    plt.xlim([-3, 3])
    plt.ylim([-.5, .5])
    plt.savefig('siggraph_out.png', dpi=300)

    print("Done!")
