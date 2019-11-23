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




