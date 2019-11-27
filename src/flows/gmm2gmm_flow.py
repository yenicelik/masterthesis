"""
    This implements a GMM to GMM flow.
    This is what we have created the synthetic GMM embeddings for,
    and also the synthetic rotation matrix M
"""
import tensorflow as tf
import tensorflow_probability
from tensorflow_probability.python.bijectors import BatchNormalization

import matplotlib.pyplot as plt

# TODO: Make an experiment config which creates a folder and dumps everything into it

import numpy as np

tfd = tf.contrib.distributions
tfb = tfd.bijectors

from src.functional.pdf import pdf_gmm_diagional_covariance
from src.synthetic.generate_embeddings import generate_synthetic_src_tgt_embedding
from src.visualize.visualize_gmm import plot_two_contour3d_input2d

def _visualize_bijector_layers(_samples):
    results = sess.run(_samples)
    f, arr = plt.subplots(1, len(results), figsize=(4 * (len(results)), 4))
    X0 = results[0]
    X0 = X0.squeeze()
    # Convert to numpy because numpy operations..
    for i in range(len(results)):
        X1 = results[i]
        X1 = X1.squeeze()
        idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10)
        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='green')
        idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue')
        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='black')
        arr[i].set_xlim([-20, 20])
        arr[i].set_ylim([-20, 20])
        arr[i].set_title(names[i])

    plt.show()

class Gmm2gmmFlow:
    """
        Implements a flow which maps one GMM to another GMM.
        We hope to be able to model one word-gaussian model into another through this.
    """

    def hyperparmaters(self):
        """
            Implements some hyperparameters that can be tuned.
            This includes learning rate, regualizration parameters, etc.
        :return:
        """
        return {}

    def __init__(self):
        # Some common config parameters
        self.num_bijectors = 8
        self.train_iters = 2e5
        self.batch_size = 1500

        # Hyperparameters (to be refactored into hyperparameters
        self.use_batchnorm = True

        self.src_dist = None
        self.tgt_dist = None

    def inject_source_dist(self, src_dist):
        self.src_dist = src_dist

    def inject_target_dist(self, tgt_dist):
        self.tgt_dist = tgt_dist

    def init_flow(self):

        # Source distribution cannot be none
        assert self.src_dist, self.src_dist
        assert self.tgt_dist, self.tgt_dist

        bijectors = []

        # use masked autoregressive flow, just because it has a nice template
        for i in range(self.num_bijectors):
            bijectors.append(
                tfb.MaskedAutoregressiveFlow(
                    tfb.masked_autoregressive_default_template(
                        hidden_layers=[512, 512]
                    )
                )
            )
            if self.use_batchnorm:
                bijectors.append(
                    BatchNormalization(name=f"batch_norm_{i}")
                )
            # Finally permute the inputs, otherwise this will not work (why again...
            # I thought this should be random shuffling.. ah no maybe its permute i.e. transpose)
            bijectors.append(tfb.Permute(permutation=[1, 0])) # What if we have more dimensions

        # TODO: Where is the activation function
        flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))

        # Now create the normalising flow using the source distribution as the base distribution
        # TODO: Is the choice of source or target distributions as the base distribution different? Shouldn't be!
        self.flow_dist = tfd.TransformedDistribution(
            distribution=self.src_dist,
            bijector=flow_bijector
        )

        print("Flow created successfully!")
        # TODO: Does the normalising flow always map to the Gaussian embeddings...?

    def forward(self, x):
        return self.flow_dist(x)


if __name__ == "__main__":
    print("Starting to create the GMM to GMM Flow using some common Tensorflow normalising flow")

    dimensions = 2
    components = 3
    quadratic_scale = 20

    sess = tf.InteractiveSession()

    # Generate the synthetic embedding spaces, including the synthetic rotation matrix
    mus_src, cov_src, mus_tgt, cov_tgt, M = generate_synthetic_src_tgt_embedding(d=dimensions, components=components)

    # TODO: Build the pdf once, then re-use it!
    # Instantiate a pdf-embedding class for this!
    pdf_src = pdf_gmm_diagional_covariance(mus_src, cov_src)
    pdf_tgt = pdf_gmm_diagional_covariance(mus_tgt, cov_tgt)

    # Turn the matrices to numpy arrays
    def lambda_pdf_src(X):
        return sess.run(pdf_src.prob(X))

    def lambda_pdf_tgt(X):
        return sess.run(pdf_tgt.prob(X))

    # Test some sampling...
    sess.run(pdf_src.sample(500))
    sess.run(pdf_tgt.sample(500))

    print("Sampling successful!")

    # Visualize the two synthetic embeddings
    plot_two_contour3d_input2d(
        pdf1=lambda_pdf_src,
        pdf2=lambda_pdf_tgt,
        x0_min=-1. * quadratic_scale,
        x0_max=1. * quadratic_scale,
        x1_min=-1. * quadratic_scale,
        x1_max=1. * quadratic_scale,
        ringcontour=True
    )

    # Is this how we assign these distributions...
    # Instantiate the normalising flow
    flow = Gmm2gmmFlow()
    flow.inject_source_dist(pdf_src)
    flow.inject_target_dist(pdf_tgt)
    flow.init_flow()

    print("Created the flow incl. target and source distributions...")

    NP_DTYPE = np.float32

    # Sample X from the source distribution
    X = sess.run(pdf_src.sample(2000))

    print("Creating the generator for the randomly shuffled dataset..")
    dataset = tf.data.Dataset.from_tensor_slices(X.astype(NP_DTYPE))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=X.shape[0])
    dataset = dataset.prefetch(3 * flow.batch_size)
    dataset = dataset.batch(flow.batch_size)
    data_iterator = dataset.make_one_shot_iterator()
    x_samples = data_iterator.get_next()
    print("Samples drawn!")

    # Visualize some samples from the source distribution
    x = pdf_src.sample(8000)
    samples = [x]
    names = [pdf_src.name]
    print("sampled...")
    for bijector in reversed(flow.flow_dist.bijector.bijectors):
        x = bijector.forward(x)
        samples.append(x)
        names.append(bijector.name)

    sess.run(tf.global_variables_initializer())

    print("Creating the generator for the randomly shuffled dataset..")

    _visualize_bijector_layers(samples)

    print("Now do training...")

    loss = -tf.reduce_mean(flow.flow_dist.log_prob(x_samples))
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

    sess.run(tf.global_variables_initializer())

    # Now train the network to go from one distribution to another..
    NUM_STEPS = int(flow.train_iters)
    global_step = []
    np_losses = []

    for _ in range(NUM_STEPS // 5000):

        _visualize_bijector_layers(samples)

        for i in range(5000):
            _, np_loss = sess.run([train_op, loss])
            if i % 1000 == 0:
                global_step.append(i)
                np_losses.append(np_loss)
            if i % int(1e3) == 0:
                print(i, np_loss)

    _visualize_bijector_layers(samples)




