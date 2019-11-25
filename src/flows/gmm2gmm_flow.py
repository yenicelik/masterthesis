"""
    This implements a GMM to GMM flow.
    This is what we have created the synthetic GMM embeddings for,
    and also the synthetic rotation matrix M
"""
import tensorflow as tf
import tensorflow_probability
from tensorflow_probability.python.bijectors import BatchNormalization

tfd = tf.contrib.distributions
tfb = tfd.bijectors

from src.functional.pdf import pdf_gmm_diagional_covariance
from src.synthetic.generate_embeddings import generate_synthetic_src_tgt_embedding
from src.visualize.visualize_gmm import plot_two_contour3d_input2d


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
            bijectors=flow_bijector
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

    # Generate the synthetic embedding spaces, including the synthetic rotation matrix
    mus_src, cov_src, mus_tgt, cov_tgt, M = generate_synthetic_src_tgt_embedding(d=dimensions, components=components)

    # TODO: Build the pdf once, then re-use it!
    # Instantiate a pdf-embedding class for this!
    pdf_src = pdf_gmm_diagional_covariance(mus_src, cov_src)
    pdf_tgt = pdf_gmm_diagional_covariance(mus_tgt, cov_tgt)

    def lambda_pdf_src(X):
        return pdf_src.prob(X)

    def lambda_pdf_tgt(X):
        return pdf_tgt.prob(X)

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

    print("Created the flow incl. target and source distributions...")

    # Sample from the source embeddings...

