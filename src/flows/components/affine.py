
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from src.config import args
from src.functional.initializers import glorot_uniform

class AffineLayer(tfb.Bijector):
    """
        Wrapper class around the Affine Bijector which encapsulates the variables as well.
    """

    def __init__(self, input_dim, r, forward_min_event_ndims=1, validate_args=False, name="affine"):
        # TODO: How do we initialize these variables?
        # TODO: No idea what r is...
        super(AffineLayer, self).__init__(forward_min_event_ndims=forward_min_event_ndims, validate_args=validate_args, name=name)
        self.initializer = glorot_uniform()
        # Use GLOROT UNIFORM
        self.V = tf.Variable(self.initializer([input_dim, r]), name='V', dtype=args.dtype)
        self.shift = tf.Variable(self.initializer([input_dim]), name='shift', dtype=args.dtype)
        self.L = tf.Variable(self.initializer([input_dim * (input_dim + 1) // 2]), name='L', dtype=args.dtype)

        # Appending this to the list of bijectors
        self.bijector = tfb.Affine(
            scale_tril=tfd.fill_triangular((self.L,)),
            scale_perturb_factor=self.V,
            shift=self.shift
        )

    def forward(self, x, name='forward', **kwargs):
        """
            Just copy the
        :param x:
        :param name:
        :param kwargs:
        :return:
        """
        return self.bijector.forward(x, name=name, **kwargs)

    def inverse(self, y, name='inverse', **kwargs):
        return self.bijector.inverse(y, name=name, **kwargs)
