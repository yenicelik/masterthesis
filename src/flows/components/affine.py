import tensorflow as tf
tf.enable_eager_execution()
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from src.config import args
from src.functional.initializers import glorot_uniform


# TODO: Replace this by a scale and by a shift operation, as affine doesn't seem compatible with tensorflow 2!

class AffineLayer(tfb.Bijector):
    """
        Wrapper class around the Affine Bijector which encapsulates the variables as well.
    """

    def __init__(self, input_dim, r, forward_min_event_ndims=1, validate_args=False, name="affine"):
        # TODO: How do we initialize these variables?
        # TODO: No idea what r is...
        super(AffineLayer, self).__init__(forward_min_event_ndims=forward_min_event_ndims, validate_args=validate_args,
                                          name=name)
        self.initializer = glorot_uniform()

        self.V = tf.Variable(self.initializer([input_dim, r]), name=f"{name}_V", dtype=args.dtype)
        self.shift = tf.Variable(self.initializer([input_dim]), name=f"{name}_shift", dtype=args.dtype)
        self.L = tf.Variable(self.initializer([input_dim * (input_dim + 1) // 2]), name=f"{name}_L", dtype=args.dtype)

        # TODO: Turn back to "Affine"
        # Appending this to the list of bijectors
        self.bijector = tfb.Affine(
            scale_tril=tfd.fill_triangular((self.L,)),
            scale_perturb_factor=self.V,
            shift=self.shift
        )

        # self.shift = tf.Variable([10.], dtype=args.dtype)
        # self.bijector = tfb.AffineScalar(shift=self.shift)

    def _forward(self, x, name='forward', **kwargs):
        """
            Just copy the
        :param x:
        :param name:
        :param kwargs:
        :return:
        """
        # TODO: See in source code what the difference is to "forward"
        return self.bijector.forward(x, name=name, **kwargs)

    def _inverse(self, y, name='inverse', **kwargs):
        return self.bijector.inverse(y, name=name, **kwargs)

    def forward_log_det_jacobian(self,
                                 x,
                                 event_ndims,
                                 name='forward_log_det_jacobian',
                                 **kwargs):
        return self.bijector.forward_min_event_ndims(
            x,
            event_ndims,
            name='forward_log_det_jacobian',
            **kwargs
        )

    def inverse_log_det_jacobian(self,
                                 y,
                                 event_ndims,
                                 name='inverse_log_det_jacobian',
                                 **kwargs):
        return self.bijector.inverse_log_det_jacobian(y,
                                                      event_ndims,
                                                      name='inverse_log_det_jacobian',
                                                      **kwargs)
