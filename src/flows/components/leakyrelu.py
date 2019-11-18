import tensorflow as tf
import tensorflow_probability as tfp

from src.functional.initializers import glorot_uniform

tfd = tfp.distributions
tfb = tfp.bijectors


# quite easy to interpret - multiplying by alpha causes a contraction in volume.
class _LeakyReLU(tfb.Bijector):
    def __init__(self, alpha=0.5, forward_min_event_ndims=1, validate_args=False, name="_leaky_relu"):
        super(_LeakyReLU, self).__init__(forward_min_event_ndims=forward_min_event_ndims, validate_args=validate_args, name=name)
        self.alpha = alpha

    def _forward(self, x):
        return tf.where(tf.greater_equal(x, 0), x, self.alpha * x)

    def _inverse(self, y):
        return tf.where(tf.greater_equal(y, 0), y, 1. / self.alpha * y)

    def _inverse_log_det_jacobian(self, y):
        event_dims = self._event_dims_tensor(y)
        I = tf.ones_like(y)
        J_inv = tf.where(tf.greater_equal(y, 0), I, 1.0 / self.alpha * I)
        # abs is actually redundant here, since this det Jacobian is > 0
        log_abs_det_J_inv = tf.log(tf.abs(J_inv))
        return tf.reduce_sum(log_abs_det_J_inv, axis=event_dims)


# TODO: Does this not exist in the default Bijector Class?
# TODO JUST MERGE BOTH

class LeakyReLULayer(tfb.Bijector):
    """
        Wrapper class around the LeakyReLU Bijector which encapsulates the variables as well.
    """

    def __init__(self, validate_args=False, name="_leaky_relu"):
        super(LeakyReLULayer, self).__init__(forward_min_event_ndims=1, validate_args=False, name="_leaky_relu")
        self.initializer = glorot_uniform()

        self.alpha = tf.abs(tf.Variable(self.initializer([]), name='alpha')) + 0.01
        self.bijector = _LeakyReLU(alpha=self.alpha)

    # TODO: Should we overwrite _forward, or forward for all bijector classes
    def forward(self, x, name='forward', **kwargs):
        self.bijector.forward(x, name='forward', **kwargs)

    def inverse(self, y, name='inverse', **kwargs):
        self.bijector.forward(y, name='inverse', **kwargs)

    def forward_log_det_jacobian(self,
                                 x,
                                 event_ndims,
                                 name='forward_log_det_jacobian',
                                 **kwargs):
        self.bijector.forward_log_det_jacobian(x,
                                               event_ndims,
                                               name='forward_log_det_jacobian',
                                               **kwargs
                                               )

    def inverse_log_det_jacobian(self,
                                 y,
                                 event_ndims,
                                 name='inverse_log_det_jacobian',
                                 **kwargs):
        self.bijector.inverse_log_det_jacobian(
            y,
            event_ndims,
            name='inverse_log_det_jacobian',
            **kwargs
        )
