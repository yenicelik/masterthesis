"""
    Implements the NVP Coupling layer as proposed in:
        - https://arxiv.org/abs/1605.08803
"""
import tensorflow as tf
import tensorflow_probability as tfp

from src.config import args

tfd = tfp.distributions
tfb = tfp.bijectors
layers = tf.contrib.layers

def net(x, out_size):
    hidden_size = 128 # 512
    return layers.stack(x, layers.fully_connected, [hidden_size, hidden_size, out_size]) # Very high size lol, should decrease this perhaps, eastman has less diimensions lol

class NVPCoupling(tfb.Bijector):
    """NVP affine coupling layer for 2D units.
    """

    def __init__(self, D, d, layer_id=0, validate_args=False, name="NVPCoupling"):
        """
        Args:
          d: First d units are pass-thru units.
        """
        # first d numbers decide scaling/shift factor for remaining D-d numbers.
        super(NVPCoupling, self).__init__(
            event_ndims=1, validate_args=validate_args, name=name)
        self.D, self.d = D, d
        self.id = layer_id
        # create variables here
        tmp = tf.placeholder(dtype=args.dtype, shape=[1, self.d])
        self.s(tmp)
        self.t(tmp)

    def s(self, xd):
        with tf.variable_scope('s%d' % self.id, reuse=tf.AUTO_REUSE):
            return net(xd, self.D - self.d)

    def t(self, xd):
        with tf.variable_scope('t%d' % self.id, reuse=tf.AUTO_REUSE):
            return net(xd, self.D - self.d)

    def _forward(self, x):
        xd, xD = x[:, :self.d], x[:, self.d:]
        yD = xD * tf.exp(self.s(xd)) + self.t(xd)  # [batch, D-d]
        return tf.concat([xd, yD], axis=1)

    def _inverse(self, y):
        yd, yD = y[:, :self.d], y[:, self.d:]
        xD = (yD - self.t(yd)) * tf.exp(-self.s(yd))
        return tf.concat([yd, xD], axis=1)

    def _forward_log_det_jacobian(self, x):
        event_dims = self._event_dims_tensor(x)
        xd = x[:, :self.d]
        return tf.reduce_sum(self.s(xd), axis=event_dims)