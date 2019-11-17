"""
    This implements a first try of a naive flow as described in this post:
    - https://github.com/ericjang/normalizing-flows-tutorial/blob/master/nf_part1_intro.ipynb
"""
import tensorflow as tf
import tensorflow_probability as tfp

from src.flows.components.leakyrelu import LeakyReLU

tfd = tfp.distributions
tfb = tfp.bijectors

# TODO: Implement an "experiment" folder with different experiments (i.e. normal to GMM; GMM to GMM, etc.)

class NaiveFlow():

    def __init__(self):
        pass

if __name__ == "__main__":
    print("Checking if the above flow model works, and to what extent")
    DTYPE = tf.float64

    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([2], DTYPE))


    # 2d is the dimension, and r is the wutttt????TODO
    d, r = 2, 2
    bijectors = []
    num_layers = 6

    variables = dict()

    class tmpSingleLayer(tf.Module):

        def __init__(self):
            self.V = tf.Variable('V', [d, r], dtype=DTYPE)
            self.shift = tf.Variable('shift', [d], dtype=DTYPE)
            self.L = tf.Variable('L', [d * (d+1) / 2], dtype=DTYPE)

            # Appending this to the list of bijectors
            bijectors.append(tfb.Affine(
                scale_tril=tfd.fill_triangular(self.L),
                scale_perturb_factor=self.V,
                shift=self.shift
            ))

            alpha = tf.abs(tf.Variable('alpha', [], dtype=DTYPE)) + 0.01
            bijectors.append( LeakyReLU(alpha=alpha) )


    layers = []
    for i in range(num_layers):
        layers.append(
            tmpSingleLayer()
        )

    mlp_bijector = tfb.Chain(list(reversed(bijectors[:-1])), name='2d_mlp_bijector')

    # Last layer is affine. Note that tfb.Chain takes a list of bijectors in the *reverse* order
    # that they are applied..

    

    # dist = tfd.TransformedDistribution(
    #     distribution=base_dist,
    #     bijector=mlp_bijector
    # )
    #
    # # visualization
    # x = base_dist.sample(512)
    # samples = [x]
    # names = [base_dist.name]
    # for bijector in reversed(dist.bijector.bijectors):
    #     x = bijector.forward(x)
    #     samples.append(x)
    #     names.append(bijector.name)
    #
    # sess.run(tf.global_variables_initializer())
    #
    # results = sess.run(samples)
    # f, arr = plt.subplots(1, len(results), figsize=(4 * (len(results)), 4))
    # X0 = results[0]
    # for i in range(len(results)):
    #     X1 = results[i]
    #     idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
    #     arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='red')
    #     idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
    #     arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='green')
    #     idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
    #     arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue')
    #     idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
    #     arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='black')
    #     arr[i].set_xlim([-2, 2])
    #     arr[i].set_ylim([-2, 2])
    #     arr[i].set_title(names[i])
    #
    # # Optimize the flow
    # loss = -tf.reduce_mean(dist.log_prob(x_samples))
    # train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    #
    # sess.run(tf.global_variables_initializer())
    #
    # NUM_STEPS = int(1e5)
    # global_step = []
    # np_losses = []
    # for i in range(NUM_STEPS):
    #     _, np_loss = sess.run([train_op, loss])
    #     if i % 1000 == 0:
    #         global_step.append(i)
    #         np_losses.append(np_loss)
    #     if i % int(1e4) == 0:
    #         print(i, np_loss)
    # start = 10
    # plt.plot(np_losses[start:])
    #
    # f, arr = plt.subplots(1, len(results), figsize=(4 * (len(results)), 4))
    # X0 = results[0]
    # for i in range(len(results)):
    #     X1 = results[i]
    #     idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
    #     arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='red')
    #     idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
    #     arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='green')
    #     idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
    #     arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue')
    #     idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
    #     arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='black')
    #     arr[i].set_xlim([-2, 2])
    #     arr[i].set_ylim([-2, 2])
    #     arr[i].set_title(names[i])
    # plt.savefig('toy2d_flow.png', dpi=300)
    #
    # X1 = sess.run(dist.sample(4000))
    # plt.scatter(X1[:, 0], X1[:, 1], color='red', s=2)
    # arr[i].set_xlim([-2.5, 2.5])
    # arr[i].set_ylim([-.5, .5])
    # plt.savefig('toy2d_out.png', dpi=300)
