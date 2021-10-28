import tensorflow as tf
from ddsp import losses


class KLRegularizer(losses.Loss):
    """
    Estimated KL divergence of p and q
    logq should be log q(z|x) - the posterior logprob of z
    logp should be log p(z) - the prior logprob of z
    """

    def __init__(self, name="kl_term"):
        super().__init__(name=name)

    def call(self, logq, logp):
        return tf.reduce_sum(logq - logp)
