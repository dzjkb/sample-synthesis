import tensorflow as tf
from ddsp import losses


class KLRegularizer(losses.Loss):
    """
    Estimated KL divergence of p and q
    logq should be log q(z|x) - the posterior logprob of z
    logp should be log p(z) - the prior logprob of z
    """

    def __init__(self, weight, name="kl_term"):
        super().__init__(name=name)
        self.weight = weight

    def call(self, logq, logp):
        return tf.constant(self.weight) * tf.reduce_sum(logq - logp)
