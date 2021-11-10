import tensorflow as tf
from ddsp import losses


class KLRegularizer(losses.Loss):
    """
    Estimated KL divergence of p and q
    logq should be log q(z|x) - the posterior logprob of z
    logp should be log p(z) - the prior logprob of z
    """

    def __init__(self, weight, name="kl_term", kl_min=0):
        super().__init__(name=name)
        self.weight = weight
        self.kl_min = kl_min

    def call(self, logq, logp):
        return tf.maximum(tf.constant(self.kl_min, dtype=tf.float32), tf.constant(self.weight, dtype=tf.float32) * tf.reduce_mean(logq - logp))
