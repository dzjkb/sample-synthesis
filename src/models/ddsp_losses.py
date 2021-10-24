import tensorflow as tf
from ddsp import losses


class SpectralELBO(losses.Loss):
    """
    Estimated lower bound with spectral loss instead of likelihood
    """

    def __init__(
        self,
        posterior,
        prior,
        name='elbo_loss',
        **kwargs,
    ):
        super().__init__(name=name)
        self._spectral_loss = losses.SpectralLoss(**kwargs)
        self._posterior = posterior
        self._prior = prior

    def call(self, z, target_audio, audio):
        # z should be constant over time, reduce time step dim
        z = tf.reduce_mean(z, axis=1, keepdims=False)

        logq = self._posterior.dist.log_prob(z, bijector_kwargs={f'vae_iaf_maf{i}': {'conditional_input': self._posterior.cond_h} for i in range(self._posterior.n_flows)})
        # the bijector name here doesn't have a 'vae_iaf' prefix since it's built here, outside of the VAE/IAF classes
        logp = self._prior.dist.log_prob(z, bijector_kwargs={f'maf{i}': {'conditional_input': self._prior.cond_h} for i in range(self._prior.n_flows)})
        kl_term = logq - logp
        spec_term = self._spectral_loss(target_audio, audio)

        return tf.reduce_mean(kl_term + spec_term)
