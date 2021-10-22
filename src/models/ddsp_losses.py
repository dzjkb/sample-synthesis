from ddsp import losses


class SpectralELBO(losses.Loss):
    """
    Estimated lower bound with spectral loss instead of standard likelihood
    """

    def __init__(
        self,
        posterior,
        prior,
        name='elbo_loss',
        **kwargs,
    ):
        super().__init__(name=name)
        self._spectral_loss = losses.SpectralLoss(
            loss_type='L1',
            mag_weight=1.0,
            logmag_weight=1.0,
        )
        self._posterior = posterior
        self._prior = prior

    def call(self, z, target_audio, audio):
        # TODO: get KL term
        kl_term = None

        spec_term = self._spectral_loss(target_audio, audio)

        return kl_term + spec_term
