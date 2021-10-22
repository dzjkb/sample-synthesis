from itertools import chain

import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import tensorflow.contrib.distributions.bijectors as tfb

import ddsp
from ddsp.training.models.model import Model
from ddsp.training import nn

from .ddsp_losses import SpectralELBO


class IAF(nn.DictLayer):
    """
    A class implementing multiple IAF steps on top of a given base distribution
    the base distribution must be a scale/loc distribution (like a diagonal gaussian), or
    more precisely should take in `loc` and `scale_diag` tensor arguments

    This expects input to be
    """

    def __init__(
        self,
        z_dims,
        base_distribution,
        n_flows,
        flow_hidden_units=(256, 256),
    ):
        super().__init__(output_keys=['z'])
        self.base_distribution = base_distribution
        self.z_dims = z_dims
        self.n_flows = n_flows

    def call(self, z):
        """
        z is expected to be of length 3x z_dim
        """

        # lol this block
        assert len(z.shape) == 3
        z = tf.squeeze(z)
        assert len(z.shape) == 2

        params = nn.split_to_dict(z, (
            ("scale", self.z_dims),
            ("loc", self.z_dims),
            ("h", self.z_dims),
        ))

        z = self.make_flow(z, params, self.base_distribution, self.get_flow_steps())
        return z[:, tf.newaxis, :]

    @staticmethod
    def make_flow(self, z, params, base_distribution, flow_steps):
        z = tfd.TransformedDistribution(
            distribution=base_distribution(
                loc=params["loc"],
                scale_diag=params["scale"],
            ),
            bijector=tfb.Chain(flow_steps),
        ).sample(bijector_kwargs={"conditional_input": params["h"]})

        assert len(z.shape) == 2
        return z

    @staticmethod
    def get_flow_steps(z_dims, n_flows, flow_hidden_units):
        return chain.from_iterable([
            tfb.Invert(tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                    params=2,
                    hidden_units=flow_hidden_units,
                    event_shape=(z_dims,),
                    conditional=True,
                    conditional_event_shape=(z_dims,),
                ),
            )),
            tfb.Permute(np.random.permutation(z_dims))
        ] for _ in n_flows)[:-1]


class IAFPrior(IAF):
    def call(self, sample_no) -> ['z']:
        z = tf.zeros((sample_no, 1, self.z_dims * 3))
        super().call(z)


class VAE(Model):
    """
    General class for variational autoencoders
    based on ddsp.training.models.Autoencoder
    """

    def __init__(
        self,
        preprocessor=None,
        encoder=None,
        posterior=None,
        prior=None,
        decoder=None,
        processor_group=None,
        losses=None,
        **kwargs,
    ):
        self.preprocessor = preprocessor
        super().__init__(**kwargs)
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.posterior = posterior
        self.prior = prior
        self.decoder = decoder
        self.processor_group = processor_group
        self.loss_objs = ddsp.core.make_iterable(losses)

    def encode(self, features, training=True):
        if self.preprocessor is not None:
            features.update(self.preprocessor(features, training=training))
        if self.encoder is not None:
            features.update(self.encoder(features))
        if self.posterior is not None:
            features.update(self.posterior(features))
        return features

    def decode(self, features, training=True):
        """Get generated audio by decoding than processing."""
        features.update(self.decoder(features, training=training))
        return self.processor_group(features)

    def get_audio_from_outputs(self, outputs):
        """Extract audio output tensor from outputs dict of call()."""
        return outputs['audio_synth']

    def call(self, features, training=True):
        """Run the core of the network, get predictions and loss."""
        features = self.encode(features, training=training)
        features.update(self.decoder(features, training=training))
    
        # Run through processor group.
        pg_out = self.processor_group(features, return_outputs_dict=True)
    
        # Parse outputs
        outputs = pg_out['controls']
        outputs['audio_synth'] = pg_out['signal']

        if training:
            self._update_losses_dict(
                self.loss_objs,
                features,
                outputs,
            )

        return outputs

    def sample(self, sample_no, f0_hz, loudness_db):
        features = {
            "f0_hz": f0_hz,
            "loudness_db": loudness_db,
            "sample_no": tf.constant(sample_no),
        }
        features.update(self.preprocessor(features, training=False))
        features.update(self.prior(features))
        return self.decode(features, training=False)

    def _update_losses_dict(self, loss_objs, features, outputs):
        for loss_obj in ddsp.core.make_iterable(loss_objs):
            # special case for ELBO since it needs more arguments
            if isinstance(loss_obj, SpectralELBO):
                losses_dict = loss_obj.get_losses_dict(
                    features['z'],
                    features['audio'],
                    self.get_audio_from_outputs(outputs),
                )
                continue
            if hasattr(loss_obj, 'get_losses_dict'):
                losses_dict = loss_obj.get_losses_dict(
                    features['audio'],
                    self.get_audio_from_outputs(outputs),
                )
                self._losses_dict.update(losses_dict)


class MultiLayerVAE(Model):
    """
    variational autoencoder with multiple stochastic layers
    based on https://arxiv.org/abs/1606.04934

    TODO: bidirectional or bottom-up sampling?
    """

    # TODO everything
    pass
