from itertools import chain

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

import ddsp
from ddsp.core import resample
from ddsp.training.models.model import Model
from ddsp.training import nn

from .ddsp_losses import SpectralELBO


class IAF(nn.DictLayer):
    """
    A class implementing multiple IAF steps on top of a given base distribution
    the base distribution must be a scale/loc distribution (like a diagonal gaussian), or
    more precisely should take in `loc` and `scale_diag` tensor arguments
    """

    def __init__(
        self,
        z_dims,
        base_distribution,
        n_flows,
        time_steps,
        flow_hidden_units=[256, 256],
    ):
        super().__init__(output_keys=['z'])
        self.base_distribution = base_distribution
        self.z_dims = z_dims
        self.n_flows = n_flows
        self.time_steps = time_steps
        self.flow_hidden_units = flow_hidden_units
        self.dist = None
        self.cond_h = None
        self.mades = [tfb.AutoregressiveNetwork(
            params=2,
            hidden_units=flow_hidden_units,
            event_shape=(z_dims,),
            conditional=True,
            conditional_event_shape=(z_dims,),
        ) for _ in range(n_flows)]

    def call(self, z):
        """
        z is expected to be of shape (batch_size, time_steps, 3*z_dim)
        """

        # reduce the time step - z should be constant over it anyway
        z = tf.reduce_mean(z, axis=1, keepdims=True)

        params = nn.split_to_dict(z, (
            ("scale", self.z_dims),
            ("loc", self.z_dims),
            ("h", self.z_dims),
        ))

        self.dist = self.make_flow(
            params,
            self.base_distribution,
            self.get_flow_steps(
                self.z_dims,
                self.n_flows,
                self.flow_hidden_units,
                self.mades,
            ),
        )
        self.cond_h = params["h"]

        z = self.dist.sample(bijector_kwargs={f'vae_iaf_maf{i}': {'conditional_input': params["h"]} for i in range(self.n_flows)})
        # assert len(z.shape) == 3
        return resample(z, self.time_steps)

    @staticmethod
    def make_flow(params, base_distribution, flow_steps):
        dist = tfd.TransformedDistribution(
            distribution=base_distribution(
                loc=params["loc"],
                scale_diag=params["scale"],
            ),
            bijector=tfb.Chain(flow_steps),
        )

        return dist

    @staticmethod
    def get_flow_steps(z_dims, n_flows, flow_hidden_units, mades):
        return list(chain.from_iterable([
            tfb.Invert(tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=mades[i],
            ), name=f'maf{i}'),
            tfb.Permute(np.random.permutation(z_dims))
        ] for i in range(n_flows)))[:-1]


class IAFPrior(IAF):
    def __init__(
        self,
        z_dims,
        base_distribution,
        n_flows,
        time_steps,
        flow_hidden_units=[256, 256],
    ):
        super().__init__(
            z_dims,
            base_distribution,
            n_flows,
            time_steps,
            flow_hidden_units=flow_hidden_units,
        )
        self._init_z = tf.zeros((1, 1, self.z_dims * 3))
        self.params = nn.split_to_dict(self._init_z, (
            ("scale", self.z_dims),
            ("loc", self.z_dims),
            ("h", self.z_dims),
        ))
        self.dist = self.make_flow(
            self.params,
            self.base_distribution,
            self.get_flow_steps(
                self.z_dims,
                self.n_flows,
                self.flow_hidden_units,
                self.mades,
            ),
        )
        self.cond_h = self.params["h"]

    def call(self, sample_no):
        z = self.dist.sample(sample_no, bijector_kwargs={f'vae_iafprior_maf{i}': {'conditional_input': self.cond_h} for i in range(self.n_flows)})
        assert len(z.shape) == 3
        return resample(z, self.time_steps)


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
