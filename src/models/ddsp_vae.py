from itertools import chain

import numpy as np
import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
# import tensorflow.keras.layers as tfkl

import ddsp
from ddsp.core import resample
from ddsp.training.models.model import Model
from ddsp.training import nn

from .ddsp_losses import KLRegularizer


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
        flow_hidden_units=[32, 32],
        name=None,
        output_keys=('z', 'logq'),
    ):
        super().__init__(output_keys=list(output_keys), name=name)
        self.base_distribution = base_distribution
        self.z_dims = z_dims
        self.n_flows = n_flows
        self.time_steps = time_steps
        self.flow_hidden_units = flow_hidden_units

        self.mades = [tfb.AutoregressiveNetwork(
            params=2,
            hidden_units=flow_hidden_units,
            event_shape=(z_dims,),
            conditional=True,
            conditional_event_shape=(z_dims,),
        ) for i in range(n_flows)]
        self.flow_steps = self.get_bijector(
            self.z_dims,
            self.n_flows,
            self.flow_hidden_units,
            self.mades
        )
        self.cond_h = None
        self.dist = None

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
            self.z_dims,
            self.base_distribution,
            self.flow_steps,
        )

        self.cond_h = tf.squeeze(params["h"])
        bij_kwargs = {f'maf{i}': {'conditional_input': self.cond_h} for i in range(self.n_flows)}
        z = self.dist.sample(bijector_kwargs=bij_kwargs)
        logq = self.dist.log_prob(z, bijector_kwargs=bij_kwargs)

        return resample(z[:, tf.newaxis, :], self.time_steps), logq

    @staticmethod
    def make_flow(params, z_dims, base_distribution, flow_steps):
        dist = tfd.TransformedDistribution(
            distribution=base_distribution(
                loc=tf.squeeze(params["loc"]),
                scale_diag=tf.squeeze(params["scale"]),
            ),
            bijector=flow_steps,
        )
        return dist

    @staticmethod
    def get_bijector(z_dims, n_flows, flow_hidden_units, mades):
        return tfb.Chain(list(chain.from_iterable([
            tfb.Invert(tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=mades[i],
            ), name=f"maf{i}"),
            tfb.Permute(np.random.permutation(z_dims))
        ] for i in range(n_flows)))[:-1])


class IAFPrior(IAF):
    def __init__(
        self,
        z_dims,
        base_distribution,
        n_flows,
        time_steps,
        flow_hidden_units=[32, 32],
    ):
        super().__init__(
            z_dims,
            base_distribution,
            n_flows,
            time_steps,
            flow_hidden_units=flow_hidden_units,
            output_keys=('z', 'logp'),
        )
        self.params = {
            "scale": tf.ones((1, 1, self.z_dims)),
            "loc": tf.zeros((1, 1, self.z_dims)),
            "h": tf.zeros((1, 1, self.z_dims)),
        }
        self.dist = self.make_flow(
            self.params,
            self.z_dims,
            self.base_distribution,
            self.flow_steps,
        )
        self.cond_h = tf.squeeze(self.params["h"])

    def sample(self, sample_no):
        bij_kwargs = {f'maf{i}': {'conditional_input': self.cond_h} for i in range(self.n_flows)}
        z = self.dist.sample(sample_no, bijector_kwargs=bij_kwargs)
        return resample(z[:, tf.newaxis, :], self.time_steps)

    def call(self, z):
        # reduce the time step - z should be constant over it anyway
        z_2d = tf.reduce_mean(z, axis=1, keepdims=True)
        bij_kwargs = {f'maf{i}': {'conditional_input': self.cond_h} for i in range(self.n_flows)}
        return z, self.dist.log_prob(z_2d, bijector_kwargs=bij_kwargs)


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
        if self.prior is not None:
            features.update(self.prior(features))
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

    # def sample(self, sample_no, f0_hz, loudness_db):
    #     features = {
    #         "f0_hz": f0_hz,
    #         "loudness_db": loudness_db,
    #         "sample_no": tf.constant(sample_no),
    #     }
    #     features.update(self.preprocessor(features, training=False))
    #     features.update(self.prior(features))
    #     return self.decode(features, training=False)

    def _update_losses_dict(self, loss_objs, features, outputs):
        for loss_obj in ddsp.core.make_iterable(loss_objs):
            # special case for the KL term
            if isinstance(loss_obj, KLRegularizer):
                losses_dict = loss_obj.get_losses_dict(
                    features['logq'],
                    features['logp'],
                )
                self._losses_dict.update(losses_dict)
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
