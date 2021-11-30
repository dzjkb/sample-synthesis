# import yaml
# import argparse
# from os.path import join
from functools import partial, reduce
from itertools import islice
import tempfile

from ddsp import spectral_ops
from ddsp.core import tf_float32
from ddsp.training import (
    summaries,
    evaluators,
    metrics,
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf

# from ..models.ddsp_models import get_trainer
from ..models.model_utils import strat
from ..models.logger import get_logger
from .fad import (
    get_fad_embeddings,
    get_fad_distance,
)
# from ..data.dataset import get_provider
# from ..data.paths import GENERATED
# from .utils import save_wav
from .ndb import (
    get_center_samples,
    get_closest_center,
    map_logmag,
    assign_samples_to_bins,
    get_cluster_counts,
    two_sample_test,
)

logger = get_logger(__name__, 'DEBUG')


def sp_summary(outputs, step):
    batch_size = int(outputs['f0_hz'].shape[0])
    have_pred = 'amps_pred' in outputs
    height = 12 if have_pred else 4
    rows = 3 if have_pred else 1

    for i in range(batch_size):
        # Amplitudes ----------------------------
        amps = np.squeeze(outputs['amps'][i])
        fig, ax = plt.subplots(nrows=rows, ncols=1, figsize=(8, height))

        if have_pred:
            ax[0].plot(amps)
            ax[0].set_title('Amplitudes - synth_params')
            amps_pred = np.squeeze(outputs['amps_pred'][i])
            ax[1].plot(amps_pred)
            ax[1].set_title('Amplitudes - pred')

            amps_diff = amps - amps_pred
            ax[2].plot(amps_diff)
            ax[2].set_title('Amplitudes - diff')

            for ax in fig.axes:
                ax.label_outer()
        else:
            ax.plot(amps)
            ax.set_title('Amplitudes - synth_params')

        summaries.fig_summary(f'amplitudes/amplitudes_{i + 1}', fig, step)

        # Harmonic Distribution ------------------
        hd = np.log(np.squeeze(outputs['harmonic_distribution'][i]) + 1e-8)
        fig, ax = plt.subplots(nrows=rows, ncols=1, figsize=(8, height))

        if have_pred:
            im = ax[0].imshow(hd.T, aspect='auto', origin='lower')
            fig.colorbar(im, ax=ax[0])
            ax[0].set_title('Harmonic Distribution (log) - synth_params')
            hd_pred = np.log(np.squeeze(outputs['hd_pred'][i]) + 1e-8)
            im = ax[1].imshow(hd_pred.T, aspect='auto', origin='lower')
            fig.colorbar(im, ax=ax[1])
            ax[1].set_title('Harmonic Distribution (log) - pred')

            hd_diff = hd - hd_pred
            im = ax[2].imshow(hd_diff.T, aspect='auto', origin='lower')
            fig.colorbar(im, ax=ax[2])
            ax[2].set_title('Harmonic Distribution (log) - diff')

            for ax in fig.axes:
                ax.label_outer()
        else:
            im = ax.imshow(hd.T, aspect='auto', origin='lower')
            fig.colorbar(im, ax=ax)
            ax.set_title('Harmonic Distribution (log) - synth_params')

        summaries.fig_summary(f'harmonic_dist/harmonic_dist_{i + 1}', fig, step)

        # Magnitudes ----------------------------
        noise = np.squeeze(outputs['noise_magnitudes'][i])
        fig, ax = plt.subplots(nrows=rows, ncols=1, figsize=(8, height))

        if have_pred:
            im = ax[0].imshow(noise.T, aspect='auto', origin='lower')
            fig.colorbar(im, ax=ax[0])
            ax[0].set_title('Noise mags - synth_params')
            noise_pred = np.squeeze(outputs['noise_pred'][i])
            im = ax[1].imshow(noise_pred.T, aspect='auto', origin='lower')
            fig.colorbar(im, ax=ax[1])
            ax[1].set_title('Noise mags - pred')

            noise_diff = noise - noise_pred
            im = ax[2].imshow(noise_diff.T, aspect='auto', origin='lower')
            fig.colorbar(im, ax=ax[2])
            ax[2].set_title('Noise mags - diff')
            for ax in fig.axes:
                ax.label_outer()
        else:
            im = ax.imshow(noise.T, aspect='auto', origin='lower')
            fig.colorbar(im, ax=ax)
            ax.set_title('Noise mags - synth_params')

        summaries.fig_summary(f'noise_mags/noise_mags_{i + 1}', fig, step)


def synth_audio_summary(outputs, step, sample_rate, synths=("harmonic", "noise")):
    for key in synths:
        audio = outputs[key]["signal"]
        summaries.audio_summary(audio, step, sample_rate=sample_rate, name=f"{key} synth - audio")


def log_grads(grads, vars, step):
    grads_norm, grads_notnorm = grads
    for g, v in zip(grads_norm, vars):
        tf.summary.histogram(f"grads/{v.name}", g, step=step)
    for g, v in zip(grads_notnorm, vars):
        tf.summary.histogram(f"unnormalized_grads/{v.name}", g, step=step)


def _rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return reduce(_getattr, [obj] + attr.split('.'))


def sample(
    model,
    data_provider,
    sample_rate,
    checkpoint_dir,
    step,
    n_gen=20,
    synth_params=False,
    fad_evaluator=None,
    weights=None,
    ndb_eval=None,
    other_evals=None,
):
    random_batch_ds = data_provider.get_batch(n_gen, shuffle=True)
    ds_iter = iter(random_batch_ds)
    batch = next(ds_iter)

    logger.debug(f"eval: starting eval of step {step}")
    with strat().scope():
        outputs = model(batch, training=False)

        audio = batch['audio'].numpy()
        audio_gen = model.get_audio_from_outputs(outputs).numpy()

        logger.debug("eval: writing reconstruction audio summaries")
        summaries.audio_summary(audio, step, sample_rate=sample_rate, name="audio original")
        summaries.audio_summary(audio_gen, step, sample_rate=sample_rate, name="audio generated")
        summaries.waveform_summary(audio, audio_gen, step, name="waveforms")
        if synth_params:
            logger.debug("eval: writing synth params summaries")
            sp_summary(outputs, step)
            synth_audio_summary(outputs, step, sample_rate=sample_rate)

        if weights:
            logger.debug("eval: writing weight histograms")
            for w in weights:
                tf.summary.histogram(f"weights/{w}", _rgetattr(model, w), step=step)

        if other_evals:
            logger.debug("eval: writing other evaluations")
            for b in islice(ds_iter, n_gen * 20):
                b_out, losses = model(b, return_losses=True, training=False)
                b_out['audio_gen'] = b_out['audio_synth']
                for e in other_evals:
                    e.evaluate(b, b_out, losses)

            for e in other_evals:
                e.flush(step)

        if hasattr(model, "sample"):
            logger.debug("eval: sampling from the model")
            sampled = model.sample(batch)
            sampled_gen = model.get_audio_from_outputs(sampled).numpy()
            summaries.audio_summary(sampled_gen, step, sample_rate=sample_rate, name="audio sampled")
            if synth_params:
                sp_summary(sampled, step)
                synth_audio_summary(sampled, step, sample_rate=sample_rate)

            sampled_audio_gen = sampled['audio_synth']

            if fad_evaluator:
                logger.debug("eval: calculating FAD")
                fad_evaluator.evaluate(None, sampled_audio_gen)
                fad_evaluator.flush(step)

            if ndb_eval:
                logger.debug("eval: calculating NDB")
                # big_batch = tf.concat(
                #     list(map(lambda d: d['audio'], islice(ds_iter, n_gen * 10))),
                #     axis=0,
                # )
                # min_dist, avg_dist = distances(big_batch, sampled_gen)

                # for i, dist in enumerate(min_dist):
                #     tf.summary.scalar(f"min_distance/sample_{i}", dist, step=step)
                # for i, dist in enumerate(avg_dist):
                #     tf.summary.scalar(f"avg_distance/sample_{i}", dist, step=step)
                ndb_eval.evaluate(None, sampled_audio_gen)
                ndb_eval.flush(step)


def get_evaluator_classes():
    return [
        evaluators.F0LdEvaluator,
    ]


class FadEvaluator(evaluators.BaseEvaluator):
    def __init__(self, sample_rate, frame_rate, trainset_stats):
        super().__init__(sample_rate, frame_rate)
        self._fad_metric = FadMetric(sample_rate, frame_rate, base_stats=trainset_stats)

    def evaluate(self, batch, outputs, losses=None):
        # audio_gen = outputs['audio_synth'].numpy()
        audio_gen = outputs
        self._fad_metric.update_state(batch, audio_gen)

    def flush(self, step):
        self._fad_metric.flush(step)


class FadMetric(metrics.BaseMetrics):
    def __init__(self, sample_rate, frame_rate, base_stats, name="fad"):
        super().__init__(sample_rate, frame_rate, name=name)
        self._base_stats_file = base_stats
        self._metrics = {
            'fad': tf.keras.metrics.Mean('fad')
        }

    @property
    def metrics(self):
        return self._metrics

    def update_state(self, batch, audio_gen):
        batch_stats_file = tempfile.NamedTemporaryFile()
        batch_stats = batch_stats_file.name
        get_fad_embeddings(
            audio_gen,
            batch_stats,
        )

        fad = get_fad_distance(batch_stats, self._base_stats_file)
        self._metrics['fad'].update_state(fad)


class NDBEvaluator(evaluators.BaseEvaluator):
    def __init__(self, sample_rate, frame_rate, train_ds, k=50, significance_level=0.05):
        super().__init__(sample_rate, frame_rate)
        logds = map_logmag(train_ds)
        self.k = k
        self.pval = significance_level
        self.center_samples = get_center_samples(logds, k=k)
        sample_to_bin = assign_samples_to_bins(logds, self.center_samples)
        self.trainset_cluster_counts = get_cluster_counts(sample_to_bin, k=k)
        self.ds_size = len(sample_to_bin)
        self.trainset_proportions = {c: v/self.ds_size for c, v in self.trainset_cluster_counts.items()}

    def evaluate(self, batch, outputs, losses=None):
        logsamples = map_logmag(outputs)
        sample_to_bin = assign_samples_to_bins(logsamples, self.center_samples)
        self.batch_cluster_counts = get_cluster_counts(sample_to_bin, k=self.k)
        self.batch_size = len(sample_to_bin)
        self.batch_proportions = {c: v/self.batch_size for c, v in self.batch_cluster_counts.items()}

    def flush(self, step):
        self._proportions_fig_summary(step)
        self._ndb_scalar_summary(step)

    def _proportions_fig_summary(self, step):
        fig, ax = plt.subplots(1, 1)
        longform_trainset = pd.DataFrame({
            "bin": list(self.trainset_proportions.keys()),
            "count": list(self.trainset_proportions.values()),
            "data": f"training set ({self.ds_size} samples)",
        })
        longform_batch = pd.DataFrame({
            "bin": list(self.batch_proportions.keys()),
            "count": list(self.batch_proportions.values()),
            "data": f"sampled batch ({self.batch_size} samples)",
        })
        longform_data = pd.concat((longform_trainset, longform_batch), axis=0).reset_index(drop=True)
        sns.barplot(data=longform_data, x="bin", y="count", hue="data", ax=ax)

        summaries.fig_summary("NDB bin proportions", fig, step)

    def _ndb_scalar_summary(self, step):
        bin_to_pval = {c: two_sample_test(
            self.trainset_cluster_counts[c],
            self.batch_cluster_counts[c],
            self.ds_size,
            self.batch_size,
        ) for c in self.trainset_cluster_counts}

        ndb = sum([pval < self.pval for pval in bin_to_pval.values()])
        tf.summary.scalar("ndb", ndb, step=step)


# def distances(batch, audio_gen, fft_sizes=(2048, 1024, 512, 256, 128, 64)):
#     """
#     for (N, n_samples) batch and (M, n_samples) audio_gen tensors returns
#     two tensors of shape (M,) indicating the minimum and average distance, respectively,
#     of each audio_gen audio sample to all batch samples
#     """

#     all_min_dists = []
#     all_avg_dists = []

#     for size in fft_sizes:
#         spec_op = partial(spectral_ops.compute_mag, size=size)
#         target_mag = spec_op(batch)
#         value_mag = spec_op(audio_gen)

#         value_dists_standard = [
#             tf.reduce_mean(tf.abs(tf_float32(target_mag) - tf_float32(tf.stack([v] * tf.shape(target_mag)[0]))), axis=(1, 2))
#             for v in tf.unstack(value_mag)
#         ]
#         value_dists_log = [
#             tf.reduce_mean(
#                 tf.abs(tf_float32(spectral_ops.safe_log(target_mag)) - tf_float32(tf.stack([v] * tf.shape(target_mag)[0]))),
#                 axis=(1, 2),
#             )
#             for v in tf.unstack(spectral_ops.safe_log(value_mag))
#         ]
#         value_dists = [standard + log for standard, log in zip(value_dists_standard, value_dists_log)]

#         min_dists = tf.stack([tf.reduce_min(d) for d in value_dists])
#         avg_dists = tf.stack([tf.reduce_mean(d) for d in value_dists])
#         all_min_dists.append(min_dists)
#         all_avg_dists.append(avg_dists)

#     min_dist = tf.reduce_mean(tf.stack(all_min_dists), axis=1)
#     avg_dist = tf.reduce_mean(tf.stack(all_avg_dists), axis=1)

#     return min_dist, avg_dist
