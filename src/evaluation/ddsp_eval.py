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
import tensorflow as tf

# from ..models.ddsp_models import get_trainer
from ..models.model_utils import strat
from .fad import (
    get_fad_embeddings,
    get_fad_distance,
)
# from ..data.dataset import get_provider
# from ..data.paths import GENERATED
# from .utils import save_wav


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
    for g, v in zip(grads, vars):
        tf.summary.histogram(v.name, g, step=step)


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
    n_gen=10,
    synth_params=False,
    fad_evaluator=None,
    weights=None,
    trainset_distance=None,
):
    random_batch_ds = data_provider.get_batch(n_gen, shuffle=True)
    ds_iter = iter(random_batch_ds)
    batch = next(ds_iter)

    with strat().scope():
        outputs = model(batch, training=False)

        audio = batch['audio'].numpy()
        audio_gen = model.get_audio_from_outputs(outputs).numpy()

        summaries.audio_summary(audio, step, sample_rate=sample_rate, name="audio original")
        summaries.audio_summary(audio_gen, step, sample_rate=sample_rate, name="audio generated")
        summaries.waveform_summary(audio, audio_gen, step, name="waveforms")
        if synth_params:
            sp_summary(outputs, step)
            synth_audio_summary(outputs, step, sample_rate=sample_rate)

        if weights:
            for w in weights:
                tf.summary.histogram(f"weights/{w}", _rgetattr(model, w), step=step)

        if hasattr(model, "sample"):
            sampled = model.sample(batch)
            sampled_gen = model.get_audio_from_outputs(sampled).numpy()
            summaries.audio_summary(sampled_gen, step, sample_rate=sample_rate, name="audio sampled")
            if synth_params:
                sp_summary(sampled, step)
                synth_audio_summary(sampled, step, sample_rate=sample_rate)

            sampled_audio_gen = sampled['audio_synth'].numpy()

            return
            if fad_evaluator:
                fad_evaluator.evaluate(None, sampled_audio_gen)

            if trainset_distance:
                big_batch = tf.concat(
                    list(map(lambda d: d['audio'], islice(ds_iter, n_gen * 10))),
                    axis=0,
                )
                min_dist, avg_dist = distances(big_batch, sampled_gen)

                for i, dist in enumerate(min_dist):
                    tf.summary.scalar(f"min_distance/sample_{i}", dist, step=step)
                for i, dist in enumerate(avg_dist):
                    tf.summary.scalar(f"avg_distance/sample_{i}", dist, step=step)


def get_evaluator_classes(dataset):
    return [
        evaluators.F0LdEvaluator,
        # partial(FadEvaluator, trainset_stats=f"{STATS_DIR}/{dataset}"),
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


def distances(batch, audio_gen, fft_sizes=(2048, 1024, 512, 256, 128, 64)):
    """
    for (N, n_samples) batch and (M, n_samples) audio_gen tensors returns
    two tensors of shape (M,) indicating the minimum and average distance, respectively,
    of each audio_gen audio sample to all batch samples
    """

    all_min_dists = []
    all_avg_dists = []

    for size in fft_sizes:
        spec_op = partial(spectral_ops.compute_mag, size=size)
        target_mag = spec_op(batch)
        value_mag = spec_op(audio_gen)

        value_dists_standard = [
            tf.reduce_mean(tf.abs(tf_float32(target_mag) - tf_float32(tf.stack([v] * tf.shape(target_mag)[0]))), axis=(1, 2))
            for v in tf.unstack(value_mag)
        ]
        value_dists_log = [
            tf.reduce_mean(
                tf.abs(tf_float32(spectral_ops.safe_log(target_mag)) - tf_float32(tf.stack([v] * tf.shape(target_mag)[0]))),
                axis=(1, 2),
            )
            for v in tf.unstack(spectral_ops.safe_log(value_mag))
        ]
        value_dists = [standard + log for standard, log in zip(value_dists_standard, value_dists_log)]

        min_dists = tf.stack([tf.reduce_min(d) for d in value_dists])
        avg_dists = tf.stack([tf.reduce_mean(d) for d in value_dists])
        all_min_dists.append(min_dists)
        all_avg_dists.append(avg_dists)

    min_dist = tf.reduce_mean(tf.stack(all_min_dists), axis=1)
    avg_dist = tf.reduce_mean(tf.stack(all_avg_dists), axis=1)

    return min_dist, avg_dist


# def main(
#     run_name: str,
#     eval_checkpoint: str,
#     dataset: str,
#     # architecture: str,
#     example_secs: int,
#     sample_rate: int,
#     frame_rate: int,
#     **kwargs,
# ):
#     data_provider = get_provider(
#         dataset,
#         example_secs,
#         sample_rate,
#         frame_rate,
#     )

#     trainer = get_trainer(
#         time_steps=frame_rate * example_secs,
#         sample_rate=sample_rate,
#         n_samples=sample_rate * example_secs,
#         restore_checkpoint=eval_checkpoint,
#     )

#     sample(
#         trainer.model,
#         data_provider,
#         sample_rate=sample_rate,
#         checkpoint=eval_checkpoint,
#     )


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cfg", "-c", type=str)
#     args = parser.parse_args()

#     with open(args.cfg) as cfg_f:
#         cfg = yaml.load(cfg_f, Loader=yaml.FullLoader)

#     main(**cfg)
