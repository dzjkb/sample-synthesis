import yaml
import argparse
# from os.path import join

from ddsp.training import summaries
import numpy as np
import matplotlib.pyplot as plt

from ..models.ddsp_models import get_trainer
from ..models.model_utils import strat
from ..data.dataset import get_provider
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


def sample(model, data_provider, sample_rate, checkpoint_dir, step, n_gen=10, synth_params=False):
    random_batch_ds = data_provider.get_batch(n_gen, shuffle=True)
    batch = next(iter(random_batch_ds))

    with strat().scope():
        outputs = model(batch, training=False)

        audio = batch['audio'].numpy()
        audio_gen = model.get_audio_from_outputs(outputs).numpy()

        summaries.audio_summary(audio, step, sample_rate=sample_rate, name="audio original")
        summaries.audio_summary(audio_gen, step, sample_rate=sample_rate, name="audio generated")
        summaries.waveform_summary(audio, audio_gen, step, name="waveforms")
        if synth_params:
            sp_summary(outputs, step)

        if hasattr(model, "sample"):
            sampled = model.sample(batch)
            sampled_gen = model.get_audio_from_outputs(sampled).numpy()
            summaries.audio_summary(sampled_gen, step, sample_rate=sample_rate, name="audio sampled")
            if synth_params:
                sp_summary(sampled, step)

    # for i in range(n_gen):
    #     save_wav(join(GENERATED, checkpoint_dir, f'{i}_eval_sample.wav'), audio[i], sr=sample_rate)
    #     save_wav(join(GENERATED, checkpoint_dir, f'{i}_gen_sample.wav'), audio_gen[i], sr=sample_rate)


def main(
    run_name: str,
    eval_checkpoint: str,
    dataset: str,
    # architecture: str,
    example_secs: int,
    sample_rate: int,
    frame_rate: int,
    **kwargs,
):
    data_provider = get_provider(
        dataset,
        example_secs,
        sample_rate,
        frame_rate,
    )

    trainer = get_trainer(
        time_steps=frame_rate * example_secs,
        sample_rate=sample_rate,
        n_samples=sample_rate * example_secs,
        restore_checkpoint=eval_checkpoint,
    )

    sample(
        trainer.model,
        data_provider,
        sample_rate=sample_rate,
        checkpoint=eval_checkpoint,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", "-c", type=str)
    args = parser.parse_args()

    with open(args.cfg) as cfg_f:
        cfg = yaml.load(cfg_f, Loader=yaml.FullLoader)

    main(**cfg)
