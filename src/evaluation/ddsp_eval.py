import yaml
import argparse
# from os.path import join

from ddsp.training import summaries

from ..models.ddsp_models import get_trainer
from ..models.model_utils import strat
from ..data.dataset import get_provider
# from ..data.paths import GENERATED
# from .utils import save_wav


def sample(model, data_provider, sample_rate, checkpoint_dir, step, n_gen=10):
    random_batch_ds = data_provider.get_batch(n_gen, shuffle=True)
    batch = next(iter(random_batch_ds))

    with strat().scope():
        outputs = model(batch, training=False)

        audio = batch['audio'].numpy()
        audio_gen = model.get_audio_from_outputs(outputs).numpy()

        summaries.audio_summary(audio, step, sample_rate=sample_rate, name="audio original")
        summaries.audio_summary(audio_gen, step, sample_rate=sample_rate, name="audio generated")
        summaries.waveform_summary(audio, audio_gen, step, name="waveforms")
        summaries.midiae_sp_summary(outputs, step)

        if hasattr(model, "sample"):
            sampled = model.sample(batch)
            sampled_gen = model.get_audio_from_outputs(sampled).numpy()
            summaries.audio_summary(sampled_gen, step, sample_rate=sample_rate, name="audio sampled")
            summaries.midiae_sp_summary(sampled, step)

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
