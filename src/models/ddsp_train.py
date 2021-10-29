import datetime as dt
import argparse
import yaml
import os.path

import tensorflow as tf

from .logger import get_logger
from .model_utils import get_save_dir
from .ddsp_models import get_trainer
from ..data.dataset import get_provider
from ..evaluation.ddsp_eval import sample

logger = get_logger(__name__, 'DEBUG')


def main(
    run_name: str,
    dataset: str,
    model_name: str = 'iaf_vae',
    lr: float = 1e-3,
    training_steps: int = 100,
    example_secs: int = 2,
    sample_rate: int = 16000,
    frame_rate: int = 250,
    batch_size: int = 32,
    steps_per_summary: int = 2000,
    **kwargs,
):
    run_timestamp = dt.datetime.now().strftime('%H-%M-%S')
    run_name = f"{run_name}_{run_timestamp}"
    save_dir = get_save_dir(run_name)

    logger.info("")
    logger.info("==============================")
    logger.info(f"Starting run {run_name}")
    logger.info("==============================")
    logger.info("")
    logger.debug(f"{run_name=}")
    logger.debug(f"{dataset=}")
    logger.debug(f"{lr=}")
    logger.debug(f"{training_steps=}")
    logger.debug(f"{example_secs=}")
    logger.debug(f"{sample_rate=}")
    logger.debug(f"{frame_rate=}")

    tf.debugging.experimental.enable_dump_debug_info(
        save_dir,
        tensor_debug_mode="FULL_HEALTH",
        circular_buffer_size=-1,
    )
    tf.summary.trace_on(
        graph=True, profiler=False
    )

    data_provider = get_provider(dataset, example_secs, sample_rate, frame_rate)
    dataset = data_provider.get_batch(batch_size, shuffle=True)
    # TODO: ds stats?
    first_example = next(iter(dataset))

    trainer = get_trainer(
        model_name=model_name,
        time_steps=frame_rate * example_secs,
        sample_rate=sample_rate,
        n_samples=sample_rate * example_secs,
        learning_rate=lr,
    )

    trainer.build(first_example)
    dataset_iter = iter(dataset)

    summary_writer = tf.summary.create_file_writer(save_dir)

    with summary_writer.as_default():
        tf.summary.trace_export("graph_summary", step=0)
        for i in range(training_steps):
            losses = trainer.train_step(dataset_iter)
            res_str = 'step: {}\t'.format(i)
            for k, v in losses.items():
                res_str += '{}: {:.2f}\t'.format(k, v)
                tf.summary.scalar(f"losses/{k}", v, step=i)
            logger.info(res_str)

            if i != 0 and i % steps_per_summary == 0:
                trainer.save(save_dir)
                sample(
                    trainer.model,
                    data_provider,
                    sample_rate=sample_rate,
                    checkpoint_dir=f"{os.path.basename(save_dir)}/{run_name}",
                    step=i,
                    n_gen=10,
                )

        trainer.save(save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", "-c", type=str)
    args = parser.parse_args()

    with open(args.cfg) as cfg_f:
        cfg = yaml.load(cfg_f, Loader=yaml.FullLoader)

    main(**cfg)
