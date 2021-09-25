import datetime as dt
import argparse
import yaml
# import json
from os.path import join

from ddsp.training import (
    train_util,
    trainers,
    data,
)

from .logger import get_logger
from .model_utils import save_model
from .ddsp_models import get_model
from ..data.paths import PROCESSED

logger = get_logger(__name__, 'DEBUG')


def main(
    run_name: str,
    dataset: str,
    lr: float = 1e-3,
    training_steps: int = 100,
    example_secs: int = 2,
    sample_rate: int = 16000,
    frame_rate: int = 250,
):
    # ====== logging
    run_timestamp = dt.datetime.now().strftime('%H-%M-%S')
    run_name = f"{run_name}_{run_timestamp}"

    logger.info(f"Starting run {run_name} with config")
    logger.debug(f"{run_name=}")
    logger.debug(f"{dataset=}")
    logger.debug(f"{lr=}")
    logger.debug(f"{training_steps=}")
    logger.debug(f"{example_secs=}")
    logger.debug(f"{sample_rate=}")
    logger.debug(f"{frame_rate=}")
    logger.info("===================================")

    # ====== training
    strategy = train_util.get_strategy()  # default tf.distribute.MirroredStrategy()
    data_provider = data.TFRecordProvider(
        join(PROCESSED, dataset) + "*",
        example_secs=example_secs,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
    )
    dataset = data_provider.get_dataset(shuffle=False)
    # TODO: ds stats?
    first_example = next(iter(dataset))

    with strategy.scope():
        model = get_model(
            time_steps=frame_rate * example_secs,
            sample_rate=sample_rate,
            n_samples=sample_rate * example_secs,
        )
        trainer = trainers.Trainer(model, strategy, learning_rate=lr)

    trainer.build(first_example)
    dataset_iter = iter(dataset)

    for i in range(training_steps):
        losses = trainer.train_step(dataset_iter)
        res_str = 'step: {}\t'.format(i)
        for k, v in losses.items():
            res_str += '{}: {:.2f}\t'.format(k, v)
        logger.info(res_str)

    save_model(model, run_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", "-c", type=str)
    args = parser.parse_args()

    with open(args.cfg) as cfg_f:
        cfg = yaml.load(cfg_f, Loader=yaml.FullLoader)

    main(**cfg)
