import datetime as dt
import argparse
import yaml
import json

from ddsp.training import (
    train_util,
    trainers,
    models
)

from .logger import get_logger
from .model_utils import save_model

logger = get_logger(__name__, 'DEBUG')


def main(
    run_name: str,
    lr: float = 1e-3,
    steps: int = 200,
):
    # ====== logging
    run_timestamp = dt.datetime.now().strftime('%H-%M-%S')
    run_name = f"{run_name}_{run_timestamp}"

    logger.info(f"Starting run {run_name} with config")
    logger.debug(f"{run_name=}")
    logger.debug(f"{lr=}")
    logger.debug(f"{steps=}")
    logger.info("===================================")

    # ====== training
    strategy = train_util.get_strategy()  # default tf.distribute.MirroredStrategy()
    # TODO dataset how where what

    with strategy.scope():
        model = models.Autoencoder()
        trainer = trainers.Trainer(model, strategy, learning_rate=lr)

    trainer.build(next(iter(dataset)))
    dataset_iter = iter(dataset)

    for i in range(steps):
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
