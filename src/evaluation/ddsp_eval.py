import yaml
import argparse
from os.path import join

from ddsp.training import (
    trainers,
    train_util,
    eval_util,
    evaluators,
    data,
)

from ..models.ddsp_models import get_model
from ..models.model_utils import load_model
from ..data.paths import PROCESSED, GENERATED


def main(
    run_name: str,
    dataset: str,
    # architecture: str,
    example_secs: int,
    sample_rate: int,
    frame_rate: int,
    **kwargs,
):
    data_provider = data.TFRecordProvider(
        join(PROCESSED, dataset) + "*",
        example_secs=example_secs,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
    )

    strategy = train_util.get_strategy()
    with strategy.scope():
        model = get_model(
            time_steps=frame_rate * example_secs,
            sample_rate=sample_rate,
            n_samples=sample_rate * example_secs,
        )
        trainer = trainers.Trainer(model, strategy)

    load_model(trainer, run_name)
    eval_util.sample(
        data_provider=data_provider,
        model=trainer.model,
        evaluator_classes=[evaluators.BasicEvaluator],
        save_dir=join(GENERATED, run_name)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", "-c", type=str)
    args = parser.parse_args()

    with open(args.cfg) as cfg_f:
        cfg = yaml.load(cfg_f, Loader=yaml.FullLoader)

    main(**cfg)
