import datetime as dt
import argparse
import yaml
import os.path
from shutil import copy

import tensorflow as tf
from ddsp.training.eval_util import evaluate

from .model_utils import load_model, get_full_checkpoint_dir
from .logger import get_logger
from .model_utils import get_save_dir
from .ddsp_models import get_trainer
from ..data.dataset import get_provider
from ..evaluation.ddsp_eval import (
    sample,
    get_evaluator_classes,
    FadEvaluator,
    log_grads,
    NDBEvaluator,
)
from ..evaluation.fad import STATS_DIR

logger = get_logger(__name__, 'DEBUG')


def main(
    run_name: str,
    dataset_name: str,
    model_name: str = 'iaf_vae',
    lr: float = 1e-3,
    training_steps: int = 100,
    example_secs: int = 2,
    sample_rate: int = 16000,
    frame_rate: int = 250,
    batch_size: int = 32,
    steps_per_summary: int = 2000,
    synth_params_summary: bool = False,
    kl_weight: int = 1,
    kl_min: int = 0,
    checkpoint_dir: str = None,
    cfg_path: str = None,
    debug_dump: bool = False,
    weight_hists: list = None,
    **kwargs,
):
    run_timestamp = dt.datetime.now().strftime('%H-%M-%S')
    run_name = f"{run_name}_{run_timestamp}"

    save_dir = get_full_checkpoint_dir(checkpoint_dir) if checkpoint_dir else get_save_dir(run_name)
    if cfg_path:
        copy(cfg_path, save_dir)

    logger.info("")
    logger.info("==============================")
    logger.info(f"Starting run {run_name}")
    if checkpoint_dir:
        logger.info(f"resuming from {checkpoint_dir}")
        logger.info("warning - model parameters must match the ones used earlier")
        logger.info("otherwise expect weird tensorflow errors")
    logger.info("==============================")
    logger.info("")
    logger.debug(f"{run_name=}")
    logger.debug(f"{dataset_name=}")
    logger.debug(f"{lr=}")
    logger.debug(f"{training_steps=}")
    logger.debug(f"{example_secs=}")
    logger.debug(f"{sample_rate=}")
    logger.debug(f"{frame_rate=}")
    logger.debug(f"{kl_weight=}")
    logger.debug(f"{kl_min=}")

    if debug_dump:
        tf.debugging.experimental.enable_dump_debug_info(
            save_dir,
            tensor_debug_mode="FULL_HEALTH",
            circular_buffer_size=-1,
        )
    tf.summary.trace_on(
        graph=True, profiler=False
    )

    data_provider = get_provider(dataset_name, example_secs, sample_rate, frame_rate)
    dataset = data_provider.get_batch(batch_size, shuffle=True)
    # TODO: ds stats?
    logger.debug("Dataset information:")
    logger.debug(f"size={dataset.cardinality()}")

    first_example = next(iter(dataset))

    trainer = get_trainer(
        model_name=model_name,
        time_steps=frame_rate * example_secs,
        sample_rate=sample_rate,
        n_samples=sample_rate * example_secs,
        learning_rate=lr,
        kl_weight=kl_weight,
        kl_min=kl_min,
    )

    trainer.build(first_example)
    dataset_iter = iter(dataset)

    if checkpoint_dir:
        load_model(trainer, checkpoint_dir)

    summary_writer = tf.summary.create_file_writer(save_dir)
    fad_evaluator = FadEvaluator(
        sample_rate,
        frame_rate,
        f"{STATS_DIR}/{dataset_name}",
    )
    ndb_evaluator = NDBEvaluator(
        sample_rate,
        frame_rate,
        data_provider.get_dataset(),
    )
    other_evaluators = [
        eval_class(sample_rate, frame_rate)
        for eval_class in get_evaluator_classes()
    ]

    with summary_writer.as_default():
        tf.summary.trace_export("graph_summary", step=1)
        for i in range(training_steps):
            step = trainer.step
            losses, grads = trainer.train_step(dataset_iter)
            res_str = 'step: {}\t'.format(step + 1)
            for k, v in losses.items():
                res_str += '{}: {:.2f}\t'.format(k, v)
                tf.summary.scalar(f"losses/{k}", v, step=step + 1)
            logger.info(res_str)

            if step != 0 and (step+1) % steps_per_summary == 0:
                trainer.save(save_dir)
                log_grads(grads, trainer.model.trainable_variables, step + 1)
                sample(
                    trainer.model,
                    data_provider,
                    sample_rate=sample_rate,
                    checkpoint_dir=f"{os.path.basename(save_dir)}/{run_name}",
                    step=step + 1,
                    synth_params=synth_params_summary,
                    weights=weight_hists,
                    # fad_evaluator=fad_evaluator,
                    ndb_eval=ndb_evaluator,
                    other_evals=other_evaluators,
                )

        trainer.save(save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", "-c", type=str)
    args = parser.parse_args()

    with open(args.cfg) as cfg_f:
        cfg = yaml.load(cfg_f, Loader=yaml.FullLoader)

    main(cfg_path=args.cfg, **cfg)
