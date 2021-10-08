from datetime import datetime
import os

from ddsp.training import train_util

from ..data.fs_utils import git_root


def get_save_dir(run_name):
    today = datetime.now().strftime('%Y-%m-%d')
    date_dir = f"{git_root()}/models/{today}"

    if not os.path.exists(date_dir):
        os.makedirs(date_dir, exist_ok=True)

    return f"{date_dir}/{run_name}"


def load_model(trainer, checkpoint_path):
    full_path = f"{git_root()}/models/{checkpoint_path}"
    trainer.restore(full_path)


STRAT = None

def strat():
    global STRAT
    if STRAT:
        return STRAT
    else:
        STRAT = train_util.get_strategy()
        return STRAT
