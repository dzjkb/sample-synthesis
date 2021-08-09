from datetime import datetime
import os

# import torch
from tensorflow import keras

from ..data.fs_utils import git_root


# def save_model(model, filename):
#     """ Creates a year-month-day folder in the 'models' directory
#         (if it doesn't exist already) and saves the given model
#         inside
#     """
#     date_day = datetime.now().strftime('%Y-%m-%d')
#     date_dir =  f'{git_root()}/models/{date_day}'
#     model_path = f'{date_dir}/{filename}.pt'

#     if not os.path.exists(date_dir):
#         os.mkdir(date_dir)

#     torch.save(model.state_dict(), model_path)


# def load_state(model, path):
#     model_path = f'{git_root()}/models/{path}'
#     state = torch.load(model_path)
#     model.load_state_dict(state)

def save_model(model, model_name):
    today = datetime.now().strftime('%Y-%m-%d')
    date_dir = f"{git_root()}/models/{today}"

    if not os.path.exists(date_dir):
        os.mkdir(date_dir)

    model.save(f"{date_dir}/{model_name}")


def load_model(path):
    full_path = f"{git_root()}/models/{path}"
    return keras.models.load_model(full_path)
