from datetime import datetime
import os

import soundfile

from ..data.fs_utils import git_root


def save_wav(run_name, filename, data, sampling_rate):
    """ creates a folder for the given training run (if it doesn't
        exist yet) and saves the given data as a .wav inside
    """
    date_day = datetime.now().strftime('%Y-%m-%d')
    out_dir =  f'{git_root()}/generated/{date_day}/{run_name}'
    wav_path = f'{out_dir}/{filename}.wav'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    soundfile.write(wav_path, data, sampling_rate)
