import numpy as np

from pathlib import Path
from torch.utils.data import TensorDataset
from torch import tensor

from preprocessing import load_raw


class AudioDataset(TensorDataset):
    """ Dataset subclass loading a whole directory of audio files
        into memory at once

        @directory - directory of audio files
        @maxlen    - duration (in seconds) all samples are resized to
        @sr        - sample rate files are converted to
    """

    DEFAULT_SR = 11025

    def __init__(self, directory, maxlen=1, sampling_rate=DEFAULT_SR):
        def getsample(path):
            s = load_raw(path, sampling_rate)
            s = np.resize((sampling_rate * maxlen))
            return s

        xs = np.array([getsample(p) for p in Path(directory).glob('*')])

        super(AudioDataset, self).__init__(tensor(xs))
