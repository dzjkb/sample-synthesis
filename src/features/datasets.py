import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from torch import tensor

from .preprocessing import load_raw


class AudioDataset(Dataset):
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
            s.resize(int(sampling_rate * maxlen))
            return s

        xs = np.array([getsample(p) for p in Path(directory).glob('*')])
        self.xs = tensor(xs)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        return self.xs[index]
