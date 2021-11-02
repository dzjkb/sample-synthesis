from os.path import join

from ddsp.training import data
from ddsp.core import midi_to_hz
from ddsp import spectral_ops
import tensorflow as tf
import numpy as np

from ..data.paths import PROCESSED

_AUTOTUNE = tf.data.experimental.AUTOTUNE


class PreprocessedTFRecordProvider(data.TFRecordProvider):
    def __init__(
        self,
        file_pattern=None,
        example_secs=4,
        sample_rate=16000,
        frame_rate=250,
        preprocess_f=None,
    ):
        super().__init__(
            file_pattern,
            example_secs,
            sample_rate,
            frame_rate,
        )
        self._pre_f = preprocess_f

    def get_dataset(self, shuffle=True):
        ds = super().get_dataset(shuffle=shuffle)
        if self._pre_f is not None:
            return ds.map(self._pre_f, num_parallel_calls=_AUTOTUNE)
        else:
            return ds


def get_provider(dataset, example_secs, sample_rate, frame_rate):
    return PreprocessedTFRecordProvider(
        join(PROCESSED, dataset) + "*",
        example_secs=example_secs,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        preprocess_f=get_preprocess_pipeline(
            dataset,
            example_secs,
            sample_rate,
            frame_rate,
        ),
    )


def get_preprocess_pipeline(ds_name, example_secs, sample_rate, frame_rate):
    # if ds_name == "nsynth":
    #     return lambda ex: _nsynth_preprocess_ex(ex, sample_rate, frame_rate)
    # else:
    #     return None
    return None  # turns out nsynth examples are already preprocessed what???


def _nsynth_preprocess_ex(ex, sample_rate, frame_rate, n_fft=2048):
    ex_out = {
        'audio':
            ex['audio'],
        'f0_hz':
            midi_to_hz(ex['pitch']),
        # 'f0_confidence':
        #     ex['f0']['confidence'],
        'loudness_db':
            spectral_ops.compute_loudness(
                ex['audio'],
                sample_rate,
                frame_rate,
                n_fft
            ).astype(np.float32),
    }
    ex_out.update({
        'pitch':
            ex['pitch'],
        'instrument_source':
            ex['instrument']['source'],
        'instrument_family':
            ex['instrument']['family'],
        'instrument':
            ex['instrument']['label'],
    })
    return ex_out
