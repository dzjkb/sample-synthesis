from os.path import join

from ddsp.training import data
from ddsp.core import midi_to_hz
from ddsp import spectral_ops
import tensorflow as tf
import numpy as np

from ..data.paths import PROCESSED

_AUTOTUNE = tf.data.experimental.AUTOTUNE


class NSynthProvider(data.DataProvider):
    def __init__(
        self,
        file_pattern=None,
        example_secs=4,
        sample_rate=16000,
        frame_rate=250,
        preprocess_f=None,
    ):
        self._file_pattern = file_pattern
        self._audio_length = example_secs * sample_rate
        self._feature_length = example_secs * frame_rate
        super().__init__(sample_rate, frame_rate)
        self._data_format_map_fn = tf.data.TFRecordDataset

    def get_dataset(self, shuffle=True):
        def parse_tfexample(record):
            return tf.io.parse_single_example(record, self.features_dict)

        def preprocess_nsynth(ex):
            return _nsynth_preprocess_ex(ex, self._sample_rate, self._frame_rate)

        filenames = tf.data.Dataset.list_files(self._file_pattern, shuffle=shuffle)
        dataset = filenames.interleave(
            map_func=self._data_format_map_fn,
            cycle_length=40,
            num_parallel_calls=_AUTOTUNE,
        )
        dataset = dataset.map(parse_tfexample, num_parallel_calls=_AUTOTUNE)
        return dataset.map(preprocess_nsynth, num_parallel_calls=_AUTOTUNE)

    @property
    def features_dict(self):
        """NSynth features taken from https://magenta.tensorflow.org/datasets/nsynth#example-features"""
        return {
            'note': tf.io.FixedLenFeature([], dtype=tf.int64),
            'note_str': tf.io.FixedLenFeature([], dtype=tf.string),
            'instrument': tf.io.FixedLenFeature([], dtype=tf.int64),
            'instrument_str': tf.io.FixedLenFeature([], dtype=tf.string),
            'pitch': tf.io.FixedLenFeature([], dtype=tf.int64),
            'velocity': tf.io.FixedLenFeature([], dtype=tf.int64),
            'sample_rate': tf.io.FixedLenFeature([], dtype=tf.int64),
            'audio': tf.io.FixedLenFeature([self._audio_length], dtype=tf.float32),
            'qualities': tf.io.VarLenFeature(tf.int64),
            'qualities_str': tf.io.VarLenFeature(tf.string),
            'instrument_family': tf.io.FixedLenFeature([], dtype=tf.int64),
            'instrument_family_str': tf.io.FixedLenFeature([], dtype=tf.string),
            'instrument_source': tf.io.FixedLenFeature([], dtype=tf.int64),
            'instrument_source_str': tf.io.FixedLenFeature([], dtype=tf.string),
        }


def get_provider(dataset, example_secs, sample_rate, frame_rate):
    if dataset == 'nsynth':
        provider = NSynthProvider
    else:
        provider = data.TFRecordProvider

    return provider(
        join(PROCESSED, dataset) + "*",
        example_secs=example_secs,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
    )


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
                n_fft,
                use_tf=True,
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
