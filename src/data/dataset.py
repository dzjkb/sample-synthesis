from os.path import join

from ddsp.training import data
import tensorflow as tf

from ..data.paths import PROCESSED

_AUTOTUNE = tf.data.experimental.AUTOTUNE


def PreprocessedTFRecordProvider(data.TFRecordProvider):
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
            tf.data.TFRecordDataset,
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
        preprocess_f=get_preprocess_pipeline(dataset),
    )


def get_preprocess_pipeline(ds_name):
    if ds_name == "nsynth":
        return _nsynth_preprocess_ex
    else:
        return None


def _nsynth_preprocess_ex(ex):
    ex_out = {
        'audio':
            ex['audio'],
        'f0_hz':
            ex['f0']['hz'],
        'f0_confidence':
            ex['f0']['confidence'],
        'loudness_db':
            ex['loudness']['db'],
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
