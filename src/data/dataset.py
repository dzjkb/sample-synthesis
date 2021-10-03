from os.path import join

from ddsp.training import data

from ..data.paths import PROCESSED


def get_provider(dataset, example_secs, sample_rate, frame_rate):
    return data.TFRecordProvider(
        join(PROCESSED, dataset) + "*",
        example_secs=example_secs,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
    )
