import collections

import apache_beam as beam
from apache_beam.io import WriteToTFRecord
# from apache_beam.transforms import window
# from apache_beam.utils import windowed_value
import tensorflow as tf

from frechet_audio_distance.create_embeddings_beam import (
    ComputeMeanAndCovariance,
    BatchedInference,
    DropKey,
    create_audio_example,
)
from frechet_audio_distance.fad_utils import (
    read_mean_and_covariances,
    frechet_distance,
)
from ..data.fs_utils import git_root

MODEL_CKPT = f"{git_root()}/data/vggish_model.ckpt"
STATS_DIR = f"{git_root()}/fad_stats"
ModelConfig = collections.namedtuple(
    'ModelConfig', 'model_ckpt embedding_dim step_size')


class CreateTfExample(beam.DoFn):
    def process(self, element):
        idx, audio = element
        name = f"sample_{idx}"
        example = create_audio_example('audio', audio, name)
        yield name, example


def get_fad_embeddings(
    audio,
    output_path,
    model_embedding_dim=128,
    model_step_size=8000,
    batch_size=64,
):
    model_cfg = ModelConfig(
        model_ckpt=MODEL_CKPT,
        embedding_dim=model_embedding_dim,
        step_size=model_step_size
    )

    pipeline = beam.Pipeline()
    examples = (
        pipeline
        | beam.Create(list(enumerate(tf.unstack(audio))))
        | "Create TF Examples" >> beam.ParDo(CreateTfExample())
    )
    embeddings = (
        examples
        | 'Batched Inference' >> beam.ParDo(
            BatchedInference(
                batch_size=batch_size,
                model=model_cfg,
                feature_key='audio')).with_outputs('raw', main='examples'))
    _ = (
        embeddings.raw
        | 'Combine Embeddings' >> beam.CombineGlobally(
            ComputeMeanAndCovariance(key_name='fad_batch_embeddings', embedding_dim=128))
        | 'DropKey' >> beam.ParDo(DropKey())
        | 'Write Stats' >> WriteToTFRecord(
            output_path,
            shard_name_template='',
            coder=beam.coders.ProtoCoder(tf.train.Example)))

    result = pipeline.run()
    result.wait_until_finish()


def get_fad_distance(test_stats, background_stats):
    mu_bg, sigma_bg = read_mean_and_covariances(background_stats)
    mu_test, sigma_test = read_mean_and_covariances(test_stats)
    fad = frechet_distance(mu_bg, sigma_bg, mu_test, sigma_test)
    return fad
