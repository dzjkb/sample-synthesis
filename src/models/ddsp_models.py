import ddsp
from ddsp.training import (
    decoders,
    encoders,
    effects,
    models,
    preprocessing,
    trainers,
)

from .model_utils import load_model, strat


def get_model(time_steps, sample_rate, n_samples):
    # Create Neural Networks.
    preprocessor = preprocessing.F0LoudnessPreprocessor(time_steps=time_steps)

    encoder = encoders.MfccTimeDistributedRnnEncoder(
        rnn_channels=512,
        rnn_type='gru',
        z_dims=32,
        z_time_steps=time_steps,
        input_keys = ('audio',),
    )

    decoder = decoders.RnnFcDecoder(rnn_channels = 256,
                                    rnn_type = 'gru',
                                    ch = 256,
                                    layers_per_stack = 1,
                                    input_keys = ('ld_scaled', 'z', 'f0_scaled'),
                                    output_splits = (('amps', 1),
                                                     ('harmonic_distribution', 45),
                                                     ('noise_magnitudes', 45)))

    # Create Processors.
    harmonic = ddsp.synths.Harmonic(n_samples=n_samples, 
                                    sample_rate=sample_rate,
                                    name='harmonic')

    noise = ddsp.synths.FilteredNoise(n_samples=n_samples,
                                      window_size=0,
                                      initial_bias=-10.0,
                                      name='noise')
    add = ddsp.processors.Add(name='add')

    reverb = ddsp.effects.Reverb(name='reverb', trainable=True)

    # Create ProcessorGroup.
    dag = [(harmonic, ['amps', 'harmonic_distribution', 'f0_hz']),
           (noise, ['noise_magnitudes']),
           (add, ['noise/signal', 'harmonic/signal']),
           (reverb, ['add/signal'])]

    processor_group = ddsp.processors.ProcessorGroup(dag=dag,
                                                     name='processor_group')

    spectral_loss = ddsp.losses.SpectralLoss(
        loss_type='L1',
        mag_weight=1.0,
        logmag_weight=1.0,
    )

    return models.Autoencoder(
        preprocessor=preprocessor,
        encoder=encoder,
        decoder=decoder,
        processor_group=processor_group,
        losses=[spectral_loss],
    )


def get_trainer(time_steps, sample_rate, n_samples, strategy=None, restore_checkpoint=None, **trainer_kwargs):
    if not strategy:
        strategy = strat()

    with strategy.scope():
        model = get_model(time_steps, sample_rate, n_samples)
        trainer = trainers.Trainer(model, strategy, **trainer_kwargs)

        if restore_checkpoint:
            load_model(trainer, restore_checkpoint)

    return trainer
