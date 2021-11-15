from ddsp.training import (
    encoders,
    nn,
)
import tensorflow.keras.layers as tfkl


class RegularizedRnn(tfkl.Layer):
    """ Single RNN layer with optional regularization.
        This is a slightly modified version of class Rnn
        from ddsp/training/nn.py
    """

    def __init__(
        self,
        dims,
        rnn_type,
        return_sequences=True,
        bidir=False,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        rnn_class = {'lstm': tfkl.LSTM, 'gru': tfkl.GRU}[rnn_type]
        self.rnn = rnn_class(
            dims,
            return_sequences=return_sequences,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )
        if bidir:
            self.rnn = tfkl.Bidirectional(self.rnn)

    def call(self, x):
        return self.rnn(x)


class MfccRegularizedRnnEncoder(encoders.MfccRnnEncoder):
    def __init__(
        self,
        rnn_channels=512,
        rnn_type='gru',
        z_dims=512,
        mean_aggregate=False,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        **kwargs,
    ):
        super().__init__(
            rnn_channels=rnn_channels,
            rnn_type=rnn_type,
            z_dims=z_dims,
            mean_aggregate=mean_aggregate,
            **kwargs,
        )

        self.rnn = RegularizedRnn(
            rnn_channels,
            rnn_type,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )
