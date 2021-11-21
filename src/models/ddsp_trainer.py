import tensorflow as tf
import tensorflow_addons as tfa
from ddsp.training import trainers


class TrainerWeightDecay(trainers.Trainer):
    def __init__(
        self,
        model,
        strategy,
        checkpoints_to_keep=100,
        learning_rate=0.001,
        weight_decay=0.0001,
        lr_decay_steps=10000,
        lr_decay_rate=0.98,
        grad_clip_norm=3.0,
        restore_keys=None,
    ):
        super().__init__(
            model,
            strategy,
            checkpoints_to_keep=checkpoints_to_keep,
            learning_rate=learning_rate,
            lr_decay_steps=lr_decay_steps,
            lr_decay_rate=lr_decay_rate,
            grad_clip_norm=grad_clip_norm,
            restore_keys=restore_keys,
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=lr_decay_steps,
            decay_rate=lr_decay_rate,
        )
        wd_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=weight_decay,
            decay_steps=lr_decay_steps,
            decay_rate=lr_decay_rate,
        )

        with self.strategy.scope():
            self.optimizer = tfa.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=lambda: None,
            )
            self.optimizer.weight_decay = lambda: wd_schedule(self.optimizer.iterations)


class TrainerGradSummaries(trainers.Trainer):
    def __init__(self, *args, **kwargs):
        self.steps_per_summary = kwargs.get('steps_per_summary')
        del kwargs['steps_per_summary']
        super().__init__(*args, **kwargs)

    @tf.function
    def step_fn(self, batch):
        """Per-Replica training step."""
        with tf.GradientTape() as tape:
            _, losses = self.model(batch, return_losses=True, training=True)

        grads = tape.gradient(losses['total_loss'], self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)

        # log gradient histograms for each variable
        if self.steps_per_summary and (self.step + 1) % self.steps_per_summary == 0:
            self._log_grads(grads, self.model.trainable_variables, self.step + 1)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return losses

    @staticmethod
    def _log_grads(grads, vars, step):
        for g, v in zip(grads, vars):
            tf.summary.histogram(v.name, g, step=step)
