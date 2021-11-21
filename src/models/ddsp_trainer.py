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
    @tf.function
    def train_step(self, inputs):
        """Distributed training step."""
        # Wrap iterator in tf.function, slight speedup passing in iter vs batch.
        batch = next(inputs) if hasattr(inputs, '__next__') else inputs
        losses, grads = self.run(self.step_fn, batch)
        # Add up the scalar losses across replicas.
        n_replicas = self.strategy.num_replicas_in_sync
        return {k: self.psum(v, axis=None) / n_replicas for k, v in losses.items()}, grads

    @tf.function
    def step_fn(self, batch):
        """Per-Replica training step."""
        with tf.GradientTape() as tape:
            _, losses = self.model(batch, return_losses=True, training=True)
        # Clip and apply gradients.
        grads = tape.gradient(losses['total_loss'], self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return losses, grads
