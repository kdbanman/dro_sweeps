import jax.numpy as jnp


class TrainingResults:
    """
    A class to record all relevant results of a training run, with a configurable
    amount of detail (e.g. logging frequency.)

    In the future, a configurable weight averaging scheme could be added.
    For now, online performant weight averaging is done via Lacoste-Julien et al.
        https://arxiv.org/abs/1212.2002

    The .record_step method should be called at _every_ training iteration.
    """
    def __init__(self, logging_period, init_weights, logger):
        self.averaged_weights = init_weights
        self.logging_period = logging_period

        self.logger = logger

    def record_step(self, step, loss, weights):
        past_weights_contribution = (1 - 2 / (step + 2)) * self.averaged_weights
        new_weights_contribution = 2 / (step + 2) * weights
        self.averaged_weights = past_weights_contribution + new_weights_contribution

        if self.logger is not None and step % self.logging_period == 0:
            self.logger.log({
                'step': step,
                'loss': loss,
                'weights_norm': jnp.linalg.norm(weights),
                'averaged_weights_norm': jnp.linalg.norm(self.averaged_weights),
            })
