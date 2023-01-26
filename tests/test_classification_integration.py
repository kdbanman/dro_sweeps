import jax.numpy as jnp
from jax import random

import dro_sweeps.classification_data_generation as cdg
import dro_sweeps.dro as dro


def test_classification():
    seed = 42069
    key = random.PRNGKey(seed)

    population_1 = {
        'size': 500,
        'input_mean': (1.0, 1.0),
        'input_covariance': ((1.0, 0.0), (0.0, 1.0)),
        'weights': jnp.array((0.5, 0.5, 0.5)),
        'noise_variance': 0.1,
    }
    population_2 = {
        'size': 50,
        'input_mean': (1.5, 1.5),
        'input_covariance': ((1.0, 0.0), (0.0, 1.0)),
        'weights': jnp.array((0.8, 0.8, 0.8)),
        'noise_variance': 0.1,
    }

    cvar_alpha = 0.1
    batch_size = 8

    steps = 1e2
    step_size = 0.005
    init_weights = jnp.array((0.1, 0.1, 0.1)).reshape((3, 1))

    key, subkey = random.split(key)
    inputs, labels = cdg.generate_dataset(key, [population_1, population_2])

    _weights, _loss_trajectory, _log_steps = dro.train_averaged_dro(
        subkey,
        inputs,
        labels,
        init_weights,
        cdg.logistic_outputs,
        dro.cross_entropy_loss,
        step_size,
        batch_size,
        cvar_alpha,
        steps,
        10,
    )