import jax.numpy as jnp
from jax import random

import dro_sweeps.regression_data_generation as dg
import dro_sweeps.dro as dro


def test_regression():
    seed = 42069
    key = random.PRNGKey(seed)

    population_1 = {
        'size': 500,
        'mean': 0.0,
        'variance': 1.0,
        'weights': jnp.array((1.5, 0.0)),
        'noise': 0.005,
    }
    population_2 = {
        'size': 50,
        'mean': 0.0,
        'variance': 1.0,
        'weights': jnp.array((0.5, 0.0)),
        'noise': 0.005,
    }

    cvar_alpha = 0.1
    batch_size = 8

    steps = 1e2
    step_size = 0.005
    init_weights = jnp.array((0.1, 0.1)).reshape((2, 1))

    key, subkey = random.split(key)
    inputs, outputs = dg.generate_dataset(subkey, [population_1, population_2])

    key, subkey = random.split(key)
    _training_results = dro.train_averaged_dro(
        subkey,
        inputs,
        outputs,
        init_weights,
        dg.linear_outputs,
        dro.squared_err_loss,
        step_size,
        batch_size,
        cvar_alpha,
        steps,
        10,
    )
