import jax.numpy as jnp
from jax import random

import dro_sweeps.data_generation
import dro_sweeps.regression_data_generation as dg


def test_shapes():
    key = random.PRNGKey(420)

    sample = dro_sweeps.data_generation.sample_gaussian(key, 0, 1, 20)
    assert sample.shape == (20, 1)

    inputs = dro_sweeps.data_generation.make_inputs(sample)
    assert inputs.shape == (20, 2)

    weights = jnp.array((0.5, 0.5))
    outputs = dro_sweeps.data_generation.linear_outputs(inputs, weights)
    assert outputs.shape == (20, 1)

    noisy_outputs = dg.noisy_linear_outputs(key, inputs, weights, 1)
    assert noisy_outputs.shape == (20, 1)

    x, y = dg.generate_samples(key, 0, 1, 20, jnp.array((0.5, 0.5)), 0.1)
    assert x.shape == (20, 2)
    assert y.shape == (20, 1)
