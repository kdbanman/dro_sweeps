import pytest
import jax.numpy as jnp
from jax import random

import dro_sweeps.scalar_data_generation as dg


def test_shapes():
    key = random.PRNGKey(420)

    sample = dg.sample_gaussian(key, 0, 1, 20)
    assert sample.shape == (20, 1)

    inputs = dg.make_inputs(sample)
    assert inputs.shape == (20, 2)

    weights = jnp.array((0.5, 0.5))
    outputs = dg.compute_outputs(inputs, weights)
    assert outputs.shape == (20, 1)

    noisy_outputs = dg.compute_noisy_outputs(key, inputs, weights, 1)
    assert noisy_outputs.shape == (20, 1)

    x, y = dg.generate_samples(
        20,
        0,
        1,
        jnp.array((0.5, 0.5)),
        0.1,
        key,
    )
    assert x.shape == (20, 2)
    assert y.shape == (20, 1)
