import jax.numpy as jnp
from jax import random

import dro_sweeps.classification_data_generation as dg


def test_shapes():
    key = random.PRNGKey(420)

    sample = dg.sample_gaussian(key, (-1.0, 1.0), ((0.1, 0), (0, 0.1)), 20)
    assert sample.shape == (20, 2)
