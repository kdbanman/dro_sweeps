import pytest

import jax.numpy as jnp
from jax import random

import dro_sweeps.classification_data_generation as cdg
import dro_sweeps.data_generation as dg


@pytest.fixture
def linear_separated_data():
    key = random.PRNGKey(420)

    sample = cdg.sample_gaussian(key, (-1.0, 1.0), ((3.0, 0), (0, 3.0)), 20)

    inputs = dg.make_inputs(sample)

    weights = jnp.array((1.0, -1.0, 0.5))

    outputs = dg.linear_outputs(inputs, weights)
    logits = cdg.logistic_outputs(inputs, weights)
    labels = cdg.label_outputs(inputs, weights)

    return sample, inputs, outputs, logits, labels


def test_shapes(linear_separated_data):
    sample, inputs, outputs, logits, labels = linear_separated_data

    assert sample.shape == (20, 2)
    assert inputs.shape == (20, 3)

    assert outputs.shape == (20, 1)
    assert logits.shape == (20, 1)
    assert labels.shape == (20, 1)


def test_ranges(linear_separated_data):
    sample, inputs, outputs, logits, labels = linear_separated_data

    assert jnp.alltrue(logits >= 0)
    assert jnp.alltrue(logits <= 1)

    assert jnp.alltrue(jnp.logical_or(labels == 1., labels == 0.))
