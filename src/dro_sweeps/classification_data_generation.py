import jax.numpy as jnp

from jax import random

import dro_sweeps.data_generation as dg


def sample_gaussian(key, mean_vector, covariance, size):
    """
    Sample a gaussian centered of shape (size, d) where d is the mean_vector dimensionality.
    """
    means = jnp.vstack([jnp.array(mean_vector)] * size)
    covariances = jnp.vstack([jnp.array(covariance)] * size).reshape((means.shape[0], means.shape[1], means.shape[1]))

    batch = random.multivariate_normal(key, means, covariances)
    return batch


def logistic_function(scalar_inputs):
    return 1 / (1 + jnp.exp(-scalar_inputs))


def logistic_outputs(inputs, weights):
    return logistic_function(dg.linear_outputs(inputs, weights))


def label_outputs(inputs, weights):
    logits = logistic_outputs(inputs, weights)
    return jnp.select([logits < 0.5, logits >= 0.5], [jnp.zeros_like(logits), jnp.ones_like(logits)])


## TODO
# - add a noise inseparability, maybe by adding noise to logit scores?
# - add a classification main.py to test with figures
# - move to test classification integration test file
