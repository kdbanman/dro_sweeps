import jax.numpy as jnp

from jax import random


def sample_gaussian(key, mean_vector, covariance, size):
    """
    Sample a gaussian centered of shape (size, d) where d is the mean_vector dimensionality.
    """
    means = jnp.vstack([jnp.array(mean_vector)] * size)
    covariances = jnp.vstack([jnp.array(covariance)] * size).reshape((means.shape[0], means.shape[1], means.shape[1]))

    batch = random.multivariate_normal(key, means, covariances)
    return batch
