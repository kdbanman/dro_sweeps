import jax.numpy as jnp

from jax import random


def sample_gaussian(key, mean, variance, size):
    """
    Returns 1d array of gaussian samples of shape (size, 1)
    """
    return jnp.sqrt(variance) * random.normal(key, (size, 1)) + mean


def make_inputs(x):
    """
    Assumes shape (size, 1) and appends a column of ones for bias.
    Resulting shape is (size, 2) as per row-wise samples in a batch.
    """
    size = x.shape[0]
    return jnp.hstack((
        x.reshape((size, 1)),
        jnp.ones_like(x).reshape((size, 1)),
    ))


def compute_outputs(x, weights):
    """
    Returns a (size, 1)-shaped array of (noiseless) outputs.
    Each row is the dot product of the corresponding x row and weights.
    """
    size = x.shape[0]
    return jnp.dot(x, weights).reshape((size, 1))


def compute_noisy_outputs(key, x, weights, noise_variance):
    """
    Returns the same as compute_outputs, but with zero-mean gaussian noise
    """
    size = x.shape[0]
    noise = jnp.sqrt(noise_variance) * random.normal(key, (size, 1))
    return compute_outputs(x, weights) + noise


def generate_samples(
        size,
        x_mean,
        x_variance,
        weights,
        noise_variance,
        key,
):
    """
    Returns (x, y) tuple
    x are scalar inputs padded by ones: x.shape == (size, 2)
    y are scalar outputs: y.shape == (size)
    x are gaussian
    y are a gaussian-noised linear function of x according to weights
    """
    x_key, noise_key = random.split(key)

    x = sample_gaussian(x_key, x_mean, x_variance, size)
    x = make_inputs(x)

    y = compute_noisy_outputs(noise_key, x, weights, noise_variance)

    return x, y
