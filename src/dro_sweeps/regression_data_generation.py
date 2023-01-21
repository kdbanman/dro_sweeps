import jax.numpy as jnp

from jax import random

from dro_sweeps.data_generation import make_inputs, linear_outputs


def sample_gaussian(key, mean, variance, size):
    """
    Returns 1d array of gaussian samples of shape (size, 1)
    """
    return jnp.sqrt(variance) * random.normal(key, (size, 1)) + mean


def noisy_linear_outputs(key, x, weights, noise_variance):
    """
    Returns the same as linear_outputs, but with zero-mean gaussian noise
    """
    size = x.shape[0]
    noise = jnp.sqrt(noise_variance) * random.normal(key, (size, 1))
    return linear_outputs(x, weights) + noise


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

    y = noisy_linear_outputs(noise_key, x, weights, noise_variance)

    return x, y


def generate_dataset(subgroup_configs, key):
    subgroup_inputs = []
    subgroup_outputs = []
    for config in subgroup_configs:
        key, subkey = random.split(key)
        X, Y = generate_samples(
            config['size'],
            config['mean'],
            config['variance'],
            config['weights'],
            config['noise'],
            subkey,
        )
        subgroup_inputs.append(X)
        subgroup_outputs.append(Y)

    return jnp.concatenate(subgroup_inputs), jnp.concatenate(subgroup_outputs)
