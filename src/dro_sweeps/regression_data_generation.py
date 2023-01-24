import jax.numpy as jnp

from jax import random

from dro_sweeps.data_generation import make_inputs, linear_outputs, sample_gaussian


def noisy_linear_outputs(key, inputs, weights, noise_variance):
    """
    Returns the same as linear_outputs, but with zero-mean gaussian noise
    """
    size = inputs.shape[0]
    noise = jnp.sqrt(noise_variance) * random.normal(key, (size, 1))
    return linear_outputs(inputs, weights) + noise


def generate_samples(key, input_mean, input_variance, size, weights, noise_variance):
    """
    Returns (inputs, outputs) tuple
    inputs are scalars padded by ones: inputs.shape == (size, 2), gaussian
    outputs are scalars: outputs.shape == (size), gaussian-noised linear
    """
    inputs_key, noise_key = random.split(key)

    inputs = sample_gaussian(inputs_key, input_mean, input_variance, size)
    inputs = make_inputs(inputs)

    outputs = noisy_linear_outputs(noise_key, inputs, weights, noise_variance)

    return inputs, outputs


def generate_dataset(key, subgroup_configs):
    subgroup_inputs = []
    subgroup_outputs = []
    for config in subgroup_configs:
        key, subkey = random.split(key)
        inputs, outputs = generate_samples(
            subkey,
            config['mean'],
            config['variance'],
            config['size'],
            config['weights'],
            config['noise'],
        )
        subgroup_inputs.append(inputs)
        subgroup_outputs.append(outputs)

    return jnp.concatenate(subgroup_inputs), jnp.concatenate(subgroup_outputs)
