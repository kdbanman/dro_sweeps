import jax.numpy as jnp

from jax import random

import dro_sweeps.data_generation as dg


def sample_multivariate_gaussian(key, mean_vector, covariance, size):
    """
    Sample a gaussian batch centered at mean_vector.
    :param key: PRNG key
    :param mean_vector: Center of batch distribution.  Tuple, list, etc. with dimensionality d.
    :param covariance: Spread of batch distribution.  Tuples, lists, etc. with dimensionality dxd
    :param size: Size of batch.
    :return: Batch of shape (size, d)
    """
    means = jnp.vstack([jnp.array(mean_vector)] * size)
    covariances = jnp.vstack([jnp.array(covariance)] * size).reshape((means.shape[0], means.shape[1], means.shape[1]))

    batch = random.multivariate_normal(key, means, covariances)
    return batch


def logistic_function(scalar_inputs):
    return 1 / (1 + jnp.exp(-scalar_inputs))


def logistic_outputs(inputs, weights):
    return logistic_function(dg.linear_outputs(inputs, weights))


def label_logits(logits):
    return jnp.select([logits < 0.5, logits >= 0.5], [jnp.zeros_like(logits), jnp.ones_like(logits)])


def label_outputs(inputs, weights):
    """
    Compute 0-1 label outputs based on a 0.5 threshold of the logits from input and weight linear combination.
    """
    return label_logits(logistic_outputs(inputs, weights))


def noisy_label_outputs(key, inputs, weights, noise_variance):
    """
    Compute the same as label_outputs, but with labels swapped around the separating hyperplane as per
    a Gaussian distribution with the provided noise variance.
    """
    logits = logistic_outputs(inputs, weights)
    logits += dg.sample_gaussian(key, 0, noise_variance, logits.shape[0])
    return label_logits(logits)


def generate_samples(
        key,
        size,
        input_mean_vector,
        input_covariance,
        weights,
        noise_variance,
):
    """
    :param key: PRNG key
    :param size: batch size
    :param input_mean_vector: Center of input batch distribution.  Tuple, list, etc. with dimensionality d.
    :param input_covariance: Spread of input batch distribution.  Tuples, lists, etc. with dimensionality dxd
    :param weights: Separating hyperplane params.  Tuple, list, etc. dimensionality d + 1
    :param noise_variance: Separating hyperplane noise.
    :return: (inputs, labels) tuple with binary outputs and inputs padded by ones:
             inputs.shape == (size, d + 1) and labels.shape == (size, 1)
    """
    inputs_key, noise_key = random.split(key)

    inputs = sample_multivariate_gaussian(key, input_mean_vector, input_covariance, size)
    inputs = dg.make_inputs(inputs)

    outputs = noisy_label_outputs(key, inputs, weights, noise_variance)
    return inputs, outputs


def generate_dataset(key, subgroup_configs):
    subgroup_inputs = []
    subgroup_outputs = []

    for config in subgroup_configs:
        key, subkey = random.split(key)
        inputs, outputs = generate_samples(
            subkey,
            config['size'],
            config['input_mean'],
            config['input_covariance'],
            config['weights'],
            config['noise'],
        )
        subgroup_inputs.append(inputs)
        subgroup_outputs.append(outputs)

    return jnp.concatenate(subgroup_inputs), jnp.concatenate(subgroup_outputs)


## TODO
# - add a classification main.py to test with figures
# - move to test classification integration test file
