from jax import numpy as jnp


def make_inputs(x):
    """
    Assumes shape (size, d) and appends a column of ones for bias.
    Resulting shape is (size, d + 1) as per row-wise samples in a batch.
    """
    size = x.shape[0]
    return jnp.hstack((
        x,
        jnp.ones(size).reshape((size, 1)),
    ))


def linear_outputs(x, weights):
    """
    Returns a (size, 1)-shaped array of (noiseless) outputs.
    Each row is the dot product of the corresponding x row and weights.
    """
    size = x.shape[0]
    return jnp.dot(x, weights).reshape((size, 1))