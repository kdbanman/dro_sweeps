import logging
import math
from functools import partial

import jax.numpy as jnp
from jax import jit, lax, random, value_and_grad


@partial(jit, static_argnames='cvar_alpha')
def cvar_batch_weights(cvar_alpha, losses):
    batch_size = losses.shape[0]

    if cvar_alpha <= 0 or cvar_alpha > 1.0:
        raise Exception(f'cvar_alpha must be in (0, 1.0]')
    elif cvar_alpha == 1.0:
        batch_weights = jnp.ones_like(losses) / batch_size
    else:  # cvar_alpha in open interval (0, 1)
        if cvar_alpha < 1 / batch_size:
            # Indexing logic below already handles this case implicitly, but we should log to the user when it happens.
            logging.warning(f'Batch size n={batch_size} can express a minimum cvar_alpha of ' +
                            f'1/n={1 / batch_size}, but {cvar_alpha} requested.  Defaulting to 1/n={1 / batch_size}.')

        # CAUTION
        # If cvar_alpha == 1.0, then cutoff_idx == batch_size, hence is out of bounds for assignment.
        # jax is perfectly happy with this:
        #   https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing
        # Don't let that path fall in here.
        #
        # If you change this function and that happens, existing unit tests seem to catch it.

        # Also, this is theoretically improvable by selecting instead of sorting:
        #   https://en.wikipedia.org/wiki/Quickselect
        # but doing this properly in JAX/XLA could be a subtle undertaking:
        #   https://github.com/google/jax/issues/4379

        cutoff_idx = math.floor(cvar_alpha * batch_size)
        surplus = 1.0 - cutoff_idx / (cvar_alpha * batch_size)

        batch_weights = jnp.zeros_like(losses, dtype='float32')
        sorted_indices = jnp.flip(jnp.argsort(losses, axis=0), axis=0)

        update_indices = lax.dynamic_slice(sorted_indices, (0, 0), (cutoff_idx, 1))
        batch_weights = batch_weights.at[update_indices].set(1.0 / (cvar_alpha * batch_size))
        batch_weights = batch_weights.at[sorted_indices[cutoff_idx]].set(surplus)

    return batch_weights


def shuffle_data(key, inputs, outputs):
    key, subkey = random.split(key)

    # Permute X, Y from the same key for identical random ordering
    inputs = random.permutation(subkey, inputs, axis=0, independent=False)
    outputs = random.permutation(subkey, outputs, axis=0, independent=False)

    return inputs, outputs


def squared_err_loss(predictions, outputs):
    return jnp.square(outputs - predictions)


def cross_entropy_loss(predictions, outputs):
    return -outputs * jnp.log(predictions) - (1 - outputs) * jnp.log(1 - predictions)


def weighted_loss(inputs, outputs, weights, predict_fn, loss_fn, batch_weights):
    predictions = predict_fn(inputs, weights)
    losses = loss_fn(predictions, outputs)
    return jnp.sum(losses * batch_weights)


@partial(jit, static_argnames=('predict_fn', 'loss_fn', 'step_size', 'cvar_alpha'))
def dro_update(inputs, outputs, weights, predict_fn, loss_fn, step_size, cvar_alpha):
    predictions = predict_fn(inputs, weights)
    losses = loss_fn(predictions, outputs)
    batch_weights = cvar_batch_weights(cvar_alpha, losses)

    loss, grads = value_and_grad(weighted_loss, argnums=2)(inputs, outputs, weights, predict_fn, loss_fn, batch_weights)

    return weights - step_size * grads, loss


def batches(inputs, batch_size):
    """
    Yields each batch as an iterator.

    Cases with inputs.shape[0] := N and batch_size := n

    - N % n == 0: N/n minibatches yielded, each size n
    - N % n == m: ⌈N/n⌉ minibatches yielded, all size n except final size m
    """
    num_samples = inputs.shape[0]
    num_batches = num_samples // batch_size

    inputs_head = inputs[:num_batches * batch_size]

    inputs_batches = inputs_head.reshape((num_batches, batch_size, inputs_head.shape[1]))

    for batch in inputs_batches:
        yield batch

    if num_samples % batch_size != 0:
        final_batch = inputs[-(num_samples % batch_size):]
        yield final_batch


def train_averaged_dro(key, inputs, outputs, weights, predict_fn, loss_fn, step_size, batch_size, cvar_alpha, steps):
    """
    optimize weights by DRO w/ ⍺-CVar (⍺=0 is conventional SGD)
    return weights averaged across final half of steps

    Averaging scheme as per https://arxiv.org/abs/1212.2002

    NOTE: Not using momentum acceleration, because "the accelerated guarantees require
    the loss to have order 1/eps-Lipschitz gradients" which only holds for the
    regularized objectives.
    """
    loss_trajectory = []
    step = 0
    mean_weights = weights
    while step < steps:
        key, subkey = random.split(key)
        inputs, outputs = shuffle_data(subkey, inputs, outputs)
        inputs_batches = batches(inputs, batch_size)
        outputs_batches = batches(outputs, batch_size)
        for input_batch, output_batch in zip(inputs_batches, outputs_batches):
            step += 1

            weights, loss = dro_update(input_batch, output_batch, weights, predict_fn, loss_fn, step_size, cvar_alpha)
            loss_trajectory.append(loss)

            mean_weights = (1 - 2 / (step + 2)) * mean_weights + 2 / (step + 2) * weights

            if step >= steps:
                break
    return weights, loss_trajectory
