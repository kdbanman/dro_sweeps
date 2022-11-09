import logging
import math
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax import value_and_grad, random, jit, lax

def sample_gaussian(key, mean, variance, size):
  '''
  Returns 1d array of gaussian samples of shape (size, 1) 
  '''
  return jnp.sqrt(variance) * random.normal(key, (size, 1)) + mean


def make_inputs(x):
  '''
  Assumes shape (size, 1) and appends a column of ones for bias.
  Resulting shape is (size, 2) as per row-wise samples in a batch.
  '''
  size = x.shape[0]
  return jnp.hstack((
      x.reshape((size, 1)),
      jnp.ones_like(x).reshape((size, 1)),
  ))


def compute_outputs(x, weights):
  '''
  Returns a (size, 1)-shaped array of (noiseless) outputs.
  Each row is the dot product of the corresponding x row and weights.
  '''
  size = x.shape[0]
  return jnp.dot(x, weights).reshape((size, 1))


def compute_noisy_outputs(key, x, weights, noise_variance):
  '''
  Returns the same as compute_outputs, but with zero-mean gaussian noise
  '''
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
  '''
  Returns (x, y) tuple
  x are scalar inputs padded by ones: x.shape == (size, 2)
  y are scalar outputs: y.shape == (size)
  x are gaussian
  y are a gaussian-noised linear function of x according to weights
  '''
  x_key, noise_key = random.split(key)

  x = sample_gaussian(x_key, x_mean, x_variance, size)
  x = make_inputs(x)

  y = compute_noisy_outputs(noise_key, x, weights, noise_variance)

  return x, y


@partial(jit, static_argnames=('cvar_alpha'))
def cvar_batch_weights(cvar_alpha, losses, debug=False):
  batch_size = losses.shape[0]

  if cvar_alpha <=0 or cvar_alpha > 1.0:
    raise Exception(f'cvar_alpha must be in (0, 1.0]')
  elif cvar_alpha == 1.0:
    batch_weights = jnp.ones_like(losses) / batch_size
  else: # cvar_alpha in open interval (0, 1)
    if debug and cvar_alpha < 1 / batch_size:
      # Indexing logic below already handles this case implicitly, but we should log to the user when it happens.
      logging.warning(f'Batch size n={batch_size} can express a minimum cvar_alpha of ' +\
                      f'1/n={1 / batch_size}, but {cvar_alpha} requested.  Defaulting to 1/n={1 / batch_size}.')

    cutoff_idx = math.floor(cvar_alpha * batch_size)
    surplus = 1.0 - cutoff_idx / (cvar_alpha * batch_size)
    
    batch_weights = jnp.zeros_like(losses, dtype='float32')
    sorted_indices = jnp.flip(jnp.argsort(losses, axis=0), axis=0)

    update_indices = lax.dynamic_slice(sorted_indices, (0, 0), (cutoff_idx, 1))
    
    batch_weights = batch_weights.at[update_indices].set(1.0 / (cvar_alpha * batch_size))

    # CAUTION
    # If cvar_alpha == 1.0, then cutoff_idx == batch_size, hence is out of bounds.
    # jax is perfectly happy with this: 
    #   https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing
    # Don't let that path fall in here.  If you change this function and that happens,
    # existing unit tests should catch it.
    batch_weights = batch_weights.at[sorted_indices[cutoff_idx]].set(surplus)
  
  return batch_weights


def test_cvar_batch_weights():
  # Minimum radius, \alpha=1.0, i.e. ERM
  batch_weights = cvar_batch_weights(cvar_alpha=1.0, losses=jnp.array([3, 2, 1]).reshape((3, 1)))
  assert batch_weights.shape == (3, 1)
  assert jnp.allclose(batch_weights, jnp.array([1 / 3] * 3).reshape((3, 1)))

  batch_weights = cvar_batch_weights(cvar_alpha=1.0, losses=jnp.array([3, 2, 4, 1]).reshape((4, 1)))
  assert batch_weights.shape == (4, 1)
  assert jnp.allclose(batch_weights, jnp.array([1 / 4] * 4).reshape((4, 1)))

  batch_weights = cvar_batch_weights(cvar_alpha=1.0, losses=jnp.array([4, 1]).reshape((2, 1)))
  assert batch_weights.shape == (2, 1)
  assert jnp.allclose(batch_weights, jnp.array([1 / 2] * 2).reshape((2, 1)))

  # Near minimum radius, \alpha=0.99
  batch_weights = cvar_batch_weights(cvar_alpha=0.99, losses=jnp.array([3, 2, 1]).reshape((3, 1)))
  assert batch_weights.shape == (3, 1)
  assert jnp.allclose(batch_weights, jnp.array([1 / (3 * 0.99), 1 / (3 * 0.99), 1.0 - 2 / (3 * 0.99)]).reshape((3, 1)))

  batch_weights = cvar_batch_weights(cvar_alpha=0.99, losses=jnp.array([3, 2, 4, 1]).reshape((4, 1)))
  assert batch_weights.shape == (4, 1)
  assert jnp.allclose(batch_weights, jnp.array([1 / (4 * 0.99), 1 / (4 * 0.99), 1 / (4 * 0.99), 1.0 - 3 / (4 * 0.99)]).reshape((4, 1)))

  # Only care about two in batch
  batch_weights = cvar_batch_weights(cvar_alpha=2/3, losses=jnp.array([3, 1, 2]).reshape(3, 1))
  assert batch_weights.shape == (3, 1)
  assert jnp.allclose(batch_weights, jnp.array([0.5, 0, 0.5]).reshape((3, 1)))

  batch_weights = cvar_batch_weights(cvar_alpha=1/2, losses=jnp.array([3, 1, 4, 2]).reshape(4, 1))
  assert batch_weights.shape == (4, 1)
  assert jnp.allclose(batch_weights, jnp.array([1/2, 0, 1/2, 0]).reshape((4, 1)))

  # Weight the biggest loss quite a bit, leave a bit for the middle, and none for the smallest.
  batch_weights = cvar_batch_weights(cvar_alpha=1/2, losses=jnp.array([3, 1, 2]).reshape(3, 1))
  assert batch_weights.shape == (3, 1)
  assert jnp.allclose(batch_weights, jnp.array([2/3, 0, 1/3]).reshape((3, 1)))

  # Weight the biggest only.
  batch_weights = cvar_batch_weights(cvar_alpha=1/3, losses=jnp.array([3, 1, 2]).reshape(3, 1))
  assert batch_weights.shape == (3, 1)
  assert jnp.allclose(batch_weights, jnp.array([1, 0, 0]).reshape((3, 1)))

  # Ensure warning when batch_size insufficiently large to express \alpha
  batch_weights = cvar_batch_weights(cvar_alpha=1/4, losses=jnp.array([3, 1, 2]).reshape(3, 1))
  # assert "warning, cannot express alpha with batch size n, defaulting to alpha = 1/n"

  # Ensure error when \alpha is zero
  # batch_weights = cvar_batch_weights(cvar_alpha=0, losses=jnp.array([3, 1, 2]).reshape(3, 1))
  # assert explosion


def shuffle_data(key, X, Y):
  key, subkey = random.split(key)

  # Permute X, Y from the same key for identical random ordering
  X = random.permutation(subkey, X, axis=0, independent=False)
  Y = random.permutation(subkey, Y, axis=0, independent=False)

  return X, Y


def batch_losses(weights, X, Y):
  predictions = compute_outputs(X, weights)
  return jnp.square(Y - predictions)


def mean_loss(weights, X, Y):
  return jnp.mean(batch_losses(weights, X, Y))


def weighted_loss(weights, batch_weights, X, Y):
  losses = batch_losses(weights, X, Y)
  return jnp.sum(losses * batch_weights)


@partial(jit, static_argnames=('step_size', 'cvar_alpha'))
def dro_update(weights, X, Y, step_size, cvar_alpha):
  losses = batch_losses(weights, X, Y)
  batch_weights = cvar_batch_weights(cvar_alpha, losses)

  loss, grads = value_and_grad(weighted_loss)(weights, batch_weights, X, Y)

  return weights - step_size * grads, loss


def batches(X, batch_size):
  '''
  Yields each batch as an iterator.

  Cases with X.shape[0] := N and batch_size := n

  - N % n == 0: N/n minibatches yielded, each size n
  - N % n == m: ⌈N/n⌉ minibatches yielded, all size n except final size m
  '''
  num_samples = X.shape[0]
  num_batches = num_samples // batch_size

  X_head = X[:num_batches * batch_size]

  batches = X_head.reshape((num_batches, batch_size, X_head.shape[1])) 
  
  for batch in batches:
    yield batch
  
  if num_samples % batch_size != 0:
    final_batch = X[-(num_samples % batch_size):]
    yield final_batch


def train_averaged_dro(key, X, Y, weights, step_size, batch_size, cvar_alpha, steps):
  '''
  optimize weights by DRO w/ ⍺-CVar (⍺=0 is conventional SGD)
  return weights averaged across final half of steps

  Averaging scheme as per https://arxiv.org/abs/1212.2002

  NOTE: Not using momentum acceleration, because "the accelerated guarantees require
  the loss to have order 1/eps-Lipschitz gradients" which only holds for the 
  regularized objectives.
  '''
  loss_trajectory = []
  step = 0
  mean_weights = weights
  while step < steps:
    key, subkey = random.split(key)
    X, Y = shuffle_data(subkey, X, Y)
    X_batches = batches(X, batch_size)
    Y_batches = batches(Y, batch_size)
    for batch_X, batch_Y in zip(X_batches, Y_batches):
      step += 1

      weights, loss = dro_update(weights, batch_X, batch_Y, step_size, cvar_alpha)
      loss_trajectory.append(loss)
      
      mean_weights = (1 - 2 / (step + 2)) * mean_weights + 2 / (step + 2) * weights

      if step >= steps:
        break
  return weights, loss_trajectory


def main():
    seed = 42069

    population_1 = {
        'size': 5000,
        'mean': 0.0,
        'variance': 1.0,
        'weights': jnp.array((1.5, 0.0)),
        'noise': 0.005,
    }
    population_2 = {
        'size': 500,
        'mean': 0.0,
        'variance': 1.0,
        'weights': jnp.array((0.5, 0.0)),
        'noise': 0.005,
    }

    key = random.PRNGKey(seed)

    key, subkey = random.split(key)
    X_1, Y_1 = generate_samples(
        population_1['size'],
        population_1['mean'],
        population_1['variance'],
        population_1['weights'],
        population_1['noise'],
        subkey,
    )

    key, subkey = random.split(key)
    X_2, Y_2 = generate_samples(
        population_2['size'],
        population_2['mean'],
        population_2['variance'],
        population_2['weights'],
        population_2['noise'],
        subkey,
    )

    plt.figure(dpi=150)
    plt.scatter(X_1[:, 0], Y_1, alpha=0.01, linewidths=0, s=2)
    plt.scatter(X_2[:, 0], Y_2, alpha=0.01, linewidths=0, s=2)
    plt.savefig('plots/populations.png')

    X = jnp.concatenate((X_1, X_2))
    Y = jnp.concatenate((Y_1, Y_2))

    cvar_alphas = jnp.logspace(-4, 0, 2)
    batch_sizes = [8, 256]

    steps = 1e3
    step_size = 0.005
    init_weights = jnp.array((0.1, 0.1)).reshape((2, 1))

    results = {}
    for batch_size in batch_sizes:
        print(f'Sweeping batch size {batch_size}...')

        results[batch_size] = {}
        results[batch_size]['averaged_weights'] = []
        results[batch_size]['loss_trajectories'] = []
        for cvar_alpha in cvar_alphas:
            weights, loss_trajectory = train_averaged_dro(
                key, 
                X, 
                Y, 
                init_weights, 
                step_size, 
                int(batch_size), 
                float(cvar_alpha), 
                steps,
            )
            results[batch_size]['averaged_weights'].append(weights)
            results[batch_size]['loss_trajectories'].append(loss_trajectory)
            print(f'⍺={cvar_alpha:0.2f} ✅ ', end='')
        print('')

    domain = make_inputs(jnp.arange(-4, 4, 0.01))

    for batch_size in results.keys():
        for cvar_alpha, losses in zip(cvar_alphas, results[batch_size]['loss_trajectories']):
            plt.plot(losses, alpha=0.5, label=f'$\\alpha={cvar_alpha:0.3f}$', color=plt.cm.cividis(cvar_alpha))
        plt.title(f'Weighted training step loss, batch size $n={batch_size}$')
        plt.legend(bbox_to_anchor=(1.3, 1.0))
        plt.semilogy()
        plt.savefig(f'plots/loss_vs_steps_{batch_size}.png')

        plt.figure(dpi=200)
        # plt.scatter(X_1[:, 0], Y_1, alpha=0.01, linewidths=0, s=2)
        plt.plot(domain[:, 0], compute_outputs(domain, population_1['weights']), color='blue', linewidth=1, linestyle='--')
        # plt.scatter(X_2[:, 0], Y_2, alpha=0.01, linewidths=0, s=2)
        plt.plot(domain[:, 0], compute_outputs(domain, population_2['weights']), color='orange', linewidth=1, linestyle='--')

        for cvar_alpha, weights in zip(cvar_alphas, results[batch_size]['averaged_weights']):
            plt.plot(domain[:, 0], compute_outputs(domain, weights), alpha=0.7, label=f'$\\alpha={cvar_alpha:0.4f}$', color=plt.cm.viridis((4 + jnp.log(cvar_alpha) / jnp.log(10)) / 4), linewidth=1)
        
        plt.title(f'Learned model, batch size $n={batch_size}$')
        plt.legend(bbox_to_anchor=(1.3, 1.0))
        plt.savefig(f'plots/learned_model_{batch_size}.png')


print('testing...')
test_cvar_batch_weights()

print('running main...')
main()