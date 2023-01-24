import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random

import dro_sweeps.data_generation as dg
import dro_sweeps.classification_data_generation as cdg
import dro_sweeps.dro as dro


def main():
    seed = 42069
    key = random.PRNGKey(seed)

    population_1 = {
        'size': 5000,
        'input_mean': (0.1, 0.2),
        'input_covariance': ((0.2, 0.0), (0.0, 0.2)),
        'weights': jnp.array((3.5, 0.9, 0.5)),
        'noise_variance': 0.005,
    }
    population_2 = {
        'size': 500,
        'input_mean': (0.05, 0.1),
        'input_covariance': ((0.1, 0.0), (0.0, 0.1)),
        'weights': jnp.array((3.0, 0.9, 0.5)),
        'noise_variance': 0.005,
    }

    key, subkey = random.split(key)
    X_1, Y_1 = cdg.generate_samples(subkey, **population_1)

    key, subkey = random.split(key)
    X_2, Y_2 = cdg.generate_samples(subkey, **population_2)


    plt.figure(dpi=150)
    plt.scatter(X_1[:, 0], X_1[:, 1], alpha=0.5, linewidths=0, s=4, c=Y_1)
    plt.savefig('plots/classification_populations_1.png')
    plt.clf()

    plt.figure(dpi=150)
    plt.scatter(X_2[:, 0], X_2[:, 1], alpha=0.5, linewidths=0, s=5, c=Y_2)
    plt.savefig('plots/classification_populations_2.png')
    plt.clf()

    X = jnp.concatenate((X_1, X_2))
    Y = jnp.concatenate((Y_1, Y_2))

    cvar_alphas = jnp.logspace(-4, 0, 2)
    batch_sizes = [8, 256]

    steps = 1e3
    step_size = 0.005
    init_weights = jnp.array((0.1, 0.1, 0.1)).reshape((3, 1))

    results = {}
    for batch_size in batch_sizes:
        print(f'Sweeping batch size {batch_size}...')

        results[batch_size] = {}
        results[batch_size]['averaged_weights'] = []
        results[batch_size]['loss_trajectories'] = []
        for cvar_alpha in cvar_alphas:
            weights, loss_trajectory = dro.train_averaged_dro(
                key,
                X,
                Y,
                init_weights,
                dg.linear_outputs,
                dro.squared_err_loss,
                step_size,
                int(batch_size),
                float(cvar_alpha),
                steps,
            )
            results[batch_size]['averaged_weights'].append(weights)
            results[batch_size]['loss_trajectories'].append(loss_trajectory)
            print(f'⍺={cvar_alpha:0.2f} ✅ ', end='')
        print('')

    for batch_size in results.keys():
        for cvar_alpha, losses in zip(cvar_alphas, results[batch_size]['loss_trajectories']):
            plt.plot(losses, alpha=0.5, label=f'$\\alpha={cvar_alpha:0.3f}$', color=plt.cm.cividis(cvar_alpha))
        plt.title(f'Weighted training step loss, batch size $n={batch_size}$')
        plt.legend(bbox_to_anchor=(1.3, 1.0))
        plt.semilogy()
        plt.savefig(f'plots/classification_loss_vs_steps_{batch_size}.png')
        plt.clf()

    # TODO plot learned models


print('running main...')
main()
