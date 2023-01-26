import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random

import dro_sweeps.data_generation
import dro_sweeps.regression_data_generation as dg
import dro_sweeps.dro as dro


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
    X_1, Y_1 = dg.generate_samples(subkey, population_1['mean'], population_1['variance'], population_1['size'],
                                   population_1['weights'], population_1['noise'])

    key, subkey = random.split(key)
    X_2, Y_2 = dg.generate_samples(subkey, population_2['mean'], population_2['variance'], population_2['size'],
                                   population_2['weights'], population_2['noise'])

    plt.figure(dpi=150)
    plt.scatter(X_1[:, 0], Y_1, alpha=0.01, linewidths=0, s=2)
    plt.scatter(X_2[:, 0], Y_2, alpha=0.01, linewidths=0, s=2)
    plt.savefig('plots/regression_populations.png')
    plt.clf()

    X = jnp.concatenate((X_1, X_2))
    Y = jnp.concatenate((Y_1, Y_2))

    cvar_alphas = jnp.logspace(-4, 0, 2)
    batch_sizes = [8, 256]

    steps = 1e3
    step_size = 0.005
    init_weights = jnp.array((0.1, 0.1)).reshape((2, 1))

    max_log_samples = 200
    log_period = steps // max_log_samples

    results = {}
    for batch_size in batch_sizes:
        print(f'Sweeping batch size {batch_size}...')

        results[batch_size] = {}
        results[batch_size]['averaged_weights'] = []
        results[batch_size]['loss_trajectories'] = []
        results[batch_size]['log_steps'] = []
        for cvar_alpha in cvar_alphas:
            weights, loss_trajectory, log_steps = dro.train_averaged_dro(
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
                log_period,
            )
            results[batch_size]['averaged_weights'].append(weights)
            results[batch_size]['loss_trajectories'].append(loss_trajectory)
            results[batch_size]['log_steps'].append(log_steps)
            print(f'⍺={cvar_alpha:0.2f} ✅ ', end='')
        print('')

    domain = dro_sweeps.data_generation.make_inputs(jnp.arange(-4, 4, 0.01).reshape((-1, 1)))

    for batch_size in results.keys():
        for cvar_alpha, losses, log_steps in zip(cvar_alphas, results[batch_size]['loss_trajectories'], results[batch_size]['log_steps']):
            plt.plot(log_steps, losses, alpha=0.5, label=f'$\\alpha={cvar_alpha:0.3f}$', color=plt.cm.cividis(cvar_alpha))
        plt.title(f'Weighted training step loss, batch size $n={batch_size}$')
        plt.legend(bbox_to_anchor=(1.3, 1.0))
        plt.semilogy()
        plt.savefig(f'plots/regression_loss_vs_steps_{batch_size}.png')
        plt.clf()

        plt.figure(dpi=200)
        # plt.scatter(X_1[:, 0], Y_1, alpha=0.01, linewidths=0, s=2)
        plt.plot(
            domain[:, 0],
            dro_sweeps.data_generation.linear_outputs(domain, population_1['weights']),
            color='blue',
            linewidth=1,
            linestyle='--',
        )
        # plt.scatter(X_2[:, 0], Y_2, alpha=0.01, linewidths=0, s=2)
        plt.plot(
            domain[:, 0],
            dro_sweeps.data_generation.linear_outputs(domain, population_2['weights']),
            color='orange',
            linewidth=1,
            linestyle='--',
        )

        for cvar_alpha, weights in zip(cvar_alphas, results[batch_size]['averaged_weights']):
            plt.plot(
                domain[:, 0],
                dro_sweeps.data_generation.linear_outputs(domain, weights),
                alpha=0.7,
                label=f'$\\alpha={cvar_alpha:0.4f}$',
                color=plt.cm.viridis((4 + jnp.log(cvar_alpha) / jnp.log(10)) / 4),
                linewidth=1,
            )

        plt.title(f'Learned model, batch size $n={batch_size}$')
        plt.legend(bbox_to_anchor=(1.3, 1.0))
        plt.savefig(f'plots/regression_learned_model_{batch_size}.png')
        plt.clf()


print('running main...')
main()
