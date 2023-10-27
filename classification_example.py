import jax.numpy as jnp
from jax import random

import wandb
from tqdm.contrib import itertools

import dro_sweeps.classification_data_generation as cdg
import dro_sweeps.dro as dro


def main():
    project_name = 'two-small-pop-dro-sweep'

    max_log_samples = 1000

    cvar_alphas = [float(x) for x in jnp.logspace(-1, 0, 5)]
    batch_sizes = [64, 512, 2048]

    seeds = range(10)

    parent_config = {
        'populations': [
            {
                'label': 'Majority',
                'size': 2048 - 256,
                'input_mean': (-0.2, -0.2),
                'input_covariance': ((0.1, 0.0), (0.0, 0.1)),
                'weights': jnp.array((3.5, 0.9, 1.0)),
                'noise_variance': 0.,
            },
            {
                'label': 'Minority',
                'size': 256,
                'input_mean': (-0., -0.),
                'input_covariance': ((0.1, 0.0), (0.0, 0.1)),
                'weights': jnp.array((3.5, 0.9, 0.0)),
                'noise_variance': 0.,
            },
        ],
        'steps': int(1e6),
        'step_size': 0.1,
        'init_weights': jnp.array((0.1, 0.1, 0.1)).reshape((3, 1)),
    }

    for batch_size, cvar_alpha, seed in itertools.product(batch_sizes, cvar_alphas, seeds):
        key = random.PRNGKey(seed)

        key, subkey = random.split(key)
        inputs_1, labels_1 = cdg.generate_samples(subkey, **parent_config['populations'][0])

        key, subkey = random.split(key)
        inputs_2, labels_2 = cdg.generate_samples(subkey, **parent_config['populations'][1])

        inputs = jnp.concatenate((inputs_1, inputs_2))
        labels = jnp.concatenate((labels_1, labels_2))

        log_period = max(1, parent_config['steps'] // max_log_samples)

        config = parent_config | {
            'seed': seed,
            'cvar_alpha': cvar_alpha,
            'batch_size': batch_size,
        }

        wandb.init(
            project=project_name,
            config=config,
            name=f'b={batch_size},⍺={cvar_alpha:0.2f},s={seed}'
        )

        key, subkey = random.split(key)
        dro.train_averaged_dro(
            subkey,
            inputs,
            labels,
            config['init_weights'],
            cdg.logistic_outputs,
            dro.cross_entropy_loss,
            config['step_size'],
            int(batch_size),
            float(cvar_alpha),
            config['steps'],
            log_period,
            wandb,
        )

        wandb.finish()


if __name__ == '__main__':
    main()
