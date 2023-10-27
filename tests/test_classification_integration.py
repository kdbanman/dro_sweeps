import jax.numpy as jnp
from jax import random

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import dro_sweeps.classification_data_generation as cdg
import dro_sweeps.dro as dro


def test_classification_cvar_smoke_test():
    """
    Using a nontrivial class balance and uncertainty set size, make sure
    CVaR DRO doesn't explode on a full run.
    """
    seed = 42069
    key = random.PRNGKey(seed)

    population_1 = {
        'size': 500,
        'input_mean': (1.0, 1.0),
        'input_covariance': ((1.0, 0.0), (0.0, 1.0)),
        'weights': jnp.array((0.5, 0.5, 0.5)),
        'noise_variance': 0.1,
    }
    population_2 = {
        'size': 50,
        'input_mean': (1.5, 1.5),
        'input_covariance': ((1.0, 0.0), (0.0, 1.0)),
        'weights': jnp.array((0.8, 0.8, 0.8)),
        'noise_variance': 0.1,
    }

    cvar_alpha = 0.1
    batch_size = 8

    steps = 1e2
    step_size = 0.005
    init_weights = jnp.array((0.1, 0.1, 0.1)).reshape((3, 1))

    key, subkey = random.split(key)
    inputs, labels = cdg.generate_dataset(key, [population_1, population_2])

    _training_results = dro.train_averaged_dro(
        subkey,
        inputs,
        labels,
        init_weights,
        cdg.logistic_outputs,
        dro.cross_entropy_loss,
        step_size,
        batch_size,
        cvar_alpha,
        steps,
        10,
    )


def test_classification_sgd_correctness():
    """
    As a basic, non-comprehensive correctness test, compare sklearn's logistic
    regression to the sgd configuration (i.e. CVaR with alpha = 0).
    """

    # TODO finish this test.  The weights can be different because inifinitely
    #      many planes separate the classes

    # seed = 42069
    # key = random.PRNGKey(seed)
    # key, subkey = random.split(key)
    # inputs, labels = cdg.generate_dataset(subkey, [{
    #     'size': 500,
    #     'input_mean': (1.0, 1.0),
    #     'input_covariance': ((1.0, 0.0), (0.0, 1.0)),
    #     'weights': jnp.array((0.5, 0.5, 0.5)),
    #     'noise_variance': 0.001,
    # }])
    #
    # reference_model = LogisticRegression(
    #     random_state=0,
    #     tol=1e-7,
    #     fit_intercept=False,
    #     C=1.0,
    #     max_iter=1000
    # ).fit(inputs, labels)
    #
    # steps = 1e4
    # step_size = 1e-1
    # init_weights = jnp.array([0.1, 0.1, 0.1]).reshape((3, 1))
    #
    # key, subkey = random.split(key)
    # _training_results = dro.train_averaged_dro(
    #     key,
    #     inputs,
    #     labels,
    #     init_weights,
    #     cdg.logistic_outputs,
    #     dro.cross_entropy_loss,
    #     step_size,
    #     inputs.shape[1],
    #     1.0,
    #     steps,
    #     10,
    # )
