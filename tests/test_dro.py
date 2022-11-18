import logging
import pytest
import jax.numpy as jnp

from dro_sweeps.dro import cvar_batch_weights


def test_cvar_min_radius():
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


def test_cvar_nonzero_radius():
    # Near minimum radius, \alpha=0.99
    batch_weights = cvar_batch_weights(cvar_alpha=0.99, losses=jnp.array([3, 2, 1]).reshape((3, 1)))
    assert batch_weights.shape == (3, 1)
    assert jnp.allclose(batch_weights,
                        jnp.array([1 / (3 * 0.99), 1 / (3 * 0.99), 1.0 - 2 / (3 * 0.99)]).reshape((3, 1)))

    batch_weights = cvar_batch_weights(cvar_alpha=0.99, losses=jnp.array([3, 2, 4, 1]).reshape((4, 1)))
    assert batch_weights.shape == (4, 1)
    assert jnp.allclose(batch_weights,
                        jnp.array([1 / (4 * 0.99), 1 / (4 * 0.99), 1 / (4 * 0.99), 1.0 - 3 / (4 * 0.99)]).reshape(
                            (4, 1)))

    # Only care about two in batch
    batch_weights = cvar_batch_weights(cvar_alpha=2 / 3, losses=jnp.array([3, 1, 2]).reshape(3, 1))
    assert batch_weights.shape == (3, 1)
    assert jnp.allclose(batch_weights, jnp.array([0.5, 0, 0.5]).reshape((3, 1)))

    batch_weights = cvar_batch_weights(cvar_alpha=1 / 2, losses=jnp.array([3, 1, 4, 2]).reshape(4, 1))
    assert batch_weights.shape == (4, 1)
    assert jnp.allclose(batch_weights, jnp.array([1 / 2, 0, 1 / 2, 0]).reshape((4, 1)))

    # Weight the biggest loss quite a bit, leave a bit for the middle, and none for the smallest.
    batch_weights = cvar_batch_weights(cvar_alpha=1 / 2, losses=jnp.array([3, 1, 2]).reshape(3, 1))
    assert batch_weights.shape == (3, 1)
    assert jnp.allclose(batch_weights, jnp.array([2 / 3, 0, 1 / 3]).reshape((3, 1)))

    # Weight the biggest only.
    batch_weights = cvar_batch_weights(cvar_alpha=1 / 3, losses=jnp.array([3, 1, 2]).reshape(3, 1))
    assert batch_weights.shape == (3, 1)
    assert jnp.allclose(batch_weights, jnp.array([1, 0, 0]).reshape((3, 1)))


def test_cvar_edge_cases(caplog):
    # Ensure warning when batch_size insufficiently large to express \alpha
    with caplog.at_level(logging.WARNING):
        cvar_batch_weights(cvar_alpha=1 / 4, losses=jnp.array([3, 1, 2]).reshape(3, 1))
        assert 'minimum' in caplog.text.lower()
        assert 'defaulting' in caplog.text.lower()

    # Ensure error when \alpha is zero
    with pytest.raises(Exception):
        cvar_batch_weights(cvar_alpha=0, losses=jnp.array([3, 1, 2]).reshape(3, 1))
