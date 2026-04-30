"""Smoke tests for nn4psych.bayesian.reduced_bayesian.

Tests verify:
- compute_rbo_forward traces under jax.jit for both contexts
- prior_sampler returns correct shapes for all parameters
- simulate_synthetic_data returns (n_trials,) arrays
- run_mcmc smoke fit returns MCMC with diverging field present
- assert_jax_devices passes with XLA_FLAGS set at package import
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import nn4psych.bayesian as bayes
from nn4psych.bayesian.reduced_bayesian import (
    assert_jax_devices,
    compute_rbo_forward,
    prior_sampler,
    reduced_bayesian_model,
    run_mcmc,
    simulate_synthetic_data,
)


def test_forward_model_traces_changepoint():
    """compute_rbo_forward compiles under jax.jit for changepoint context."""
    pred_errors = jnp.zeros(10)
    params = {"H": jnp.array(0.125), "LW": jnp.array(0.7), "UU": jnp.array(0.9)}

    @jax.jit
    def _run():
        return compute_rbo_forward(params, pred_errors, "changepoint")

    lr, norm_upd, omega, tau = _run()
    assert lr.shape == (10,), f"expected (10,), got {lr.shape}"
    assert norm_upd.shape == (10,), f"expected (10,), got {norm_upd.shape}"
    assert omega.shape == (10,), f"expected (10,), got {omega.shape}"
    assert tau.shape == (11,), f"expected (11,), got {tau.shape}"


def test_forward_model_traces_oddball():
    """compute_rbo_forward compiles under jax.jit for oddball context."""
    pred_errors = jnp.zeros(10)
    params = {"H": jnp.array(0.125), "LW": jnp.array(0.7), "UU": jnp.array(0.9)}

    @jax.jit
    def _run():
        return compute_rbo_forward(params, pred_errors, "oddball")

    lr, norm_upd, omega, tau = _run()
    assert lr.shape == (10,), f"expected (10,), got {lr.shape}"


def test_prior_sampler_shapes():
    """prior_sampler returns dict with expected keys and shapes (num_samples,)."""
    rng_key = jax.random.PRNGKey(0)
    num_samples = 5
    samples = prior_sampler(reduced_bayesian_model, num_samples=num_samples, rng_key=rng_key)

    expected_keys = {"H", "LW", "UU", "sigma_motor", "sigma_LR"}
    assert expected_keys.issubset(set(samples.keys())), (
        f"expected keys {expected_keys}, got {set(samples.keys())}"
    )
    for key in expected_keys:
        assert samples[key].shape == (num_samples,), (
            f"expected shape ({num_samples},) for '{key}', got {samples[key].shape}"
        )


def test_simulate_synthetic_data_shapes():
    """simulate_synthetic_data returns (n_trials,) for both outputs."""
    n_trials = 50
    params = {
        "H": 0.125,
        "LW": 0.7,
        "UU": 0.9,
        "sigma_motor": 5.0,
        "sigma_LR": 0.3,
    }
    bag, bucket = simulate_synthetic_data(
        params, n_trials=n_trials, hazard=0.125, context="changepoint", seed=0
    )
    assert bag.shape == (n_trials,), f"expected ({n_trials},), got {bag.shape}"
    assert bucket.shape == (n_trials,), f"expected ({n_trials},), got {bucket.shape}"


@pytest.mark.slow
def test_run_mcmc_smoke():
    """run_mcmc smoke fit returns MCMC with diverging field in extra_fields."""
    params = {
        "H": 0.125,
        "LW": 0.7,
        "UU": 0.9,
        "sigma_motor": 5.0,
        "sigma_LR": 0.3,
    }
    bag, bucket = simulate_synthetic_data(
        params, n_trials=30, hazard=0.125, context="changepoint", seed=1
    )
    mcmc = run_mcmc(
        bag,
        bucket,
        context="changepoint",
        num_warmup=20,
        num_samples=20,
        num_chains=2,
        seed=0,
        progress_bar=False,
    )
    samples = mcmc.get_samples()
    expected_params = {"H", "LW", "UU", "sigma_motor", "sigma_LR"}
    assert expected_params.issubset(set(samples.keys())), (
        f"expected parameters {expected_params}, got {set(samples.keys())}"
    )
    extra = mcmc.get_extra_fields()
    assert "diverging" in extra, (
        f"'diverging' field missing from extra_fields; got {list(extra.keys())}"
    )


def test_assert_jax_devices():
    """assert_jax_devices passes when XLA_FLAGS set at package import."""
    # XLA_FLAGS is set in nn4psych.bayesian.__init__ before any jax import,
    # so jax.local_device_count() should return 4.
    assert_jax_devices(expected=4)  # should not raise
