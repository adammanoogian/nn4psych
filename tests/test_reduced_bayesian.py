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


def test_matlab_parity():
    """Guard against silent drift in the JAX forward model.

    Runs compute_rbo_forward and the NumPy port of frugFun5.m on the same
    synthetic bag positions with a small but representative parameter set.
    Fails if max-abs divergence exceeds tolerance on alpha or omega.

    Tolerance note (04-03a): 5e-2 on max-abs accounts for the accepted
    deviation in the truncated-normal normalization (normcdf(300,B,totSig)
    - normcdf(0,B,totSig)) which MATLAB applies to pI but NumPyro's
    compute_rbo_forward cannot apply because it only receives pred_errors,
    not absolute bucket positions.  The median divergence is < 1e-5; the
    max-abs deviation of ~3.7% occurs only at CP trials where the bucket
    is near 0 or 300.  See 04-03a-SUMMARY.md for full analysis.
    """
    import numpy as np
    import jax.numpy as jnp
    from nn4psych.bayesian.reduced_bayesian import compute_rbo_forward, SIGMA_N
    from nn4psych.bayesian._frugfun_reference import frugfun5_reference

    rng = np.random.default_rng(42)
    bag = np.clip(rng.normal(150.0, SIGMA_N, size=64), 0, 300)

    # MATLAB self-bucketing: derive bucket from MATLAB belief trajectory
    m = frugfun5_reference(bag, Hazard=0.125, noise=SIGMA_N, likeWeight=1.0)
    bucket = m["B"][:-1]
    pred_errors = bag - bucket

    params = {"H": jnp.asarray(0.125), "LW": jnp.asarray(1.0), "UU": jnp.asarray(1.0)}
    lr, _upd, omega, _tau = compute_rbo_forward(params, jnp.asarray(pred_errors), "changepoint")

    alpha_diff = float(np.max(np.abs(np.asarray(lr) - m["alpha"])))
    omega_diff = float(np.max(np.abs(np.asarray(omega) - m["pCha"])))

    # Accepted deviation: truncated-normal correction for pI is ~3.7% max
    # (cannot apply without absolute bucket position in pred_error space).
    # Median deviation < 1e-5 confirms the core math is correct.
    assert alpha_diff < 5e-2, (
        f"alpha max-abs diff {alpha_diff:.4e} exceeds tolerance 5e-2 — "
        f"check compute_rbo_forward tot_sig and tau update equations"
    )
    assert omega_diff < 5e-2, (
        f"omega max-abs diff {omega_diff:.4e} exceeds tolerance 5e-2 — "
        f"check compute_rbo_forward U_log/N_log or log_change_ratio formula"
    )
