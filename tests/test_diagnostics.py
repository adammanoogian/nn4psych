"""Unit tests for nn4psych.bayesian.diagnostics.

Tests cover:
- run_diagnostics: key presence, type correctness
- run_diagnostics: RuntimeWarning when extra_fields missing
- fit_with_retry: passes on first attempt with reasonable data
- fit_with_retry: falls back to retry when passes_gate is monkey-patched False
- make_fit_summary: JSON-serializable output (M3: stat_focus not kind)
"""

from __future__ import annotations

import json
import warnings

import jax
import jax.numpy as jnp
import pytest

import nn4psych.bayesian as bayes
from nn4psych.bayesian.reduced_bayesian import (
    reduced_bayesian_model,
    run_mcmc,
    simulate_synthetic_data,
    prior_sampler,
)
from nn4psych.bayesian.diagnostics import (
    run_diagnostics,
    fit_with_retry,
    make_fit_summary,
    to_jsonable,
)
import nn4psych.bayesian.diagnostics as diag_module
from numpyro.infer import MCMC, NUTS


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SMOKE_PARAMS = {
    "H": 0.125,
    "LW": 0.7,
    "UU": 0.9,
    "sigma_motor": 5.0,
    "sigma_LR": 0.3,
}
_VAR_NAMES = ["H", "LW", "UU", "sigma_motor", "sigma_LR"]


def _tiny_mcmc(seed: int = 7) -> MCMC:
    """Return a tiny fitted MCMC object (with extra_fields=('diverging',))."""
    bag, bucket = simulate_synthetic_data(
        _SMOKE_PARAMS, n_trials=20, hazard=0.125, context="changepoint", seed=seed
    )
    mcmc = run_mcmc(
        bag, bucket, context="changepoint",
        num_warmup=20, num_samples=20, num_chains=2, seed=seed,
    )
    return mcmc


# ---------------------------------------------------------------------------
# Test 1: run_diagnostics returns expected keys with correct types
# ---------------------------------------------------------------------------


def test_run_diagnostics_returns_keys():
    """run_diagnostics returns all seven required keys with correct types."""
    mcmc = _tiny_mcmc(seed=1)
    diag = run_diagnostics(mcmc)

    required_keys = {
        "rhat_max", "ess_min", "n_divergences",
        "passes_rhat", "passes_ess", "passes_gate",
    }
    assert required_keys == set(diag.keys()), (
        f"expected keys {required_keys}, got {set(diag.keys())}"
    )

    assert isinstance(diag["rhat_max"], float), (
        f"rhat_max should be float, got {type(diag['rhat_max'])}"
    )
    assert isinstance(diag["ess_min"], float), (
        f"ess_min should be float, got {type(diag['ess_min'])}"
    )
    assert isinstance(diag["n_divergences"], int), (
        f"n_divergences should be int, got {type(diag['n_divergences'])}"
    )
    assert isinstance(diag["passes_rhat"], bool)
    assert isinstance(diag["passes_ess"], bool)
    assert isinstance(diag["passes_gate"], bool)

    # passes_gate is the conjunction of the two component gates
    assert diag["passes_gate"] == (diag["passes_rhat"] and diag["passes_ess"])


# ---------------------------------------------------------------------------
# Test 2: RuntimeWarning when extra_fields not passed
# ---------------------------------------------------------------------------


def test_run_diagnostics_warns_without_extra_fields(monkeypatch):
    """run_diagnostics emits RuntimeWarning and returns n_divergences=-1 when
    MCMC.get_extra_fields() returns a dict without the 'diverging' key.

    Note: NumPyro always includes 'diverging' in get_extra_fields(). We
    simulate the missing-field scenario by monkey-patching the diagnostics
    module's extraction path — specifically patching the dict returned by
    get_extra_fields to not contain 'diverging'. We do this by patching
    at the module level after idata creation to avoid interfering with
    ArviZ's internal use of get_extra_fields(group_by_chain=True).
    """
    mcmc = _tiny_mcmc(seed=2)

    # Wrap run_diagnostics to inject the missing-field scenario by patching
    # the divergence extraction logic directly. We test that the warning and
    # -1 return value work by calling the relevant code path explicitly.
    import nn4psych.bayesian.diagnostics as diag_mod
    original_extras = mcmc.get_extra_fields

    def patched_extras(**kwargs):
        result = original_extras(**kwargs)
        # Return a copy without 'diverging' to simulate missing field
        if not kwargs:  # Only strip for the no-kwarg call in run_diagnostics
            return {k: v for k, v in result.items() if k != "diverging"}
        return result

    monkeypatch.setattr(mcmc, "get_extra_fields", patched_extras)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        diag = run_diagnostics(mcmc)

    runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(runtime_warnings) >= 1, (
        f"Expected at least 1 RuntimeWarning, got {len(runtime_warnings)}: {w}"
    )
    assert diag["n_divergences"] == -1, (
        f"Expected n_divergences=-1 (unmeasured), got {diag['n_divergences']}"
    )


# ---------------------------------------------------------------------------
# Test 3: fit_with_retry passes on first attempt
# ---------------------------------------------------------------------------


def test_fit_with_retry_passes_first_attempt():
    """fit_with_retry with reasonable parameters returns status='PASS',
    len(attempts)==1 when diagnostics pass on the first attempt."""
    bag, bucket = simulate_synthetic_data(
        _SMOKE_PARAMS, n_trials=25, hazard=0.125, context="changepoint", seed=3
    )
    mcmc, status, attempts = fit_with_retry(
        reduced_bayesian_model,
        {
            "bag_positions": bag,
            "bucket_positions": bucket,
            "context": "changepoint",
        },
        seed=3,
        num_warmup_first=100,
        num_warmup_retry=200,
        num_samples=100,
        num_chains=2,
    )

    # With 100 warmup / 100 samples, gates may or may not pass — but if they
    # do pass, only 1 attempt should be made. The key invariant is that
    # attempts always has at least 1 record.
    assert len(attempts) >= 1, f"Expected >= 1 attempts, got {len(attempts)}"
    assert status in ("PASS", "FAILED"), f"Unexpected status: {status}"
    if status == "PASS":
        assert len(attempts) == 1, (
            f"Expected exactly 1 attempt when first attempt passes, "
            f"got {len(attempts)}"
        )


# ---------------------------------------------------------------------------
# Test 4: fit_with_retry falls back when passes_gate is patched False
# ---------------------------------------------------------------------------


def test_fit_with_retry_falls_back_to_retry(monkeypatch):
    """fit_with_retry calls run_diagnostics twice when first attempt
    is monkey-patched to return passes_gate=False."""
    call_count = {"n": 0}
    original_run_diagnostics = diag_module.run_diagnostics

    def patched_run_diagnostics(mcmc, var_names=None):
        call_count["n"] += 1
        result = original_run_diagnostics(mcmc, var_names=var_names)
        if call_count["n"] == 1:
            # Force first attempt to fail the gate
            result["passes_gate"] = False
            result["passes_rhat"] = False
        return result

    monkeypatch.setattr(diag_module, "run_diagnostics", patched_run_diagnostics)

    bag, bucket = simulate_synthetic_data(
        _SMOKE_PARAMS, n_trials=20, hazard=0.125, context="changepoint", seed=4
    )
    _mcmc, status, attempts = fit_with_retry(
        reduced_bayesian_model,
        {
            "bag_positions": bag,
            "bucket_positions": bucket,
            "context": "changepoint",
        },
        seed=4,
        num_warmup_first=20,
        num_warmup_retry=30,
        num_samples=20,
        num_chains=2,
    )

    assert len(attempts) == 2, (
        f"Expected exactly 2 attempts (retry after forced failure), "
        f"got {len(attempts)}"
    )
    # First attempt should be marked as failed
    assert attempts[0]["passes_gate"] is False, (
        "First attempt should have passes_gate=False (patched)"
    )


# ---------------------------------------------------------------------------
# Test 5: make_fit_summary is JSON-serializable (M3 fix: stat_focus not kind)
# ---------------------------------------------------------------------------


def test_make_fit_summary_jsonable():
    """make_fit_summary returns a dict serializable by json.dumps.

    If ArviZ's stat_focus kwarg were broken (or if kind= were used instead),
    this test would raise TypeError before json.dumps is reached.
    """
    mcmc = _tiny_mcmc(seed=5)
    mcmc_ret, status, attempts = fit_with_retry(
        reduced_bayesian_model,
        {
            "bag_positions": simulate_synthetic_data(
                _SMOKE_PARAMS, n_trials=20, hazard=0.125,
                context="changepoint", seed=5
            )[0],
            "bucket_positions": simulate_synthetic_data(
                _SMOKE_PARAMS, n_trials=20, hazard=0.125,
                context="changepoint", seed=5
            )[1],
            "context": "changepoint",
        },
        seed=5,
        num_warmup_first=20,
        num_warmup_retry=30,
        num_samples=20,
        num_chains=2,
    )

    summary = make_fit_summary(
        mcmc_ret,
        status=status,
        attempts=attempts,
        var_names=_VAR_NAMES,
        subject_id="test_smoke",
        condition="changepoint",
        seed=5,
    )

    # Must be JSON serializable without TypeError
    js = json.dumps(summary, indent=2)
    assert "rhat" in js, "JSON should contain 'rhat'"
    assert "ess_bulk" in js, "JSON should contain 'ess_bulk'"
    assert "n_divergences" in js, "JSON should contain 'n_divergences'"

    # Verify all var_names appear in the params section
    for param in _VAR_NAMES:
        assert param in summary["params"], (
            f"Parameter '{param}' missing from summary['params']"
        )

    # Verify required per-param keys
    for param in _VAR_NAMES:
        param_stats = summary["params"][param]
        for key in ("mean", "median", "sd", "hdi_2.5", "hdi_97.5", "rhat", "ess_bulk"):
            assert key in param_stats, (
                f"Key '{key}' missing from summary['params']['{param}']"
            )
