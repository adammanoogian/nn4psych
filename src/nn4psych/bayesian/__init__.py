"""Bayesian subpackage for nn4psych.

Phase 4: paper-aligned priors live in ``reduced_bayesian.py``. Legacy
``numpyro_models.py`` placeholder priors are kept for archival reference
but not exported.

Provides NumPyro-based MCMC inference for normative models (Nassar 2010+2021
Reduced Bayesian Observer). Enforces JAX CPU-only operation for PyTorch
coexistence.
"""
import os

# Must be set before any JAX import — enforces CPU backend
# and prevents pre-allocation that conflicts with PyTorch GPU memory.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# 4 virtual CPU devices for 4-chain NUTS; must be set before any jax import
# (RESEARCH.md Pitfall 3).
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from nn4psych.bayesian.reduced_bayesian import (
    reduced_bayesian_model,
    run_mcmc,
    compute_rbo_forward,
    prior_sampler,
    simulate_synthetic_data,
    assert_jax_devices,
    SIGMA_N,
)
from nn4psych.bayesian.diagnostics import (
    run_diagnostics,
    fit_with_retry,
    make_fit_summary,
    to_jsonable,
)

__all__ = [
    "reduced_bayesian_model",
    "run_mcmc",
    "compute_rbo_forward",
    "prior_sampler",
    "simulate_synthetic_data",
    "assert_jax_devices",
    "SIGMA_N",
    "run_diagnostics",
    "fit_with_retry",
    "make_fit_summary",
    "to_jsonable",
]
