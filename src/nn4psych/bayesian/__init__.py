"""
Bayesian subpackage for nn4psych.

Provides NumPyro-based MCMC inference for normative models.
Enforces JAX CPU-only operation for PyTorch coexistence.
"""
import os

# Must be set before any JAX import — enforces CPU backend
# and prevents pre-allocation that conflicts with PyTorch GPU memory.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from nn4psych.bayesian.numpyro_models import (
    run_mcmc,
    summarize_posterior,
    posterior_predictive,
    compute_waic,
    get_map_estimate,
    compute_normative_model,
)

__all__ = [
    "run_mcmc",
    "summarize_posterior",
    "posterior_predictive",
    "compute_waic",
    "get_map_estimate",
    "compute_normative_model",
]
