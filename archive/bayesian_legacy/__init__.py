"""
Bayesian Models for Predictive Inference Analysis

This module contains Bayesian normative models for the predictive inference task.
These models can be fit to behavioral data to understand optimal inference strategies.

Two implementations are available:
1. PyEM (recommended): Fast, optimized using scipy and pyEM framework
2. PyMC: Full Bayesian inference with MCMC sampling (experimental)

Quick Start
-----------
For PyEM (recommended):
    from bayesian import fit_bayesian_model, norm2alpha, norm2beta

    results = fit_bayesian_model(bucket_positions, bag_positions, context='changepoint')

For PyMC (experimental):
    from bayesian import BayesianModel

    model = BayesianModel(states, model_type='changepoint')
    model.run_mle()
"""

# PyEM implementation (recommended for point estimates)
from bayesian.pyem_models import fit, norm2alpha, norm2beta

# NumPyro implementation (recommended for full Bayesian inference)
from bayesian.numpyro_models import (
    run_mcmc,
    summarize_posterior,
    posterior_predictive,
    compute_waic,
    get_map_estimate,
)

# PyMC implementation (experimental, has known bugs)
from bayesian.bayesian_models import BayesianModel

# Convenience aliases
fit_bayesian_model = fit  # More intuitive name for main fitting function

__all__ = [
    # PyEM functions (MLE/MAP estimation)
    "fit",
    "fit_bayesian_model",  # Alias
    "norm2alpha",
    "norm2beta",
    # NumPyro functions (full Bayesian MCMC)
    "run_mcmc",
    "summarize_posterior",
    "posterior_predictive",
    "compute_waic",
    "get_map_estimate",
    # PyMC class (experimental)
    "BayesianModel",
]
