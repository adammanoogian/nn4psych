"""
NumPyro-based Bayesian Normative Model Implementation

This module implements the Bayesian normative model using NumPyro for full
Bayesian inference via MCMC (No-U-Turn Sampler). Built on JAX for fast
computation with automatic differentiation and JIT compilation.

Advantages over PyEM:
- Full posterior distributions (not just point estimates)
- Uncertainty quantification for all parameters
- Posterior predictive checks
- GPU acceleration support

Reference:
    Loosen et al. (2023) - https://link.springer.com/article/10.3758/s13428-024-02427-y
    McGuire et al. (2014) - Functionally dissociable influences on learning rate
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jax_norm
from jax.scipy.stats import uniform as jax_uniform
from jax.scipy.special import expit as jax_expit
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import hpdi
import arviz as az
from typing import Dict, Optional, Tuple


def sigmoid_transform(x: jnp.ndarray, max_val: float = 5.0) -> jnp.ndarray:
    """
    Transform unbounded parameter to [0, max_val] using sigmoid.

    Parameters
    ----------
    x : jnp.ndarray
        Unbounded parameter value
    max_val : float
        Maximum value (default: 5.0)

    Returns
    -------
    jnp.ndarray
        Transformed value in [0, max_val]
    """
    return max_val / (1 + jnp.exp(-x))


def logistic_transform(x: jnp.ndarray) -> jnp.ndarray:
    """
    Transform unbounded parameter to [0, 1] using logistic function.

    Parameters
    ----------
    x : jnp.ndarray
        Unbounded parameter value

    Returns
    -------
    jnp.ndarray
        Transformed value in [0, 1]
    """
    return jax_expit(x)


def compute_normative_model(
    params: Dict[str, jnp.ndarray],
    pred_errors: jnp.ndarray,
    context: str,
    sigma_N: float = 20.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute normative model outputs given parameters and prediction errors.

    This is the core forward model implementing Equations 1-5 from Loosen et al. (2023).

    Parameters
    ----------
    params : dict
        Dictionary with keys: 'H', 'LW', 'UU', 'sigma_motor', 'sigma_LR'
    pred_errors : jnp.ndarray
        Prediction errors (bag - bucket) for each trial
    context : str
        'changepoint' or 'oddball'
    sigma_N : float
        Standard deviation of bag placement (default: 20.0)

    Returns
    -------
    tuple
        (learning_rates, normative_updates, omegas, taus)
        All arrays have shape (n_trials,) except taus which has (n_trials+1,)
    """
    H = params['H']
    LW = params['LW']
    UU = params['UU']

    n_trials = len(pred_errors)

    # Initialize arrays
    learning_rate = jnp.zeros(n_trials)
    omega = jnp.zeros(n_trials)
    tau = jnp.zeros(n_trials + 1)
    normative_update = jnp.zeros(n_trials)

    # Initial uncertainty
    tau_0 = 0.5 / UU
    tau = tau.at[0].set(tau_0)

    def step_fn(carry, t):
        """Single trial update using JAX scan for efficiency."""
        tau_prev = carry

        # Current prediction error
        delta = pred_errors[t]

        # ===================================================================
        # EQUATION 4: Changepoint Probability (Ω_t)
        # ===================================================================
        # Uniform component (extreme outcomes)
        U_val = jax_uniform.pdf(delta, loc=0, scale=300) ** LW

        # Normal component (expected outcomes)
        sigma_t = sigma_N / tau_prev
        N_val = jax_norm.pdf(delta, loc=0, scale=sigma_t) ** LW

        # Changepoint probability
        omega_t = (H * U_val) / (H * U_val + (1 - H) * N_val + 1e-10)

        # ===================================================================
        # EQUATION 5: Relative Uncertainty (τ_t)
        # ===================================================================
        # Precision-weighted integration
        numerator = ((omega_t * sigma_N) +
                    ((1 - omega_t) * sigma_t * tau_prev) +
                    (omega_t * (1 - omega_t) * (delta * (1 - tau_prev))**2))
        denominator = numerator + sigma_N
        this_tau = numerator / denominator

        # Apply uncertainty underestimation
        tau_next = this_tau / UU

        # ===================================================================
        # EQUATIONS 2 & 3: Learning Rate (α_t)
        # ===================================================================
        if context == 'changepoint':
            # Eq. 2: α_t = Ω_t + τ_t - (Ω_t × τ_t)
            lr_t = omega_t + tau_prev - (omega_t * tau_prev)
        else:  # oddball
            # Eq. 3: α_t = τ_t - (Ω_t × τ_t)
            lr_t = tau_prev - (omega_t * tau_prev)

        # ===================================================================
        # EQUATION 1: Normative Update
        # ===================================================================
        norm_update_t = lr_t * delta

        return tau_next, (lr_t, norm_update_t, omega_t, tau_next)

    # Run scan over all trials
    _, (learning_rate, normative_update, omega, tau_scan) = jax.lax.scan(
        step_fn, tau_0, jnp.arange(n_trials)
    )

    # Combine initial tau with scanned taus
    tau = jnp.concatenate([jnp.array([tau_0]), tau_scan])

    return learning_rate, normative_update, omega, tau


def normative_model(
    bucket_positions: Optional[jnp.ndarray] = None,
    bag_positions: Optional[jnp.ndarray] = None,
    context: str = 'changepoint',
    prior_scale: float = 1.0,
) -> None:
    """
    NumPyro probabilistic model for Bayesian normative learning.

    This defines the full generative model with priors and likelihood.
    Use with MCMC for posterior inference.

    Parameters
    ----------
    bucket_positions : jnp.ndarray, optional
        Observed bucket positions (agent actions)
    bag_positions : jnp.ndarray, optional
        Observed bag positions (outcomes)
    context : str
        'changepoint' or 'oddball'
    prior_scale : float
        Scale for prior distributions (default: 1.0)
        Higher values = more diffuse priors

    Notes
    -----
    This is a NumPyro effect handler that samples from priors and
    conditions on observed data. Call with MCMC to get posterior samples.

    Priors:
    - H ~ Beta(2, 2): Centered at 0.5, weakly informative
    - LW ~ Beta(2, 2): Centered at 0.5, weakly informative
    - UU ~ Beta(2, 2): Centered at 0.5, weakly informative
    - sigma_motor ~ HalfNormal(1.0): Weakly regularizes motor noise
    - sigma_LR ~ HalfNormal(1.0): Weakly regularizes variance slope
    """
    # ===================================================================
    # PRIORS
    # ===================================================================
    # Use weakly informative priors
    # Beta(2, 2) is centered at 0.5 with moderate uncertainty
    H = numpyro.sample('H', dist.Beta(2, 2))
    LW = numpyro.sample('LW', dist.Beta(2, 2))
    UU = numpyro.sample('UU', dist.Beta(2, 2))

    # Half-normal priors for variance parameters
    # Centered at small values, tail toward larger values
    sigma_motor = numpyro.sample('sigma_motor',
                                 dist.HalfNormal(prior_scale))
    sigma_LR = numpyro.sample('sigma_LR',
                              dist.HalfNormal(prior_scale))

    # ===================================================================
    # FORWARD MODEL
    # ===================================================================
    if bucket_positions is not None and bag_positions is not None:
        n_trials = len(bucket_positions)

        # Compute prediction errors
        pred_errors = bag_positions - bucket_positions

        # Package parameters
        params = {
            'H': H,
            'LW': LW,
            'UU': UU,
            'sigma_motor': sigma_motor,
            'sigma_LR': sigma_LR,
        }

        # Compute normative model
        learning_rate, normative_update, omega, tau = compute_normative_model(
            params, pred_errors, context
        )

        # Compute observed bucket updates
        bucket_update = jnp.diff(bucket_positions, prepend=bucket_positions[0])

        # ===================================================================
        # EQUATION 7: Update Variance
        # ===================================================================
        sigma_update = sigma_motor + jnp.abs(normative_update) * sigma_LR

        # ===================================================================
        # EQUATION 6: Update Likelihood
        # ===================================================================
        # Likelihood: observed updates ~ Normal(normative_update, sigma_update)
        with numpyro.plate('trials', n_trials):
            numpyro.sample(
                'bucket_update',
                dist.Normal(normative_update, sigma_update),
                obs=bucket_update
            )

        # Store derived quantities for posterior predictive checks
        numpyro.deterministic('learning_rate', learning_rate)
        numpyro.deterministic('normative_update', normative_update)
        numpyro.deterministic('omega', omega)
        numpyro.deterministic('tau', tau[:-1])  # Remove last element for alignment


def run_mcmc(
    bucket_positions: np.ndarray,
    bag_positions: np.ndarray,
    context: str = 'changepoint',
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    seed: int = 42,
    progress_bar: bool = True,
) -> MCMC:
    """
    Run MCMC sampling for Bayesian normative model.

    Parameters
    ----------
    bucket_positions : np.ndarray
        Observed bucket positions
    bag_positions : np.ndarray
        Observed bag positions
    context : str
        'changepoint' or 'oddball'
    num_warmup : int
        Number of warmup/adaptation steps (default: 1000)
    num_samples : int
        Number of posterior samples per chain (default: 2000)
    num_chains : int
        Number of MCMC chains (default: 4)
    seed : int
        Random seed for reproducibility (default: 42)
    progress_bar : bool
        Whether to show progress bar (default: True)

    Returns
    -------
    MCMC
        NumPyro MCMC object with posterior samples

    Examples
    --------
    >>> mcmc = run_mcmc(bucket_positions, bag_positions, context='changepoint')
    >>> samples = mcmc.get_samples()
    >>> print(f"H posterior mean: {samples['H'].mean():.3f}")
    """
    # Convert to JAX arrays
    bucket_jax = jnp.array(bucket_positions)
    bag_jax = jnp.array(bag_positions)

    # Set up NUTS sampler
    kernel = NUTS(normative_model)

    # Run MCMC
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )

    # Sample from posterior
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(
        rng_key,
        bucket_positions=bucket_jax,
        bag_positions=bag_jax,
        context=context,
    )

    return mcmc


def summarize_posterior(mcmc: MCMC, prob: float = 0.89) -> Dict:
    """
    Summarize posterior samples from MCMC.

    Parameters
    ----------
    mcmc : MCMC
        NumPyro MCMC object after sampling
    prob : float
        Probability mass for credible intervals (default: 0.89)

    Returns
    -------
    dict
        Summary statistics for each parameter:
        - mean: Posterior mean
        - std: Posterior standard deviation
        - median: Posterior median
        - hpdi_low: Lower bound of HPDI
        - hpdi_high: Upper bound of HPDI
    """
    samples = mcmc.get_samples()

    summary = {}
    param_names = ['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR']

    for param in param_names:
        if param in samples:
            param_samples = np.array(samples[param])

            # Compute HPDI (Highest Posterior Density Interval)
            hpdi_interval = hpdi(samples[param], prob=prob)

            summary[param] = {
                'mean': param_samples.mean(),
                'std': param_samples.std(),
                'median': np.median(param_samples),
                'hpdi_low': float(hpdi_interval[0]),
                'hpdi_high': float(hpdi_interval[1]),
            }

    return summary


def posterior_predictive(
    mcmc: MCMC,
    bucket_positions: np.ndarray,
    bag_positions: np.ndarray,
    context: str,
    num_samples: int = 1000,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Generate posterior predictive samples.

    Useful for:
    - Model checking (do predictions match observations?)
    - Uncertainty quantification
    - Generating predictions for new data

    Parameters
    ----------
    mcmc : MCMC
        Fitted MCMC object
    bucket_positions : np.ndarray
        Bucket positions (can be observed or new data)
    bag_positions : np.ndarray
        Bag positions (can be observed or new data)
    context : str
        'changepoint' or 'oddball'
    num_samples : int
        Number of posterior predictive samples (default: 1000)
    seed : int
        Random seed

    Returns
    -------
    dict
        Posterior predictive samples for 'bucket_update' and derived quantities
    """
    bucket_jax = jnp.array(bucket_positions)
    bag_jax = jnp.array(bag_positions)

    # Get posterior samples
    posterior_samples = mcmc.get_samples()

    # Generate predictions
    predictive = Predictive(
        normative_model,
        posterior_samples=posterior_samples,
        num_samples=num_samples,
    )

    rng_key = jax.random.PRNGKey(seed)
    predictions = predictive(
        rng_key,
        bucket_positions=bucket_jax,
        bag_positions=bag_jax,
        context=context,
    )

    # Convert to numpy
    predictions_np = {k: np.array(v) for k, v in predictions.items()}

    return predictions_np


def compute_waic(mcmc: MCMC, bucket_positions: np.ndarray,
                 bag_positions: np.ndarray, context: str) -> Dict[str, float]:
    """
    Compute WAIC (Watanabe-Akaike Information Criterion) for model comparison.

    WAIC is a fully Bayesian information criterion that accounts for
    posterior uncertainty.

    Parameters
    ----------
    mcmc : MCMC
        Fitted MCMC object
    bucket_positions : np.ndarray
        Observed bucket positions
    bag_positions : np.ndarray
        Observed bag positions
    context : str
        'changepoint' or 'oddball'

    Returns
    -------
    dict
        WAIC statistics:
        - waic: WAIC value (lower is better)
        - p_waic: Effective number of parameters
        - se: Standard error of WAIC
    """
    # Get posterior predictive log-likelihood
    posterior_samples = mcmc.get_samples()

    # Convert to ArviZ InferenceData for WAIC calculation
    bucket_jax = jnp.array(bucket_positions)
    bag_jax = jnp.array(bag_positions)

    idata = az.from_numpyro(
        mcmc,
        posterior_predictive=posterior_predictive(
            mcmc, bucket_positions, bag_positions, context
        )
    )

    # Compute WAIC
    waic_result = az.waic(idata)

    return {
        'waic': float(waic_result.waic),
        'p_waic': float(waic_result.p_waic),
        'se': float(waic_result.se),
    }


def get_map_estimate(mcmc: MCMC) -> Dict[str, float]:
    """
    Get Maximum A Posteriori (MAP) point estimate from posterior samples.

    This finds the mode of the posterior distribution for comparison
    with MLE estimates from PyEM.

    Parameters
    ----------
    mcmc : MCMC
        Fitted MCMC object

    Returns
    -------
    dict
        MAP estimates for each parameter
    """
    samples = mcmc.get_samples()

    map_estimates = {}
    param_names = ['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR']

    for param in param_names:
        if param in samples:
            # Use mean as point estimate (for normal posteriors, mean ≈ mode)
            # For better mode estimation, could use kernel density estimation
            map_estimates[param] = float(np.array(samples[param]).mean())

    return map_estimates
