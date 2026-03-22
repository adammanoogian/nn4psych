"""
Model Comparison Tools for Bayesian Normative Models

This module provides functions for comparing different Bayesian models using
information criteria (BIC, AIC) and cross-validation approaches.

Reference:
    Loosen et al. (2023) - https://link.springer.com/article/10.3758/s13428-024-02427-y
    McGuire et al. (2014) - Functionally Dissociable Influences on Learning Rate
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from bayesian.pyem_models import fit


def calculate_bic(negll: float, n_params: int, n_trials: int) -> float:
    """
    Calculate Bayesian Information Criterion.

    BIC = k * ln(n) + 2 * NegLL

    Lower BIC indicates better model fit with parsimony penalty.

    Parameters
    ----------
    negll : float
        Negative log-likelihood of the model
    n_params : int
        Number of free parameters in the model
    n_trials : int
        Number of data points (trials)

    Returns
    -------
    float
        BIC value (lower is better)
    """
    return n_params * np.log(n_trials) + 2 * negll


def calculate_aic(negll: float, n_params: int) -> float:
    """
    Calculate Akaike Information Criterion.

    AIC = 2k + 2 * NegLL

    Lower AIC indicates better model fit with parsimony penalty.

    Parameters
    ----------
    negll : float
        Negative log-likelihood of the model
    n_params : int
        Number of free parameters in the model

    Returns
    -------
    float
        AIC value (lower is better)
    """
    return 2 * n_params + 2 * negll


def calculate_aicc(negll: float, n_params: int, n_trials: int) -> float:
    """
    Calculate corrected AIC (AICc) for small sample sizes.

    AICc = AIC + (2k^2 + 2k) / (n - k - 1)

    Use when n/k < 40 (sample size to parameters ratio is small).

    Parameters
    ----------
    negll : float
        Negative log-likelihood of the model
    n_params : int
        Number of free parameters in the model
    n_trials : int
        Number of data points (trials)

    Returns
    -------
    float
        AICc value (lower is better)
    """
    aic = calculate_aic(negll, n_params)
    correction = (2 * n_params**2 + 2 * n_params) / (n_trials - n_params - 1)
    return aic + correction


def compare_contexts(
    bucket_positions: np.ndarray,
    bag_positions: np.ndarray,
    fitted_params_cp: np.ndarray,
    fitted_params_ob: np.ndarray,
    prior: Optional[callable] = None,
) -> Dict[str, float]:
    """
    Compare changepoint vs oddball model fits for the same data.

    Parameters
    ----------
    bucket_positions : np.ndarray
        Observed bucket positions
    bag_positions : np.ndarray
        Observed bag positions
    fitted_params_cp : np.ndarray
        Fitted parameters for changepoint model
    fitted_params_ob : np.ndarray
        Fitted parameters for oddball model
    prior : callable, optional
        Prior function for parameter regularization

    Returns
    -------
    dict
        Dictionary containing comparison metrics:
        - cp_negll: Changepoint negative log-likelihood
        - ob_negll: Oddball negative log-likelihood
        - cp_bic: Changepoint BIC
        - ob_bic: Oddball BIC
        - cp_aic: Changepoint AIC
        - ob_aic: Oddball AIC
        - best_model: 'changepoint' or 'oddball'
        - bic_difference: BIC(oddball) - BIC(changepoint)
        - evidence_ratio: Relative likelihood of best model
    """
    n_trials = len(bucket_positions)
    n_params = len(fitted_params_cp)

    # Get negative log-likelihoods
    cp_negll = fit(
        fitted_params_cp,
        bucket_positions,
        bag_positions,
        context='changepoint',
        prior=prior,
        output='nll'
    )

    ob_negll = fit(
        fitted_params_ob,
        bucket_positions,
        bag_positions,
        context='oddball',
        prior=prior,
        output='nll'
    )

    # Calculate information criteria
    cp_bic = calculate_bic(cp_negll, n_params, n_trials)
    ob_bic = calculate_bic(ob_negll, n_params, n_trials)

    cp_aic = calculate_aic(cp_negll, n_params)
    ob_aic = calculate_aic(ob_negll, n_params)

    # Determine best model
    bic_diff = ob_bic - cp_bic  # Positive means CP is better
    best_model = 'changepoint' if bic_diff > 0 else 'oddball'

    # Calculate evidence ratio (Kass & Raftery, 1995)
    # exp(-0.5 * BIC_diff) gives Bayes factor approximation
    evidence_ratio = np.exp(-0.5 * abs(bic_diff))

    return {
        'cp_negll': cp_negll,
        'ob_negll': ob_negll,
        'cp_bic': cp_bic,
        'ob_bic': ob_bic,
        'cp_aic': cp_aic,
        'ob_aic': ob_aic,
        'best_model': best_model,
        'bic_difference': bic_diff,
        'evidence_ratio': evidence_ratio,
    }


def cross_validate_k_fold(
    bucket_positions: np.ndarray,
    bag_positions: np.ndarray,
    context: str,
    initial_params: np.ndarray,
    k: int = 5,
    prior: Optional[callable] = None,
) -> Dict[str, any]:
    """
    Perform k-fold cross-validation for model evaluation.

    Parameters
    ----------
    bucket_positions : np.ndarray
        Observed bucket positions
    bag_positions : np.ndarray
        Observed bag positions
    context : str
        'changepoint' or 'oddball'
    initial_params : np.ndarray
        Initial parameter values for optimization
    k : int
        Number of folds (default: 5)
    prior : callable, optional
        Prior function for regularization

    Returns
    -------
    dict
        Cross-validation results:
        - cv_negll_mean: Mean negative log-likelihood across folds
        - cv_negll_std: Standard deviation of negative log-likelihood
        - fold_negll: Array of negative log-likelihoods for each fold
        - fold_params: List of fitted parameters for each fold
    """
    from scipy.optimize import minimize

    n_trials = len(bucket_positions)
    fold_size = n_trials // k

    fold_negll = np.zeros(k)
    fold_params = []

    for i in range(k):
        # Create train/test split
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k - 1 else n_trials

        # Test indices
        test_idx = np.arange(test_start, test_end)

        # Train indices (everything except test)
        train_idx = np.concatenate([
            np.arange(0, test_start),
            np.arange(test_end, n_trials)
        ])

        # Train model
        result = minimize(
            fit,
            initial_params,
            args=(
                bucket_positions[train_idx],
                bag_positions[train_idx],
                context,
                prior,
                'nll'
            ),
            method='Nelder-Mead',
            options={'maxiter': 5000}
        )

        fitted_params = result.x
        fold_params.append(fitted_params)

        # Test on held-out data
        test_negll = fit(
            fitted_params,
            bucket_positions[test_idx],
            bag_positions[test_idx],
            context,
            prior=None,  # Don't apply prior to test likelihood
            output='nll'
        )

        fold_negll[i] = test_negll

    return {
        'cv_negll_mean': fold_negll.mean(),
        'cv_negll_std': fold_negll.std(),
        'fold_negll': fold_negll,
        'fold_params': fold_params,
    }


def likelihood_ratio_test(
    negll_full: float,
    negll_reduced: float,
    df_diff: int,
    alpha: float = 0.05,
) -> Dict[str, any]:
    """
    Perform likelihood ratio test between nested models.

    Tests whether the full model (more parameters) provides significantly
    better fit than the reduced model (fewer parameters).

    Parameters
    ----------
    negll_full : float
        Negative log-likelihood of full model
    negll_reduced : float
        Negative log-likelihood of reduced model
    df_diff : int
        Difference in degrees of freedom (number of parameters)
    alpha : float
        Significance level (default: 0.05)

    Returns
    -------
    dict
        Test results:
        - lr_statistic: Likelihood ratio test statistic
        - p_value: P-value from chi-square distribution
        - significant: Whether difference is significant at alpha level
        - dof: Degrees of freedom
    """
    from scipy.stats import chi2

    # LR statistic = 2 * (LL_full - LL_reduced) = 2 * (NegLL_reduced - NegLL_full)
    lr_stat = 2 * (negll_reduced - negll_full)

    # P-value from chi-square distribution
    p_value = 1 - chi2.cdf(lr_stat, df_diff)

    return {
        'lr_statistic': lr_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'dof': df_diff,
    }


def model_weights_aic(aic_values: np.ndarray) -> np.ndarray:
    """
    Calculate Akaike weights for model averaging.

    Weights indicate relative likelihood of each model being the best model.

    Parameters
    ----------
    aic_values : np.ndarray
        AIC values for different models

    Returns
    -------
    np.ndarray
        Akaike weights (sum to 1)

    Reference
    ---------
    Burnham & Anderson (2002) - Model Selection and Multimodel Inference
    """
    # Calculate delta AIC (difference from best model)
    min_aic = np.min(aic_values)
    delta_aic = aic_values - min_aic

    # Calculate relative likelihood
    rel_likelihood = np.exp(-0.5 * delta_aic)

    # Normalize to get weights
    weights = rel_likelihood / rel_likelihood.sum()

    return weights


def print_comparison_summary(comparison: Dict[str, float]) -> None:
    """
    Print formatted summary of model comparison results.

    Parameters
    ----------
    comparison : dict
        Results from compare_contexts()
    """
    print("="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print("\nNegative Log-Likelihood:")
    print(f"  Changepoint:  {comparison['cp_negll']:8.2f}")
    print(f"  Oddball:      {comparison['ob_negll']:8.2f}")
    print("\nBayesian Information Criterion (BIC):")
    print(f"  Changepoint:  {comparison['cp_bic']:8.2f}")
    print(f"  Oddball:      {comparison['ob_bic']:8.2f}")
    print(f"  Δ BIC (OB-CP): {comparison['bic_difference']:7.2f}")
    print("\nAkaike Information Criterion (AIC):")
    print(f"  Changepoint:  {comparison['cp_aic']:8.2f}")
    print(f"  Oddball:      {comparison['ob_aic']:8.2f}")
    print("\nBest Model: " + comparison['best_model'].upper())
    print(f"Evidence Ratio: {comparison['evidence_ratio']:.4f}")

    # Interpret evidence strength (Kass & Raftery, 1995)
    bic_diff = abs(comparison['bic_difference'])
    if bic_diff < 2:
        strength = "Weak evidence"
    elif bic_diff < 6:
        strength = "Positive evidence"
    elif bic_diff < 10:
        strength = "Strong evidence"
    else:
        strength = "Very strong evidence"

    print(f"Evidence Strength: {strength}")
    print("="*60)
