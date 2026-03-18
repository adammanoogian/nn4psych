"""
PyEM-based Bayesian Normative Model Implementation

This module implements the Bayesian normative model for the helicopter-bag task
using the PyEM framework for parameter estimation.

The model computes optimal learning rates based on changepoint detection and
uncertainty estimation, following the mathematical framework described in
Loosen et al. (2023) and McGuire et al. (2014).

Reference:
    Loosen, A. M., Skvortsova, V., & Hauser, T. U. (2023).
    pyEM: A Python package for EM-based Bayesian estimation of hierarchical cognitive models.
    Behavior Research Methods.
    https://link.springer.com/article/10.3758/s13428-024-02427-y

    McGuire, J. T., Nassar, M. R., Gold, J. I., & Kable, J. W. (2014).
    Functionally dissociable influences on learning rate in a dynamic environment.
    Neuron, 84(4), 870-881.
"""

import numpy as np
from scipy.special import expit
from scipy import stats
import sys


def calc_fval(negll, params, prior=None, output='npl'):
    """
    Calculate function value for optimization.

    Simple replacement for pyEM.math.calc_fval to avoid external dependency.
    Combines negative log-likelihood with optional prior for MAP estimation.

    Parameters
    ----------
    negll : float
        Negative log-likelihood of the data given parameters
    params : array-like
        Parameter values (not used if prior is None)
    prior : callable or None
        Prior function that takes params and returns negative log prior.
        If None, returns MLE objective (no prior).
    output : str
        'nll' for negative log-likelihood only (maximum likelihood)
        'npl' for negative posterior likelihood (maximum a posteriori)

    Returns
    -------
    float
        Objective function value to minimize

    Notes
    -----
    When prior is provided, this implements MAP estimation:
        argmin_θ [ -log P(D|θ) - log P(θ) ]
    """
    if output == 'nll':
        return negll
    elif output == 'npl':
        if prior is not None:
            return negll + prior(params)
        else:
            return negll
    else:
        raise ValueError(f"Unknown output type: {output}")


def norm2beta(x, max_val=5):
    """
    Transform normalized parameter to [0, max_val] using sigmoid.

    Parameters
    ----------
    x : float or np.ndarray
        Value(s) in normalized space (-inf, inf)
    max_val : float
        Maximum value of output range (default: 5)

    Returns
    -------
    float or np.ndarray
        Value(s) in range [0, max_val]

    Notes
    -----
    Uses sigmoid transformation:
        β(x) = max_val / (1 + exp(-x))

    At x=0, output is max_val/2.
    """
    return max_val / (1 + np.exp(-x))


def norm2alpha(x):
    """
    Transform normalized parameter to [0, 1] using logistic function.

    Parameters
    ----------
    x : float or np.ndarray
        Value(s) in normalized space (-inf, inf)

    Returns
    -------
    float or np.ndarray
        Value(s) in range [0, 1]

    Notes
    -----
    Uses logistic (sigmoid) function:
        α(x) = 1 / (1 + exp(-x))

    At x=0, output is 0.5.
    This is equivalent to scipy.special.expit(x).
    """
    return expit(x)

def fit(params, bucket_positions, bag_positions, context, prior=None, output='npl'):
    """
    Fit Bayesian normative model to behavioral data.

    Implements the normative model described in Loosen et al. (2023) Supplement.
    The model computes optimal learning rates based on changepoint detection (Ω)
    and relative uncertainty (τ), then evaluates likelihood of observed updates.

    Parameters
    ----------
    params : array-like, shape (5,)
        Model parameters in normalized space:
        [0] H - Hazard rate (transformed to [0,1])
        [1] LW - Likelihood weight (transformed to [0,1])
        [2] UU - Uncertainty underestimation (transformed to [0,1])
        [3] sigma_motor - Motor noise (transformed to [0,5])
        [4] sigma_LR - LR variance slope (transformed to [0,5])
    bucket_positions : np.ndarray, shape (n_trials,)
        Observed bucket positions (agent's actions)
    bag_positions : np.ndarray, shape (n_trials,)
        Observed bag positions (outcomes)
    context : str
        Task context: 'changepoint' or 'oddball'
        Determines learning rate computation (Eq. 2 vs Eq. 3)
    prior : callable, optional
        Prior function returning negative log prior probability.
        If None, maximum likelihood estimation.
    output : str
        Return type:
        'npl' - Negative posterior likelihood (for MAP estimation)
        'nll' - Negative log-likelihood (for MLE estimation)
        'all' - Full dictionary of model outputs

    Returns
    -------
    float or dict
        If output='npl' or 'nll': Returns objective value to minimize
        If output='all': Returns dict with keys:
            - params: [H, LW, UU, sigma_motor, sigma_LR]
            - bucket_positions: Input bucket positions
            - bag_positions: Input bag positions
            - context: Task context
            - pred_bucket_placement: Model predictions
            - learning_rate: Learning rate α_t (Eq. 2 or 3)
            - pred_error: Prediction errors δ_t
            - omega: Changepoint probability Ω_t (Eq. 4)
            - tau: Relative uncertainty τ_t (Eq. 5)
            - U_val: Uniform likelihood values
            - N_val: Normal likelihood values
            - bucket_update: Observed updates
            - normative_update: Normative updates (Eq. 1)
            - L_normative_update: Update likelihoods (Eq. 6)
            - negll: Negative log-likelihood
            - BIC: Bayesian Information Criterion

    References
    ----------
    Loosen et al. (2023) Supplement:
    https://link.springer.com/article/10.3758/s13428-024-02427-y#Sec43

    Equations referenced:
    - Eq. 1: Normative update = α_t × δ_t
    - Eq. 2: Learning rate (CP) = Ω_t + τ_t - (Ω_t × τ_t)
    - Eq. 3: Learning rate (OB) = τ_t - (Ω_t × τ_t)
    - Eq. 4: Changepoint probability Ω_t
    - Eq. 5: Relative uncertainty τ_t
    - Eq. 6: Update likelihood
    - Eq. 7: Update variance
    """
    nparams = len(params)

    # Transform parameters from normalized to constrained space
    H = norm2alpha(params[0])           # Hazard rate [0,1]: prior frequency of extreme events
    LW = norm2alpha(params[1])          # Likelihood weight [0,1]: extremeness sensitivity
    UU = norm2alpha(params[2])          # Uncertainty underestimation [0,1]: inappropriate uncertainty reduction
    sigma_motor = norm2beta(params[3])  # Motor noise [0,5]: base update variance
    sigma_LR = norm2beta(params[4])     # LR variance slope [0,5]: variance scaling with update size

    # Task constant: standard deviation of bag placement around helicopter
    sigma_N = 20  # N(bag | helicopter, σ=20)    
    
    # make sure params are in range
    all_bounds = [0, 1]
    for p in [H, LW, UU,]:
        if (p < all_bounds[0]) or (p > all_bounds[1]):
            return 1e7

    ntrials = bucket_positions.shape[0]
    pred_error = bag_positions - bucket_positions  # δ_t = bag - bucket

    # Initialize output arrays
    learning_rate          = np.zeros((ntrials,))     # α_t (Eq. 2 or 3)
    omega                  = np.zeros((ntrials,))     # Ω_t (Eq. 4)
    tau                    = np.zeros((ntrials+1,))   # τ_t (Eq. 5)
    U_val                  = np.zeros((ntrials,))     # Uniform likelihood component
    N_val                  = np.zeros((ntrials,))     # Normal likelihood component
    pred_bucket_placement  = np.zeros((ntrials,))     # Model's predicted bucket position
    bucket_update          = np.zeros((ntrials,))     # Observed bucket update
    normative_update       = np.zeros((ntrials,))     # Normative update (Eq. 1)
    L_normative_update     = np.zeros((ntrials,))     # Update likelihood (Eq. 6)
    negll                  = np.zeros((ntrials,))     # Trial-wise negative log-likelihood

    # Initialize relative uncertainty
    tau_0 = 0.5 / UU  # Initial uncertainty (somewhat uncertain)

    # Trial-by-trial computation
    for t in range(ntrials):
        # ===================================================================
        # EQUATION 4: Changepoint Probability (Ω_t)
        # ===================================================================
        # Ω_t = (H × U(δ)^LW) / (H × U(δ)^LW + (1-H) × N(δ)^LW)

        # Uniform component: extreme outcomes are equally likely anywhere
        U_val[t] = stats.uniform.pdf(pred_error[t], 0, 300) ** LW

        # Normal component: outcomes near current belief are more likely
        if t == 0:
            sigma_t = sigma_N / tau_0
        else:
            sigma_t = sigma_N / tau[t]
        N_val[t] = stats.norm.pdf(pred_error[t], 0, sigma_t) ** LW

        # Compute changepoint probability (Eq. 4)
        omega[t] = (H * U_val[t]) / (H * U_val[t] + (1 - H) * N_val[t])

        # ===================================================================
        # EQUATION 5: Relative Uncertainty (τ_t)
        # ===================================================================
        # Precision-weighted integration of old and new information
        # τ_t = [precision-weighted numerator] / [total precision + σ_N]

        if t == 0:
            # First trial: use initial uncertainty
            numerator = ((omega[t] * sigma_N) +
                        ((1 - omega[t]) * sigma_t * tau_0) +
                        (omega[t] * (1 - omega[t]) * (pred_error[t] * (1 - tau_0))**2))
            denominator = numerator + sigma_N
            this_tau = numerator / denominator
        else:
            # Subsequent trials: use previous uncertainty
            numerator = ((omega[t] * sigma_N) +
                        ((1 - omega[t]) * sigma_t * tau[t]) +
                        (omega[t] * (1 - omega[t]) * (pred_error[t] * (1 - tau[t]))**2))
            denominator = numerator + sigma_N
            this_tau = numerator / denominator

        # Apply uncertainty underestimation
        tau[t+1] = this_tau / UU

        # ===================================================================
        # EQUATIONS 2 & 3: Learning Rate (α_t)
        # ===================================================================
        if context == 'changepoint':
            # Eq. 2: α_t = Ω_t + τ_t - (Ω_t × τ_t)
            # Higher when either changepoint detected OR uncertain
            learning_rate[t] = omega[t] + tau[t] - (omega[t] * tau[t])
        elif context == 'oddball':
            # Eq. 3: α_t = τ_t - (Ω_t × τ_t)
            # Only increases with uncertainty, decreases with changepoint detection
            learning_rate[t] = tau[t] - (omega[t] * tau[t])

        # ===================================================================
        # EQUATION 1: Normative Update
        # ===================================================================
        # update_t = α_t × δ_t
        normative_update[t] = learning_rate[t] * pred_error[t]

        # ===================================================================
        # EQUATION 7: Update Variance
        # ===================================================================
        # σ_update = σ_motor + |normative_update| × σ_LR
        sigma_update = sigma_motor + abs(normative_update[t]) * sigma_LR

        # ===================================================================
        # EQUATION 6: Update Likelihood
        # ===================================================================
        # L(observed_update | normative_update, σ_update)
        bucket_update[t] = bucket_positions[t] - bucket_positions[t-1]
        L_normative_update[t] = stats.norm.pdf(
            bucket_update[t],
            loc=normative_update[t],
            scale=sigma_update
        )

        # Model's prediction for where bucket should be placed
        pred_bucket_placement[t] = bucket_positions[t-1] + normative_update[t]

        # Negative log-likelihood for this trial
        negll[t] = -np.log(L_normative_update[t] + 1e-10)  # Add small constant for stability
    
    sum_negll = np.nansum(negll)
    # CALCULATE NEGATIVE POSTERIOR LIKELIHOOD FROM NEGLL AND PRIOR (OR NEGLL)
    if (output == 'npl') or (output == 'nll'):
        fval = calc_fval(sum_negll, params, prior=prior, output=output)
        return fval
    
    elif output == 'all':
        subj_dict = {'params'               : [H, LW, UU, sigma_motor, sigma_LR],
                     'bucket_positions'     :bucket_positions, 
                     'bag_positions'        :bag_positions, 
                     'context'              :context,
                     'pred_bucket_placement':pred_bucket_placement,
                     'learning_rate'        :learning_rate,
                     'pred_error'           :pred_error,
                     'omega'                :omega,
                     'tau'                  :tau,
                     'U_val'                :U_val,
                     'N_val'                :N_val,
                     'bucket_update'        :bucket_update,
                     'normative_update'     :normative_update,
                     'L_normative_update'   :L_normative_update,
                     'negll'                :sum_negll,
                     'BIC'                  : nparams * np.log(ntrials) + 2*sum_negll}
        return subj_dict

# Example usage (commented out to avoid running on import)
# To use this example:
# 1. Uncomment the code below
# 2. Update file_dir to point to your data location
# 3. Run: python -m bayesian.pyem_models

# if __name__ == "__main__":
#     import os
#     import pickle
#
#     # Pull data from get_behavior.py
#     # NOTE: Update this path to your actual data location
#     file_dir = "data/rnn_behav/model_params_101000/"
#
#     # Load file_dir/gamma_cp_list.pkl
#     with open(os.path.join(file_dir, 'gamma_cp_list.pkl'), 'rb') as f:
#         cp_array = pickle.load(f)
#
#     # Load file_dir/gamma_ob_list.pkl
#     with open(os.path.join(file_dir, 'gamma_ob_list.pkl'), 'rb') as f:
#         ob_array = pickle.load(f)
#
#     # Load model_list_ob
#     with open(os.path.join(file_dir, 'gamma_dict.pkl'), 'rb') as f:
#         model_list = pickle.load(f)
#
#     # Each array has: [trials, bucket_positions, bag_positions, helicopter_positions, hazard_triggers]
#     agent_list_cp = [[x[1], x[2], 'changepoint'] for x in cp_array]
#     agent_list_ob = [[x[1], x[2], 'oddball'] for x in ob_array]
#
#     # outfit_cp = EMfit(agent_list_cp, fit, ['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR'], mstep_maxit=20, convergence_custom='relative_npl', verbose=2)
#     # outfit_ob = EMfit(agent_list_ob, fit, ['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR', 'sigma_LR'], mstep_maxit=20, convergence_custom='relative_npl', verbose=2)