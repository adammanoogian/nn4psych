import numpy as np
from scipy.special import expit
from scipy import stats
import sys
from pyEM.math import calc_fval

def norm2beta(x, max_val=20):
    return max_val / (1 + np.exp(-x))

def norm2alpha(x):
    return expit(x)

def fit(params, bucket_positions, bag_positions, context, prior=None, output='npl'):
    nparams = len(params)

    H = norm2alpha(params[0]) # hazard rate: frequency that the model expected extreme events 
    LW = norm2alpha(params[1]) # likelihood weight: the degree to which extremeness of an outcome factored into identification of change-points/oddballs
    UU = norm2alpha(params[2]) # uncertainty underestimation: degree to which uncertainty is inappropriately reduced on each trial
    sigma_motor = norm2alpha(params[3]) # update variance: the base width of the distribution over possible bucket positions centred on the inferred helicopter location
    sigma_LR = norm2alpha(params[4]) # update variance slope: the degree to which the width of the distribution over bucket positions increases with larger normative updates
    sigma = norm2beta(params[5]) 
    drift_scale = norm2alpha(params[6]) # drift scale: the rate at which the helicopter was assumed to be drifting in the oddball condition
    # TODO: there is also `sigma_persev` and `S` (Equation 8), but ignoring for now
    # that's a separate model so wouldn't worry about that for now

    # make sure params are in range
    all_bounds = [0, 1]
    for p in [H, LW, UU, sigma_motor, sigma_LR, drift_scale]:
        if (p < all_bounds[0]) or (p > all_bounds[1]):
            return 1e7

    ntrials = bucket_positions.shape[0]
    pred_error = bag_positions - bucket_positions

    learning_rate    = np.zeros((ntrials,))
    omega            = np.zeros((ntrials,))
    tau              = np.zeros((ntrials,))
    U_val            = np.zeros((ntrials,))
    N_val            = np.zeros((ntrials,))
    pred_bucket_placement = np.zeros((ntrials,))
    negll  = 0

    for t in range(ntrials):
        # TODO: need to figure this out -- also prob better to change to be in log-space: e.g., `LW * stats.uniform.logpdf(pred_error[t], 0, 300)`
        U_val[t] = stats.uniform.pdf(pred_error[t], 0, 300) ** LW
        N_val[t] = stats.norm.pdf(pred_error[t], 0, sigma) ** LW

        # changepoint probability (𝛺) 
        omega[t] = (H * U_val[t]) / (H * U_val[t] + (1 - H) * N_val[t])
        
        # TODO: need to figure this out...
        # relative uncertainty (𝜏)
        if t == 0:
            if context == 'changepoint':
                tau[t] = 1 / sigma
            elif context == 'oddball':
                tau[t] = (1 / sigma) + drift_scale
        else:
            tau[t] = tau[t-1] / UU
        
        #adam suggestion- take out the trial dependency & also the context dependency (because we had a random walk in CP)
        # sigma[t] = tau[t] + drift_scale
        # tau[t] = tau[t-1] / UU
       
        if context == 'changepoint':
            learning_rate[t] = omega[t] + tau[t] - (omega[t] * tau[t])
        elif context == 'oddball':
            learning_rate[t] = tau[t] - (omega[t] * tau[t])

        normative_update = learning_rate[t] * pred_error[t]
        sigma_update = sigma_motor + normative_update * sigma_LR
        L_normative_update = stats.norm.pdf(bucket_positions[t]-bucket_positions[t-1], loc=normative_update, scale=sigma_update)

        pred_bucket_placement[t] = bucket_positions[t-1] + normative_update

        negll -= np.log(L_normative_update)
            
    # CALCULATE NEGATIVE POSTERIOR LIKELIHOOD FROM NEGLL AND PRIOR (OR NEGLL)
    if (output == 'npl') or (output == 'nll'):
        fval = calc_fval(negll, params, prior=prior, output=output)
        return fval
    
    elif output == 'all':
        subj_dict = {'params'               : [H, LW, UU, sigma_motor, sigma_LR, sigma, drift_scale],
                     'bucket_positions'     :bucket_positions, 
                     'bag_positions'        :bag_positions, 
                     'context'              :context,
                     'pred_bucket_placement':pred_bucket_placement,
                     'learning_rate'        :learning_rate,
                     'omega'                :omega,
                     'tau'                  :tau,
                     'U_val'                :U_val,
                     'N_val'                :N_val,
                     'BIC'                  : nparams * np.log(ntrials) + 2*negll}
        return subj_dict