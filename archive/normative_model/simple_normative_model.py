#%%
'''
Get statistics of CP and OB task
'''
from tasks import PIE_CP_OB
import copy
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats

# Simulate some observed data for illustrative purposes
seed = 42
np.random.seed(seed)
true_helicopter_position = 0.0
true_variance = 0.1
n_samples = 10
observed_data = np.random.normal(true_helicopter_position, np.sqrt(true_variance), size=n_samples)


def from_posterior(param, samples):
    """Create a new prior distribution from posterior samples using KDE."""
    smin, smax = samples.min().item(), samples.max().item()
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # Extend the domain and use a linear approximation of density
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return pm.Interpolated(param, x, y)

def fit_bayes_model(data, true_heli_data, true_var_data, heli_min, heli_max, iterations=10, batch_size=3):
    rng = np.random.default_rng(seed=seed)
    traces = []
    current_data = []

    for i in range(iterations):
        # Append new samples based on batch size
        new_samples = rng.normal(loc=true_heli_data, scale=np.sqrt(true_var_data), size=batch_size)
        current_data.extend(new_samples)

        # Bayesian model
        with pm.Model() as model:
            if traces:
                # Use the posterior from traces as prior for new model
                heli_position = from_posterior("heli_position", az.extract(traces[-1], var_names="heli_position"))
                bag_sd = from_posterior("bag_sd", az.extract(traces[-1], var_names="bag_sd"))
            else:
                # Initial Uniform and HalfCauchy
                heli_position = pm.Uniform('heli_position', lower=heli_min, upper=heli_max)
                bag_sd = pm.HalfCauchy('bag_sd', beta=1)

            # Likelihood
            likelihood = pm.Normal('likelihood', mu=heli_position, sigma=bag_sd, observed=current_data)

            # Sample from the posterior
            trace = pm.sample(draws=2_000, tune=500, progressbar=False, random_seed=rng)
            traces.append(trace)

    return traces

traces = fit_bayes_model(observed_data, true_helicopter_position, true_variance, -1, 1)

# Plot the final trace
ax = az.plot_trace(traces[-1], compact=True)
plt.gcf().suptitle("Trace for Final Model", fontsize=16)
plt.show()


#%%

epochs = 1
trials = 10
max_time = 300

train_cond = False # show helicopter?
max_displacement = 20 # 1 unit of action to move left or right


for epoch in range(epochs):
    for task_type in ["change-point"]:
        env = PIE_CP_OB(condition=task_type,max_time=max_time, 
                        total_trials=trials, train_cond=train_cond,
                        max_displacement=max_displacement)
        
        for trial in range(trials):
            obs, done = env.reset()
            total_reward = 0

            while not done:
                action = env.action_space.sample()  # For testing, we use random actions
                next_obs, reward, done = env.step(action)
                total_reward += reward

                # print(env.trial, env.time, obs, action, next_obs, np.round(reward,3), done)

                obs = copy.copy(next_obs)

        # states = np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])
        states = env.render()

true_helicopter_pos = states[3]
obs_bag_pos = states[2]



trace = fit_bayes_model(obs_bag_pos,true_helicopter_pos, 20, 0, 300)