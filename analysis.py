
#%%
#load data

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random, vmap

#hardcode
# Example LSTM activations (400 episodes, 50 epochs, 64 hidden units)
activations = np.load(r'C:\Users\aman0087\Documents\Github\nn4psych\data\activity_contextual.npy')
activations = activations.reshape(400 * 50, 64)

history = np.load(r'C:\Users\aman0087\Documents\Github\nn4psych\data\history_contextual.npy')

#%% 
gammaPlot = .5
prediction_errors = []

for t in range(len(history) - 1):
    reward_t = history[t][0]
    reward_t_plus_1 = history[t + 1][0]
    prediction_error = reward_t + gammaPlot * (reward_t_plus_1 - reward_t)
    prediction_errors.append(prediction_error)


window_size = 50
volatility_over_time = np.array([
    np.var(prediction_errors[i:i + window_size]) 
    for i in range(len(prediction_errors) - window_size + 1)
])

#hardfix because last 50 doesn't have a volatility
random_values = np.random.rand(window_size)
volatility_over_time = np.concatenate((volatility_over_time, random_values))

#%% 
#calc proportion of significant hidden units over time
#to-do: take out sig threshold for easier debugging

proportions = []
significance_threshold = .05

def custom_linregress(x, y):
    n = x.size
    x_mean = jnp.mean(x)
    y_mean = jnp.mean(y)
    xy_cov = jnp.mean(x * y) - x_mean * y_mean
    xx_cov = jnp.mean(x * x) - x_mean * x_mean

    slope = xy_cov / xx_cov
    intercept = y_mean - slope * x_mean
    r_value = xy_cov / jnp.sqrt(xx_cov * (jnp.mean(y * y) - y_mean * y_mean))
    p_value = 2 * (1 - jax.scipy.stats.norm.cdf(jnp.abs(r_value) * jnp.sqrt((n - 2) / (1 - r_value ** 2))))
    std_err = jnp.sqrt((jnp.mean((y - (slope * x + intercept)) ** 2)) / (n - 2))

    return slope, intercept, r_value, p_value, std_err

def perform_regression(timepoint_volatility, unit_activation, threshold):
    _, _, _, p_value, _ = custom_linregress(timepoint_volatility, unit_activation)
    return p_value < threshold

# Vectorize the perform_regression function to apply it to each hidden unit
vectorized_regression = vmap(perform_regression, in_axes=(None, 0, None))

# Perform regression for each time point
for t in range(activations.shape[0]):
    timepoint_volatility = volatility_over_time[t]
    threshold = significance_threshold / activations.shape[1]

    # Apply the vectorized regression function to all hidden units at this time point
    significant_units = jnp.sum(vectorized_regression(timepoint_volatility, activations[t], threshold))
    proportions.append(significant_units / activations.shape[1])

# Convert proportions to a NumPy array before saving
proportions_np = np.array(proportions)
np.save('data/proportions.npy', proportions_np)

#%%
# Plotting the results

# Plot the proportion of significant hidden units over time
plt.figure(figsize=(10, 6))
plt.plot(range(len(proportions_np)), proportions_np, label='Proportion of Significant Hidden Units', color='red')
plt.xlabel('Time')
plt.ylabel('Proportion of Significant Units')
plt.title('Proportion of Significant Hidden Units Over Time')
plt.grid(True)
plt.legend()
plt.show()

# Save the plot
plt.savefig('plots/proportion_significant_units.png')


