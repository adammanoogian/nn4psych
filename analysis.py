# code to analyze RNN dynamics
# Veronica

#%%
#load data

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

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


proportions = []

# Perform regression for each time point
for t in range(activations.shape[0]):
    significant_units = 0

  # Perform regression for each hidden unit at this time point
    for unit in range(activations.shape[1]):
        # Single timepoint data for regression (considering the deviation from the mean)
        unit_activation = activations[t, unit]
        timepoint_volatility = volatility_over_time[t]
        
        # Linear regression requires more than one data point, but for the purpose of this example, we're using single points.
        slope, intercept, r_value, p_value, std_err = stats.linregress([timepoint_volatility], [unit_activation])
        
        # Bonferroni correction
        threshold = 0.10 / activations.shape[1]
        
        # Check if this hidden unit is significant at this time point
        if p_value < threshold:
            significant_units += 1
    
    # Calculate the proportion of significant units at this time point
    proportion_significant = significant_units / activations.shape[1]
    proportions.append(proportion_significant)

#%%
# Plotting the results

# Plot the proportion of significant hidden units over time
plt.figure(figsize=(10, 6))
plt.plot(range(len(proportions)), proportions, label='Proportion of Significant Hidden Units', color='red')
plt.xlabel('Time')
plt.ylabel('Proportion of Significant Units')
plt.title('Proportion of Significant Hidden Units Over Time')
plt.grid(True)
plt.legend()
plt.show()

# %%
