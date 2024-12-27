#%%
#Load all data 

import numpy as np
import matplotlib.pyplot as plt

states = np.load('data/pt_rnn_context/env_data.npy') #[trials, bucket_position, bag_position, helicopter_position]

# Extract prediction error (PE) and state (s) and predicted state (s_hat)
true_state = states[2]
predicted_state = states[3]
prediction_error = abs(true_state - predicted_state)
prediction_error = np.minimum(prediction_error, 100)

update = abs(np.diff(predicted_state))
learning_rate = np.where(prediction_error[:-1] != 0, update / prediction_error[:-1], 0)

# Calculate the slope of the learning rate
slope, intercept = np.polyfit(prediction_error[:-1], learning_rate, 1)

#%%
#Plots

def plot_update_by_prediction_error(prediction_error, update, slope, intercept):
    # Plot update x pe
    plt.figure(figsize=(10, 6))
    plt.scatter(prediction_error[:-1], update, alpha=0.5, color='blue', label='Data Points')
    plt.plot(prediction_error[:-1], slope * prediction_error[:-1] + intercept, color='red', label=f'Slope: {slope:.2f}')
    plt.xlabel('Prediction Error')
    plt.ylabel('Update (Predicted State at t+1 - State at t)')
    plt.title('Update by prediction error')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Save the plot
    plt.savefig('plots/update_by_prediction_error.png')

def plot_learning_rate_by_prediction_error(prediction_error, learning_rate, slope, intercept):
    # Plot learning rate by prediction error
    plt.figure(figsize=(10, 6))
    plt.scatter(prediction_error[:-1], learning_rate, alpha=0.5, color='green', label='Data Points')
    # Plot the regression line
    plt.plot(prediction_error[:-1], slope * prediction_error[:-1] + intercept, color='orange', label=f'Slope: {slope:.2f}')
    plt.xlabel('Prediction Error')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate by Prediction Error')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Save the plot
    plt.savefig('plots/learning_rate_by_prediction_error.png')

def plot_states_and_learning_rate(true_state, predicted_state, learning_rate):
    trials = np.arange(len(true_state))
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Trial')
    ax1.set_ylabel('State', color='blue')
    ax1.plot(trials, true_state, label='True State', color='blue')
    ax1.plot(trials, predicted_state, label='Predicted State', color='cyan')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color='green')
    ax2.plot(trials[:-1], learning_rate, label='Learning Rate', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    plt.title('True State, Predicted State, and Learning Rate over Trials')
    plt.grid(True)
    plt.show()
    plt.savefig('plots/states_and_learning_rate_over_trials.png')

# %%


# Call the functions to generate the plots
plot_update_by_prediction_error(prediction_error, update, slope, intercept)
plot_learning_rate_by_prediction_error(prediction_error, learning_rate, slope, intercept)
plot_states_and_learning_rate(true_state, predicted_state, learning_rate)