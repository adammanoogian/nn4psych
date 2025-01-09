#%%
#Load all data 

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d


def extract_states(states):
    # originally by Adam
    # Extract prediction error (PE) and state (s) and predicted state (s_hat)
    true_state = states[2]  # bag position
    predicted_state = states[1]  # bucket position
    prediction_error = abs(true_state - predicted_state)
    prediction_error = np.minimum(prediction_error, 100)
    prediction_error = prediction_error[:-1] 

    update = abs(np.diff(predicted_state))
    learning_rate = np.where(prediction_error != 0, update / prediction_error, 0)

    hazard_trials = states[4]
    hazard_indexes = np.where(states[4] == 1)[0]
    hazard_distance = np.zeros(len(states[0]), dtype=int)
    current = 0
    for i in range(len(states[0])):
        if i in hazard_indexes:
            current = 0
        hazard_distance[i] = current
        current += 1
    return prediction_error, update, learning_rate, true_state, predicted_state,hazard_distance, hazard_trials

#Plots

def plot_update_by_prediction_error(prediction_error, update, condition):
    # Plot update x pe
    plt.figure(figsize=(3, 2))
    plt.scatter(prediction_error, update, alpha=0.5, color='blue', label='Data Points')

    slope, intercept, r_value, p_value, std_err = linregress(prediction_error, update)
    plt.plot(prediction_error, slope * prediction_error + intercept, color='k', label=f'm={slope:.3f}, c={intercept:.2f},r={r_value:.3f}, p={p_value:.3f}')

    sorted_indices = np.argsort(prediction_error)
    prediction_error_sorted = prediction_error[sorted_indices]
    update_sorted = update[sorted_indices]

    window_size = 10
    smoothed_update = uniform_filter1d(update_sorted, size=window_size)
    plt.plot(prediction_error_sorted, smoothed_update, color='red', label='Smooth')

    plt.xlabel('Prediction Error')
    plt.ylabel('Update (Predicted State at t+1 - State at t)')
    plt.title(f'{condition}: Update by prediction error')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Save the plot
    plt.savefig(f'plots/update_by_prediction_error_{condition}.png')

def plot_learning_rate_by_prediction_error(prediction_error, learning_rate, condition):
    # Plot learning rate by prediction error
    plt.figure(figsize=(3, 2))
    plt.scatter(prediction_error, learning_rate, alpha=0.5, color='green', label='Data Points')
    
    slope, intercept, r_value, p_value, std_err = linregress(prediction_error, learning_rate)
    # Plot the regression line
    plt.plot(prediction_error, slope * prediction_error + intercept, color='orange', label=f'm={slope:.3f}, c={intercept:.2f},r={r_value:.3f}, p={p_value:.3f}')

    sorted_indices = np.argsort(prediction_error)
    prediction_error_sorted = prediction_error[sorted_indices]
    learning_rate_sorted = learning_rate[sorted_indices]

    window_size = 10
    smoothed_learning_rate = uniform_filter1d(learning_rate_sorted, size=window_size)
    plt.plot(prediction_error_sorted, smoothed_learning_rate, color='red', label='Smooth')

    plt.xlabel('Prediction Error')
    plt.ylabel('Learning Rate')
    plt.title(f'{condition}: Learning Rate by Prediction Error')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Save the plot
    plt.savefig(f'plots/learning_rate_by_prediction_error_{condition}.png')

def plot_states_and_learning_rate(true_state, predicted_state, learning_rate, condition):
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

    plt.title(f'{condition}: True State, Predicted State, and Learning Rate over Trials')
    plt.grid(True)
    plt.show()
    plt.savefig(f'plots/states_and_learning_rate_over_trials_{condition}.png')

def plot_learning_rate_histogram(learning_rate, condition):
    bins = np.arange(0, 1.1, 0.1)
    plt.figure(figsize=(10, 6))
    plt.hist(learning_rate, bins=bins, edgecolor='black')
    plt.xlabel('Learning Rate')
    plt.ylabel('Frequency')
    plt.title(f'{condition}: Histogram of Learning Rate')
    plt.grid(True)
    plt.show()
    plt.savefig(f'plots/learning_rate_histogram_{condition}.png')


def plot_lr_after_hazard(learning_rate, hazard_distance, condition):

    bins = np.arange(0, 1.1, 0.1)
    bin_indices = np.digitize(learning_rate, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    update_size_indices = np.where(bin_indices == 0, 0, np.where((bin_indices >= 1) & (bin_indices <= 8), 1, 2))
    # Count intersections between hazard_distance and update_size_indices
    interaction_counts = {
        "small": {},
        "medium": {},
        "large": {}
    }

    for hd, usi in zip(hazard_distance, update_size_indices):
        if usi == 0:
            category = "small"
        elif usi == 1:
            category = "medium"
        else:
            category = "large"
        
        if hd in interaction_counts[category]:
            interaction_counts[category][hd] += 1
        else:
            interaction_counts[category][hd] = 1


    # Prepare data for plotting
    categories = ["small", "medium", "large"]
    colors = ["blue", "green", "red"]
    x = sorted({hd for counts in interaction_counts.values() for hd in counts.keys()})

    plt.figure(figsize=(10, 6))

    for category, color in zip(categories, colors):
        counts = [interaction_counts[category].get(hd, 0) for hd in x]
        plt.plot(x, counts, label=category.capitalize(), color=color)

    plt.xlabel('Hazard Distance')
    plt.ylabel('Count')
    plt.title(f'{condition}: Interactions by Hazard Distance and Size')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'plots/interactions_line_graph_{condition}.png')
# %%






# %%

if __name__ == "__main__":
    states = np.load('data/pt_rnn_context/env_data.npy') #[trials, bucket_position, bag_position, helicopter_position]

    prediction_error, update, learning_rate, true_state, predicted_state,hazard_distance, hazard_trials = extract_states(states) #ERROR? Is this supposed to return slope

    # Call the functions to generate the plots
    plot_update_by_prediction_error(prediction_error, update)
    plot_learning_rate_by_prediction_error(prediction_error, learning_rate)
    plot_states_and_learning_rate(true_state, predicted_state, learning_rate)
    plot_learning_rate_histogram(learning_rate)
    plot_lr_after_hazard(learning_rate, hazard_trials)

# %%
