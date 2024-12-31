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

hazard_trials = states[4]
hazard_indexes = np.where(states[4] == 1)[0]
hazard_distance = np.zeros(len(states[0]), dtype=int)
current = 0
for i in range(len(states[0])):
    if i in hazard_indexes:
        current = 0
    hazard_distance[i] = current
    current += 1

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

def plot_learning_rate_histogram(learning_rate):
    bins = np.arange(0, 1.1, 0.1)
    plt.figure(figsize=(10, 6))
    plt.hist(learning_rate, bins=bins, edgecolor='black')
    plt.xlabel('Learning Rate')
    plt.ylabel('Frequency')
    plt.title('Histogram of Learning Rate')
    plt.grid(True)
    plt.show()
    plt.savefig('plots/learning_rate_histogram.png')


def plot_lr_after_hazard(learning_rate, hazard_distance):

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
    plt.title('Interactions by Hazard Distance and Size')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('plots/interactions_line_graph.png')
# %%






# %%

# Call the functions to generate the plots
plot_update_by_prediction_error(prediction_error, update, slope, intercept)
plot_learning_rate_by_prediction_error(prediction_error, learning_rate, slope, intercept)
plot_states_and_learning_rate(true_state, predicted_state, learning_rate)
plot_learning_rate_histogram(learning_rate)
plot_lr_after_hazard(learning_rate, hazard_trials)
