#%%
#Load all data 

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d

#out of date - use get_area in get_behavior.py to get data
#or run for loop in analyze_rnn with set weights to get data
# def extract_states(states):
#     # originally by Adam
#     # Extract prediction error (PE) and state (s) and predicted state (s_hat)
#     true_state = states[2]  # bag position
#     predicted_state = states[1]  # bucket position
#     prediction_error = abs(true_state - predicted_state)
#     prediction_error = np.minimum(prediction_error, 100)
#     prediction_error = prediction_error[:-1] 

#     update = abs(np.diff(predicted_state))
#     learning_rate = np.where(prediction_error != 0, update / prediction_error, 0)

#     hazard_trials = states[4]
#     hazard_indexes = np.where(states[4] == 1)[0]
#     hazard_distance = np.zeros(len(states[0]), dtype=int)
#     current = 0
#     for i in range(len(states[0])):
#         if i in hazard_indexes:
#             current = 0
#         hazard_distance[i] = current
#         current += 1
#     return prediction_error, update, learning_rate, true_state, predicted_state,hazard_distance, hazard_trials

#Plots

def plot_update_by_prediction_error(prediction_error, update, condition="change-point"):
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

def plot_learning_rate_by_prediction_error(prediction_error, learning_rate, condition="change-point"):
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

def plot_states_and_learning_rate(true_state, predicted_state, learning_rate, condition="change-point"):
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

def plot_learning_rate_histogram(learning_rate, condition="change-point"):
    bins = np.arange(0, 1.1, 0.1)
    plt.figure(figsize=(10, 6))
    plt.hist(learning_rate, bins=bins, edgecolor='black')
    plt.xlabel('Learning Rate')
    plt.ylabel('Frequency')
    plt.title(f'{condition}: Histogram of Learning Rate')
    plt.grid(True)
    plt.show()
    plt.savefig(f'plots/learning_rate_histogram_{condition}.png')


def plot_lr_after_hazard(learning_rate, hazard_distance, condition="change-point"):

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


def get_lrs_v2(states, threshold=20):
    '''
    -takes in state vector
    -threshold is the cutoff for prediction error to be considered a learning rate
    -returns prediction error and learning rate sorted by prediction error
    '''
    true_state = states[2]  # bag position
    predicted_state = states[1]  # bucket position
    prediction_error = (true_state - predicted_state)[:-1]
    update = np.diff(predicted_state)

    idx = prediction_error != 0
    prediction_error = prediction_error[idx]
    update = update[idx]
    learning_rate = update / prediction_error

    prediction_error = abs(prediction_error)
    idx = prediction_error > threshold
    pes = prediction_error[idx]
    lrs = np.clip(learning_rate, 0, 1)[idx]
    #sort for easy plotting
    sorted_indices = np.argsort(pes)
    prediction_error_sorted = pes[sorted_indices]
    learning_rate_sorted = lrs[sorted_indices]

    area = np.trapz(learning_rate_sorted, prediction_error_sorted)

    return prediction_error_sorted, learning_rate_sorted, pes, lrs, area

def plot_lrs(states, scale=0.1):
    epochs = states.shape[0]
    pess, lrss, area = [], [], []
    for c in range(2):
        pes, lrs = [], []
        for e in range(epochs):
            pe, lr = get_lrs_v2(states[e, c])

            pes.append(pe)
            lrs.append(lr)

        pes = np.concatenate(pes)
        lrs = np.concatenate(lrs)
        sorted_indices = np.argsort(pes)
        prediction_error_sorted = pes[sorted_indices]
        learning_rate_sorted = lrs[sorted_indices]

        pess.append(prediction_error_sorted)
        lrss.append(learning_rate_sorted)
        area.append(np.trapz(learning_rate_sorted, prediction_error_sorted))

    plt.figure(figsize=(3, 2))
    colors = ['orange', 'brown']
    labels = ['CP', 'OB']
    for i in range(2):
        window_size = int(len(lrss[i]) * scale)
        smoothed_learning_rate = uniform_filter1d(lrss[i], size=window_size)
        plt.plot(pess[i], smoothed_learning_rate, color=colors[i], linewidth=2, label=labels[i])
    plt.legend()
    plt.xlabel('Prediction error')
    plt.ylabel('Learning rate')
    plt.title(f'CB={area[0]:.1f}, OB={area[1]:.1f}, A={(area[0] - area[1]):.1f}')
    plt.tight_layout()
    return pess, lrss, area

def plot_lrs_v2_batch(behav_dict, scale=0.1):
    """
    nassar2021 fig2
    For each RNN parameter in behav_dict, group runs by the parameter value (extracted from model_list),
    then create one figure with one subplot per unique parameter value.
    In each subplot, plot CP (solid) and OB (dashed) curves for each run.
    """
    #run cap option (just take a few runs)
    run_cap = 3
    for rnn_param, data in behav_dict.items():
        model_groups = data['model_list']
        # Group runs by parameter value (from the model_groups values)
        group_dict = {}
        for key, val in model_groups.items():
            # key is a tuple (e.g. (run_index, group_index)) and val is a set containing a numeric value and a string
            param_val = next((x for x in val if isinstance(x, (int, float))), None)
            if param_val is None:
                continue
            # use the first element of the key as the run index
            run_id = key[0]
            group_dict.setdefault(param_val, []).append(run_id)

        # Apply run_cap to limit the number of runs per unique parameter value
        for param_val in group_dict:
            group_dict[param_val] = group_dict[param_val][:run_cap]

        unique_param_vals = sorted(group_dict.keys())
        n_plots = len(unique_param_vals)
        fig, axs = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots))
        if n_plots == 1:
            axs = [axs]

        cmap = plt.cm.viridis

        for ax, param_val in zip(axs, unique_param_vals):
            runs = group_dict[param_val]
            run_indices = np.array(runs, dtype=float)
            norm = plt.Normalize(vmin=run_indices.min(), vmax=run_indices.max())

            for run in runs:
                # Plot CP data (solid)
                pe_cp = data['pe_sorted_cp'][run]
                lr_cp = data['lr_sorted_cp'][run]
                window_size_cp = max(1, int(len(lr_cp) * scale))
                smoothed_lr_cp = uniform_filter1d(lr_cp, size=window_size_cp)
                color = cmap(norm(run))
                ax.plot(pe_cp, smoothed_lr_cp, linestyle='-', color=color, linewidth=2, label=f"Run {run} CP")

                # Plot OB data (dashed)
                pe_ob = data['pe_sorted_ob'][run]
                lr_ob = data['lr_sorted_ob'][run]
                window_size_ob = max(1, int(len(lr_ob) * scale))
                smoothed_lr_ob = uniform_filter1d(lr_ob, size=window_size_ob)
                ax.plot(pe_ob, smoothed_lr_ob, linestyle='--', color=color, linewidth=2, label=f"Run {run} OB")

            area_cp_vals = [data['area_cp'][run] for run in runs]
            area_ob_vals = [data['area_ob'][run] for run in runs]
            avg_area_cp = np.mean(area_cp_vals)
            avg_area_ob = np.mean(area_ob_vals)
            avg_area_diff = avg_area_cp - avg_area_ob
            ax.set_title(f"{rnn_param} = {param_val}, avg area CP={avg_area_cp:.2f}, OB={avg_area_ob:.2f}, diff={avg_area_diff:.2f}")
            ax.set_title(f"{rnn_param} = {param_val}, avg area CP={avg_area_cp:.2f}, OB={avg_area_ob:.2f}")
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Learning Rate')
            ax.grid(True)
            ax.legend()

        fig.tight_layout()
        plt.show()
    
def plot_lr_bins_post_hazard_batch(behav_dict, num_runs=1):
    """
    nassar2021 fig3a-d (modified)
    For each RNN parameter, group runs by the unique parameter value (extracted from model_list),
    then create one figure with two subplots per unique value (one for CP and one for OB)
    showing histograms of learning rate frequencies (bin size = 0.1) using at most num_runs runs.
    """
    bins = np.arange(0, 1.1, 0.1)
    
    for rnn_param, data in behav_dict.items():
        # Group runs by unique parameter value
        group_dict = {}
        for key, val in data['model_list'].items():
            param_val = next((x for x in val if isinstance(x, (int, float))), None)
            if param_val is None:
                continue
            run_id = key[0]
            group_dict.setdefault(param_val, []).append(run_id)

        # Limit runs per unique parameter value to num_runs
        for param_val in group_dict:
            group_dict[param_val] = group_dict[param_val][:num_runs]

        unique_param_vals = sorted(group_dict.keys())
        n_plots = len(unique_param_vals)
        # Create subplots with 2 columns: one for CP and one for OB
        fig, axs = plt.subplots(n_plots, 2, figsize=(16, 4 * n_plots))
        if n_plots == 1:
            axs = axs.reshape(1, -1)

        for i, param_val in enumerate(unique_param_vals):
            runs = group_dict[param_val]
            # Get unsorted learning rates for CP and OB
            cp_lr_all = np.concatenate([data['lr_unsorted_cp'][run] for run in runs])
            ob_lr_all = np.concatenate([data['lr_unsorted_ob'][run] for run in runs])
            
            # Left subplot: CP histogram
            ax_cp = axs[i, 0]
            ax_cp.hist(cp_lr_all, bins=bins, alpha=0.7, label='CP', edgecolor='black')
            ax_cp.set_title(f'{rnn_param} = {param_val} CP')
            ax_cp.set_xlabel('Learning Rate')
            ax_cp.set_ylabel('Frequency')
            ax_cp.legend()
            ax_cp.grid(True)
            
            # Right subplot: OB histogram
            ax_ob = axs[i, 1]
            ax_ob.hist(ob_lr_all, bins=bins, alpha=0.7, label='OB', edgecolor='black')
            ax_ob.set_title(f'{rnn_param} = {param_val} OB')
            ax_ob.set_xlabel('Learning Rate')
            ax_ob.set_ylabel('Frequency')
            ax_ob.legend()
            ax_ob.grid(True)
        
        fig.suptitle(f'{rnn_param}: Learning Rate Histogram (CP and OB) Across Unique Parameter Values', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def compute_hazard_distance(hazard_triggers):
    """
    Compute hazard distance from hazard trigger array.
    For each trial, the hazard distance is reset to 0 when a hazard is triggered (value==1),
    and otherwise is incremented by 1.
    returns: hazard distance array. same size as hazard_triggers.
    """
    hd = np.zeros(len(hazard_triggers), dtype=int)
    current = 0
    for i, trigger in enumerate(hazard_triggers):
        if trigger == 1:
            current = 0
        hd[i] = current
        current += 1
    return hd

def plot_lr_curve_post_hazard(behav_data):
    """
    For each unique RNN parameter value, using one run per unique value, plot 6 subplots:
      - Left column: CP data for one run corresponding to the unique parameter value.
      - Right column: OB data for that run.
    Rows:
      Row 1: Non-updates (lr < 0.1)
      Row 2: Moderate updates (0.1 ≤ lr < 0.9)
      Row 3: Large updates (lr ≥ 0.9)
      
    Each subplot shows the frequency (or probability for CP) of hazard distance values.
    Hazard distance is computed from the hazard trigger array (index 4) in the states array.
    """

    for rnn_param, data in behav_data.items():
        # Group runs by unique parameter value using the model_list
        group_dict = {}
        for key, val in data['model_list'].items():
            # Extract the numeric value from the model_list tuple (if present)
            param_val = next((x for x in val if isinstance(x, (int, float))), None)
            if param_val is None:
                continue
            run_id = key[0]
            group_dict.setdefault(param_val, []).append(run_id)

        unique_param_vals = sorted(group_dict.keys())
        # For each unique parameter value, select one run (the first)
        for param_val in unique_param_vals:
            run = group_dict[param_val][0]

            cp_data = data['cp_array'][run]
            ob_data = data['ob_array'][run]

            # Compute hazard distances using the hazard trigger array (index 4)
            cp_hd = compute_hazard_distance(cp_data[4])
            ob_hd = compute_hazard_distance(ob_data[4])

            # Determine common x-axis limits from both CP and OB data
            max_hd = max(np.max(cp_hd), np.max(ob_hd))

            # Get learning rate vectors for CP and OB for the selected run
            cp_lr = data['lr_unsorted_cp'][run]
            ob_lr = data['lr_unsorted_ob'][run]

            categories = {
                'Non-updates': lambda lr: lr < 0.1,
                'Moderate updates': lambda lr: (lr >= 0.1) & (lr < 0.9),
                'Large updates': lambda lr: lr >= 0.9
            }

            # Create figure with 3 rows (update categories) and 2 columns (CP and OB)
            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
            fig.suptitle(f'{rnn_param} = {param_val}: LR Post-Hazard Analysis', fontsize=16)

            # Loop through each update category to build subplots
            for row_idx, (cat_label, condition) in enumerate(categories.items()):
                # Find indices where learning rates satisfy the condition for CP and OB
                cp_inds = np.where(condition(cp_lr))[0]
                ob_inds = np.where(condition(ob_lr))[0]

                # Filter the corresponding hazard distances
                cp_hd_filtered = cp_hd[cp_inds]
                ob_hd_filtered = ob_hd[ob_inds]

                # Count frequency of hazard distances for CP and convert to probabilities
                cp_counts = {}
                for hd in cp_hd_filtered:
                    cp_counts[hd] = cp_counts.get(hd, 0) + 1
                total_cp = sum(cp_counts.values())
                cp_keys = sorted(cp_counts.keys())
                cp_values = [cp_counts[k] / total_cp for k in cp_keys] if total_cp > 0 else []

                # Count frequency of hazard distances for OB
                ob_counts = {}
                for hd in ob_hd_filtered:
                    ob_counts[hd] = ob_counts.get(hd, 0) + 1
                ob_keys = sorted(ob_counts.keys())
                total_ob = sum(ob_counts.values())
                ob_values = [ob_counts[k] / total_ob for k in ob_keys] if total_ob > 0 else []

                # Plot CP histogram on the left column
                ax_cp = axs[row_idx, 0]
                ax_cp.bar(cp_keys, cp_values, color='blue', alpha=0.7)
                ax_cp.set_title(f'CP - {cat_label}')
                ax_cp.set_xlabel('Hazard Distance')
                ax_cp.set_ylabel('P(update)')
                ax_cp.grid(True)
                ax_cp.set_xlim(0, max_hd)

                # Plot OB histogram on the right column
                ax_ob = axs[row_idx, 1]
                ax_ob.bar(ob_keys, ob_values, color='green', alpha=0.7)
                ax_ob.set_title(f'OB - {cat_label}')
                ax_ob.set_xlabel('Hazard Distance')
                ax_ob.set_ylabel('Count')
                ax_ob.grid(True)
                ax_ob.set_xlim(0, max_hd)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

# %% Setup data from get_behavior.py

import utils as utils
from scipy.ndimage import uniform_filter1d
import numpy as np

def get_batch_behav(RNN_param_list = ["gamma", "preset", "rollout", "scale"]):
    '''
    -takes in a list of RNN parameters to analyze made from get_behavior.py
    -returns a dictionary of the data for each parameter
    #to-do: average out the runs for each parameter (need same task.env if unsorted) 
    '''
#np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])
    results = {}
    for rnn_param in RNN_param_list:
        cp_array, ob_array, model_list = utils.unpickle_state_vector(RNN_param=rnn_param)
        pe_sorted_cp, lr_sorted_cp, pe_unsorted_cp, lr_unsorted_cp, area_cp = zip(*[get_lrs_v2(cp_array[i]) for i in range(len(model_list))])
        pe_sorted_ob, lr_sorted_ob, pe_unsorted_ob, lr_unsorted_ob, area_ob = zip(*[get_lrs_v2(ob_array[i]) for i in range(len(model_list))])
        
        results[rnn_param] = {
            'cp_array': cp_array,
            'pe_sorted_cp': pe_sorted_cp,
            'lr_sorted_cp': lr_sorted_cp,
            'pe_unsorted_cp': pe_unsorted_cp,
            'lr_unsorted_cp': lr_unsorted_cp,
            'area_cp': area_cp,
            'ob_array': ob_array,
            'pe_sorted_ob': pe_sorted_ob,
            'lr_sorted_ob': lr_sorted_ob,
            'pe_unsorted_ob': pe_unsorted_ob,
            'lr_unsorted_ob': lr_unsorted_ob,
            'area_ob': area_ob,
            'model_list': model_list
        }
    return results

behav_dict = get_batch_behav()
plot_lrs_v2_batch(behav_dict, scale=0.1)
plot_lr_bins_post_hazard_batch(behav_dict, num_runs=3)
plot_lr_curve_post_hazard(behav_dict)



#%% Run analysis on data

# if __name__ == "__main__":
    #out of date
        #states = np.load('data/bayesian_models/model_predictions.npy') #[trials, bucket_position, bag_position, helicopter_position]
        #states = np.load('data/pt_rnn_context-point.npy') #[trials, bucket_position, bag_position, helicopter_position]
        #prediction_error, update, learning_rate, true_state, predicted_state,hazard_distance, hazard_trials = extract_states(states) #ERROR? Is this supposed to return slope


    # Call the functions to generate the plots
    # plot_update_by_prediction_error(prediction_error, update)
    # plot_learning_rate_by_prediction_error(prediction_error, learning_rate)
    # plot_states_and_learning_rate(true_state, predicted_state, learning_rate)
    # plot_learning_rate_histogram(learning_rate)
    # plot_lr_after_hazard(learning_rate, hazard_trials)

# %%
