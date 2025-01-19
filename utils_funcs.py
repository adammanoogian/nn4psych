import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d

def saveload(filename, variable, opt):
    import pickle
    if opt == 'save':
        with open(f"{filename}.pickle", "wb") as file:
            pickle.dump(variable, file)
        print('file saved')
    else:
        with open(f"{filename}.pickle", "rb") as file:
            return pickle.load(file)


def get_lrs(states):
    true_state = states[2]  # bag position
    predicted_state = states[1]  # bucket position
    prediction_error = abs((true_state - predicted_state))[:-1]
    update = abs(np.diff(predicted_state))
    learning_rate = np.where(prediction_error !=0, update / prediction_error)
    
    sorted_indices = np.argsort(prediction_error)
    prediction_error_sorted = prediction_error[sorted_indices]
    learning_rate_sorted = learning_rate[sorted_indices]

    window_size = 10
    smoothed_learning_rate = uniform_filter1d(learning_rate_sorted, size=window_size)
    return prediction_error_sorted, smoothed_learning_rate

def get_lrs_v2(states):
    true_state = states[2]  # bag position
    predicted_state = states[1]  # bucket position
    prediction_error = (true_state - predicted_state)[:-1]
    update = np.diff(predicted_state)

    idx = prediction_error !=0
    prediction_error= prediction_error[idx]
    update = update[idx]
    learning_rate = update / prediction_error

    prediction_error = abs(prediction_error)
    idx = prediction_error>20
    pes = prediction_error[idx]
    lrs = np.clip(learning_rate,0,1)[idx]

    sorted_indices = np.argsort(pes)
    prediction_error_sorted = pes[sorted_indices]
    learning_rate_sorted = lrs[sorted_indices]

    pad_pes = np.pad(prediction_error_sorted,(0, len(true_state)-len(prediction_error_sorted)-1), 'constant', constant_values=-1)
    pad_lrs = np.pad(learning_rate_sorted,(0, len(true_state)-len(learning_rate_sorted)-1), 'constant', constant_values=-1)

    return pad_pes, pad_lrs


