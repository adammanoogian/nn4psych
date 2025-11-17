import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.nn import init

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, gain=1.5, noise=0.0, bias=False):
        super(ActorCritic, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gain = gain
        self.noise = noise  # Include the noise variance as an argument
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity='tanh',bias=bias)
        self.actor = nn.Linear(hidden_dim, action_dim,bias=bias)
        self.critic = nn.Linear(hidden_dim, 1,bias=bias)
        self.init_weights()

    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                init.normal_(param, mean=0, std=1/(self.input_dim**0.5))
            elif 'weight_hh' in name:
                init.normal_(param, mean=0, std=self.gain / self.hidden_dim**0.5)
            elif 'bias_ih' in name or 'bias_hh' in name:
                init.constant_(param, 0)

        for layer in [self.actor, self.critic]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.normal_(param, mean=0, std=1/self.hidden_dim)
                elif 'bias' in name:
                    init.constant_(param, 0)

    def forward(self, x, hx):
        r, h = self.rnn(x, hx)
        r = r.squeeze(1)
        critic_value = self.critic(r)

        return self.actor(r), critic_value, h

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

def get_lrs_v2(states, threshold=20):
    true_state = states[2]  # bag position
    predicted_state = states[1]  # bucket position
    prediction_error = (true_state - predicted_state)[:-1]
    update = np.diff(predicted_state)

    idx = prediction_error !=0
    prediction_error= prediction_error[idx]
    update = update[idx]
    learning_rate = update / prediction_error

    prediction_error = abs(prediction_error)
    idx = prediction_error>threshold
    pes = prediction_error[idx]
    lrs = np.clip(learning_rate,0,1)[idx]

    sorted_indices = np.argsort(pes)
    prediction_error_sorted = pes[sorted_indices]
    learning_rate_sorted = lrs[sorted_indices]

    pad_pes = np.pad(prediction_error_sorted,(0, len(true_state)-len(prediction_error_sorted)-1), 'constant', constant_values=-1)
    pad_lrs = np.pad(learning_rate_sorted,(0, len(true_state)-len(learning_rate_sorted)-1), 'constant', constant_values=-1)

    return pad_pes, pad_lrs


def plot_behavior(states, context,epoch, ax=None):
    if ax is None:
        plt.figure(figsize=(10, 6))
    trials, bucket_positions, bag_positions, helicopter_positions, hazard_triggers = states
    # plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='blue')
    plt.plot(trials, bag_positions, label='Bag', color='red', marker='o', linestyle='-.', alpha=0.5, ms=2)
    plt.plot(trials, helicopter_positions, label='Heli', color='green', linestyle='--',ms=2)
    plt.plot(trials, bucket_positions, label='Bucket', color='b',marker='o', linestyle='-.', alpha=0.5,ms=2)

    plt.ylim(-10, 310)  # Set y-axis limit from 0 to 300
    plt.xlabel('Trial')
    plt.ylabel('Position')
    plt.title(f"{context}, E:{epoch}")
    plt.legend(fontsize=6)
