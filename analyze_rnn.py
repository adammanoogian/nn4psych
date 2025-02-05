#%%
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from tasks import PIE_CP_OB_v2
import matplotlib.pyplot as plt
from torch.nn import init
from utils_funcs import get_lrs_v2, saveload, plot_behavior
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
from copy import deepcopy


contexts = ["change-point","oddball"] #"change-point","oddball"
num_contexts = len(contexts)
train_cond = True
reward_size= 7.5
max_displacement=15
max_time = 300
n_trials = 200
epochs = 100

input_dim = 6+2  # set this based on your observation space. observation vector is length 4 [helicopter pos, bucket pos, bag pos, bag-bucket pos], context vector is length 2.  
hidden_dim = 64  # size of RNN
action_dim = 3  # set this based on your action space. 0 is left, 1 is right, 2 is confirm.

seed = 2025
np.random.seed(seed)
torch.manual_seed(seed)

model_path = "./model_params/Falseheli_25000e_v3_25000e_0.75r_64n_0.99g_8.223684210526316e-05lr_15md_7.5rz_0s.pth"


class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, gain=1.5):
        super(ActorCritic, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gain =gain
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity='tanh')
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        self.init_weights()

    def init_weights(self):
        for name, param in self.rnn.named_parameters(): # initialize the input and rnn weights 
            if 'weight_ih' in name:
                # initialize input weights using 1/sqrt(fan_in). if 1/fan_in, more feature learning. 
                init.normal_(param, mean=0, std=1/(self.input_dim**0.5))
            elif 'weight_hh' in name:
                # initialize rnn weights in a stable (gain=1.0) or chaotic regime (gain=1.5)
                init.normal_(param, mean=0, std=self.gain / self.hidden_dim**0.5)
            elif 'bias_ih' in name or 'bias_hh' in name:
                # Set RNN biases to zero
                init.constant_(param, 0)
        
        for layer in [self.actor, self.critic]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    # initialize input weights using 1/fan_in to induce feature learning 
                    init.normal_(param, mean=0, std=1/self.hidden_dim)
                elif 'bias' in name:
                    init.constant_(param, 0)

    def forward(self, x, hx):
        r, h = self.rnn(x, hx)
        r = r.squeeze(1)
        return self.actor(r), self.critic(r), h

model = ActorCritic(input_dim, hidden_dim, action_dim)
if model_path is not None:
    model.load_state_dict(torch.load(model_path))
    print('Load Model')


all_states = np.zeros([epochs, num_contexts, 5, n_trials])

# get rnn, actor, critic activity
for epoch in range(epochs):
    Hs = []
    As = []
    Cs = []
    Rs = []
    Os = []
    for tt, context in enumerate(contexts):
        env = PIE_CP_OB_v2(condition=context, max_time=max_time, total_trials=n_trials, 
                train_cond=train_cond, max_displacement=max_displacement, reward_size=reward_size)
        
        h, a, c, r, o = [],[],[], [], []
        hx = torch.randn(1, 1, hidden_dim) * 1/hidden_dim
        for trial in range(n_trials):

            next_obs, done = env.reset()
            norm_next_obs = env.normalize_states(next_obs)
            next_state = np.concatenate([norm_next_obs, env.context])
            next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

            while not done:
                actor_logits, critic_value, hx = model(next_state, hx)
                probs = Categorical(logits=actor_logits)
                action = probs.sample()

                # Take action and observe reward
                next_obs, reward, done = env.step(action.item())

                h.append(hx[0,0]), a.append(actor_logits[0]), c.append(critic_value[0]), r.append(reward), o.append(env.hazard_trigger)

                # Prep next state
                norm_next_obs = env.normalize_states(next_obs)
                next_state = np.concatenate([norm_next_obs, env.context])
                next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

        Hs.append(h), As.append(a), Cs.append(c), Rs.append(r), Os.append(o)

        all_states[epoch, tt] = np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])



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

    return prediction_error_sorted, learning_rate_sorted


def plot_lrs(states, scale=0.1):
    epochs = states.shape[0]
    pess, lrss, area = [],[], []
    for c in range(2):
        pes,lrs = [],[]
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
    

    plt.figure(figsize=(3,2))
    colors = ['orange', 'brown']
    labels = ['CP', 'OB']
    for i in range(2):
        window_size = int(len(lrss[i])*scale)
        smoothed_learning_rate = uniform_filter1d(lrss[i], size=window_size)
        plt.plot(pess[i], smoothed_learning_rate, color=colors[i], linewidth=2,label=labels[i])
    plt.legend()
    plt.xlabel('Prediction error')
    plt.ylabel('Learning rate')
    plt.title(f'CB={area[0]:.1f}, OB={area[1]:.1f}, A={(area[0]-area[1]):.1f}')
    plt.tight_layout()
    return pess, lrss, area

_,_,area = plot_lrs(all_states,scale=0.25)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def plot_combined_state_space(Hs, Rs, Os, contexts):
    plt.figure(figsize=(12, 6))
    
    for i, (h_list, r_list, o_list, context) in enumerate(zip(Hs, Rs, Os, contexts)):
        # Convert lists to numpy arrays
        h_array = torch.stack(h_list).detach().numpy()
        r_array = np.array(r_list)
        o_array = np.array(o_list)

        # Perform PCA for dimensionality reduction to 2D
        pca = PCA(n_components=2)
        h_proj = pca.fit_transform(h_array)

        # Calculate differences between consecutive hidden states for vector field
        vectors = h_proj[1:] - h_proj[:-1]

        # Determine marker colors for reward and hazards
        colors = np.where(r_array > 0, 'green', 'gray') # Default colors based on reward
        # Overlay red color where hazard is indicated (o_array > 0)
        colors[o_array > 0] = 'red'

        # Determine marker sizes based on reward
        sizes = 20# + ((r_array - 0) / (1 - 0)) * (200 - 20)  # Scale sizes from 20 to 200

        # Plot PCA projection with vector field
        plt.subplot(1, len(contexts), i+1)
        
        # Draw paths with a quiver plot
        plt.quiver(h_proj[:-1, 0], h_proj[:-1, 1],
                   vectors[:, 0], vectors[:, 1],
                   angles='xy', scale_units='xy', scale=1, color='purple', alpha=0.7)
        
        # Scatter plot the hidden states with color coding
        plt.scatter(h_proj[:, 0], h_proj[:, 1], c=colors, s=sizes, alpha=1)
        
        # Highlight start and end points
        plt.scatter(h_proj[0, 0], h_proj[0, 1], color='blue', s=100, alpha=1, marker='s', label='Start Point')
        plt.scatter(h_proj[-1, 0], h_proj[-1, 1], color='blue', s=100, alpha=1, marker='x', label='End Point')
        
        plt.scatter([],[], label=context, color='r')
        plt.scatter([],[], label='Bag Drop', color='g')
        plt.scatter([],[], label='Bucket Mvmt', color='gray')

        plt.title(f'Combined State Space - {context}')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

# Call the combined function with hidden states, rewards, hazard indications, and contexts
plot_combined_state_space(Hs, Rs, Os, contexts)