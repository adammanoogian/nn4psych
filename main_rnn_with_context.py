#%%
'''
Idea is to pretrain a vanilla RNN using RL on several epochs of the helicopter task so that it sort of knows what to do. 
The model weights are saved for further analysis or additional training
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from tasks import PIE_CP_OB
import matplotlib.pyplot as plt
from torch.nn import init
from behav_figures import plot_update_by_prediction_error, plot_learning_rate_by_prediction_error, plot_learning_rate_histogram, plot_lr_after_hazard, plot_states_and_learning_rate, extract_states

# Assuming that PIE_CP_OB is a gym-like environment
# from your_environment_file import PIE_CP_OB

# task parameters
n_trials = 100  # number of trials per epoch for each condition.
max_time = 300  # number of time steps available for each trial. After max_time, the bag is dropped and the next trial begins after.

contexts = ["change-point","oddball"] #"change-point","oddball"
num_contexts = len(contexts)

# Model Parameters
input_dim = 4+2  # set this based on your observation space. observation vector is length 4 [helicopter pos, bucket pos, bag pos, bag-bucket pos], context vector is length 2.  
hidden_dim = 128  # size of RNN
action_dim = 3  # set this based on your action space. 0 is left, 1 is right, 2 is confirm.
learning_rate = 0.0001  # set learning rate to 0, or 0.0001. The model should have learned the task and doesnt need synaptic plasticty.
gamma = 0.99
reset_memory = 20  # reset RNN activity after T trials

# Actor-Critic Network with RNN
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, gain=1.0):
        super(ActorCritic, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gain =gain
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity='tanh')
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def init_weights(self):
        for name, param in self.rnn.named_parameters(): # initialize the input and rnn weights 
            if 'weight_ih' in name:
                # initialize input weights using 1/sqrt(fan_in). if 1/fan_in, more feature learning. 
                init.normal_(param, mean=0, std=1/self.input_dim)
            elif 'weight_hh' in name:
                # initialize rnn weights in a stable (gain=1.0) or chaotic regime (gain=1.5)
                init.normal_(param, mean=0, std=self.gain / self.hidden_dim**0.5)
        
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
    

    
def to_numpy(tensor):
    return tensor.cpu().detach().numpy()



def train(env, model, optimizer, n_trials, gamma):

    totG = []
    totloss = []
    hx = torch.randn(1, 1, hidden_dim) *0.1  # initialize RNN activity with random

    for trial in range(n_trials):
        obs, done = env.reset()  # reset env at the start of every trial to change helocopter pos based on hazard rate

        norm_obs = env.normalize_states(obs)  # normalize vector to bound between something resonable for the RNN to handle

        state = np.concatenate([norm_obs,env.context])
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # add batch and seq dim

        # Detach hx to prevent backpropagation through the entire history
        hx = hx.detach()
        if trial % reset_memory == 0:
            # Initialize the RNN hidden state
            hx = torch.randn(1, 1, hidden_dim) *0.1

        log_probs = []
        values = []
        rewards = []
        totR = 0 

        while not done: #allows multiple actions in one trial (incrementally moving bag_position)
            # Forward pass
            actor_logits, critic_value, hx = model(state, hx)
            probs = Categorical(logits=actor_logits)
            action = probs.sample()

            log_probs.append(probs.log_prob(action))
            values.append(critic_value)

            next_obs, reward, done = env.step(action.item())
            norm_next_obs = env.normalize_states(next_obs)
            next_state = np.concatenate([norm_next_obs,env.context])
            rewards.append(reward)
            totR += reward
            state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

            if done:
                break
        
        # print("trial:", env.trial, "time:", env.time, "obs:", obs, "actor:", actor_logits, "action:", action, "reward:", reward, "next_obs:", next_obs)

        # Calculate returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze()

        # Compute advantages
        advantages = returns - values.detach()

        # Calculate loss
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = ((returns - values)**2).mean()
        loss = actor_loss + critic_loss

        #  train network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totG.append(totR)
        totloss.append(to_numpy(loss))

    return np.array(totG), np.array(totloss)


model = ActorCritic(input_dim, hidden_dim, action_dim)
model_path = './model_params/model_params_30.pth'
model.load_state_dict(torch.load(model_path))

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

all_G = np.zeros([num_contexts, n_trials])
all_loss = np.zeros([num_contexts, n_trials])
all_states = np.zeros([num_contexts, 5, n_trials])

for tt, task_type in enumerate(contexts):

    env = PIE_CP_OB(condition=task_type, max_time=max_time, total_trials=n_trials, train_cond=False)

    totG, totloss = train(env, model, optimizer, n_trials=n_trials, gamma=gamma)

    all_G[tt] = totG
    all_loss[tt] = totloss
    all_states[tt] = env.render()

# bag collection performance.
# G is the total reward. should be close to 1 for all trials. If not 1, means did not get the bag
# Loss is the model training loss. Should be close to 0 for all trials. If non-zero means high error.
f,ax = plt.subplots(1,2,figsize=(3*2,2*2))
ax[0].plot(all_G[0], color='b', label='CP')
ax[0].plot(all_G[1],color='r',label='OB')
ax[0].legend()
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('G')

ax[1].plot(all_loss[0], color='b', label='CP')
ax[1].plot(all_loss[1],color='r',label='OB')
ax[1].legend()
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
f.tight_layout()

print(np.mean(all_G,axis=1))

cp_states =all_states[0]
ob_states = all_states[1]

for context, states in zip(contexts, [cp_states, ob_states]):
    prediction_error, update, learning_rate, true_state, predicted_state,hazard_distance, hazard_trials = extract_states(states)

    # Call the functions to generate the plots
    plot_update_by_prediction_error(prediction_error, update,context)
    plot_learning_rate_by_prediction_error(prediction_error, learning_rate, context)
    plot_states_and_learning_rate(true_state, predicted_state, learning_rate,context)
    plot_learning_rate_histogram(learning_rate,context)
    plot_lr_after_hazard(learning_rate, hazard_trials,context)
