#%%
'''
Idea is to pretrain a vanilla RNN using RL on several epochs of the helicopter task 
so that it sort of knows what to do. 
The model weights are saved to run a single epoch of 100 trials of each condition, 
similar to Nassar et al. 2021 
and for additional analyses
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from tasks import PIE_CP_OB_v2
import matplotlib.pyplot as plt
from torch.nn import init
from utils_funcs import get_lrs_v2, saveload, plot_behavior
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
from copy import deepcopy
# Assuming that PIE_CP_OB is a gym-like environment
# from your_environment_file import PIE_CP_OB

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
device = torch.device("cpu")



# Env parameters
n_epochs = 2  # number of epochs to train the model on. Similar to the number of times the agent is trained on the helicopter task. 
n_trials = 200  # number of trials per epoch for each condition.
trratio = 0.5

train_epochs = n_epochs*trratio #n_epochs*0.5  # number of epochs where the helicopter is shown to the agent. if 0, helicopter is never shown.
no_train_epochs = []  # epoch in which the agent weights are not updated using gradient descent. To see if the model can use its dynamics to solve the task instead.
contexts = ["change-point","oddball"] #"change-point","oddball"
num_contexts = len(contexts)

# Task parameters
max_displacement = 10 # number of units each left or right moves.
max_time = 300
step_cost = 0 #-1/300  # penalize every additional step that the agent does not confirm. 
reward_size = 5 # smaller value means a tighter margin to get reward.
alpha = 1

# Model Parameters
input_dim = 6+3  # set this based on your observation space. observation vector is length 4 [helicopter pos, bucket pos, bag pos, bag-bucket pos], context vector is length 2.  
hidden_dim = 64  # size of RNN
action_dim = 3  # set this based on your action space. 0 is left, 1 is right, 2 is confirm.
params = hidden_dim*(input_dim+hidden_dim + action_dim+1)
bias = [0, 0, 0]
beta_ent = 0.0

# knobs
gamma = 0.95
rollout_size = 10
reset_memory = 1.0  # reset RNN activity after T trials
tdnoise = 0.0
tdlb = None
tdub = None

learning_rate = 0.0001/100 * rollout_size 

seed = 2025
np.random.seed(seed)
torch.manual_seed(seed)

model_path = None

exptname = f"v3_{n_epochs}e_{trratio}r_{hidden_dim}n_{gamma}g_{learning_rate}lr_{max_displacement}md_{reward_size}rz_{seed}s"
print(exptname)

# Actor-Critic Network with RNN
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, gain=1.0):
        super(ActorCritic, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gain =gain
        bias = False
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity='tanh', bias=bias)
        self.actor = nn.Linear(hidden_dim, action_dim, bias=bias)
        self.critic = nn.Linear(hidden_dim, 1, bias=bias)

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
    
def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


class RolloutBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.entropies = []

    def add_experience(self, state, action, reward, value, log_prob, entropy):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.entropies.clear()

def compute_advantages_and_returns(buffer, gamma, device):
    returns = []
    G = 0.0
    advantages = []
    buffer_size = len(buffer.rewards)


    for reward, value in zip(reversed(buffer.rewards), reversed(buffer.values)):
        G = reward + gamma * G
        returns.insert(0, G)

        # Calculate TD error
        td_error = G - value.detach().cpu().numpy()

        # noise = np.random.normal(0, tdnoise, size=td_error.shape)
        noise = np.random.gamma(1.0,1.0, size=td_error.shape) * tdnoise
        td_error += noise

        # Apply bounds if specified
        if tdlb is not None or tdub is not None:
            td_error = np.clip(td_error, tdlb, tdub)

        advantages.insert(0, td_error)

    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    log_probs = torch.stack(buffer.log_probs)
    values = torch.stack(buffer.values).squeeze()

    return returns, advantages, log_probs, values

def train_with_rollouts(env, model, optimizer, epoch, n_trials, gamma, rollout_buffer_size):
    buffer = RolloutBuffer(rollout_buffer_size)

    totG = []
    totloss = []
    tottime = []

    hx = (torch.randn(1, 1, hidden_dim) * 1/hidden_dim**0.5).to(device)
    trial_counter = 0
    
    while trial_counter < n_trials:
    
        next_obs, done = env.reset()
        norm_next_obs = env.normalize_states(next_obs)
        next_state = np.concatenate([norm_next_obs, env.context, np.array([0.0])])
        next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(device)

        # Detach hx to prevent backpropagation through the entire history
        hx = hx.detach()
        # if trial_counter % reset_memory == 0:
        # if np.random.random_sample()< reset_memory:
        #     hx += (torch.randn(1, 1, hidden_dim) * 1/hidden_dim).to(device)
        totR = 0
        
        while not done:

            if np.random.random_sample()< reset_memory:
                hx = (torch.randn(1, 1, hidden_dim) * 1/hidden_dim).to(device)

            actor_logits, critic_value, hx = model(next_state, hx)
            bias_tensor = torch.tensor(bias, dtype=actor_logits.dtype, device=actor_logits.device)
            actor_logits += bias_tensor
            probs = Categorical(logits=actor_logits)
            action = probs.sample()

            # Take action and observe reward
            next_obs, reward, done = env.step(action.item())
            totR += reward

            # Append experiences to rollout buffer
            buffer.add_experience(next_state, action, reward, critic_value, probs.log_prob(action), probs.entropy())

            # Prep next state
            norm_next_obs = env.normalize_states(next_obs)
            next_state = np.concatenate([norm_next_obs, env.context, np.array([reward])])
            next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(device)

        trial_counter += 1

        if len(buffer.rewards) >= rollout_buffer_size:
            # Computation for advantages and returns
            returns, advantages, log_probs, values = compute_advantages_and_returns(buffer, gamma, device)

            # Calculate losses
            actor_loss = -(log_probs * advantages).mean()
            critic_loss = ((returns - values)**2).mean()
            entropy_loss = -torch.stack(buffer.entropies).mean()

            loss = actor_loss + 0.5 * critic_loss + beta_ent * entropy_loss

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            buffer.clear()

            totloss.append(to_numpy(loss))

        totG.append(totR)
        tottime.append(env.time)

    return np.array(totG), np.array(totloss), np.array(tottime)

# initialize untrained model
model = ActorCritic(input_dim, hidden_dim, action_dim).to(device)
store_params = []
store_params.append(deepcopy(model.state_dict()))
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # initialize optimizer for training

# load pretrained, if any
if model_path is not None:
    print('Load Model')
    model.load_state_dict(torch.load(model_path))

# store variables
epoch_perf = np.zeros([n_epochs, num_contexts, 4, n_trials])
all_states = np.zeros([n_epochs, num_contexts, 5, n_trials])
all_lrs = np.zeros([n_epochs, num_contexts, n_trials-1])
all_pes = np.zeros([n_epochs, num_contexts, n_trials-1])


# training loop
for epoch in range(n_epochs):

    if epoch < train_epochs:
        train_cond = True  # give helicopter position for these epochs to simulate train condition
    else:
        train_cond = False # dont give helicopter position for these epochs to simulate test condition
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # reinitialize optimizer for test conditon

    idxs = np.random.choice(np.arange(2), size=2, replace=False)  # randomize CP and OB conditions

    for tt in idxs:

        task_type = contexts[tt]

        env = PIE_CP_OB_v2(condition=task_type, max_time=max_time, total_trials=n_trials, 
                        train_cond=train_cond, max_displacement=max_displacement, reward_size=reward_size, step_cost=step_cost, alpha=alpha)

        totG, totloss,tottime = train_with_rollouts(env, model, optimizer, epoch=epoch, n_trials=n_trials, gamma=gamma, rollout_buffer_size=rollout_size)
        totloss = np.tile(np.mean(totloss),200)

        # save performance
        all_states[epoch, tt] = np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])
        all_pes[epoch,tt], all_lrs[epoch, tt] = get_lrs_v2(all_states[epoch, tt])
        perf = abs(all_states[epoch, tt, 3] - all_states[epoch, tt,1])
        epoch_perf[epoch,tt] = np.array([totG, totloss, tottime, perf])

        print(f"Epoch {epoch}, Task {task_type}, G {np.mean(totG):.3f}, d {np.mean(epoch_perf[epoch,tt, 3]):.3f}, s {np.mean(all_lrs[epoch, tt]):.3f}")

        if epoch == n_epochs-1 or epoch == train_epochs-1:
            #plot last epochs behav data
            _ = env.render(epoch)


    perf = np.mean(abs(all_states[epoch,:, 3] - all_states[epoch,:,1]))
    if epoch == train_epochs-1 and perf < 32:
        store_params.append(deepcopy(model.state_dict()))
        # model_path = f'./model_params/Trueheli_{epoch+1}e_{exptname}.pth'
        # torch.save(model.state_dict(), model_path)

    if epoch == n_epochs-1 and len(store_params)>1:
        store_params.append(deepcopy(model.state_dict()))
        # model_path = f'./model_params/Falseheli_{epoch+1}e_{exptname}.pth'
        # torch.save(model.state_dict(), model_path)

colors = ['orange', 'brown']
labels = ['CP', 'OB']
plt.figure(figsize=(4*2,3*5))

plt.subplot(521)
for i in range(2):
    plt.plot(np.mean(epoch_perf[:,i,3],axis=1), color=colors[i], label=labels[i])
plt.xlabel('Epoch')
plt.ylabel('Heli - Bucket error') # should become more positive.
plt.axhline(32, color='r')
plt.axvline(train_epochs, color='b')

plt.subplot(522)
for i in range(2):
    plt.plot(np.mean(epoch_perf[:,i, 0],axis=1), color=colors[i], label=labels[i])
plt.xlabel('Epoch')
plt.ylabel('G')
plt.axvline(train_epochs, color='b')


plt.subplot(523)
for i in range(2):
    plt.plot(np.mean(epoch_perf[:,i,2],axis=1), color=colors[i], label=labels[i])
plt.xlabel('Epoch')
plt.ylabel('Time to Confirm')
plt.axvline(train_epochs, color='b')

scores = np.mean(all_lrs[:,0],axis=1) - np.mean(all_lrs[:,1],axis=1)
plt.subplot(524)
plt.plot(scores, color='tab:green')
slope, intercept, r_value, p_value, std_err = linregress(np.arange(n_epochs), scores)
plt.plot(slope*np.arange(n_epochs)+intercept, color='k')
plt.xlabel('Epoch')
plt.ylabel('Avg LR Diff') # should become more positive.
plt.title(f'm={slope:.3f}, R={r_value:.3f}, p={p_value:.3f}')

idxs = [int(train_epochs)-1, int(n_epochs)-1]
titles = ['With Heli', 'Without Heli']

for i,id in enumerate(idxs):

    plt.subplot(5,2,i+5)
    for c in range(2):

        pes = all_pes[id, c]
        lrs = all_lrs[id,c]

        pes = pes[pes>=0]
        lrs = lrs[lrs>=0]

        plt.scatter(pes, lrs, color=colors[c],alpha=0.1, s=2)
        
        window_size = 30
        smoothed_learning_rate = uniform_filter1d(lrs, size=window_size)
        plt.plot(pes, smoothed_learning_rate, color=colors[c], linewidth=5)
        print(np.mean(smoothed_learning_rate))

    plt.xlabel('Prediction Error')
    plt.ylabel('Learning Rate')
    plt.title(titles[i])

j=7
for c in range(2):
    for i,id in enumerate(idxs):
        plot_behavior(all_states[id, c], contexts[c], id, ax=plt.subplot(5,2,j))
        j+=1
plt.tight_layout()


if len(store_params)==3:
    plt.savefig(f'./figs/{exptname}.png')
    print('Fig saved')
    saveload('./state_data/'+exptname, [epoch_perf, all_states, all_pes, all_lrs, store_params], 'save')
    print('Data saved')
    model_path = f'./model_params/Falseheli_{epoch+1}e_{exptname}.pth'
    torch.save(model.state_dict(), model_path)
