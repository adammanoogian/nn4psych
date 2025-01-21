#%%
'''
Idea is to pretrain a vanilla RNN using RL on several epochs of the helicopter task 
so that it sort of knows what to do. 
The model weights are saved to run a single epoch of 100 trials of each condition, 
similar to Nassar et al. 2021 
and for additional analyses
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, required=False, help='epochs', default=1000)
parser.add_argument('--trials', type=int, required=False, help='trials', default=200)
parser.add_argument('--maxt', type=int, required=False, help='maxt', default=300)
parser.add_argument('--maxdisp', type=int, required=False, help='maxdisp', default=15)
parser.add_argument('--rewardsize', type=float, required=False, help='rewardsize', default=7.5)
parser.add_argument('--gamma', type=float, required=False, help='gamma', default=0.95)
parser.add_argument('--nrnn', type=int, required=False, help='nrnn', default=64)
parser.add_argument('--seed', type=int, required=False, help='seed', default=0)
parser.add_argument('--ratio', type=float, required=False, help='ratio', default=0.95)
parser.add_argument('--rollsz', type=int, required=False, help='rollsz', default=30)

parser.add_argument('--cliptd', type=int, required=False, help='rollsz', default=30)

args, unknown = parser.parse_known_args()
print(args)

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

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

device = torch.device("cpu")
print(device)

# Env parameters
n_epochs = args.epochs  # number of epochs to train the model on. Similar to the number of times the agent is trained on the helicopter task. 
n_trials = args.trials  # number of trials per epoch for each condition.

train_epochs = n_epochs*args.ratio #n_epochs*0.5  # number of epochs where the helicopter is shown to the agent. if 0, helicopter is never shown.
contexts = ["change-point","oddball"] #"change-point","oddball"
num_contexts = len(contexts)

# Task parameters
max_displacement = args.maxdisp # number of units each left or right moves.
max_time = args.maxt # int(5*300//max_displacement)
step_cost = 0 #-1/300  # penalize every additional step that the agent does not confirm. 
reward_size = args.rewardsize # smaller value means a tighter margin to get reward.
alpha = 1

# Model Parameters
input_dim = 6+2  # set this based on your observation space. observation vector is length 4 [helicopter pos, bucket pos, bag pos, bag-bucket pos], context vector is length 2.  
hidden_dim = args.nrnn  # size of RNN
action_dim = 3  # set this based on your action space. 0 is left, 1 is right, 2 is confirm.
params = hidden_dim*(input_dim+hidden_dim + action_dim+1)
learning_rate = 1/params/2.5
gamma = args.gamma
reset_memory = n_trials  # reset RNN activity after T trials
bias = [0, 0, 0]
beta_ent = 0.0
seed = args.seed

model_path = None
rollout_size = args.rollsz

np.random.seed(seed)
torch.manual_seed(seed)

model_path = None

exptname = f"V4_{rollout_size}roll_{n_epochs}e_{args.ratio}r_{reward_size}rz_{hidden_dim}n_{gamma}g_{max_displacement}md_{seed}s"
print(exptname)


# Actor-Critic Network with RNN
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
        self.dones = []

    def add_experience(self, state, action, reward, value, log_prob, entropy, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.entropies.clear()
        self.dones.clear()

def compute_advantages_and_returns(buffer, gamma, device):
    returns = []
    G = 0.0
    advantages = []
    buffer_size = len(buffer.rewards)
    
    for reward, value in zip(reversed(buffer.rewards), reversed(buffer.values)):
        G = reward + gamma * G
        returns.insert(0, G)
        advantages.insert(0, G - value.detach().cpu().numpy())

    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    log_probs = torch.stack(buffer.log_probs)
    values = torch.stack(buffer.values).squeeze()

    return returns, advantages, log_probs, values

def compute_td(buffer, gamma, device, next_value):

    buffer_size = len(buffer.rewards)
    tds = torch.zeros(buffer_size).to(device)

    for i in range(buffer_size):
        reward = buffer.rewards[i]
        value = buffer.values[i].detach().cpu().numpy()

        if i < buffer_size - 1:
            next_value = buffer.values[i+1].detach().cpu().numpy()
        else:
            next_value = 0  # Next value is 0 at terminal state.

        td = reward + gamma * next_value - value
        tds.append(td)

    returns = torch.tensor(buffer.rewards, dtype=torch.float32).to(device)
    tds = torch.tensor(tds, dtype=torch.float32).to(device)
    log_probs = torch.stack(buffer.log_probs)
    values = torch.stack(buffer.values).squeeze()

    return returns, tds, log_probs, values

def compute_gae(buffer, gamma, device, next_value, gae_lambda=0.95):
    rewards = torch.tensor(buffer.rewards, dtype=torch.float32).to(device)
    values = torch.tensor(buffer.values, dtype=torch.float32).to(device)
    dones = torch.tensor(buffer.dones, dtype=torch.float32).to(device)
    advantages = torch.zeros(len(rewards)).to(device)
    last_gae_lam = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[-1]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
    
    returns = advantages + values
    return advantages, returns

def train_with_rollouts(env, model, optimizer, epoch, n_trials, gamma, rollout_buffer_size):
    buffer = RolloutBuffer(rollout_buffer_size)

    totG = []
    totloss = []
    tottime = []

    hx = (torch.randn(1, 1, hidden_dim) * 1/hidden_dim).to(device)
    trial_counter = 0
    
    while trial_counter < n_trials:
    
        next_obs, done = env.reset()
        norm_next_obs = env.normalize_states(next_obs)
        next_state = np.concatenate([norm_next_obs, env.context])
        next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(device)

        # Detach hx to prevent backpropagation through the entire history
        hx = hx.detach()
        if trial_counter % reset_memory == 0:
            hx = (torch.randn(1, 1, hidden_dim) * 1/hidden_dim).to(device)

        totR = 0
        
        while not done:
            actor_logits, critic_value, hx = model(next_state, hx)
            bias_tensor = torch.tensor(bias, dtype=actor_logits.dtype, device=actor_logits.device)
            actor_logits += bias_tensor
            probs = Categorical(logits=actor_logits)
            action = probs.sample()

            # Take action and observe reward
            next_obs, reward, done = env.step(action.item())
            totR += reward

            # Append experiences to rollout buffer
            buffer.add_experience(next_state, action, reward, critic_value, probs.log_prob(action), probs.entropy(), done)

            # Prep next state
            norm_next_obs = env.normalize_states(next_obs)
            next_state = np.concatenate([norm_next_obs, env.context])
            next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(device)

        trial_counter += 1

        if len(buffer.rewards) >= rollout_buffer_size:
            # Computation for advantages and returns
            # returns, advantages, log_probs, values = compute_advantages_and_returns(buffer, gamma, device)
            with torch.no_grad():
                _, critic_value, _ = model(next_state, hx)
                tds, returns = compute_gae(buffer, gamma, device, critic_value)

            # Calculate losses
            log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32).to(device)
            actor_loss = -(log_probs * tds.detach()).mean()
            # critic_loss = (tds**2).mean()
            critic_loss = (returns - torch.stack(buffer.values).squeeze()).pow(2).mean()
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

gap = 5
for i,id in enumerate(idxs):

    plt.subplot(5,2,i+5)
    for c in range(2):

        pes = []
        lrs = []
        for g in range(gap):
            pes.append(all_pes[id-g, c])
            lrs.append(all_lrs[id-g,c])
        pes = np.concatenate(pes).flatten()
        lrs = np.concatenate(lrs).flatten()

        pes = pes[pes>=0]
        lrs = lrs[lrs>=0]

        idx = np.argsort(pes)
        pes = pes[idx]
        lrs = lrs[idx]

        plt.scatter(pes, lrs, color=colors[c],alpha=0.1, s=5)
        
        window_size = max_displacement*2
        smoothed_learning_rate = uniform_filter1d(lrs, size=window_size)
        plt.plot(pes, smoothed_learning_rate, color=colors[c], linewidth=5)
        print(np.mean(smoothed_learning_rate))

    plt.xlabel('Prediction Error')
    plt.ylabel('Learning Rate')
    plt.title(titles[i])

j=7
for i,id in enumerate(idxs):
    for c in range(2):
        plot_behavior(all_states[id, c], contexts[c], id, ax=plt.subplot(5,2,j))
        j+=1


plt.tight_layout()



if len(store_params)==3:
    plt.savefig(f'./figs/{exptname}.png')
    print('Fig saved')
    saveload('./state_data/'+exptname, [epoch_perf, all_states, all_pes, all_lrs, store_params], 'save')
    print('Data saved')