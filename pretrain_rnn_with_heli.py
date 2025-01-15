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
from tasks import PIE_CP_OB
import matplotlib.pyplot as plt
from torch.nn import init
from utils_funcs import plot_analysis, get_lrs
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
# Assuming that PIE_CP_OB is a gym-like environment
# from your_environment_file import PIE_CP_OB

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
device = torch.device("cpu")



# Env parameters
n_epochs = 1000  # number of epochs to train the model on. Similar to the number of times the agent is trained on the helicopter task. 
n_trials = 200  # number of trials per epoch for each condition.
max_time = 300  # number of time steps available for each trial. After max_time, the bag is dropped and the next trial begins after.

train_epochs = n_epochs*0.5 #n_epochs*0.5  # number of epochs where the helicopter is shown to the agent. if 0, helicopter is never shown.
no_train_epochs = []  # epoch in which the agent weights are not updated using gradient descent. To see if the model can use its dynamics to solve the task instead.
contexts = ["change-point","oddball"] #"change-point","oddball"
num_contexts = len(contexts)

# Task parameters
max_displacement = 15 # number of units each left or right moves.
step_cost = 0 #-1/300  # penalize every additional step that the agent does not confirm. 
reward_size = 5 # smaller value means a tighter margin to get reward.
alpha = 1

# Model Parameters
input_dim = 4+2  # set this based on your observation space. observation vector is length 4 [helicopter pos, bucket pos, bag pos, bag-bucket pos], context vector is length 2.  
hidden_dim = 64  # size of RNN
action_dim = 3  # set this based on your action space. 0 is left, 1 is right, 2 is confirm.
params = hidden_dim*(input_dim+hidden_dim + action_dim+1)
learning_rate = 1/params
gamma = 0.95
reset_memory = n_trials  # reset RNN activity after T trials
bias = [0, 0, 0]
beta_ent = 0.0

model_path = None#f'./model_params/pre_model_params_{max_displacement}_heliTrue.pth'

exptname = f"{n_trials}t_{max_displacement}md_{reward_size}rz_{hidden_dim}n_{gamma}g_{learning_rate}lr"
print(exptname)

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
                init.normal_(param, mean=0, std=1/(self.input_dim))
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



def train(env, model, optimizer,epoch, n_trials, gamma):

    totG = []
    totloss = []
    tottime = []

    hx = (torch.randn(1, 1, hidden_dim) * 0.00).to(device)
    for trial in range(n_trials):

        next_obs, done = env.reset()  # reset env at the start of every trial to change helocopter pos based on hazard rate

        norm_next_obs = env.normalize_states(next_obs)  # normalize vector to bound between something resonable for the RNN to handle
        next_state = np.concatenate([norm_next_obs,env.context])
        next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(device)  # add batch and seq dim

        # Detach hx to prevent backpropagation through the entire history
        hx = hx.detach()
        if trial % reset_memory == 0:
            # Initialize the RNN hidden state
            hx = (torch.randn(1, 1, hidden_dim) * 0.00).to(device)

        log_probs = []
        values = []
        rewards = []
        entropies = []
        totR = 0 

        while not done: #allows multiple actions in one trial (incrementally moving bag_position)

            # choose action given state
            actor_logits, critic_value, hx = model(next_state, hx)
            bias_tensor = torch.tensor(bias, dtype=actor_logits.dtype, device=actor_logits.device)
            actor_logits += bias_tensor
            probs = Categorical(logits=actor_logits)
            action = probs.sample()

            # take action in env
            next_obs, reward, done = env.step(action.item())

            # log state, action, reward
            log_probs.append(probs.log_prob(action))
            values.append(critic_value)
            entropies.append(probs.entropy())
            rewards.append(reward)
            totR += reward

            # prep next state
            norm_next_obs = env.normalize_states(next_obs)
            next_state = np.concatenate([norm_next_obs,env.context])
            next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(device)

        
        # print("trial:", env.trial, "time:", env.time, "obs:", obs, "actor:", actor_logits, "action:", action, "reward:", reward, "next_obs:", next_obs)

        # Calculate returns
        returns = []
        G = 0.0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze()

        # Compute advantages
        advantages = returns - values.detach()

        # Calculate loss
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = ((returns - values)**2).mean()
        entropy_loss = -torch.stack(entropies).mean()

        loss = actor_loss + 0.5 * critic_loss + beta_ent * entropy_loss

        #  train network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totG.append(totR)
        totloss.append(to_numpy(loss))
        tottime.append(env.time)

    return np.array(totG), np.array(totloss), np.array(tottime)

# initialize untrained model
model = ActorCritic(input_dim, hidden_dim, action_dim).to(device)
store_params = []
store_params.append(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # initialize optimizer for training

# load pretrained, if any
if model_path is not None:
    print('Load Model')
    model.load_state_dict(torch.load(model_path))

# store variables
epoch_G = np.zeros([n_epochs, num_contexts, n_trials])
epoch_loss = np.zeros([n_epochs, num_contexts, n_trials])
epoch_time = np.zeros([n_epochs, num_contexts, n_trials])
all_states = np.zeros([n_epochs, num_contexts, 5, n_trials])
all_lrs = np.zeros([n_epochs, num_contexts, n_trials-1])
all_pes = np.zeros([n_epochs, num_contexts, n_trials-1])
all_scores = np.zeros([n_epochs,num_contexts])


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

        env = PIE_CP_OB(condition=task_type, max_time=max_time, total_trials=n_trials, 
                        train_cond=train_cond, max_displacement=max_displacement, reward_size=reward_size, step_cost=step_cost, alpha=alpha)

        totG, totloss,tottime = train(env, model, optimizer, epoch=epoch, n_trials=n_trials, gamma=gamma)

        epoch_G[epoch, tt] = totG
        epoch_loss[epoch, tt] = totloss
        epoch_time[epoch, tt] = tottime

        all_states[epoch, tt] = np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])
        all_pes[epoch,tt], all_lrs[epoch, tt] = get_lrs(all_states[epoch, tt])
        idx = all_pes[epoch,tt]>max_displacement
        all_scores[epoch, tt] = np.trapz(all_lrs[epoch, tt][idx], all_pes[epoch,tt][idx])

        print(f"Epoch {epoch}, Task {task_type}, G {np.mean(totG):.3f}, t {np.mean(tottime):.3f}, s {np.mean(all_scores[epoch, tt]):.3f}")

        if epoch == n_epochs-1 or epoch == train_epochs-1:
            #plot last epochs behav data
            _ = env.render(epoch)


    perf = np.mean(abs(all_states[epoch,:, 3] - all_states[epoch,:,1]))
    if epoch == train_epochs-1 and perf < 15:
        store_params.append(model.state_dict())
        # model_path = f'./model_params/Trueheli_{epoch+1}e_{exptname}.pth'
        # torch.save(model.state_dict(), model_path)

    if epoch == n_epochs-1 and perf < 15:
        store_params.append(model.state_dict())
        # model_path = f'./model_params/Falseheli_{epoch+1}e_{exptname}.pth'
        # torch.save(model.state_dict(), model_path)


# Calculate the difference in learning rates between CP and OB conditions. Should be positive. 
cp_vs_ob = plot_analysis(epoch_G, epoch_loss, epoch_time, all_states[-1])

plt.figure(figsize=(3,2*2))
scores = all_scores[:,0] - all_scores[:,1]
plt.subplot(311)
plt.plot(scores)
plt.xlabel('Epoch')
plt.ylabel('CP vs OB') # should become more positive.

perf = np.mean(abs(all_states[:,:, 3] - all_states[:,:,1]),axis=2)
plt.subplot(312)
plt.plot(perf)
plt.xlabel('Epoch')
plt.ylabel('Heli-Bucket') # should become more positive.
plt.axhline(max_displacement*2, color='k')
plt.tight_layout()

plt.figure(figsize=(3*2,2))
gap = 10
idxs = [int(train_epochs)-1, int(n_epochs)-1]
colors = ['orange', 'brown']
for i,id in enumerate(idxs):
    plt.subplot(1,2,i+1)
    for c in range(2):
        pes = []
        lrs = []
        for s,states in enumerate(all_states[id-gap:id]):
            pe, lr = get_lrs(states[c])
            pes.append(pe)
            lrs.append(lr)
        pes = np.array(pes).reshape(-1)
        lrs = np.array(lrs).reshape(-1)

        sorted_indices = np.argsort(pes)
        prediction_error_sorted = pes[sorted_indices]
        learning_rate_sorted = lrs[sorted_indices]
        idx = np.argmax(prediction_error_sorted>15)
        
        window_size = 15
        smoothed_learning_rate = uniform_filter1d(learning_rate_sorted, size=window_size)
        plt.plot(prediction_error_sorted[idx:], smoothed_learning_rate[idx:], color=colors[c])
        print(np.trapz(smoothed_learning_rate[idx:], prediction_error_sorted[idx:]))

    plt.xlabel('Prediction Error')
    plt.ylabel('Learning Rate')
plt.tight_layout()

