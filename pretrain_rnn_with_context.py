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
# Assuming that PIE_CP_OB is a gym-like environment
# from your_environment_file import PIE_CP_OB

# task parameters
n_epochs = 500  # number of epochs to train the model on. Similar to the number of times the agent is trained on the helicopter task. 
n_trials = 100  # number of trials per epoch for each condition.
max_time = 300  # number of time steps available for each trial. After max_time, the bag is dropped and the next trial begins after.

train_epochs = n_epochs*0.75  # number of epochs where the helicopter is shown to the agent. if 0, helicopter is never shown.
no_train_epochs = []  # epoch in which the agent weights are not updated using gradient descent. To see if the model can use its dynamics to solve the task instead.

contexts = ["change-point","oddball"] #"change-point","oddball"
num_contexts = len(contexts)
max_displacement = 30 # number of units each left or right moves.

# Model Parameters
input_dim = 4+2  # set this based on your observation space. observation vector is length 4 [helicopter pos, bucket pos, bag pos, bag-bucket pos], context vector is length 2.  
hidden_dim = 128  # size of RNN
action_dim = 3  # set this based on your action space. 0 is left, 1 is right, 2 is confirm.
learning_rate = 0.0001
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



def train(env, model, optimizer,epoch, n_trials, gamma):

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
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epoch_G = np.zeros([n_epochs, num_contexts, n_trials])
epoch_loss = np.zeros([n_epochs, num_contexts, n_trials])

for epoch in range(n_epochs):

    if epoch < train_epochs:
        train_cond = True  # give helicopter position for these epochs to simulate train condition
    else:
        train_cond = False # dont give helicopter position for these epochs to simulate test condition

    for tt, task_type in enumerate(contexts):

        env = PIE_CP_OB(condition=task_type, max_time=max_time, total_trials=n_trials, train_cond=train_cond, max_displacement=max_displacement)

        totG, totloss = train(env, model, optimizer, epoch=epoch, n_trials=n_trials, gamma=gamma)

        epoch_G[epoch, tt] = totG
        epoch_loss[epoch, tt] = totloss

        print(f"Epoch {epoch}, Task {task_type}, G {np.mean(totG):.3f}, L {np.mean(totloss):.3f}")

        if epoch == 0 or epoch == n_epochs-1:
            #save last epochs behav data
            states = env.render(epoch)


model_path = f'./model_params/model_params_{max_displacement}.pth'
torch.save(model.state_dict(), model_path)

model2 = ActorCritic(input_dim, hidden_dim, action_dim)
model2.load_state_dict(torch.load(model_path))

f,ax = plt.subplots(2,2,figsize=(3*2,2*2))
ax[0,0].plot(np.mean(epoch_G[:,0],axis=1), color='b', label='CP')
ax[0,0].plot(np.mean(epoch_G[:,1],axis=1),color='r',label='OB')
ax[0,0].legend()
ax[0,0].set_xlabel('Epoch')
ax[0,0].set_ylabel('G')

ax[0,1].plot(np.mean(epoch_loss[:,0],axis=1), color='b', label='CP')
ax[0,1].plot(np.mean(epoch_loss[:,1],axis=1),color='r',label='OB')
ax[0,1].legend()
ax[0,1].set_xlabel('Epoch')
ax[0,1].set_ylabel('Loss')

ax[1,0].plot(epoch_G[0,0], color='b')
ax[1,0].plot(epoch_G[0,1],color='r')
ax[1,0].set_xlabel('Epoch 0: Trial')
ax[1,0].set_ylabel('G')

ax[1,1].plot(epoch_G[-1,0], color='b')
ax[1,1].plot(epoch_G[-1,1],color='r')
ax[1,1].set_xlabel(f'Epoch {n_epochs}: Trial')
ax[1,1].set_ylabel('G')
f.tight_layout()

print(np.max(np.mean(epoch_G,axis=2),axis=0))


