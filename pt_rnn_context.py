#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from tasks import PIE_CP_OB
import matplotlib.pyplot as plt

# Assuming that PIE_CP_OB is a gym-like environment
# from your_environment_file import PIE_CP_OB

# task parameters
n_epochs = 100
n_trials = 100
max_time = 300
train_epochs = n_epochs*0.5
no_train_epochs = []
contexts = ["change-point","oddball"] #"change-point","oddball"
num_contexts = len(contexts)

# Model Parameters
input_dim = 4+2  # set this based on your observation space
hidden_dim = 128
action_dim = 3  # set this based on your action space
learning_rate = 0.0001
gamma = 0.99
reset_memory = 20  # reset RNN activity after trials

# Actor-Critic Network with RNN
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hx):
        x, hx = self.rnn(x, hx)
        x = x.squeeze(1)
        return self.actor(x), self.critic(x), hx
    
def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def train(env, model, optimizer,epoch, n_trials, gamma):

    totG = []
    store_vars = []
    hx = torch.randn(1, 1, hidden_dim) *0.1

    for trial in range(n_trials):
        obs, done = env.reset()
        norm_obs = env.normalize_states(obs)
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

        store_s=[]
        store_h = []
        store_a = []
        store_v = []

        while not done:
            # Forward pass
            actor_logits, critic_value, hx = model(state, hx)
            probs = Categorical(logits=actor_logits)
            action = probs.sample()

            log_probs.append(probs.log_prob(action))
            values.append(critic_value)

            store_s.append(to_numpy(state))
            store_h.append(to_numpy(hx))
            store_a.append(action.item())
            store_v.append(to_numpy(critic_value))

            next_obs, reward, done = env.step(action.item())
            norm_next_obs = env.normalize_states(next_obs)
            next_state = np.concatenate([norm_next_obs,env.context])
            rewards.append(reward)
            totR += reward

            # print(env.trial, env.time, obs,actor_logits, action, reward, next_obs)

            state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

            if done:
                break

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

        if epoch+1 in no_train_epochs:
            store_vars.append([store_s, store_h, store_a, store_v])
        else:
            #  train network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        totG.append(totR)

    return np.array(totG), store_vars


model = ActorCritic(input_dim, hidden_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epoch_G = np.zeros([n_epochs, num_contexts, n_trials])

for epoch in range(n_epochs):

    if epoch < train_epochs:
        train_cond = True  # give helicopter position for these epochs to simulate train condition
    else:
        train_cond = False # dont give helicopter position for these epochs to simulate test condition

    for tt, task_type in enumerate(contexts):

        env = PIE_CP_OB(condition=task_type, max_time=max_time, total_trials=n_trials, train_cond=train_cond)

        totG, store_vars = train(env, model, optimizer, epoch=epoch, n_trials=n_trials, gamma=gamma)

        epoch_G[epoch, tt] = totG

        print(f"Epoch {epoch}, Task {task_type}, G {np.mean(totG)}")

        if epoch == 0 or epoch == n_epochs-1:
            env.render()


f,ax = plt.subplots(3,1,figsize=(3,2*3))
ax[0].plot(np.mean(epoch_G[:,0],axis=1), color='b', label='CP')
ax[0].plot(np.mean(epoch_G[:,1],axis=1),color='r',label='OB')
ax[0].legend()
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('G')
ax[1].plot(epoch_G[0,0], color='b')
ax[1].plot(epoch_G[0,1],color='r')
ax[1].set_xlabel('Epoch 0: Trial')
ax[1].set_ylabel('G')
ax[2].plot(epoch_G[-1,0], color='b')
ax[2].plot(epoch_G[-1,1],color='r')
ax[2].set_xlabel(f'Epoch {n_epochs}: Trial')
ax[2].set_ylabel('G')
f.tight_layout()

print(np.max(np.mean(epoch_G,axis=2),axis=0))