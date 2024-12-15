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
train_cond = False
max_time = 300
contexts = ["change-point","oddball"] #"change-point","oddball"
num_contexts = len(contexts)

# Model Parameters
input_dim = 4+2  # set this based on your observation space
hidden_dim = 128
action_dim = 3  # set this based on your action space
learning_rate = 0.0001

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
    


def train(env, model, optimizer, n_trials=500, gamma=0.99):

    totG = []
    for _ in range(n_trials):
        obs, done = env.reset()
        norm_obs = env.normalize_states(obs)
        state = np.concatenate([norm_obs,env.context])
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # add batch and seq dim

        # Initialize the RNN hidden state
        hx = torch.zeros(1, 1, hidden_dim)

        log_probs = []
        values = []
        rewards = []

        while not done:
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

        # Take a policy step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totG.append(G)
    return np.array(totG)


model = ActorCritic(input_dim, hidden_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epoch_G = np.zeros([n_epochs, num_contexts, n_trials])

for epoch in range(n_epochs):

    for tt, task_type in enumerate(contexts):

        env = PIE_CP_OB(condition=task_type, max_time=max_time, total_trials=n_trials, train_cond=train_cond)

        totG = train(env, model, optimizer, n_trials=n_trials)

        epoch_G[epoch, tt] = totG

        print(f"Epoch {epoch}, Task {task_type}, G {np.mean(totG)}")

        if epoch == 0 or epoch == n_epochs-1:
            env.render()


f,ax = plt.subplots(3,1,figsize=(3,2*3))
ax[0].plot(np.mean(epoch_G[:,0],axis=1), color='b', label='CP')
ax[0].plot(np.mean(epoch_G[:,1],axis=1),color='r',label='OB')
ax[0].legend()
ax[0].set_xlabel('Epoch')
ax[1].plot(epoch_G[0,0], color='b')
ax[1].plot(epoch_G[0,1],color='r')
ax[1].set_xlabel('Epoch 0: Trial')
ax[2].plot(epoch_G[-1,0], color='b')
ax[2].plot(epoch_G[-1,1],color='r')
ax[2].set_xlabel(f'Epoch {n_epochs}: Trial')
f.tight_layout()

print(np.max(np.mean(epoch_G,axis=2),axis=0))