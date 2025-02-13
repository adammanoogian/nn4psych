#%%

import numpy as np
import matplotlib.pyplot as plt
from utils_funcs import ActorCritic
import torch
from tasks import PIE_CP_OB_v2
from torch.distributions import Categorical
import glob


def get_area(model_path, epochs=100, reset_memory=0.0):
    hidden_dim = 64
    trials = 200

    model = ActorCritic(9, hidden_dim, 3)
    model.load_state_dict(torch.load(model_path))

    # print(f'Load Model {model_path}')
    contexts = ["change-point","oddball"] #"change-point","oddball"

    all_states = np.zeros([epochs, 2, 5, trials])
    for epoch in range(epochs):
        for tt, context in enumerate(contexts):
            env = PIE_CP_OB_v2(condition=context, max_time=300, total_trials=trials, 
                    train_cond=False, max_displacement=10, reward_size=2)
            
            hx = torch.randn(1, 1, hidden_dim) * 1/hidden_dim**0.5
            for trial in range(trials):

                next_obs, done = env.reset()
                norm_next_obs = env.normalize_states(next_obs)
                next_state = np.concatenate([norm_next_obs, env.context, np.array([0.0])])
                next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

                hx = hx.detach()
                # if trial_counter % reset_memory == 0:
                # if np.random.random_sample()< reset_memory:
                #     hx += (torch.randn(1, 1, hidden_dim) * 1/hidden_dim**0.5)

                while not done:

                    if np.random.random_sample()< reset_memory:
                        hx = (torch.randn(1, 1, hidden_dim) * 1/hidden_dim**0.5)

                    actor_logits, critic_value, hx = model(next_state, hx)
                    probs = Categorical(logits=actor_logits)
                    action = probs.sample()

                    # Take action and observe reward
                    next_obs, reward, done = env.step(action.item())

                    # Prep next state
                    norm_next_obs = env.normalize_states(next_obs)
                    next_state = np.concatenate([norm_next_obs, env.context, np.array([reward])])
                    next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

            all_states[epoch, tt] = np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])


    return np.array(all_states)




analysis = 'gamma'
epochs = 1

data_dir = "./model_params_101000/"
bias = False


if analysis == 'gamma':
    # influence of gamma

    gammas = [0.99, 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    all_param_states = {'gammas':gammas,'states':[]}

    for g, gamma in enumerate(gammas):
        
        file_names= data_dir+f"*_V3_{gamma}g_0.0rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        models = glob.glob(file_names)
        print(gamma, len(models))

        for m,model in enumerate(models):
            
            all_states = get_area(model, epochs=epochs)
            all_param_states['states'].append(all_states)



if analysis == 'rollout':
    # influence of rollout
    rollouts = [5, 10,20, 30, 40, 50, 75, 100, 150, 200] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    all_param_states = {'rollouts':rollouts,'states':[]}
    for g, rollout in enumerate(rollouts):

        file_names= data_dir+f"*_V3_0.95g_0.0rm_{rollout}bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        models = glob.glob(file_names)
        print(rollout, len(models))

        for m,model in enumerate(models):
            
            all_states = get_area(model, epochs=epochs)
            all_param_states['states'].append(all_states)

# introduce variables into sampling
if analysis == 'preset':
    # influence of rollout

    presets = [0.0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    all_param_states = {'presets':presets,'states':[]}
    for g, preset in enumerate(presets):

        file_names = data_dir+f"*_V3_0.95g_{preset}rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        models = glob.glob(file_names)
        print(preset, len(models))

        for m,model in enumerate(models):
            all_states = get_area(model, epochs=epochs)
            all_param_states['states'].append(all_states)



if analysis == 'scale':
    # influence of rollout

    scales = [0.1, 0.25, 0.5,0.75, 0.9, 1.0, 1.1, 1.25, 1.5] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    all_param_states = {'scales':scales,'states':[]}
    for g, scale in enumerate(scales):

        # file_names= data_dir+f"*_V5_0.95g_0.0rm_50bz_0.0td_{scale}tds_64n_50000e_10md_5.0rz_*s.pth"
        file_names = data_dir+f"*_V3_0.95g_0.0rm_100bz_0.0td_{scale}tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth" 
        models = glob.glob(file_names)


        for m,model in enumerate(models):
            all_states = get_area(model, epochs=epochs)
            all_param_states['states'].append(all_states)


