#%%

import numpy as np
import matplotlib.pyplot as plt
from utils_funcs import ActorCritic
import torch
from tasks import PIE_CP_OB_v2
from torch.distributions import Categorical
import glob

import os
import pickle



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




analysis = 'all'
epochs = 1 #must adjust format of saved variables if you increase from 1

data_dir = "./model_params_101000/"
save_dir = "data/rnn_behav/model_params_101000/"
os.makedirs(save_dir, exist_ok=True)
bias = False


if analysis == 'gamma' or "all":
    # influence of gamma

    gammas = [0.99, 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    all_param_states = {'gammas':gammas,'states':[]}

    gamma_dict = {}
    gamma_cp_list = [] 
    gamma_ob_list = []
    
    for g, gamma in enumerate(gammas):
        
        file_names= data_dir+f"*_V3_{gamma}g_0.0rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        models = glob.glob(file_names)
        print(gamma, len(models))

        for m, model in enumerate(models):
            
            all_states = get_area(model, epochs=epochs)
            all_param_states['states'].append(all_states)

            gamma_dict[m, g] = {"gamma", gamma}
            gamma_cp_list.append(all_states[0,0])
            gamma_ob_list.append(all_states[0,1])   

            with open(os.path.join(save_dir, "gamma_all_param_states.pkl"), "wb") as f:
                pickle.dump(all_param_states, f)
            with open(os.path.join(save_dir, "gamma_dict.pkl"), "wb") as f:
                pickle.dump(gamma_dict, f)
            with open(os.path.join(save_dir, "gamma_cp_list.pkl"), "wb") as f:
                pickle.dump(gamma_cp_list, f)
            with open(os.path.join(save_dir, "gamma_ob_list.pkl"), "wb") as f:
                pickle.dump(gamma_ob_list, f)


if analysis == 'rollout' or 'all':
    # influence of rollout
    rollouts = [5, 10,20, 30, 40, 50, 75, 100, 150, 200] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    all_param_states = {'rollouts':rollouts,'states':[]}

    rollout_dict = {}
    rollout_cp_list = [] 
    rollout_ob_list = []
    
    for g, rollout in enumerate(rollouts):

        file_names= data_dir+f"*_V3_0.95g_0.0rm_{rollout}bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        models = glob.glob(file_names)
        print(rollout, len(models))

        for m,model in enumerate(models):
            
            all_states = get_area(model, epochs=epochs)
            all_param_states['states'].append(all_states)

            rollout_dict[m, g] = {"rollout", rollout}
            rollout_cp_list.append(all_states[0,0])
            rollout_ob_list.append(all_states[0,1])   

            with open(os.path.join(save_dir, "rollout_all_param_states.pkl"), "wb") as f:
                pickle.dump(all_param_states, f)
            with open(os.path.join(save_dir, "rollout_dict.pkl"), "wb") as f:
                pickle.dump(rollout_dict, f)
            with open(os.path.join(save_dir, "rollout_cp_list.pkl"), "wb") as f:
                pickle.dump(rollout_cp_list, f)
            with open(os.path.join(save_dir, "rollout_ob_list.pkl"), "wb") as f:
                pickle.dump(rollout_ob_list, f)

# introduce variables into sampling
if analysis == 'preset' or 'all':
    # influence of rollout

    presets = [0.0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    all_param_states = {'presets':presets,'states':[]}

    preset_dict = {}
    preset_cp_list = [] 
    preset_ob_list = []
    
    for g, preset in enumerate(presets):

        file_names = data_dir+f"*_V3_0.95g_{preset}rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        models = glob.glob(file_names)
        print(preset, len(models))

        for m,model in enumerate(models):
            all_states = get_area(model, epochs=epochs)
            all_param_states['states'].append(all_states)

            preset_dict[m, g] = {"preset", preset}
            preset_cp_list.append(all_states[0,0])
            preset_ob_list.append(all_states[0,1])   

            with open(os.path.join(save_dir, "preset_all_param_states.pkl"), "wb") as f:
                pickle.dump(all_param_states, f)
            with open(os.path.join(save_dir, "preset_dict.pkl"), "wb") as f:
                pickle.dump(preset_dict, f)
            with open(os.path.join(save_dir, "preset_cp_list.pkl"), "wb") as f:
                pickle.dump(preset_cp_list, f)
            with open(os.path.join(save_dir, "preset_ob_list.pkl"), "wb") as f:
                pickle.dump(preset_ob_list, f)

if analysis == 'scale' or 'all':
    # influence of rollout

    scales = [0.1, 0.25, 0.5,0.75, 0.9, 1.0, 1.1, 1.25, 1.5] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    all_param_states = {'scales':scales,'states':[]}

    scale_dict = {}
    scale_cp_list = [] 
    scale_ob_list = []
    
    for g, scale in enumerate(scales):

        # file_names= data_dir+f"*_V5_0.95g_0.0rm_50bz_0.0td_{scale}tds_64n_50000e_10md_5.0rz_*s.pth"
        file_names = data_dir+f"*_V3_0.95g_0.0rm_100bz_0.0td_{scale}tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth" 
        models = glob.glob(file_names)


        for m,model in enumerate(models):
            all_states = get_area(model, epochs=epochs)
            all_param_states['states'].append(all_states)

            scale_dict[m, g] = {"scale", scale}
            scale_cp_list.append(all_states[0,0])
            scale_ob_list.append(all_states[0,1])   

            with open(os.path.join(save_dir, "scale_all_param_states.pkl"), "wb") as f:
                pickle.dump(all_param_states, f)
            with open(os.path.join(save_dir, "scale_dict.pkl"), "wb") as f:
                pickle.dump(scale_dict, f)
            with open(os.path.join(save_dir, "scale_cp_list.pkl"), "wb") as f:
                pickle.dump(scale_cp_list, f)
            with open(os.path.join(save_dir, "scale_ob_list.pkl"), "wb") as f:
                pickle.dump(scale_ob_list, f)

#combined_dict doesn't work because of overlapping keys

# if analysis == 'all': 
#     combined_dict = {**gamma_dict, **rollout_dict, **preset_dict, **scale_dict}
#     cp_array = [gamma_cp_list, rollout_cp_list, preset_cp_list, scale_cp_list]
#     ob_array = [gamma_ob_list, rollout_ob_list, preset_ob_list, scale_ob_list]

#     with open(os.path.join(save_dir, "combined_dict.pkl"), "wb") as f:
#         pickle.dump(combined_dict, f)
#     with open(os.path.join(save_dir, "combined_cp_array.pkl"), "wb") as f:
#         pickle.dump(cp_array, f)
#     with open(os.path.join(save_dir, "combined_ob_array.pkl"), "wb") as f:
#         pickle.dump(ob_array, f)
