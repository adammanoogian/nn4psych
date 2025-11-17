

Code outline

Main - no input into the RNN (just to make sure the structure is working)
Context- can put in reward, action, context input & integrate these together
RNN helicopter - all of above and synced to continuous helicopter task


analysis.py - analysis of rnn unit activations (incomplete)
analyze_compiled.py - compiles and plots data from best run
analyze_hyperparams_gamma

tasks.py - holds the discrete / continuous helicopter environment. 
    -'step' to interact and get [bucket_pos, bag_pos, prediction_error] at each trial

pyem_models.py - contains current model fitting

bayesian_models.py - contains (incomplete) PYMC model setup. also setup to simulate model predictions given priors

?pretrain_rnn_with_heli_v5.py - most recent pretraining of RNN

behav_figures.py - various behav figures for individual and batch_data

utils.py - calcs for PYMC model setup, unpickle state vector, filter to exclude models based on performance

get_behavior.py - runs trained models through task to reproduce and save behavior

?compile.py - (can be merged into utils?)

OUT OF DATE / USE? 

utils.funcs.py - holds generic actor critic model, lr and plotting functions (out of date?)
analysis.py - contained prev Wang 2018 reproduction, may be out of data now
code_figures_behaviour - fig 2 Nassar
behav_figures.py - first ~4 sets of figures that are for one run at a time (maybe keep for paper)
analyze_normative.py - bayesian model sim and plotting? (out of date? or put elsewhere? )

Workflow 

Setup RNN -> run with param sweep -> get set of models -> analyze behav or fp 

Main things used for analysis- 

'state vector' - 
all_states[epoch, tt] = np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])

all RNN hyper parameters - 

    gammas  = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25, 0.1]
    rollouts = [5, 10, 20, 30, 50, 75, 100, 150, 200] #skipping 40 
    presets = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    scales  = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

Trained model weight paths - 
    -number at beginning of string indicates the ΔArea (on last epoch?) (proxy for performance)
    -'val' = 
    "gamma": (
        gammas,
        "*_V3_{val}g_0.0rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
    ),
    "rollout": (
        rollouts,
        "*_V3_0.95g_0.0rm_{val}bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
    ),
    "preset": (
        presets,
        "*_V3_0.95g_{val}rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
    ),
    "scale": (
        scales,
        "*_V3_0.95g_0.0rm_100bz_0.0td_{val}tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"



#to-do 
clean up how data is saved / separate figures from running model
softcode actions to responses in task