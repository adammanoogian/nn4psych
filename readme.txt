

Code outline

Main - no input into the RNN (just to make sure the structure is working)
Context- can put in reward, action, context input & integrate these together
RNN helicopter - all of above and synced to continuous helicopter task


tasks.py - holds the discrete / continuous helicopter environment. 
    -'step' to interact and get [bucket_pos, bag_pos, prediction_error] at each trial

pyem_models.py - contains current model fitting 

bayesian_models.py - contains (incomplete) PYMC model setup. also setup to simulate model predictions given priors

?pretrain_rnn_with_heli_v5.py - most recent pretraining of RNN

behav_figures.py - various behav figures




OUT OF DATE / USE? 

utils.funcs.py - holds generic actor critic model
utils.py - contains (old?) extract states equation, some calcs for PYMC model setup
analysis.py - contained prev Wang 2018 reproduction, may be out of data now
code_figures_behaviour - fig 2 Nassar


Workflow 

Setup RNN -> run with param sweep -> get set of models -> analyze behav or fp 

Main things used for analysis
all_states[epoch, tt] = np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])





#to-do 
clean up how data is saved / separate figures from running model
softcode actions to responses in task