#%%
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import copy

class DiscretePredictiveInferenceEnv(gym.Env):
    def __init__(self, condition="change-point"):
        super(DiscretePredictiveInferenceEnv, self).__init__()
        
        self.action_space = spaces.Discrete(5)        
        # Observation: currentCurrent bucket position, last bag position, and prediction error
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), 
                                            high=np.array([4, 4, 4]), dtype=np.float32)
        
        # Initialize variables
        self.helicopter_pos = 2
        self.bucket_pos = 2
        self.bag_pos = self._generate_bag_position()
        
        # Task type: "change-point" or "oddball"
        self.task_type = condition
        
        # Trial counter and data storage for rendering
        self.trial = 0
        self.trials = []
        self.bucket_positions = []
        self.bag_positions = []

        # Hazard rates for the different conditions
        self.change_point_hazard = 0.125
        self.oddball_hazard = 0.125

    def reset(self):
        self.helicopter_pos = 2
        self.bucket_pos = 2
        self.bag_pos = self._generate_bag_position()
        self.trial = 0
        
        # Reset data storage
        self.trials = []
        self.bucket_positions = []
        self.bag_positions = []
        self.helicopter_positions = []
        
        return np.array([self.bucket_pos, self.bag_pos, abs(self.bag_pos - self.bucket_pos)], dtype=np.float32)

    def step(self, action):
        # Update bucket position based on action
        if action == 0:
            self.bucket_pos = 0
        elif action == 1:
            self.bucket_pos = 1
        elif action == 2:
            self.bucket_pos = 2
        elif action == 3:
            self.bucket_pos = 3
        elif action == 4:
            self.bucket_pos = 4
        
        # Determine bag position based on task type
        if self.task_type == "change-point":
            if np.random.rand() < self.change_point_hazard:
                self.helicopter_pos = np.random.randint(0, 4)
            self.bag_pos = self._generate_bag_position()  # Bag follows the stable helicopter position
        else:  # "oddball"
            if np.random.rand() < self.oddball_hazard:
                self.bag_pos = np.random.randint(0, 4)  # Oddball event
            else:
                self.bag_pos = self._generate_bag_position()
        
        # Store positions for rendering
        self.trials.append(self.trial)
        self.bucket_positions.append(self.bucket_pos)
        self.bag_positions.append(self.bag_pos)
        self.helicopter_positions.append(self.helicopter_pos)

        # Calculate reward
        reward = 1-abs(self.bag_pos - self.bucket_pos)
        
        # Increment trial count
        self.trial += 1
        
        # Compute the new observation
        observation = np.array([self.bucket_pos, self.bag_pos, abs(self.bag_pos - self.bucket_pos)], dtype=np.float32)
        
        # Determine if the episode should end (e.g., after 100 trials)
        done = self.trial >= 100
        
        return observation, reward, done, {}
    
    def _generate_bag_position(self):
        """Generate a new bag position around the current helicopter location within bounds."""
        bag_pos = self.helicopter_pos
        # add or subtract 1 from the helicopter position at 20% chance if at 0 or 4; 10% if in 1,2,3
        if self.helicopter_pos == 0:
            if np.random.rand() < 0.2:
                bag_pos += 1
        elif self.helicopter_pos == 4:
            if np.random.rand() < 0.2:
                bag_pos -= 1
        else: # 1,2,3: 80% chance to stay, 10% move left, or 10% right
            if np.random.rand() < 0.8:
                pass
            elif np.random.rand() < 0.5:
                bag_pos -= 1
            else:
                bag_pos += 1
        
        # Ensure the bag position is within the 0-300 range
        return max(0, min(4, bag_pos))

    def render(self, mode='human'):
        plt.figure(figsize=(10, 6))
        plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='blue')
        plt.plot(self.trials, self.bag_positions, label='Bag Position', color='red', marker='o', linestyle='-.', alpha=0.5)
        plt.plot(self.trials, self.helicopter_positions, label='Helicopter', color='green', linestyle='--')

        plt.ylim(-.2, 4.2)  # Set y-axis limit from 0 to 300
        plt.xlabel('Trial')
        plt.ylabel('Position')
        plt.title(f"Task: {self.task_type.capitalize()} Condition")
        plt.legend()
        plt.show()

    def close(self):
        pass

class ContinuousPredictiveInferenceEnv(gym.Env):
    def __init__(self, condition="change-point", total_trials=200):
        super(ContinuousPredictiveInferenceEnv, self).__init__()
        
        # Observation: currentCurrent bucket position, last bag position, and prediction error
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), 
                                            high=np.array([300, 300, 300]), dtype=np.float32)
        self.total_trials = total_trials
        # Initialize variables
        self.helicopter_pos = 150
        self.bucket_pos = 150
        self.bag_pos = self._generate_bag_position()
        
        # Task type: "change-point" or "oddball"
        self.task_type = condition
        
        # Trial counter and data storage for rendering
        self.trial = 0
        self.trials = []
        self.bucket_positions = []
        self.bag_positions = []

        # Hazard rates for the different conditions
        self.change_point_hazard = 0.125
        self.oddball_hazard = 0.125

    def reset(self):
        self.helicopter_pos = 150
        self.bucket_pos = 150
        self.bag_pos = self._generate_bag_position()
        self.trial = 0
        
        # Reset data storage
        self.trials = []
        self.bucket_positions = []
        self.bag_positions = []
        self.helicopter_positions = []
        
        return np.array([self.bucket_pos, self.bag_pos, self.bag_pos - self.bucket_pos], dtype=np.float32)

    def step(self, action):
        # Update bucket position based on action
        if action == 0:  # Move left
            self.bucket_pos = max(0, self.bucket_pos - 30)
        elif action == 1:  # Move right
            self.bucket_pos = min(300, self.bucket_pos + 30)
        
        # Determine bag position based on task type
        if self.task_type == "change-point":
            if np.random.rand() < self.change_point_hazard:
                self.helicopter_pos = np.random.randint(30, 270)
            self.bag_pos = self._generate_bag_position()  # Bag follows the stable helicopter position
        else:  # "oddball"
            if np.random.rand() < self.oddball_hazard:
                self.bag_pos = np.random.randint(0, 300)  # Oddball event
            else:
                self.bag_pos = self._generate_bag_position()
        
        # Store positions for rendering
        self.trials.append(self.trial)
        self.bucket_positions.append(self.bucket_pos)
        self.bag_positions.append(self.bag_pos)
        self.helicopter_positions.append(self.helicopter_pos)

        # Calculate reward
        reward = -abs(self.bag_pos - self.bucket_pos)
        
        # Increment trial count
        self.trial += 1
        
        # Compute the new observation
        observation = np.array([self.bucket_pos, self.bag_pos, self.bag_pos - self.bucket_pos], dtype=np.float32)
        
        # Determine if the episode should end (e.g., after 100 trials)
        done = self.trial >= self.total_trials
        
        return observation, reward, done, {}
    
    def _generate_bag_position(self):
        """Generate a new bag position around the current helicopter location within bounds."""
        bag_pos = int(np.random.normal(self.helicopter_pos, 20))
        # Ensure the bag position is within the 0-300 range
        return max(0, min(300, bag_pos))

    def render(self, mode='human'):
        plt.figure(figsize=(10, 6))
        # plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='blue')
        plt.plot(self.trials, self.bag_positions, label='Bag Position', color='red', marker='o', linestyle='-.', alpha=0.5)
        plt.plot(self.trials, self.helicopter_positions, label='Helicopter', color='green', linestyle='--')

        plt.ylim(-10, 310)  # Set y-axis limit from 0 to 300
        plt.xlabel('Trial')
        plt.ylabel('Position')
        plt.title(f"Task: {self.task_type.capitalize()} Condition")
        plt.legend()
        plt.show()

    def close(self):
        pass


class PIE_CP_OB:
    def __init__(self, condition="change-point", total_trials=100,max_time=300, train_cond=False, max_displacement=30):
        super(PIE_CP_OB, self).__init__()
        
        # Observation: currentCurrent bucket position, last bag position, and prediction error
        self.action_space = spaces.Discrete(3)
        self.observation_space = 4 # helicopter pos during training, bucket pos, bag pos, bag-bucket pos, CP or OB context
        self.max_time = max_time

        self.min_obs_size = 1
        self.max_obs_size = 301
        self.bound_helicopter = 30
        self.total_trials = total_trials
        self.hide_variable = 0 # this means that if a variable is 0, it is supposed to be hidden from the agent's consideration

        # Initialize variables
        self.helicopter_pos =  np.random.randint(self.min_obs_size+self.bound_helicopter, self.max_obs_size-self.bound_helicopter)
        self.bucket_pos = np.random.randint(self.min_obs_size+self.bound_helicopter, self.max_obs_size-self.bound_helicopter)
        self.prev_bag_pos = copy.copy(self.hide_variable)
        self.prev_pred_error = copy.copy(self.hide_variable)
        self.sample_bag_pos = self._generate_bag_position(self.helicopter_pos)
        self.reward = 0
        self.max_disp = max_displacement #changed to 1 from 30

        # Task type: "change-point" or "oddball"
        self.task_type = condition
        self.train_cond = train_cond  # either True or False, if True, helicopter position is shown to agent. if False helicopter position is 0

        if condition == "change-point":
            self.context =  np.array([1,0])
        elif condition == "oddball":
            self.context =  np.array([0,1])

        
        # Trial counter and data storage for rendering
        self.trial = 0
        self.trials = []
        self.bucket_positions = []
        self.bag_positions = []
        self.helicopter_positions = []
        self.hazard_triggers = []

        # Hazard rates for the different conditions
        self.change_point_hazard = 0.125
        self.oddball_hazard = 0.125
    
    def normalize_states_(self,x):
        # normalize states to be between -1 to 1 to feed to network
        # return np.array([x[0]/self.maxobs, x[1]/self.maxobs , x[2]/(self.maxobs/2)])
        ranges = np.array([[0, 300], [0, 300], [0, 300], [-300, 300]])
        normalized_vector = np.array([2 * (x[i] - ranges[i, 0]) / (ranges[i, 1] - ranges[i, 0]) - 1 for i in range(len(x))])
        return normalized_vector
    
    def normalize_states(self,x):
        # normalize states to be between 0 to 1 to feed to network
        return x/300

    def reset(self):
        # reset at the start of every trial. Observation inclues: helicopter 
        self.hazard_trigger = 0

        if self.task_type == "change-point":
            if np.random.rand() < self.change_point_hazard:
                self.helicopter_pos = np.random.randint(30, 270)  # change helicopter position based on hazard rate
                self.hazard_trigger = 1
            self.sample_bag_pos = self._generate_bag_position(self.helicopter_pos)  # Bag follows the stable helicopter position

        else:  # "oddball"

            # slow change in helicopter position in the oddball condition with small SD
            slow_shift = int(np.random.normal(0, 7.5))
            self.helicopter_pos += slow_shift
            self.helicopter_pos = np.clip(self.helicopter_pos, self.min_obs_size + self.bound_helicopter,self.max_obs_size-self.bound_helicopter)

            if np.random.rand() < self.oddball_hazard:
                self.sample_bag_pos = np.random.randint(0, 300)  # Oddball event
                self.hazard_trigger = 1
            else:
                self.sample_bag_pos = self._generate_bag_position(self.helicopter_pos)
        self.time = 0

        if self.train_cond:
            self.obs = np.array([self.helicopter_pos, self.bucket_pos, copy.copy(self.hide_variable), self.prev_pred_error], dtype=np.float32)  # initialize initial observation. assume bag = bucket
        else:
            self.obs = np.array([copy.copy(self.hide_variable), self.bucket_pos, copy.copy(self.hide_variable), self.prev_pred_error], dtype=np.float32)  # initialize initial observation. assume bag = bucket

        self.done = False
        self.bag_dropped = False
        return self.obs, self.done
    
    def step(self, action):
        # idea is to have 2 separate phases within each trial. Phase 1: allow the agent to move the bucket to a desired position. Phase 2: press confirmation button to start bag drop

        # Phase 1:
        # Update bucket position based on action before confirmation
        self.gt = 0
        if action == 0: 
            # Move left
            self.gt = -self.max_disp
        if action == 1:
            # Move right
            self.gt = self.max_disp

        self.bucket_pos += self.gt
        self.bucket_pos = np.clip(self.bucket_pos, a_min=self.min_obs_size,a_max=self.max_obs_size)
        self.obs = copy.copy(self.obs)
        self.obs[1] = self.bucket_pos
        # reward = 0

        # if self.bag_dropped:
        #     # Increment trial count
        #     self.trial += 1
        #     self.done = True
        #     reward = -abs(self.prev_bag_pos - self.bucket_pos)/self.max_obs_size  # reward is negative scalar, proportional to distance between bucket and bag. Faster to train agent
        #     # self.gt = 0
        
        if self.time>= self.max_time:
            self.done = True

        # Phase 2:
        self.reward = 0# -1/self.max_obs_size # punish for every timestep
        # confirm bucket position to start bag drop
        if action == 2 or self.time >= self.max_time-1:
            self.prev_bag_pos = copy.copy(self.sample_bag_pos)
            self.prev_pred_error = self.prev_bag_pos - self.bucket_pos     

            # Compute the new observation
            if self.train_cond:
                self.obs = np.array([self.helicopter_pos, self.bucket_pos, self.prev_bag_pos, self.prev_pred_error], dtype=np.float32)  # initialize initial observation. assume bag = bucket
            else:
                self.obs = np.array([copy.copy(self.hide_variable), self.bucket_pos, self.prev_bag_pos, self.prev_pred_error], dtype=np.float32)  # initialize initial observation. assume bag = bucket
            
            # Calculate reward/negative prediction error that the agent maximizes for
            # self.reward = np.random.choice(np.arange(1,4),1)*(abs(self.prev_bag_pos - self.bucket_pos) <20)  # reward = 1 if bucket is close to bag pos for 10 units. Slower to train agent
            # self.reward = -abs(self.prev_bag_pos - self.bucket_pos)/self.max_obs_size  # reward is negative scalar, proportional to distance between bucket and bag. Faster to train agent
            
            # reward or punish inactivity
            if np.random.uniform()<0.0:
                # randomly punish for not catching bag
                self.reward = -abs(self.prev_bag_pos - self.bucket_pos)/self.max_obs_size  # reward is negative scalar, proportional to distance between bucket and bag. Faster to train agent
            else:
                # reward follows gaussian distribution. the closer the bucket is to the bag positin, the higher the reward.
                df = ((self.prev_bag_pos - self.bucket_pos)/30)**2
                self.reward = np.exp(-0.5*df)
                # self.reward = np.random.randint(1,4)*(abs(self.prev_bag_pos - self.bucket_pos) <20)  # reward = 1 if bucket is close to bag pos for 10 units. Slower to train agent

            # penalize if agent doesnt choose to confirm
            if self.time >= self.max_time-1:
                self.reward -= 1
                # self.reward =0
                # print(f'T {self.trial}, t {self.time}, -- Penalize')

            self.trial += 1
            self.done = True

            # Store positions for rendering
            self.trials.append(self.trial)
            self.bucket_positions.append(self.bucket_pos)
            self.bag_positions.append(self.prev_bag_pos)
            self.helicopter_positions.append(self.helicopter_pos)
            self.hazard_triggers.append(self.hazard_trigger)
            self.bag_dropped = True

        self.time += 1
        # reward = -abs(self.prev_bag_pos - self.bucket_pos)/self.max_obs_size  # reward is negative scalar, proportional to distance between bucket and bag. Faster to train agent
        return self.obs, self.reward, self.done
    
    def _generate_bag_position(self, helicopter_pos):
        """Generate a new bag position around the current helicopter location within bounds."""
        bag_pos = int(np.random.normal(helicopter_pos, 20))
        # Ensure the bag position is within the 0-300 range
        return np.clip(bag_pos, 0,self.max_obs_size)

    def render(self, epoch=0):
        plt.figure(figsize=(10, 6))
        # plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='blue')
        plt.plot(self.trials, self.bag_positions, label='Bag Position', color='red', marker='o', linestyle='-.', alpha=0.5)
        plt.plot(self.trials, self.helicopter_positions, label='Helicopter', color='green', linestyle='--')
        plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='b',marker='o', linestyle='-.', alpha=0.5)

        plt.ylim(-10, 310)  # Set y-axis limit from 0 to 300
        plt.xlabel('Trial')
        plt.ylabel('Position')
        plt.title(f"Task: {self.task_type.capitalize()} Condition - Epoch: {epoch}")
        plt.legend()
        plt.show()

        return np.array([self.trials, self.bucket_positions, self.bag_positions, self.helicopter_positions, self.hazard_triggers])


# Run
if __name__ == "__main__":

    # for task_type in ["change-point", "oddball"]:
    #     env = ContinuousPredictiveInferenceEnv(condition=task_type,) #DiscretePredictiveInferenceEnv(condition=task_type)
        
    #     obs = env.reset()
    #     done = False
    #     total_reward = 0

    #     print(obs)
        
    #     while not done:
    #         action = env.action_space.sample()  # For testing, we use random actions
    #         obs, reward, done,_ = env.step(action)
    #         total_reward += reward

    #         print(env.trial, env.time, action, obs, reward, done)
        
    #     env.render()
    #     print(f"Total Reward for {task_type.capitalize()} Condition: {total_reward}")
    #     env.close()


    store_obs = []
    store_obs_ = []
    trials = 100
    train_cond = True
    max_time = 300
    for task_type in ["change-point"]:
        env = PIE_CP_OB(condition=task_type,max_time=max_time, total_trials=trials, train_cond=train_cond) #DiscretePredictiveInferenceEnv(condition=task_type)
        
        for trial in range(trials):
            obs, done = env.reset()
            
            while not done:
                action = env.action_space.sample()  # For testing, we use random actions

                # action = np.random.choice(np.arange(2),1)
                next_obs, reward, done = env.step(action)

                print(env.trial, env.time, obs, action, next_obs, reward, done)

                obs = copy.copy(next_obs)

                store_obs.append(env.normalize_states(obs))
                store_obs_.append(env.normalize_states_(obs))
            
        env.render()

    store_obs = np.array(store_obs)
    store_obs_ = np.array(store_obs_)
    plt.figure()
    plt.hist(store_obs.reshape(-1), alpha=0.2, bins=100)
    plt.hist(store_obs_.reshape(-1), alpha=0.2, bins=100)
    plt.title(f'Max {np.max(store_obs):.3f}, Min {np.min(store_obs):.3f} \n Max {np.max(store_obs_):.3f}, Min {np.min(store_obs_):.3f}')
# %%
