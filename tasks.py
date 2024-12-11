#%%
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt


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


class PIE_CP_OB(gym.Env):
    def __init__(self, condition="change-point", total_trials=200):
        super(PIE_CP_OB, self).__init__()
        
        # Observation: currentCurrent bucket position, last bag position, and prediction error
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0], dtype=np.float32), 
                                            high=np.array([300.0, 300.0, 300.0], dtype=np.float32), 
                                            dtype=np.float32)
        self.maxobs = 300
        self.total_trials = total_trials
        # Initialize variables
        self.helicopter_pos = 150
        self.bucket_pos = 150
        self.bag_pos = self._generate_bag_position()
        self.obs = np.zeros(3)
        
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
    
    def normalize_states(self,x):
        # normalize states to be between -1 to 1 to feed to network
        # return np.array([x[0]/self.maxobs, x[1]/self.maxobs , x[2]/(self.maxobs/2)])
        return x/(self.maxobs/2) -1

    def reset(self):
        self.helicopter_pos = 150
        self.bucket_pos = 150
        self.bag_pos = self._generate_bag_position()  # why is bag position given to the agent?
        self.trial = 0
        
        # Reset data storage
        self.trials = []
        self.bucket_positions = []
        self.bag_positions = []
        self.helicopter_positions = []
        self.obs = np.array([self.bucket_pos, 150, 150], dtype=np.float32)  # initialize initial observation. assume bag = bucket
        self.done = False
        return self.obs
    
    def step(self, action):
        # idea is to have 2 separate phases within each trial. Phase 1: allow the agent to move the bucket to a desired position. Phase 2: press confirmation button to start bag drop

        # Phase 1:
        # Update bucket position based on action before confirmation
        if action == 0:  # Move left
            self.bucket_pos -= 1
        elif action == 1:  # Move right
            self.bucket_pos += 1
        self.bucket_pos = np.clip(self.bucket_pos, a_min=0,a_max=300)
        self.obs[0] = self.bucket_pos

        # no reward for moving bucket
        reward = 0

        # Phase 2:
        # confirm bucket position to start bag drop
        if action == 2:
            if self.task_type == "change-point":
                if np.random.rand() < self.change_point_hazard:
                    self.helicopter_pos = np.random.randint(30, 270)  # change helicopter position based on hazard rate
                self.bag_pos = self._generate_bag_position()  # Bag follows the stable helicopter position
            else:  # "oddball"
                # slow change in helicopter position in the oddball condition with small SD
                slow_shift = int(np.random.normal(0, 7.5))
                self.helicopter_pos += slow_shift
                self.helicopter_pos = np.clip(self.helicopter_pos,30,270)

                if np.random.rand() < self.oddball_hazard:
                    self.bag_pos = np.random.randint(0, 300)  # Oddball event
                else:
                    self.bag_pos = self._generate_bag_position()
        
            # Store positions for rendering
            self.trials.append(self.trial)
            self.bucket_positions.append(self.bucket_pos)
            self.bag_positions.append(self.bag_pos)
            self.helicopter_positions.append(self.helicopter_pos)

            # Calculate reward/negative prediction error that the agent maximizes for
            reward = 1*(abs(self.bag_pos - self.bucket_pos) <3)  # reward = 1 if bucket is close to bag pos for 10 units. Slower to train agent
            # reward = -abs(self.bag_pos - self.bucket_pos)/self.maxobs  # reward is negative scalar, proportional to distance between bucket and bag. Faster to train agent
            
            # Increment trial count
            self.trial += 1
        
            # Compute the new observation
            self.obs = np.array([self.bucket_pos, self.bag_pos, self.bag_pos - self.bucket_pos], dtype=np.float32)
            
            # Determine if the episode should end (e.g., after 100 trials)
            self.done = self.trial >= self.total_trials
        
        return self.obs, reward, self.done, {}
    
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
        plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='b',marker='o', linestyle='-.', alpha=0.5)

        plt.ylim(-10, 310)  # Set y-axis limit from 0 to 300
        plt.xlabel('Trial')
        plt.ylabel('Position')
        plt.title(f"Task: {self.task_type.capitalize()} Condition")
        plt.legend()
        plt.show()

    def close(self):
        pass


# Run
if __name__ == "__main__":
    trials = 100

    for task_type in ["change-point", "oddball"]:
        env = PIE_CP_OB(condition=task_type, total_trials=trials) #DiscretePredictiveInferenceEnv(condition=task_type)
            
        obs = env.reset()
        done = False
        total_reward = 0

        print(obs)
        
        while not done:
            action = env.action_space.sample()  # For testing, we use random actions
            obs, reward, done, _ = env.step(action)
            total_reward += reward

            print(action, obs, reward, done)
        
        env.render()
        print(f"Total Reward for {task_type.capitalize()} Condition: {total_reward}")
        env.close()