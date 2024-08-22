import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class ContinuousPredictiveInferenceEnv(gym.Env):
    def __init__(self, condition="change-point"):
        super(PredictiveInferenceEnv, self).__init__()
        
        # Observation: currentCurrent bucket position, last bag position, and prediction error
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), 
                                            high=np.array([300, 300, 300]), dtype=np.float32)
        
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
            self.bucket_pos = max(0, self.bucket_pos - 1)
        elif action == 1:  # Move right
            self.bucket_pos = min(300, self.bucket_pos + 1)
        
        # Determine bag position based on task type
        if self.task_type == "change-point":
            if np.random.rand() < self.change_point_hazard:
                self.helicopter_pos = np.random.randint(0, 300)
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
        done = self.trial >= 100
        
        return observation, reward, done, {}
    
    def _generate_bag_position(self):
        """Generate a new bag position around the current helicopter location within bounds."""
        bag_pos = int(np.random.normal(self.helicopter_pos, 20))
        # Ensure the bag position is within the 0-300 range
        return max(0, min(300, bag_pos))

    def render(self, mode='human'):
        plt.figure(figsize=(10, 6))
        plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='blue')
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

# Run
for task_type in ["change-point", "oddball"]:
    env = ContinuousPredictiveInferenceEnv(condition=task_type)
        
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = env.action_space.sample()  # For testing, we use random actions
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    
    env.render()
    print(f"Total Reward for {task_type.capitalize()} Condition: {total_reward}")
    env.close()