import numpy as np
from physics_sim import PhysicsSim

class Env2TaskAdaptor():

    def __init__(self, env):
        self.env = env
        self.action_repeat = 3

        self.state_size = self.action_repeat * np.prod(self.env.observation_space.shape)
        self.action_low = env.action_space.low[0]
        self.action_high = env.action_space.high[0]
        self.action_size = np.prod(env.action_space.shape)

    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        observation_all = []
        for _ in range(self.action_repeat):
            observation, reward_from_one_step, done, info  = self.env.step(action)
            reward += reward_from_one_step 
            observation_all.append(observation)
        next_state = np.concatenate(observation_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        observation = self.env.reset()
        state = np.concatenate([observation] * self.action_repeat) 
        return state
