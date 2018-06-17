import numpy as np

class TotalRewardRecordingAgentWrapper():

    def __init__(self, delegate):
        self.delegate = delegate
        self.best_total_reward = -np.inf
        self.reset_episode()

    def reset_episode(self):
        state = self.delegate.reset_episode()
        self.total_reward = 0.0
        return state

    def step(self, action, reward, next_state, done):
        self.delegate.step(action, reward, next_state, done)
        self.total_reward += reward
        if done:
            self.best_total_reward = max(self.best_total_reward, self.total_reward)

    def act(self, state):
        return self.delegate.act(state)

