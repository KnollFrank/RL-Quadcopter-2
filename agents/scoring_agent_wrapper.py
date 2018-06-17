import numpy as np

class ScoringAgentWrapper():

    def __init__(self, delegate):
        self.delegate = delegate
        self.best_score = -np.inf
        self.reset_episode()

    def reset_episode(self):
        state = self.delegate.reset_episode()
        self.total_reward = 0.0
        self.count = 0
        return state

    def step(self, action, reward, next_state, done):
        self.delegate.step(action, reward, next_state, done)
        self.total_reward += reward
        self.count += 1
        if done:
            self.score = self.total_reward # / float(self.count)
            if self.score > self.best_score:
                self.best_score = self.score

    def act(self, state):
        return self.delegate.act(state)

