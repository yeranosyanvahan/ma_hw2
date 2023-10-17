from Bandit import Bandit
import numpy as np
import matplotlib.pyplot as plt


class Thompsonbandit(Bandit):
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0
    
    def __repr__(self):
        return 'An Arm with {} Win Rate'.format(self.p)
    
    def pull(self):
        return np.random.randn() + self.p
    
    def update(self, x):
        self.N += 1
        self.p_estimate = (1 - 1.0/self.N)*self.p_estimate + 1.0/self.N*x

    def sample(self):
        return np.random.beta(self.a, self.b)
    
    def update(self, x):
        self.a += x
        self.b += 1 - x
        self.N += 1

class Thompsonsampling:

    BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]
    NUM_TRIALS = 100000
    def experiment():
        bandits = [Thompsonbandit(p) for p in Thompsonsampling.BANDIT_PROBABILITIES]
        rewards = np.zeros(NUM_TRIALS)
        for i in range(Thompsonsampling.NUM_TRIALS):
            j = np.argmax([b.sample() for b in bandits])
            x = bandits[j].pull()
            rewards[i] = x
            bandits[j].update(x)

        return rewards
