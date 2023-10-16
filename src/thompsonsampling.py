from Bandit import Bandit
import numpy as np
import matplotlib.pyplot as plt


class Thompsonbandit(Bandit):
    def sample(self):
        return np.random.beta(self.a, self.b)
    def update(self, x):
        self.a += x
        self.b += 1 - x
        self.N += 1

class Thompsonsampling:
    def experiment():
        bandits = [Thompsonbandit(p) for p in BANDIT_PROBABILITIES]
        sample_points = [5,10,20,50,100,200,500,1000,1500,1999]
        rewards = np.zeros(NUM_TRIALS)
        for i in range(NUM_TRIALS):
            j = np.argmax([b.sample() for b in bandits])
        # plot the posteriors
        if i in sample_points:
            plt(bandits, i)
        x = bandits[j].pull()
        rewards[i] = x
        bandits[j].update(x)
        # print total reward
        print(f"Total Reward Earned: {rewards.sum()}")
        print(f"Overall Win Rate: {rewards.sum() / NUM_TRIALS}")
        print(f"NUmber of times selected each bandit: {[b.N for b in bandits]}")