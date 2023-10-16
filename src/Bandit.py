"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
import numpy as np
from logs import *

logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():

    def plot1(self):
        # Visualize the performance of each bandit: linear and log
        pass

    def plot2(self):
        # Compare E-greedy and thompson sampling cummulative rewards
        # Compare E-greedy and thompson sampling cummulative regrets
        pass

#--------------------------------------#

class EpsilonGreedy(Bandit):
    def experiment(bandit_probs,N=NUM_TRIALS):
        bandits = [BanditArm(p) for p in bandit_probs]

        means = np.array(bandit_probs) # count number of suboptimal choices
        true_best = np.argmax(means)  
        count_suboptimal = 0

        data = np.empty(N)

        for t in range(N):
            eps = 1/t if t!=0 else 1
            p = np.random.random()
            if p < eps:
                j = np.random.choice(len(bandits))
            else:
                j = np.argmax([b.p_estimate for b in bandits])
            x = bandits[j].pull()
            bandits[j].update(x)

            if j != true_best:
                count_suboptimal += 1

            # for the plot
            data[t] = x

        cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
        
        for b in bandits:
            print("Estimated average reward where epsilon={0}:{1}".format(eps,b.p_estimate))  
        print("Percent suboptimal where epsilon={0}: {1}".format( eps, float(count_suboptimal) / N))
        print("--------------------------------------------------")
        return cumulative_average

#--------------------------------------#

class ThompsonSampling(Bandit):
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0
    
    def __repr__(self):
        return 'An Arm with {} Win Rate'.format(self.p)
    
    def pull(self):
        return np.random.randn() + self.p

    def experiment(self):
        bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
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
    
    def sample(self):
        return np.random.beta(self.a, self.b)
    
    def update(self, x):
        self.a += x
        self.b += 1 - x
        self.N += 1


    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass


def comparison():
    # think of a way to compare the performances of the two algorithms VISUALLY and 
    pass

if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

