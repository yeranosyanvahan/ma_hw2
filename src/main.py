from banditarm import BanditArm 
from banditarm import BanditArm as Bandit
import numpy as np
#from Bandit import Bandit

class EpsilonGreedy(Bandit):
    def __init__(self, p, epsilon):
        self.epsilon = epsilon
        super().__init__(p)

    # Implement Epsilon Greedy experiment
    def experiment(self, num_trials):
        # Initialize bandit arms
        arms = [BanditArm(p) for p in self.p]
        self.arms = arms

        # Lists to store rewards and optimal actions
        rewards = []
        optimal_actions = []

        for t in range(1, num_trials + 1):
            # Exploration vs. exploitation
            if np.random.rand() < self.epsilon:
                # Explore: choose a random arm
                arm_idx = np.random.choice(len(arms))
            else:
                # Exploit: choose the arm with the highest estimated reward
                arm_idx = np.argmax([arm.p_estimate for arm in arms])

            # Pull the selected arm
            reward = arms[arm_idx].pull()
            arms[arm_idx].update(reward)

            # Calculate regret (difference between optimal and selected arm)
            optimal_arm_idx = np.argmax([arm.p for arm in arms])
            regret = arms[optimal_arm_idx].p - arms[arm_idx].p

            # Store results
            rewards.append(reward)
            optimal_actions.append(1 if arm_idx == optimal_arm_idx else 0)

        return rewards, optimal_actions

    def report(self, num_trials):
        rewards, optimal_actions = self.experiment(num_trials)

        # Calculate cumulative rewards and regrets
        cumulative_rewards = np.cumsum(rewards)
        cumulative_regrets = np.cumsum([arm.p - max([a.p for a in arms]) for arm, a in zip(arms, self.p)])

        # Visualize the learning process
        plt.plot(cumulative_rewards)
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.title("Epsilon Greedy Learning")
        plt.show()

        # Store data in a CSV file
        data = {'Bandit': self.p, 'Reward': rewards, 'Algorithm': 'Epsilon Greedy'}
        df = pd.DataFrame(data)
        df.to_csv('epsilon_greedy_results.csv')

        # Print cumulative reward and regret
        print(f"Cumulative Reward: {cumulative_rewards[-1]}")
        print(f"Cumulative Regret: {cumulative_regrets[-1]}")


class ThompsonSampling(Bandit):
    def __init__(self, p, precision):
        self.precision = precision
        super().__init__(p)

    # Implement Thompson Sampling experiment
    def experiment(self, num_trials):
        # Initialize bandit arms
        arms = [BanditArm(p) for p in self.p]

        # Lists to store rewards and optimal actions
        rewards = []
        optimal_actions = []

        for t in range(1, num_trials + 1):
            # Sample from Beta distribution for each arm
            samples = [np.random.beta(arm.N + 1, arm.N + 1) for arm in arms]

            # Choose the arm with the highest sampled value
            arm_idx = np.argmax(samples)

            # Pull the selected arm
            reward = arms[arm_idx].pull()
            arms[arm_idx].update(reward)

            # Calculate regret (difference between optimal and selected arm)
            optimal_arm_idx = np.argmax([arm.p for arm in arms])
            regret = arms[optimal_arm_idx].p - arms[arm_idx].p

            # Store results
            rewards.append(reward)
            optimal_actions.append(1 if arm_idx == optimal_arm_idx else 0)

        return rewards, optimal_actions

    def report(self, num_trials):
        rewards, optimal_actions = self.experiment(num_trials)

        # Calculate cumulative rewards and regrets
        cumulative_rewards = np.cumsum(rewards)
        cumulative_regrets = np.cumsum([arm.p - max([a.p for a in arms]) for arm, a in zip(arms, self.p)])

        # Visualize the learning process
        plt.plot(cumulative_rewards)
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.title("Thompson Sampling Learning")
        plt.show()

        # Store data in a CSV file
        data = {'Bandit': self.p, 'Reward': rewards, 'Algorithm': 'Thompson Sampling'}
        df = pd.DataFrame(data)
        df.to_csv('thompson_sampling_results.csv')

        # Print cumulative reward and regret
        print(f"Cumulative Reward: {cumulative_rewards[-1]}")
        print(f"Cumulative Regret: {cumulative_regrets[-1]}")


if __name__ == '__main__':
    Bandit_Reward = [1, 2, 3, 4]
    NumberOfTrials = 20000

    epsilon = 0.1  # Choose an appropriate epsilon value
    epsilon_greedy_bandit = EpsilonGreedy(Bandit_Reward, epsilon)
    epsilon_greedy_bandit.report(NumberOfTrials)

    precision = 0.01  # Choose an appropriate precision value
    thompson_sampling_bandit = ThompsonSampling(Bandit_Reward, precision)
    thompson_sampling_bandit.report(NumberOfTrials)