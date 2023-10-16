import numpy as np
from banditarm import BanditArm

NUM_TRIALS = 1000
class DecayingEpsilonGreedy:
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