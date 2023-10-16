import numpy as np

class BanditArm:
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

