import numpy as np

def binaryBanditB():
    p = [0.8, 0.9]
    action = np.random.randint(0, 2) 
    return 1 if np.random.rand() < p[action] else 0