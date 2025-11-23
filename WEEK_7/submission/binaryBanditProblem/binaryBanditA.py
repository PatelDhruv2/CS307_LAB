import numpy as np

def binaryBanditA():
    p = [0.1, 0.2]
    action = np.random.randint(0, 2)  
    return 1 if np.random.rand() < p[action] else 0