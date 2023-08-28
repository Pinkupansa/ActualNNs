import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def atan(x):
    return np.arctan(x)/np.pi