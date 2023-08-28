import numpy as np

def xor_dataset(N):
    # Generate a dataset of N random in [0, 1]^2 and their xor
    X = np.random.rand(2, N)
    Y = np.zeros(N)
    for i in range(N):
        if (X[0][i] < 0.5 and X[1][i] < 0.5) or (X[0][i] > 0.5 and X[1][i] > 0.5):
            Y[i] = 0
        else:
            Y[i] = 1
    return X, Y

def circle_dataset(N):
    X = np.random.rand(2, N)
    Y = np.zeros(N)
    for i in range(N):
        if (X[0][i] - 0.5)**2 + (X[1][i] - 0.5)**2 < 0.125:
            Y[i] = 1
        else:
            Y[i] = 0
    return X, Y

def linear_dataset(N):
    #random linear decision boundary
    X = np.random.rand(2, N)
    Y = np.zeros(N)
    for i in range(N):
        if X[1][i] < X[0][i]:
            Y[i] = 1
        else:
            Y[i] = 0
    return X, Y