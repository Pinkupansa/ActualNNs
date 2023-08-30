import numpy as np

def mse(out_true, out_pred):
    return np.sum((out_true - out_pred)**2)/len(out_true)
def cross_entropy_loss(out_true, out_pred):
    epsilon = 1e-12
    out_pred = np.clip(out_pred, epsilon, 1. - epsilon)
    return -np.sum(out_true * np.log(out_pred))