import numpy as np

def feature_scale(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_scaled = (X - mu) / sigma
    return X_scaled, mu, sigma


def scale_input(X, mu, sigma):
    return (X - mu) / sigma