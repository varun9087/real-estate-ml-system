import numpy as np

def compute_cost(X, y, w, b):
    m = X.shape[0]
    predictions = X.dot(w) + b
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


def compute_gradient(X, y, w, b):
    m = X.shape[0]

    predictions = X.dot(w) + b
    errors = predictions - y

    dj_dw = (1 / m) * X.T.dot(errors)
    dj_db = (1 / m) * np.sum(errors)

    return dj_dw, dj_db