import numpy as np

def predict(x, w, b):
    return w * x + b

def compute_cost(x, y, w, b):
    m = len(x)
    cost = 0
    for i in range(m):
        cost += (predict(x[i], w, b) - y[i]) ** 2
    return cost / (2 * m)

def compute_gradient(x, y, w, b):
    m = len(x)
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        error = predict(x[i], w, b) - y[i]
        dj_dw += error * x[i]
        dj_db += error

    return dj_dw / m, dj_db / m
