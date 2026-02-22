import numpy as np
from models.linear_regression import compute_cost, compute_gradient

def gradient_descent(X, y, alpha=0.01, iterations=1000):

    m, n = X.shape
    w = np.zeros(n)
    b = 0

    J_history = []

    for i in range(iterations):

        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(X, y, w, b)
        J_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost:.4f}")

    return w, b, J_history