from models.linear_regression import compute_gradient, compute_cost

def train(x, y, w_init, b_init, alpha, iterations):
    w = w_init
    b = b_init
    cost_history = []

    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

        cost = compute_cost(x, y, w, b)
        cost_history.append(cost)

    return w, b, cost_history
