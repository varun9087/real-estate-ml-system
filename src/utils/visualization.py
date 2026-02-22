import matplotlib.pyplot as plt

def plot_cost(J_history):
    plt.plot(J_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost vs Iterations")
    plt.show()