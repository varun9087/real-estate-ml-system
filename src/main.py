import pandas as pd
from training.trainer import train
from models.linear_regression import predict
from utils.visualization import plot_cost

# Load data
data = pd.read_csv("../data/housing_v1.csv")
x = data["size"].values
y = data["price"].values

# Train model
w, b, cost_history = train(x, y, 0, 0, 0.01, 1000)

print("Training Complete")
print(f"w = {w:.4f}, b = {b:.4f}")

plot_cost(cost_history)
