import pandas as pd
import numpy as np

from preprocessing.scaler import feature_scale
from training.trainer import gradient_descent
from utils.visualization import plot_cost

# Load dataset
data = pd.read_csv("../data/housing_v1.csv")

# Separate features and target
X = data[["size", "bedrooms", "age"]].values
y = data["price"].values

# Feature scaling
X_scaled, mu, sigma = feature_scale(X)

# Train model
w, b, J_history = gradient_descent(X_scaled, y, alpha=0.01, iterations=1000)

print("\nTraining Finished")
print(f"Final weights: {w}")
print(f"Final bias: {b}")

# Plot cost vs iterations
plot_cost(J_history)

# Example prediction
example = np.array([[2000, 4, 3]])
example_scaled = (example - mu) / sigma
prediction = example_scaled.dot(w) + b

print(f"\nPrediction for 2000 sqft house: {prediction[0]:.2f}")