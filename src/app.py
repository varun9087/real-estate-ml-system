import streamlit as st
import numpy as np
import pandas as pd

from preprocessing.scaler import feature_scale, scale_input
from training.trainer import gradient_descent

# Load data
data = pd.read_csv("../data/housing_v1.csv")
X = data[["size", "bedrooms", "age"]].values
y = data["price"].values

# Scale and train
X_scaled, mu, sigma = feature_scale(X)
w, b, _ = gradient_descent(X_scaled, y, alpha=0.01, iterations=1000)

st.title("🏠 Real Estate ML System - Multi Feature")

size = st.number_input("Size (sqft)")
bedrooms = st.number_input("Bedrooms")
age = st.number_input("Age of house")

if st.button("Predict Price"):

    X_input = np.array([[size, bedrooms, age]])
    X_input_scaled = scale_input(X_input, mu, sigma)

    prediction = X_input_scaled.dot(w) + b

    st.success(f"Predicted Price: {prediction[0]:.2f}")