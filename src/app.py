import streamlit as st
import pandas as pd
from training.trainer import train
from models.linear_regression import predict

st.title(" Real Estate Price Predictor")
st.write("Phase 1 - Linear Regression (From Scratch)")

# Load dataset
data = pd.read_csv("../data/housing_v1.csv")
x = data["size"].values
y = data["price"].values

# Train model
w, b, _ = train(x, y, 0, 0, 0.01, 1000)

# User input
size = st.number_input("Enter house size (1000 sqft units):", min_value=0.5, max_value=5.0, step=0.1)

if st.button("Predict"):
    prediction = predict(size, w, b)
    st.success(f"Estimated Price: ${prediction * 1000:.2f}")
