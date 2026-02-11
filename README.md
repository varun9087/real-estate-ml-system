# 🏠 Real Estate ML System – Phase 1

This project is the first phase of a long-term Machine Learning system designed to predict real estate prices based on property size.

In this phase, Linear Regression is implemented completely **from scratch using Gradient Descent**, without using machine learning libraries like scikit-learn. The system also includes a **Streamlit-based web interface** for real-time price prediction.

---

## 🚀 Version Information

- Version: 1.0
- Model: Linear Regression (Single Feature)
- Optimization: Gradient Descent
- Dataset: Custom Housing Dataset
- Interface: Streamlit Web App

---

## 🎯 Problem Statement

Build a scalable machine learning system that predicts house prices based on input features.  
Phase 1 focuses on a single feature (house size) and establishes the foundational ML pipeline, including:

- Model representation
- Cost function
- Gradient computation
- Gradient Descent optimization
- Cost convergence visualization
- Web-based user interface

---

## 🧠 Concepts Implemented

- Linear Regression (f(x) = wx + b)
- Mean Squared Error (MSE)
- Cost Function minimization
- Gradient Descent algorithm
- Cost vs Iterations analysis
- Modular ML system architecture
- UI integration using Streamlit

---

## 🏗 Project Architecture

real-estate-ml-system/
│
├── data/
│ └── housing_v1.csv
│
├── src/
│ ├── models/
│ │ └── linear_regression.py
│ │
│ ├── training/
│ │ └── trainer.py
│ │
│ ├── utils/
│ │ └── visualization.py
│ │
│ ├── main.py
│ └── app.py
│
├── requirements.txt
└── README.md


---

## 📊 Features

### ✅ Model Training
- Gradient Descent implemented from scratch
- Cost tracking during optimization

### 📉 Cost vs Iterations Plot
- Visualizes convergence behavior
- Helps analyze learning rate effectiveness

### 🌐 Web Interface
- Built using Streamlit
- Users can input house size
- Real-time price prediction displayed

---

## ▶️ How to Run (CLI Version)

```bash
cd src
python main.py
🌐 How to Run (Web UI Version)
cd src
python -m streamlit run app.py
The application will open in your browser.

📈 Sample Output
Cost decreases steadily over iterations

Model converges to optimal parameters

Accurate predictions based on training data

Interactive UI for user input

🔮 Future Phases
This project is designed to evolve:

Phase 2 → Multiple Features (Multivariate Regression)

Phase 3 → Feature Scaling

Phase 4 → Train/Test Split & Model Evaluation

Phase 5 → Logistic Regression

Phase 6 → Neural Networks

Phase 7 → Deployment & Model Versioning

