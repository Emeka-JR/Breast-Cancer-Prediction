import streamlit as st
import numpy as np
import joblib

# Loading the trained model
model = joblib.load("breast_cancer_model.pkl")

st.title("Breast Cancer Prediction App")
st.write("Input the patient's diagnostic features below:")

# 30 input features from the original breast cancer dataset
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

# Create number inputs for each feature
inputs = []
for feature in feature_names:
    value = st.number_input(f"{feature}", step=0.01)
    inputs.append(value)

# Predict button
if st.button("Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)
    result = "Malignant" if prediction[0] == 0 else "Benign"
    st.success(f"The predicted tumor type is: {result}")
