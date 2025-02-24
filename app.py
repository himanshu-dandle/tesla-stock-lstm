import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
import matplotlib.pyplot as plt

# Define file paths
model_path = os.path.join("results", "lstm_model_fixed.keras")  # ✅ Load correct model
scaler_path = os.path.join("results", "scaler.pkl")
plot_path = os.path.join("results", "actual_vs_predicted.png")  # ✅ Path to evaluation plot



# Load the trained model & scaler with error handling
try:
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)  
        st.write("✅ Model loaded successfully!")
    else:
        st.error("❌ Model file not found! Please check the path.")
        model = None
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    model = None

try:
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        st.write("✅ Scaler loaded successfully!")
    else:
        st.error("❌ Scaler file not found! Please check the path.")
        scaler = None
except Exception as e:
    st.error(f"❌ Error loading scaler: {str(e)}")
    scaler = None

# Streamlit App UI
st.title("📈 Tesla Stock Price Prediction")
st.write("Enter the last 60 days of closing prices to predict the next day's price.")

# User input fields
input_data = []
for i in range(60):
    input_value = st.number_input(f"Day {i+1} Closing Price", min_value=0.0, format="%.2f")
    input_data.append(input_value)

# Predict Button
if st.button("Predict Stock Price"):
    if len(input_data) == 60 and model is not None and scaler is not None:
        try:
            # Convert input data to model format
            input_array = np.array(input_data).reshape(1, 60, 1)
            scaled_prediction = model.predict(input_array)
            predicted_price = scaler.inverse_transform(scaled_prediction)[0][0]
            st.success(f"📊 Predicted Stock Price: ${predicted_price:.2f}")
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
    else:
        st.error("❌ Missing input data or model is not loaded properly!")

# Display Actual vs Predicted Plot
st.subheader("📊 Actual vs Predicted Stock Prices")
if os.path.exists(plot_path):
   ## st.image(plot_path, caption="Actual vs Predicted Prices", use_column_width=True
    st.image(plot_path, caption="Actual vs Predicted Prices", use_container_width=True)

else:
    st.warning("⚠️ Prediction plot not found. Please run model evaluation.")
