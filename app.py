import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("crop_recommendation_model.pkl")

# Define crop labels
crop_labels = ["apple", "banana", "blackgram", "chickpea", "coconut", "coffee", 
               "cotton", "grapes", "jute", "kidneybeans", "lentil", "maize", "mango",
               "mothbeans", "mungbean", "muskmelon", "orange", "papaya", 
               "pigeonpeas", "pomegranate", "rice", "watermelon"]

# Streamlit UI
st.title("🌱 Crop Recommendation System")
st.write("Enter soil and climate conditions to get the best crop recommendation.")

# User inputs
N = st.number_input("Nitrogen Level", min_value=0, max_value=150, value=50)
P = st.number_input("Phosphorus Level", min_value=0, max_value=150, value=50)
K = st.number_input("Potassium Level", min_value=0, max_value=150, value=50)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0)

# Predict button
if st.button("Predict Crop"):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)
    predicted_crop = crop_labels[prediction[0]]
    
    st.success(f"🌾 Recommended Crop: **{predicted_crop}**")

