import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration
st.set_page_config(page_title="Movie Gross Revenue Predictor", layout="centered")

# Load the pre-trained model and scaler
@st.cache_resource
def load_models():
    with open('RandomForest_model_movies.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('RandomForest_scaler_movies.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_models()

st.title("🎬 Movie Gross Revenue Predictor")
st.write("Enter the movie details below to predict its gross revenue using the trained Random Forest model.")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    budget = st.number_input("Budget ($)", min_value=0.0, value=15000000.0, step=500000.0)
    score = st.slider("IMDb Score", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    votes = st.number_input("Number of Votes", min_value=0.0, value=50000.0, step=1000.0)
    year = st.number_input("Release Year", min_value=1980, max_value=2030, value=2020, step=1)

with col2:
    runtime = st.number_input("Runtime (minutes)", min_value=0.0, value=105.0, step=1.0)
    
    # Rating mapping based on the notebook logic
    rating = st.selectbox("Rating Classification", options=["G", "PG", "PG-13", "R", "Other"])
    rating_map = {'G': 4, 'PG': 3, 'PG-13': 2, 'R': 1, 'Other': 0}
    rating_encoded = rating_map[rating]
    
    # USA origin mapping
    country = st.selectbox("Is the movie produced in the USA?", options=["Yes", "No"])
    is_usa = 1 if country == "Yes" else 0
    
    # ROI input (Required because it was included in the training features)
    roi = st.number_input("Expected ROI (e.g., 2.5 for 2.5x Budget)", value=1.0, step=0.1)

# Prediction Button
if st.button("Predict Gross Revenue 🚀"):
    # Organize input into a DataFrame to match the Scaler's expected feature names
    input_df = pd.DataFrame([[
        budget, score, votes, runtime, year, rating_encoded, is_usa, roi
    ]], columns=['budget', 'score', 'votes', 'runtime', 'year', 'rating_encoded', 'is_usa', 'ROI'])
    
    try:
        # Scale the features
        input_scaled = scaler.transform(input_df)
        
        # Predict the gross revenue
        prediction = model.predict(input_scaled)[0]
        
        # Display the prediction (Balloons removed)
        st.success(f"### Predicted Gross Revenue: **${prediction:,.2f}**")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")