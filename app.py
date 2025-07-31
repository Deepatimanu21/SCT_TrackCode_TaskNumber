import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('house_price_model.pkl', 'rb'))

st.title("🏡 House Price Predictor")
st.write("Enter property details to estimate price")

# Input fields
sqft = st.number_input("Square Footage", min_value=300, max_value=10000, step=10)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1.0, max_value=10.0, step=0.5)

# Predict button
if st.button("Predict Price"):
    # Prepare input for model
    input_data = np.array([[sqft, bedrooms, bathrooms]])
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    # Add error margin using MAE
    mae = 36569  # From your model evaluation
    lower_bound = prediction - mae
    upper_bound = prediction + mae
    
    st.success(f"Estimated Price: ${prediction:,.2f}")
    st.write(f"Possible Range: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
