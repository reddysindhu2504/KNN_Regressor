import streamlit as st
import pickle
import pandas as pd

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("🧬 Breast Cancer Detection App (KNN Model)")

st.write("Enter the feature values below:")

# Breast cancer dataset feature names (IMPORTANT: must match training order)
feature_names = [
    'mean perimeter', 'mean concave points', 'worst radius',
       'worst perimeter', 'worst concave points']

# Collect user inputs
user_input = []

for feature in feature_names:
    value = st.number_input(feature, value=0.0)
    user_input.append(value)

# Prediction
if st.button("Predict"):
    
    #   Convert to DataFrame 
    input_df = pd.DataFrame([user_input], columns=feature_names)

    prediction = model.predict(input_df)

    # Output
    if prediction[0] == 1:
        st.success(" Result: Benign Tumor (Non-cancerous)")
    else:
        st.error(" Result: Malignant Tumor (Cancerous)")