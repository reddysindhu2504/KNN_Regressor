import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('model.pkl','rb'))

st.title('Welcome to Breast Cancer Prediction')

st.write('Enter your feature values below')

feature_names = ['mean perimeter', 'mean concave points', 'worst radius',
       'worst perimeter', 'worst concave points']

user_input = []
for feature in feature_names:
    value = st.number_input(feature,value=0.0)
    user_input.append(value)

if st.button('Predict'):
    input_df = pd.DataFrame([user_input],columns = feature_names)
    pred = model.predict(input_df)

    if pred == 1:
        st.success('Tumour is Benign, Non Cancerous')
    else:
        st.error('Tumour is Malignant, Cancerous')