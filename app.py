import streamlit as st
import joblib
import pandas as pd

# Load trained model and feature names
model = joblib.load('models/calibrated_model.pkl')
x_train = pd.read_parquet('data/x_train.parquet')

feature_names = x_train.columns.tolist()

numeric_feature = x_train.select_dtypes(include='number').columns.tolist()
categorical_feature = x_train.select_dtypes(include='object').columns.tolist()

st.title('Customer Churn Prediction')
st.write('Prototype : Predict whether a customer will churn')

# dynamic user inputs collection
input_data = {}

st.header('Enter customer Details')

for feature in feature_names: 
    if 'id' in feature.lower():
        continue # skip IDs

    elif feature in categorical_feature: # handling categorical data
        categories = list(x_train[feature].dropna().unique())
        value = st.selectbox(f'{feature}',categories)
        input_data[feature] = value

    else:
        default_val = float(x_train[feature].median())
        value = st.number_input(f'{feature}', value = default_val,step=1.0) # handling numeric data
        input_data[feature] = value

input_df = pd.DataFrame([input_data])[feature_names]   # convert to dataframe

# Prediction

if st.button('Predict'):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader('Results')
    st.write('Prediction: ','churn' if prediction == 1 else 'Not Churn')
    st.write('Churn Probability: ', round(proba,3))
    st.progress(min(int(proba*100), 100))