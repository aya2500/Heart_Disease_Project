# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load saved models and scaler
# -----------------------------
rf_model = joblib.load(r'D:\Heart_Disease_Project\Heart_Disease_Project\models\Random_Forest_model.pkl')
lr_model = joblib.load(r'D:\Heart_Disease_Project\Heart_Disease_Project\models\Logistic_Regression_model.pkl')
dt_model = joblib.load(r'D:\Heart_Disease_Project\Heart_Disease_Project\models\Decision_Tree_model.pkl')
svm_model = joblib.load(r'D:\Heart_Disease_Project\Heart_Disease_Project\models\SVM_model.pkl')
scaler = joblib.load(r'D:\Heart_Disease_Project\Heart_Disease_Project\models\scaler.pkl')
# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Heart Disease Prediction App ❤️")

st.markdown("""
Enter patient data below to predict the risk of heart disease.
""")

# Example: dynamically create input fields based on feature names
# Load selected features
df_features = pd.read_csv(r'D:\Heart_Disease_Project\Heart_Disease_Project\data\heart_disease_selected_features.csv')
feature_names = df_features.drop('target', axis=1).columns

# Create user input dictionary
user_input = {}
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0)
    user_input[feature] = val

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# -----------------------------
# Prediction button
# -----------------------------
model_choice = st.selectbox("Choose model for prediction:", ['Random Forest', 'Logistic Regression', 'Decision Tree', 'SVM'])

if st.button("Predict"):
    # Scale input if needed
    if model_choice in ['Random Forest', 'Decision Tree']:
        X_input = input_df.values
    else:
        X_input = scaler.transform(input_df)
    
    # Select model
    if model_choice == 'Random Forest':
        model = rf_model
    elif model_choice == 'Logistic Regression':
        model = lr_model
    elif model_choice == 'Decision Tree':
        model = dt_model
    else:
        model = svm_model
    
    # Make prediction
    pred = model.predict(X_input)[0]
    pred_prob = model.predict_proba(X_input)[0][1]  # probability of disease
    
    # Display results
    if pred == 1:
        st.error(f"⚠️ Patient is predicted to have heart disease with probability {pred_prob:.2f}")
    else:
        st.success(f"✅ Patient is predicted to be healthy with probability {1-pred_prob:.2f}")
