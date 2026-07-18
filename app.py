import streamlit as st
import pandas as pd
import joblib

# 1. LOAD ASSETS
@st.cache_resource
def load_assets():
    model = joblib.load('growup_investor_model.pkl')
    scaler = joblib.load('growup_scaler.pkl')
    feature_names = joblib.load('model_columns.pkl')
    return model, scaler, feature_names

model, scaler, feature_names = load_assets()

# 2. UI
st.title("📈 Grow-Up Investor Prediction")
# Use the numeric values found in your notebook
duration = st.selectbox("Investment Duration", [1, 2, 3, 4]) 
# Add your other inputs here...

def get_model_input():
    # Initialize all 38 columns to 0
    data = {col: 0 for col in feature_names}
    
    # Map Numerical inputs directly
    data['Age'] = age 
    
    # CORRECT LOGIC FOR drop_first=True:
    # If the user selects the "first" category (e.g., 1), you do nothing (leave as 0).
    # If the user selects any other category (2, 3, or 4), you set that specific column to 1.
    if duration != 1:
        col_name = f'Investment_Duration_{duration}'
        if col_name in data:
            data[col_name] = 1
            
    return pd.DataFrame([data])

# 3. PREDICTION
if st.button("Predict"):
    input_df = get_model_input()
    # Ensure column order matches the model exactly
    input_df = input_df[feature_names] 
    prob = model.predict_proba(scaler.transform(input_df))[0][1]
    st.write(f"Probability: {prob*100:.2f}%")
