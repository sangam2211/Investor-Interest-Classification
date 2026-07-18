import streamlit as st
import pandas as pd
import joblib

# Load Assets
model, scaler, feature_names = load_assets()

st.title("📈 Grow-Up Hedge Funding")

# Use Tabs to group inputs
tab1, tab2 = st.tabs(["👤 Personal Details", "💰 Investment Preferences"])

with tab1:
    age = st.slider("Age", 18, 60, 25)
    gender = st.radio("Gender", ["Male", "Female"])

with tab2:
    rank_fd = st.slider("Rank: Fixed Deposits (1=Best, 7=Worst)", 1, 7, 3)
    exp_return = st.selectbox("Expected Return", ["10%-20%", "20%-30%", "30%-40%"])

# ---------------------------------------------------------
# HELPER: CONVERT USER INPUTS TO MODEL COLUMNS
# ---------------------------------------------------------
def get_model_input():
    # 1. Start with all zeros
    data = {col: 0 for col in feature_names}
    
    # 2. Map direct values
    data['Age'] = age
    data['Rank_FixedDeposits'] = rank_fd
    
    # 3. Map One-Hot Encoded Categories (Crucial Step!)
    # If Gender_Male was a column, this turns it on/off
    if f'Gender_{gender}' in data:
        data[f'Gender_{gender}'] = 1
        
    # Map the Expected Return dropdown
    if f'Expected_Return_{exp_return}' in data:
        data[f'Expected_Return_{exp_return}'] = 1
        
    return pd.DataFrame([data])

# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------
if st.button("Predict Investment Likelihood"):
    input_df = get_model_input()
    scaled_input = scaler.transform(input_df)
    
    # Predict... (same logic as before)
