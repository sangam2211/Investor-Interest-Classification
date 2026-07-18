import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# 1. LOAD ASSETS
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load('growup_investor_model.pkl')
    scaler = joblib.load('growup_scaler.pkl')
    feature_names = joblib.load('model_columns.pkl')
    return model, scaler, feature_names

model, scaler, feature_names = load_assets()

# ---------------------------------------------------------
# 2. UI LAYOUT
# ---------------------------------------------------------
st.set_page_config(page_title="Investor Prediction", layout="centered")
st.title("📈 Grow-Up Investor Prediction")

tab1, tab2 = st.tabs(["👤 Personal Details", "💰 Investment Preferences"])

with tab1:
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    invests_av = st.selectbox("Do you currently invest?", ["Yes", "No"])

with tab2:
    # We use these exact strings. If your model column names are 
    # 'Question_Answer', make sure the dictionary keys match those substrings.
    duration = st.selectbox("Investment Duration", ["1-3 years", "3-5 years", "More than 5 years"])
    monitor = st.selectbox("Monitoring Frequency", ["Monthly", "Weekly"])
    returns = st.selectbox("Expected Return", ["10%-20%", "20%-30%", "30%-40%"])
    factor = st.selectbox("Primary Factor", ["Returns", "Risk"])

# ---------------------------------------------------------
# 3. MAPPING LOGIC (The Fix for KeyError)
# ---------------------------------------------------------
def get_model_input():
    # Start with a clean slate: 0 for all 38 columns
    data = {col: 0 for col in feature_names}
    
    # 1. Map Numerical Columns (Simple assignment)
    if 'Age' in data:
        data['Age'] = age
        
    # 2. Map Categorical Columns (One-Hot Matching)
    # We loop through all your 38 column names and look for the user's choice.
    # This prevents KeyErrors because we look for the substring in the column name.
    
    selections = [gender, invests_av, duration, monitor, returns, factor]
    
    for col in feature_names:
        for choice in selections:
            # We look for columns like "Question_Text_Option"
            # If the user chose "Male" and column is "Gender_Male", it matches.
            if f"_{choice}" in col or col.endswith(f"_{choice}"):
                data[col] = 1
                
    return pd.DataFrame([data])

# ---------------------------------------------------------
# 4. PREDICTION
# ---------------------------------------------------------
if st.button("Predict Likelihood", type="primary"):
    try:
        input_df = get_model_input()
        
        # Enforce exact column order required by model
        input_df = input_df[feature_names]
        
        # Scale and Predict
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1] * 100
        
        if prediction == 1:
            st.success(f"### Likely to Invest: {prob:.1f}%")
        else:
            st.error(f"### Unlikely to Invest: {prob:.1f}%")
            
    except Exception as e:
        st.error(f"Error: {e}")
        st.write("Debug: Check if your selection matches the column names in your data.")
