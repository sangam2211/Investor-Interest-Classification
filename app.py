import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------
# Load assets once and cache them
@st.cache_resource
def load_assets():
    model = joblib.load('growup_investor_model.pkl')
    scaler = joblib.load('growup_scaler.pkl')
    feature_names = joblib.load('model_columns.pkl')
    return model, scaler, feature_names

model, scaler, feature_names = load_assets()

# ---------------------------------------------------------
# 2. UI (Define variables here)
# ---------------------------------------------------------
st.title("📈 Grow-Up Investor Prediction")

age = st.slider("Age", 18, 65, 30)
# Use the values found in your notebook [1, 2, 3, 4]
duration = st.selectbox("Investment Duration", [1, 2, 3, 4]) 

# ---------------------------------------------------------
# 3. MAPPING LOGIC (Safe & Scope-Protected)
# ---------------------------------------------------------
def get_prediction(age_val, dur_val):
    # Initialize a dataframe of 0s with the exact 38 column names
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Map inputs
    # Check if 'Age' exists in the trained columns before setting it
    if 'Age' in input_df.columns:
        input_df['Age'] = age_val
        
    # Map Duration (Handling the drop_first=True logic)
    # If the user selects 1, we do nothing (it remains 0, which is correct for dropped columns)
    # If user selects 2, 3, or 4, we set that specific column to 1
    duration_col = f'Investment_Duration_{dur_val}'
    if duration_col in input_df.columns:
        input_df[duration_col] = 1
        
    return input_df

# ---------------------------------------------------------
# 4. EXECUTION
# ---------------------------------------------------------
if st.button("Predict Likelihood"):
    try:
        # Build the input
        df_input = get_prediction(age, duration)
        
        # Scale and Predict
        # We assume the scaler was fitted on the 38 features
        scaled_data = scaler.transform(df_input)
        
        prob = model.predict_proba(scaled_data)[0][1] * 100
        
        if prob > 50:
            st.success(f"Likely to Invest: {prob:.1f}%")
        else:
            st.warning(f"Unlikely to Invest: {prob:.1f}%")
            
    except Exception as e:
        st.error(f"Mapping error: {e}")
        st.write("Current columns:", feature_names)
