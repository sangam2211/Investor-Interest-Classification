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
# 2. UI - INPUT FIELDS
# ---------------------------------------------------------
st.title("📈 Grow-Up Hedge Funding")
st.write("Enter customer survey details to predict investment potential.")

# Dynamic input collection based on feature_names
user_inputs = {}
for col in feature_names:
    if 'Age' in col:
        user_inputs[col] = st.sidebar.slider(col, 18, 60, 25)
    elif 'Rank' in col:
        user_inputs[col] = st.sidebar.slider(col, 1, 7, 3)
    elif 'Expected_Return' in col:
        user_inputs[col] = st.sidebar.selectbox(col, [0, 1, 2])
    else:
        # Default for one-hot encoded binary features
        user_inputs[col] = st.sidebar.checkbox(col, value=False)

# ---------------------------------------------------------
# 3. CONSTRUCT DATAFRAME
# ---------------------------------------------------------
input_df = pd.DataFrame(user_inputs, index=[0])

# Ensure columns match training order exactly
input_df = input_df[feature_names]

# ---------------------------------------------------------
# 4. SCALE & PREDICT
# ---------------------------------------------------------
if st.button("Predict Investment Likelihood"):
    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1] * 100
        
        st.subheader("Prediction Results")
        if prediction == 1:
            st.success(f"Likely to invest (Probability: {probability:.1f}%)")
        else:
            st.error(f"Unlikely to invest (Probability: {probability:.1f}%)")
    except Exception as e:
        st.error(f"Error: {e}")
