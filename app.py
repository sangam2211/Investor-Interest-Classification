import streamlit as st
import pandas as pd
import joblib

# Load the model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load('growup_investor_model.pkl')
    scaler = joblib.load('growup_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# Build the UI
st.title("📈 Grow-Up Hedge Funding: Investor Prediction App")
st.write("Enter the customer's survey responses below to predict if they will invest in the stock market.")

st.sidebar.header("Customer Survey Inputs")

# Example inputs (Update these to match your exact X_encoded columns)
age = st.sidebar.slider("Age", min_value=21, max_value=35, value=27)
rank_fd = st.sidebar.slider("Rank: Fixed Deposits (1=Best, 7=Worst)", 1, 7, 3)
rank_ppf = st.sidebar.slider("Rank: Public Provident Fund", 1, 7, 3)

expected_return = st.sidebar.selectbox("Expected Return", ["10%-20%", "20%-30%", "30%-40%"])
return_map = {'10%-20%': 0, '20%-30%': 1, '30%-40%': 2}
expected_return_encoded = return_map[expected_return]

# Format input data (Ensure this dictionary matches ALL 38 features in your training data)
input_data = {
    'Age': age,
    'Rank_FixedDeposits': rank_fd,
    'Rank_PPF': rank_ppf,
    'Expected_Return': expected_return_encoded,
    # Add your remaining 34 features here with default values
}

input_df = pd.DataFrame([input_data])

# Scale and Predict
if st.button("Predict Investment Likelihood"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1] * 100
    
    st.subheader("Prediction Results")
    if prediction == 1:
        st.success(f"✅ The customer is LIKELY to invest in the stock market. (Probability: {probability:.1f}%)")
    else:
        st.error(f"❌ The customer is UNLIKELY to invest in the stock market. (Probability: {probability:.1f}%)")
