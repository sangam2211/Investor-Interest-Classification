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
# 2. USER INTERFACE
# ---------------------------------------------------------
st.set_page_config(page_title="Investor Prediction", layout="wide")
st.title("📈 Grow-Up Investor Prediction")

tab1, tab2, tab3 = st.tabs(["👤 Profile", "📊 Asset Rankings", "🎯 Intentions"])

with tab1:
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    invests_av = st.selectbox("Currently Invest in Avenues?", ["Yes", "No"])

with tab2:
    st.write("Rank preferences from 1 (Best) to 7 (Worst)")
    r_mf = st.slider("Mutual Funds", 1, 7, 1)
    r_eq = st.slider("Equity Market", 1, 7, 2)
    r_deb = st.slider("Debentures", 1, 7, 3)
    r_gb = st.slider("Govt Bonds", 1, 7, 4)
    r_fd = st.slider("Fixed Deposits", 1, 7, 5)
    r_ppf = st.slider("PPF", 1, 7, 6)
    r_gold = st.slider("Gold", 1, 7, 7)

with tab3:
    # These options match your exact dataset values
    factor = st.selectbox("Primary Factor", ['Returns', 'Risk', 'Locking Period'])
    duration = st.selectbox("Investment Duration", ['3-5 years', '1-3 years', 'More than 5 years', 'Less than 1 year'])
    monitor = st.selectbox("Monitoring Frequency", ['Monthly', 'Weekly', 'Daily'])

# ---------------------------------------------------------
# 3. MAPPING LOGIC
# ---------------------------------------------------------
def get_model_input():
    # Start with all zeros
    data = {col: 0 for col in feature_names}
    
    # Map numerical/rank columns
    data['Age'] = age
    data['Rank_MutualFunds'] = r_mf
    data['Rank_EquityMarket'] = r_eq
    data['Rank_Debentures'] = r_deb
    data['Rank_GovtBonds'] = r_gb
    data['Rank_FixedDeposits'] = r_fd
    data['Rank_PPF'] = r_ppf
    data['Rank_Gold'] = r_gold
    
    # Map Categorical selections
    # We dynamically set 1 for the column that matches the selection
    selections = [gender, invests_av, factor, duration, monitor]
    
    for col in feature_names:
        for val in selections:
            # Matches columns like "Gender_Male" or "Duration_3-5 years"
            if col.endswith(f"_{val}"):
                data[col] = 1
                
    return pd.DataFrame([data])

# ---------------------------------------------------------
# 4. PREDICTION
# ---------------------------------------------------------
if st.button("Predict Likelihood", type="primary"):
    try:
        input_df = get_model_input()
        input_df = input_df[feature_names] # Force correct order
        
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1] * 100
        
        if prediction == 1:
            st.success(f"### Likely to Invest: {prob:.1f}%")
        else:
            st.error(f"### Unlikely to Invest: {prob:.1f}%")
    except Exception as e:
        st.error(f"Error mapping inputs: {e}")
