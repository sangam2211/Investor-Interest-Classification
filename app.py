import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# 1. LOAD ASSETS AND DEFINE FEATURE ORDER
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load('growup_investor_model.pkl')
    scaler = joblib.load('growup_scaler.pkl')
    # Exact ordered feature names from your model data
    feature_names = [
        'Gender', 'Age', 'Invests_Avenues', 'Rank_MutualFunds', 'Rank_EquityMarket',
        'Rank_Debentures', 'Rank_GovtBonds', 'Rank_FixedDeposits', 'Rank_PPF', 'Rank_Gold',
        'Investment_Duration', 'Monitor_Frequency', 'Expected_Return', 'High_Return_Seeker',
        'Risk_Appetite_Score', 'Prefers_Equity', 'Investment_Factor_Returns', 'Investment_Factor_Risk',
        'Investment_Objective_Growth', 'Investment_Objective_Income',
        'Investment_Purpose_Savings for Future', 'Investment_Purpose_Wealth Creation',
        'Primary_Avenue_Fixed Deposits', 'Primary_Avenue_Mutual Fund', 'Primary_Avenue_Public Provident Fund',
        'Savings_Objective_Health Care', 'Savings_Objective_Retirement Plan',
        'Reason_Equity_Dividend', 'Reason_Equity_Liquidity',
        'Reason_MutualFunds_Fund Diversification', 'Reason_MutualFunds_Tax Benefits',
        'Reason_GovtBonds_Safe Investment', 'Reason_GovtBonds_Tax Incentives',
        'Reason_FD_High Interest Rates', 'Reason_FD_Risk Free',
        'Info_Source_Internet', 'Info_Source_Newspapers and Magazines', 'Info_Source_Television'
    ]
    return model, scaler, feature_names

model, scaler, feature_names = load_assets()

# ---------------------------------------------------------
# 2. USER INTERFACE LAYOUT
# ---------------------------------------------------------
st.set_page_config(page_title="Grow-Up Hedge Funding", page_icon="📈", layout="wide")
st.title("📈 Grow-Up Hedge Funding: Investor Prediction Engine")
st.write("Determine the likelihood of a customer investing in the stock market by completing the fields below.")
st.markdown("---")

# Use three human-readable tabs to cleanly group the 38 features
tab1, tab2, tab3 = st.tabs(["👤 Investor Profile", "📊 Asset Preferences & Rankings", "🎯 Intentions & Motives"])

with tab1:
    st.subheader("Demographics & Basic Behavior")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 70, 30)
        gender = st.selectbox("Gender", [("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
        invests_av = st.selectbox("Currently Invests in Any Avenues?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        inv_duration = st.slider("Typical Investment Duration", 1, 5, 3)
        monitor_freq = st.slider("Portfolio Monitoring Frequency", 1, 5, 3)
        exp_return = st.selectbox("Expected Return Band", [("Low (10%-20%)", 0), ("Medium (20%-30%)", 1), ("High (30%-40%)", 2)], format_func=lambda x: x[0])[1]
    with col2:
        high_seeker = st.selectbox("Is a High Return Seeker?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        risk_appetite = st.slider("Risk Appetite Score", 1, 5, 3)
        prefers_equity = st.selectbox("Explicitly Prefers Equity?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        factor_returns = st.slider("Importance Factor: Expected Returns", 1, 5, 3)
        factor_risk = st.slider("Importance Factor: Risk Level", 1, 5, 3)
        obj_growth = st.selectbox("Objective: Capital Growth", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        obj_income = st.selectbox("Objective: Regular Income", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

with tab2:
    st.subheader("Asset Rankings (1 = Highly Preferred, 7 = Least Preferred)")
    col3, col4 = st.columns(2)
    with col3:
        r_mf = st.slider("Rank: Mutual Funds", 1, 7, 3)
        r_eq = st.slider("Rank: Equity Market", 1, 7, 3)
        r_deb = st.slider("Rank: Debentures", 1, 7, 4)
        r_gb = st.slider("Rank: Government Bonds", 1, 7, 4)
    with col4:
        r_fd = st.slider("Rank: Fixed Deposits", 1, 7, 3)
        r_ppf = st.slider("Rank: Public Provident Fund", 1, 7, 3)
        r_gold = st.slider("Rank: Gold", 1, 7, 4)

with tab3:
    st.subheader("Reasons, Preferences & Sources")
    col5, col6 = st.columns(2)
    with col5:
        inv_purpose = st.selectbox("Primary Investment Purpose", ["Savings for Future", "Wealth Creation", "Other"])
        prim_avenue = st.selectbox("Primary Preferred Avenue", ["Fixed Deposits", "Mutual Fund", "Public Provident Fund", "Other"])
        sav_objective = st.selectbox("Savings Goal Focus", ["Health Care", "Retirement Plan", "Other"])
        info_source = st.selectbox("Main Information Source", ["Internet", "Newspapers and Magazines", "Television", "Other"])
    with col6:
        r_equity_choice = st.selectbox("Top Driver to Choose Equity", ["Dividend", "Liquidity", "Other"])
        r_mf_choice = st.selectbox("Top Driver to Choose Mutual Funds", ["Fund Diversification", "Tax Benefits", "Other"])
        r_gb_choice = st.selectbox("Top Driver to Choose Govt Bonds", ["Safe Investment", "Tax Incentives", "Other"])
        r_fd_choice = st.selectbox("Top Driver to Choose Fixed Deposits", ["High Interest Rates", "Risk Free", "Other"])

# ---------------------------------------------------------
# 3. BACKEND DICTIONARY MAPPING AND DATAFRAME CREATION
# ---------------------------------------------------------
def process_inputs():
    # Initialize all 38 features with 0
    data = {col: 0 for col in feature_names}
    
    # Apply baseline continuous/ordinal/binary fields
    data['Gender'] = gender
    data['Age'] = age
    data['Invests_Avenues'] = invests_av
    data['Rank_MutualFunds'] = r_mf
    data['Rank_EquityMarket'] = r_eq
    data['Rank_Debentures'] = r_deb
    data['Rank_GovtBonds'] = r_gb
    data['Rank_FixedDeposits'] = r_fd
    data['Rank_PPF'] = r_ppf
    data['Rank_Gold'] = r_gold
    data['Investment_Duration'] = inv_duration
    data['Monitor_Frequency'] = monitor_freq
    data['Expected_Return'] = exp_return
    data['High_Return_Seeker'] = high_seeker
    data['Risk_Appetite_Score'] = risk_appetite
    data['Prefers_Equity'] = prefers_equity
    data['Investment_Factor_Returns'] = factor_returns
    data['Investment_Factor_Risk'] = factor_risk
    data['Investment_Objective_Growth'] = obj_growth
    data['Investment_Objective_Income'] = obj_income
    
    # Handle One-Hot Encoded strings safely via dynamic category switches
    if f'Investment_Purpose_{inv_purpose}' in data: data[f'Investment_Purpose_{inv_purpose}'] = 1
    if f'Primary_Avenue_{prim_avenue}' in data: data[f'Primary_Avenue_{prim_avenue}'] = 1
    if f'Savings_Objective_{sav_objective}' in data: data[f'Savings_Objective_{sav_objective}'] = 1
    if f'Info_Source_{info_source}' in data: data[f'Info_Source_{info_source}'] = 1
    if f'Reason_Equity_{r_equity_choice}' in data: data[f'Reason_Equity_{r_equity_choice}'] = 1
    if f'Reason_MutualFunds_{r_mf_choice}' in data: data[f'Reason_MutualFunds_{r_mf_choice}'] = 1
    if f'Reason_GovtBonds_{r_gb_choice}' in data: data[f'Reason_GovtBonds_{r_gb_choice}'] = 1
    if f'Reason_FD_{r_fd_choice}' in data: data[f'Reason_FD_{r_fd_choice}'] = 1
    
    # Construct a structured pandas row, enforcing exact column positioning
    final_df = pd.DataFrame([data])
    return final_df[feature_names]

# ---------------------------------------------------------
# 4. SCALE AND GENERATE PREDICTION
# ---------------------------------------------------------
st.markdown("---")
if st.button("Predict Target Conversion", type="primary", use_container_width=True):
    try:
        input_data = process_inputs()
        
        # Scale input vectors using the loaded standardizer
        scaled_data = scaler.transform(input_data)
        
        # Run Random Forest inferencing
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1] * 100
        
        st.subheader("Classification Outcome")
        if prediction == 1:
            st.success(f"### Target Class: YES (Investor) \nConversion Probability: **{probability:.2f}%**")
            st.balloons()
        else:
            st.error(f"### Target Class: NO (Non-Investor) \nConversion Probability: **{probability:.2f}%**")
            
    except Exception as error:
        st.error(f"Pipeline Pipeline execution failure: {error}")
