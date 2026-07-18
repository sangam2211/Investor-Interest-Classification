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
# 2. COMPLETE USER INTERFACE (ALL 23 QUESTIONS)
# ---------------------------------------------------------
st.set_page_config(page_title="Grow-Up Investor Prediction", layout="wide")
st.title("📈 Grow-Up Investor Prediction")

tab1, tab2, tab3 = st.tabs(["👤 Profile", "📊 Rankings", "🎯 Strategy & Reasons"])

with tab1:
    st.subheader("Personal Details")
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    inv_av = st.selectbox("Do you currently invest in Investment Avenues?", ['Yes', 'No'])
    source = st.selectbox("Primary source of information for investments", ['Financial Consultants', 'Newspapers and Magazines', 'Television', 'Internet'])

with tab2:
    st.subheader("Rank your preferences (1 = Best, 7 = Worst)")
    col1, col2 = st.columns(2)
    with col1:
        r_mf = st.slider("Mutual Funds", 1, 7, 1)
        r_eq = st.slider("Equity Market", 1, 7, 2)
        r_deb = st.slider("Debentures", 1, 7, 3)
        r_gb = st.slider("Government Bonds", 1, 7, 4)
    with col2:
        r_fd = st.slider("Fixed Deposits", 1, 7, 5)
        r_ppf = st.slider("Public Provident Fund", 1, 7, 6)
        r_gold = st.slider("Gold", 1, 7, 7)

with tab3:
    st.subheader("Investment Behavior")
    col3, col4 = st.columns(2)
    with col3:
        factors = st.selectbox("Factors considered while investing", ['Returns', 'Risk', 'Locking Period'])
        objective = st.selectbox("Investment objective", ['Capital Appreciation', 'Growth', 'Income'])
        purpose = st.selectbox("Purpose behind investment", ['Wealth Creation', 'Savings for Future', 'Returns'])
        duration = st.selectbox("How long do you prefer to keep your money?", ['Less than 1 year', '1-3 years', '3-5 years', 'More than 5 years'])
        monitor = st.selectbox("How often do you monitor your investment?", ['Monthly', 'Weekly', 'Daily'])
        exp_return = st.selectbox("Expected return", ['10%-20%', '20%-30%', '30%-40%'])
        avenue = st.selectbox("Which investment avenue do you mostly invest in?", ['Equity', 'Mutual Fund', 'Fixed Deposits', 'Public Provident Fund'])
    
    with col4:
        sav_obj = st.selectbox("Savings objectives", ['Education', 'Retirement Plan', 'Health Care'])
        rsn_eq = st.selectbox("Reasons for Equity Market", ['Capital Appreciation', 'Dividend', 'Liquidity'])
        rsn_mf = st.selectbox("Reasons for Mutual Funds", ['Fund Diversification', 'Better Returns', 'Tax Benefits'])
        rsn_gb = st.selectbox("Reasons for Govt Bonds", ['Assured Returns', 'Safe Investment', 'Tax Incentives'])
        rsn_fd = st.selectbox("Reasons for Fixed Deposits", ['Fixed Returns', 'High Interest Rates', 'Risk Free'])

# Package all UI inputs into a single dictionary
ui_data = {
    'AGE': age,
    'Rank_MutualFunds': r_mf, 'Rank_EquityMarket': r_eq, 'Rank_Debentures': r_deb,
    'Rank_GovtBonds': r_gb, 'Rank_FixedDeposits': r_fd, 'Rank_PPF': r_ppf, 'Rank_Gold': r_gold,
    'GENDER': gender, 'Invest_Avenues': inv_av, 'Factors': factors, 'Objective': objective,
    'Purpose': purpose, 'Duration': duration, 'Monitor': monitor, 'Return': exp_return,
    'Avenue': avenue, 'Savings_Obj': sav_obj, 'Reason_Equity': rsn_eq, 'Reason_MF': rsn_mf,
    'Reason_Bonds': rsn_gb, 'Reason_FD': rsn_fd, 'Source': source
}

# ---------------------------------------------------------
# 3. SMART MAPPING ENGINE
# ---------------------------------------------------------
def get_model_input(ui_data):
    # Start with a clean DataFrame of zeros using exact trained columns
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # 1. Map Age
    for col in feature_names:
        if col.upper() == 'AGE':
            input_df[col] = ui_data['AGE']
            
    # 2. Map Ranks
    rank_mapping = {
        'Mutual Funds': ui_data['Rank_MutualFunds'], 'Equity Market': ui_data['Rank_EquityMarket'],
        'Debentures': ui_data['Rank_Debentures'], 'Government Bonds': ui_data['Rank_GovtBonds'],
        'Fixed Deposits': ui_data['Rank_FixedDeposits'], 'Public Provident Fund': ui_data['Rank_PPF'],
        'Gold': ui_data['Rank_Gold']
    }
    for col in feature_names:
        if 'Rank' in col:
            for key, val in rank_mapping.items():
                if key in col:
                    input_df[col] = val

    # 3. Map Categorical (Drop-first safe)
    cat_mapping = {
        'GENDER': ui_data['GENDER'],
        'invest in Investment Avenues': ui_data['Invest_Avenues'],
        'factors considered': ui_data['Factors'],
        'investment objective': ui_data['Objective'],
        'purpose behind investment': ui_data['Purpose'],
        'How long do you prefer': ui_data['Duration'],
        'How often do you monitor': ui_data['Monitor'],
        'return do you expect': ui_data['Return'],
        'mostly invest in': ui_data['Avenue'],
        'savings objectives': ui_data['Savings_Obj'],
        'Reasons for investing in Equity': ui_data['Reason_Equity'],
        'Reasons for investing in Mutual Funds': ui_data['Reason_MF'],
        'Reasons for investing in Government Bonds': ui_data['Reason_Bonds'],
        'Reasons for investing in Fixed Deposits': ui_data['Reason_FD'],
        'sources of information': ui_data['Source']
    }
    
    for col in feature_names:
        for q_key, user_choice in cat_mapping.items():
            # If both the question and the user's answer are in the column name, it's a match
            if q_key in col and str(user_choice) in col:
                if col.endswith(f"_{user_choice}"):
                    input_df[col] = 1

    # Fallback just in case you manually renamed Duration to numbers in the notebook
    if 'Investment_Duration' in feature_names:
        dur_map = {'Less than 1 year': 1, '1-3 years': 2, '3-5 years': 3, 'More than 5 years': 4}
        input_df['Investment_Duration'] = dur_map[ui_data['Duration']]

    return input_df

# ---------------------------------------------------------
# 4. PREDICTION
# ---------------------------------------------------------
st.write("---")
if st.button("Predict Likelihood to Invest in Stock Market", type="primary", use_container_width=True):
    try:
        # Generate the mapped input dataframe
        df_input = get_model_input(ui_data)
        
        # Enforce exact column order
        df_input = df_input[feature_names]
        
        # Scale & Predict
        scaled_data = scaler.transform(df_input)
        prob = model.predict_proba(scaled_data)[0][1] * 100
        
        if prob > 50:
            st.success(f"### Likely to Invest in Stock Market: {prob:.1f}%")
            st.balloons()
        else:
            st.error(f"### Unlikely to Invest in Stock Market: {prob:.1f}%")
            
    except Exception as e:
        st.error(f"Execution Error: {e}")
