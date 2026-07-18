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
# 2. POLISHED & COMPACT USER INTERFACE WITH IMAGES
# ---------------------------------------------------------
st.set_page_config(page_title="Grow-Up Investor Prediction", layout="wide", page_icon="📈")

# --- SIDEBAR ---
with st.sidebar:
    # Sidebar Image (Stock market trend)
    st.image("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80", use_container_width=True)
    st.title("Grow-Up Hedge Funding")
    st.markdown("""
    **Welcome to the Investor Conversion Predictor.**
    
    Use this tool to analyze client profiles and predict whether they are likely to invest in the stock market.
    
    *Adjust the parameters in the main window to see the prediction update.*
    """)
    st.write("---")
    st.caption("Powered by Machine Learning")

# --- MAIN HEADER ---
st.title("📈 Grow-Up Investor Prediction")
# Header Image (Finance/Trading desk)
st.image("https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", use_container_width=True)
st.markdown("Fill out the investor profile below to predict their likelihood of investing in the stock market.")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["👤 Profile Details", "📊 Asset Rankings", "🎯 Strategy & Reasons"])

with tab1:
    st.subheader("Personal Details")
    # Using 3 columns to keep the Age slider and dropdowns compact
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        age = st.slider("Age", 18, 65, 30)
    with col_p2:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        inv_av = st.selectbox("Invest in Investment Avenues?", ['Yes', 'No'])
    with col_p3:
        source = st.selectbox("Primary information source", ['Financial Consultants', 'Newspapers and Magazines', 'Television', 'Internet'])

with tab2:
    st.subheader("Rank your preferences (1 = Best, 7 = Worst)")
    # Using 4 columns prevents the sliders from stretching across the screen
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        r_mf = st.slider("Mutual Funds", 1, 7, 1)
        r_eq = st.slider("Equity Market", 1, 7, 2)
    with col2:
        r_deb = st.slider("Debentures", 1, 7, 3)
        r_gb = st.slider("Government Bonds", 1, 7, 4)
    with col3:
        r_fd = st.slider("Fixed Deposits", 1, 7, 5)
        r_ppf = st.slider("Public Provident Fund", 1, 7, 6)
    with col4:
        r_gold = st.slider("Gold", 1, 7, 7)

with tab3:
    st.subheader("Investment Behavior")
    # Using 3 columns makes the dropdowns neat and constrained
    col5, col6, col7 = st.columns(3)
    with col5:
        factors = st.selectbox("Factors considered", ['Returns', 'Risk', 'Locking Period'])
        objective = st.selectbox("Investment objective", ['Capital Appreciation', 'Growth', 'Income'])
        purpose = st.selectbox("Purpose behind investment", ['Wealth Creation', 'Savings for Future', 'Returns'])
        duration = st.selectbox("Preferred duration", ['Less than 1 year', '1-3 years', '3-5 years', 'More than 5 years'])
    
    with col6:
        monitor = st.selectbox("Monitoring frequency", ['Monthly', 'Weekly', 'Daily'])
        exp_return = st.selectbox("Expected return", ['10%-20%', '20%-30%', '30%-40%'])
        avenue = st.selectbox("Main investment avenue", ['Equity', 'Mutual Fund', 'Fixed Deposits', 'Public Provident Fund'])
        sav_obj = st.selectbox("Savings objectives", ['Education', 'Retirement Plan', 'Health Care'])
        
    with col7:
        rsn_eq = st.selectbox("Reasons for Equity Market", ['Capital Appreciation', 'Dividend', 'Liquidity'])
        rsn_mf = st.selectbox("Reasons for Mutual Funds", ['Fund Diversification', 'Better Returns', 'Tax Benefits'])
        rsn_gb = st.selectbox("Reasons for Govt Bonds", ['Assured Returns', 'Safe Investment', 'Tax Incentives'])
        rsn_fd = st.selectbox("Reasons for Fixed Deposits", ['Fixed Returns', 'High Interest Rates', 'Risk Free'])

# Package all UI inputs into a dictionary
ui_data = {
    'AGE': age, 'Rank_MutualFunds': r_mf, 'Rank_EquityMarket': r_eq, 'Rank_Debentures': r_deb,
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
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    for col in feature_names:
        if col.upper() == 'AGE':
            input_df[col] = ui_data['AGE']
            
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
            if q_key in col and str(user_choice) in col:
                if col.endswith(f"_{user_choice}"):
                    input_df[col] = 1

    if 'Investment_Duration' in feature_names:
        dur_map = {'Less than 1 year': 1, '1-3 years': 2, '3-5 years': 3, 'More than 5 years': 4}
        input_df['Investment_Duration'] = dur_map[ui_data['Duration']]

    return input_df

# ---------------------------------------------------------
# 4. PREDICTION BUTTON & OUTPUT
# ---------------------------------------------------------
st.write("---")

# Use columns to shrink and center the button
button_col1, button_col2, button_col3 = st.columns([1, 1, 1])

with button_col2:
    predict_clicked = st.button("Predict Likelihood to Invest", type="primary", use_container_width=True)

if predict_clicked:
    try:
        df_input = get_model_input(ui_data)
        df_input = df_input[feature_names]
        
        scaled_data = scaler.transform(df_input)
        probabilities = model.predict_proba(scaled_data)[0]
        
        prob_no = probabilities[0] * 100
        prob_yes = probabilities[1] * 100
        
        st.write("") # Spacer
        res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
        with res_col2:
            if prob_yes >= 50.0:
                st.success(f"### 📈 Likely to Invest: {prob_yes:.1f}%")
                st.balloons()
            else:
                st.warning(f"### 📉 Unlikely to Invest: {prob_no:.1f}%")
                
    except Exception as e:
        st.error(f"Execution Error: {e}")
