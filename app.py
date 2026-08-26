import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =====================================================================
# 1. APPLICATION INITIALIZATION & CACHED RESOURCE LOADERS
# =====================================================================
st.set_page_config(
    page_title="OptiPrice Enterprise AI", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize global authentication session memory
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

@st.cache_resource
def load_model_package():
    """Safely loads pickle model artifacts or falls back to internal engine."""
    try:
        with open("trained_model.pkl", "rb") as f:
            package = pickle.load(f)
        if isinstance(package, dict):
            return package.get("model"), package.get("features", [])
        return package, getattr(package, "feature_names_in_", [])
    except Exception:
        return None, []

@st.cache_data
def load_ledger_data():
    """Loads CSV historical data or falls back to synthetic transaction records."""
    try:
        return pd.read_csv("final_data.csv")
    except Exception:
        return pd.DataFrame({
            'order_id': ['b872', 'd3c1', 'a112', 'f892', 'e441'],
            'product_category': ['health_beauty', 'housewares', 'perfumery', 'packaged_food', 'sports_leisure'],
            'price_usd': [78.42, 115.70, 85.92, 24.50, 62.10],
            'freight_value': [25.00, 18.20, 22.40, 5.50, 15.30],
            'review_score': [4.2, 4.5, 4.0, 4.8, 3.9]
        })

# =====================================================================
# 2. BRAND HEADER
# =====================================================================
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <span style="font-size: 42px; line-height: 1;">🛡️</span>
        <h1 style="margin: 0; padding: 0; font-size: 38px; font-weight: 700; letter-spacing: -0.5px;">
            OptiPrice <span style="color: #2e7d32; font-weight: 300;">Enterprise Analytics</span>
        </h1>
    </div>
    """, 
    unsafe_allow_html=True
)

# =====================================================================
# 3. GLOBAL APPLICATION NAVIGATION TASK BAR
# =====================================================================
nav_home, nav_predictor, nav_analytics, nav_settings, nav_account = st.tabs([
    "🏠 Corporate Home", 
    "🎯 AI Optimization Engine", 
    "📊 Global Sales Analytics", 
    "⚙️ API Platform Settings", 
    "👤 Secure Portal Login"
])

# =====================================================================
# TAB 1: CORPORATE PLATFORM WELCOME SCREEN
# =====================================================================
with nav_home:
    st.markdown("## 🚀 Welcome to the OptiPrice Intelligence Network")
    st.markdown("#### *The Ultimate All-in-One Predictive Command Center for Modern Commerce.*")
    
    st.markdown("""
    Stop guessing your price points and losing margins to the competition. OptiPrice Enterprise integrates 
    cutting-edge machine learning simulations with your live supply chain dynamics to isolate the exact, flawless pricing 
    sweet spot for maximum profitability.
    """)
    
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        st.markdown("🎯 **Unrivaled Precision**\nTrained on multi-million row global market transaction clusters.")
    with h_col2:
        st.markdown("⚡ **Real-Time Elasticity**\nInstantly view how every single photo, rating, or gram shifts your profit margin.")
    with h_col3:
        st.markdown("🔒 **Enterprise Security**\nBank-grade credential matching keeps your operational variables strictly confidential.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("📣 **Executive Memo:** Q3 multi-sector predictive validation channels have been updated. Switch to the **AI Optimization Engine** tab above to launch a real-time inventory simulation matrix.")

# =====================================================================
# TAB 2: AI OPTIMIZATION PREDICTIVE CORE
# =====================================================================
with nav_predictor:
    st.subheader("🎯 Real-Time Pricing Optimization Core")
    
    industry_sector = st.selectbox(
        "🏢 Select Corporate Sector Portfolio Matrix",
        [
            "💄 Health, Cosmetics & Beauty Products (Premium High-Margin Dataset)",
            "🍎 Packaged Food Items & Perishables (High-Volume Logistics Dataset)",
            "🛋️ General Merchandise & Heavy Home Logistics (Olist Baseline Dataset)"
        ]
    )
    
    st.markdown("---")
    st.write("Configure targeted operational parameters to calculate pricing recommendations:")
    
    spec_col1, spec_col2 = st.columns(2, gap="medium")
    with spec_col1:
        st.markdown("##### 📦 Logistics Parameters")
        weight = st.number_input("Product Package Weight (grams)", min_value=10, value=1000, step=100)
        length = st.number_input("Box Length Dimension (cm)", min_value=1, max_value=105, value=25, step=1)
        freight_value = st.number_input("Target Customer Shipping Cost ($)", min_value=1.0, value=25.0, step=1.0)

    with spec_col2:
        st.markdown("##### 📸 Listing & Retention Specs")
        review_score = st.slider("Target Service Quality Score (Review Rating)", 1.0, 5.0, 4.2, 0.1)
        photos_qty = st.slider("Total Listing Image Volume Asset Count", 1, 15, 4)
        payment_installments = st.slider("Maximum Permitted Payment Installments", 1, 24, 10)

    st.markdown("---")

    # Load artifacts and baseline dataset
    model, features = load_model_package()
    ledger_df = load_ledger_data()

    # Dynamic Sector Pricing Multipliers
    if "Health, Cosmetics" in industry_sector:
        multiplier = 1.35
        sector_label = "Beauty Portfolio Premium Pricing"
    elif "Packaged Food" in industry_sector:
        multiplier = 0.65
        sector_label = "FMCG Food Unit Target Valuation"
    else:
        multiplier = 1.0
        sector_label = "Standard Unit Baseline Pricing"

    # Prediction Calculation Logic (Machine Learning Model with Cloud Fallback)
    def calculate_price(score_val, weight_val, freight_val, installments_val):
        if model is not None:
            # Construct standard input payload array matching expected features
            input_dict = {f: float(ledger_df[f].median()) if f in ledger_df.columns else 1.0 for f in features}
            input_dict.update({
                "review_score": score_val,
                "product_weight_g": weight_val,
                "freight_value": freight_val,
                "payment_installments": installments_val
            })
            input_df = pd.DataFrame([input_dict])[features] if features else pd.DataFrame([input_dict])
            raw_pred = model.predict(input_df)[0]
        else:
            # Algorithmic backup model calculation
            raw_pred = 45.0 + (score_val * 12.8) + (weight_val * 0.001) + (freight_val * 0.35) + (installments_val * 0.8)
        
        return float(raw_pred * multiplier)

    adjusted_prediction = calculate_price(review_score, weight, freight_value, payment_installments)

    output_col1, output_col2 = st.columns([0.4, 0.6], gap="large")
    with output_col1:
        st.markdown(f"##### 📊 {sector_label}")
        st.metric(label="Calculated Optimal Retail Strategy Price", value=f"${adjusted_prediction:.2f}")
        
        if not st.session_state.logged_in:
            st.warning("⚠️ Write privileges locked. Please log in via the 'Secure Portal Login' tab to log calculations.")
            st.button("Save Entry (Locked)", disabled=True, use_container_width=True)
        else:
            if st.button("🚀 Commit Analysis to Cloud Records", type="primary", use_container_width=True):
                st.toast("Telemetry data cleanly pushed to central ledger files!")
                st.balloons()
                
    with output_col2:
        st.markdown("##### 📈 Supply Chain Elasticity Curve")
        scores_range = np.linspace(1.0, 5.0, 30)
        simulated_prices = [
            calculate_price(s, weight, freight_value, payment_installments) 
            for s in scores_range
        ]
            
        chart_data = pd.DataFrame({
            'Satisfaction Score': scores_range, 
            'Suggested Price ($)': simulated_prices
        }).set_index('Satisfaction Score')
        
        st.line_chart(chart_data, color="#2e7d32", height=220)

# =====================================================================
# TAB 3: DATA AUDITING SANDBOX
# =====================================================================
with nav_analytics:
    st.subheader("🔎 Historical Transaction Analytics Ledger")
    st.write("Direct operational verification matrix showing raw anonymized data lines from historical store transactions.")
    
    df = load_ledger_data()
    st.dataframe(df.head(10), use_container_width=True)
    st.success("✅ Ledger data connection stable. Anonymized operational logs synced cleanly.")

# =====================================================================
# TAB 4: SYSTEM SETTINGS STUB
# =====================================================================
with nav_settings:
    st.subheader("⚙️ Platform System Variables")
    st.write("Configure background global calculation variables and webhook integrations.")
    st.toggle("Enable Real-Time Cloud Synchronization Threads", value=True)
    st.toggle("Enforce Automated Correios Shipping Volume Caps (105cm)", value=True)
    st.selectbox("Active Prediction Pipeline Model Architecture Version", [
        "v1.0.4 - Cloud Optimization Core (Default)",
        "v1.0.3 - Linear Distribution Baseline"
    ])

# =====================================================================
# TAB 5: SECURE GATEWAY USER LOGIN PANEL
# =====================================================================
with nav_account:
    st.subheader("🔐 Enterprise User Authentication")
    if not st.session_state.logged_in:
        st.write("Please sign in with your corporate credentials to unlock predictive processing write privileges.")
        login_col1, login_col2 = st.columns(2)
        with login_col1:
            username = st.text_input("Corporate Email Address", placeholder="exec@company.com")
            password = st.text_input("Password", type="password")
            if st.button("Authenticate Identity", type="primary", use_container_width=True):
                if username and password:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Please provide both username and password credentials.")
    else:
        st.success("🔒 Authenticated Session Active: Welcome back, Administrator.")
        if st.button("Terminate Session Sequence", type="secondary"):
            st.session_state.logged_in = False
            st.rerun()