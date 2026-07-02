import os
import subprocess
import sys

# --- FORCED CLOUD INSTALLATION ---
# This forces Streamlit to install shap and matplotlib at runtime if it missed them
try:
    import shap
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "matplotlib"])
    import shap
    import matplotlib.pyplot as plt
# ---------------------------------

import streamlit as st
import pandas as pd
import altair as alt
import joblib
import numpy as np
from db_handler import get_trade_history

# Configure the page layout
st.set_page_config(page_title="Monsoon Quant V2", page_icon="⛈️", layout="wide")

st.title("⛈️ Monsoon Quant: Agrochemical Alpha Engine")
st.markdown("This institutional-grade model uses domestic weather anomalies and cross-sectional momentum to predict alpha in Indian agricultural equities.")
st.divider()

# Create standard terminal tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Cross-Sectional Backtest", 
    "🎛️ Live ML Inference Engine", 
    "🏦 Live Execution Ledger"
])

# ==========================================
# TAB 1: HISTORICAL BACKTEST
# ==========================================
with tab1:
    st.sidebar.header("⚙️ Strategy Controls")
    df = get_trade_history()

    if df.empty:
        st.warning("No signal data found. Please run the backend pipeline.")
    else:
        unique_tickers = ["ALL"] + list(df['ticker'].unique())
        selected_ticker = st.sidebar.selectbox("Filter by Asset", unique_tickers)
        
        min_conf, max_conf = float(df['confidence'].min()), float(df['confidence'].max())
        confidence_threshold = st.sidebar.slider(
            "Minimum Confidence Threshold", 
            min_value=min_conf, max_value=max_conf, value=0.60, step=0.01, format="%.2f"
        )

        filtered_df = df[df['confidence'] >= confidence_threshold].copy()
        if selected_ticker != "ALL":
            filtered_df = filtered_df[filtered_df['ticker'] == selected_ticker]
            
        st.subheader("📈 Strategy Performance Visualization")
        if not filtered_df.empty:
            chart_df = filtered_df.iloc[::-1].copy()
            chart_df['Cumulative Return'] = (1 + chart_df['forward_return'].astype(float)).cumprod() - 1
            
            equity_chart = alt.Chart(chart_df).mark_area(
                line={'color': '#1f77b4'},
                color=alt.Gradient(
                    gradient='linear',
                    stops=[alt.GradientStop(color='#1f77b4', offset=0), alt.GradientStop(color='white', offset=1)],
                    x1=1, x2=1, y1=1, y2=0
                )
            ).encode(
                x=alt.X('date:T', title='Timeline'),
                y=alt.Y('Cumulative Return:Q', title='Compounded Alpha', axis=alt.Axis(format='%')),
                tooltip=['date', 'ticker', 'forward_return', 'Cumulative Return']
            ).properties(height=350)
            
            st.altair_chart(equity_chart, use_container_width=True)
        else:
            st.info("No trades match your current filter settings.")

        st.subheader("📊 Cross-Sectional Backtest Ledger")
        display_df = filtered_df.copy()
        display_df['confidence'] = (display_df['confidence'].astype(float) * 100).map("{:.2f}%".format)
        display_df['forward_return'] = (display_df['forward_return'].astype(float) * 100).map("{:+.2f}%".format)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ==========================================
# TAB 2: LIVE ML INFERENCE ENGINE
# ==========================================
with tab2:
    st.subheader("🎛️ Scenario Stress Testing")
    st.markdown("Adjust the normalized feature parameters below to see how the XGBoost algorithm evaluates probability in real-time.")
    
    # Check if the model exists
    MODEL_PATH = "monsoon_xgboost.joblib"
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. Please run your backend pipeline to export the model.")
    else:
        # Load the model silently
        model = joblib.load(MODEL_PATH)
        
        # Build the interactive UI grid
        col_w1, col_w2 = st.columns(2)
        
        with col_w1:
            st.markdown("**☁️ Climate Parameters (Z-Scores)**")
            rain_val = st.slider("21-Day Cumulative Rainfall", -3.0, 3.0, 0.0, 0.1)
            temp_val = st.slider("21-Day Average Temperature", -3.0, 3.0, 0.0, 0.1)
            
        with col_w2:
            st.markdown("**📈 Financial Parameters (Normalized)**")
            bench_val = st.slider("NIFTY 50 Momentum", -3.0, 3.0, 0.0, 0.1)
            vol_val = st.slider("Asset Volatility", -3.0, 3.0, 0.0, 0.1)
            target_val = st.slider("Target Asset Momentum", -3.0, 3.0, 0.0, 0.1)
            price_ma_val = st.slider("Price vs 21-MA", -3.0, 3.0, 0.0, 0.1)
            rel_vol_val = st.slider("Relative Volume", -3.0, 3.0, 0.0, 0.1)

        # Feature names must match training data exactly for SHAP to label the chart
        feature_names = [
            'price_vs_ma21', 'relative_volume_21d', 'volatility_21', 
            'target_return_21d', 'bench_return_21d', 
            'weather_rain_cum_21d', 'weather_temp_avg_21d'
        ]
        
        # Convert inputs to a DataFrame so SHAP can read the column names
        live_df = pd.DataFrame([[
            price_ma_val, rel_vol_val, vol_val, target_val, bench_val, rain_val, temp_val
        ]], columns=feature_names)
        
        # Run live probability inference
        probability = model.predict_proba(live_df)[0][1]
        
        st.divider()
        st.subheader("🧠 Live Algorithmic Output")
        
        # Display the output with conditional color logic
        if probability >= 0.60:
            st.success(f"**EXECUTE TRADE:** The model predicts a {probability*100:.2f}% probability of outperformance.")
        elif probability >= 0.50:
            st.warning(f"**HOLD CASH:** Positive expectation ({probability*100:.2f}%) but fails the 60% capital preservation threshold.")
        else:
            st.error(f"**NEGATIVE EXPECTATION:** The model predicts a {probability*100:.2f}% probability. Strict cash allocation.")

        # --- NEW: SHAP ALGORITHMIC TRANSPARENCY VISUALIZATION ---
        st.divider()
        st.subheader("🔍 Algorithmic Transparency (SHAP Analysis)")
        st.markdown("This Waterfall Chart visualizes the mathematical forces driving the current prediction. **Red bars** represent variables pushing the probability higher; **Blue bars** represent variables pushing it lower.")
        
        with st.spinner("Calculating SHAP mathematical attributions..."):
            # Initialize the Explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(live_df)
            
            # Generate the plot
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.plots.waterfall(shap_values[0], show=False)
            
            # Adjust layout to fit Streamlit cleanly
            plt.tight_layout()
            st.pyplot(fig)

# ==========================================
# TAB 3: LIVE EXECUTION LEDGER (PAPER TRADING)
# ==========================================
with tab3:
    st.subheader("🏦 Autonomous Execution Ledger")
    st.markdown("This ledger tracks the live, forward-looking paper trades executed by the autonomous daily daemon.")
    
    from db_handler import get_paper_ledger
    ledger_df = get_paper_ledger()
    
    if ledger_df.empty:
        st.info("No paper trades have been executed yet. The daemon will log trades here at 3:15 PM.")
    else:
        # Calculate Total Virtual Capital Deployed
        total_deployed = ledger_df['total_value'].sum()
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Total Trades Executed", len(ledger_df))
        with col_m2:
            st.metric("Total Capital Deployed", f"₹{total_deployed:,.2f}")
            
        st.divider()
        
        # Format the dataframe nicely
        display_ledger = ledger_df.copy()
        display_ledger['fill_price'] = display_ledger['fill_price'].map("₹{:.2f}".format)
        display_ledger['total_value'] = display_ledger['total_value'].map("₹{:.2f}".format)
        
        st.dataframe(display_ledger, use_container_width=True, hide_index=True)