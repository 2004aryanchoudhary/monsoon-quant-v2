import streamlit as st
import pandas as pd
from db_handler import get_trade_history

# Configure the page layout
st.set_page_config(page_title="Monsoon Quant V2", page_icon="⛈️", layout="wide")

st.title("⛈️ Monsoon Quant: Agrochemical Alpha Engine")
st.markdown("""
This institutional-grade model uses domestic weather anomalies and cross-sectional momentum to predict alpha in Indian agricultural equities. 
* **Universe:** KSCL, RALLIS, COROMANDEL, DHANUKA
* **Risk Protocol:** Strict 60% confidence threshold for capital preservation.
""")

st.divider()

# Fetch data from the SQLite database
df = get_trade_history()

if df.empty:
    st.warning("No signal data found in the database. Please run the backend pipeline.")
else:
    # 1. Display the LATEST Signal (Tomorrow's Trade)
    latest_signal = df.iloc[0]
    
    st.subheader("📡 Active Market Signal")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Target Asset", value=latest_signal['ticker'])
    with col2:
        st.metric(label="Algorithm Confidence", value=f"{float(latest_signal['confidence'])*100:.2f}%")
    with col3:
        st.metric(label="Signal Date", value=latest_signal['date'])

    # 2. Display the Historical Ledger
    st.divider()
    st.subheader("📊 Cross-Sectional Backtest Ledger")
    
    # Format the dataframe for display
    display_df = df.copy()
    display_df['confidence'] = (display_df['confidence'].astype(float) * 100).map("{:.2f}%".format)
    display_df['forward_return'] = (display_df['forward_return'].astype(float) * 100).map("{:+.2f}%".format)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)