import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as si
from arch import arch_model
import plotly.graph_objects as go
import yaml, os, time, requests

# Optional ML
try:
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

# -----------------------------------------------------------------------------
# PAGE CONFIG & CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AlphaQuant Terminal Pro", layout="wide")
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0c29, #302b63, #24243e); color: white; }
    .stButton>button { border-radius:8px; background:linear-gradient(135deg, #667eea, #764ba2); color:white; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
defaults = {
    'live_mode': False, 'selected_market': "Crypto", 'long_hist_data': {},
    'ml_model_trained': False, 'paper_balance': 100000, 'trade_journal': []
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (Refactored for Stability)
# -----------------------------------------------------------------------------
def yf_download_retry(ticker, period="2y", interval="1d", max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if not data.empty: return data
        except: pass
        time.sleep(1)
    return pd.DataFrame()

def calculate_hurst_exponent(ts):
    if len(ts) < 20: return np.nan
    lags = range(2, min(20, len(ts)//5))
    try:
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0
    except: return np.nan

# -----------------------------------------------------------------------------
# ANALYTICS PLOT: HURST
# -----------------------------------------------------------------------------
def plot_hurst(asset_choice):
    # Retrieve data from session state safely
    data_bundle = st.session_state.get('long_hist_data', {})
    if asset_choice not in data_bundle: return None
    
    close = data_bundle[asset_choice]['Close'].squeeze()
    if len(close) < 120: return None
    
    log_prices = np.log(close)
    hurst_series = log_prices.rolling(60).apply(lambda x: calculate_hurst_exponent(x.values), raw=False)
    df = pd.DataFrame({'Close': close, 'Hurst': hurst_series}).dropna()
    
    if df.empty: return None
    
    current_hurst = float(df['Hurst'].iloc[-1])
    regime = "MEAN REVERTING" if current_hurst < 0.45 else "TRENDING" if current_hurst > 0.55 else "RANDOM WALK"
    color_theme = '#FF3333' if "MEAN" in regime else '#00FF00' if "TREND" in regime else '#FFA500'
    
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,6), gridspec_kw={'height_ratios':[1.5,1]})
    ax1.plot(df.index, df['Close'], color='white')
    ax1.set_title(f'Market Regime: {regime}', color='white')
    
    ax2.plot(df.index, df['Hurst'], color=color_theme, linewidth=2)
    ax2.axhline(0.5, color='white', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Hurst Exponent')
    
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# MAIN UI
# -----------------------------------------------------------------------------
st.title("AlphaQuant Terminal Pro")
asset_choice = st.sidebar.selectbox("Select Asset", ["Bitcoin", "Ethereum", "Nifty 50"])

# Placeholder for data fetching logic
if st.sidebar.button("Fetch Data"):
    with st.spinner("Downloading market data..."):
        # Example fetching
        ticker_map = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Nifty 50": "^NSEI"}
        df = yf_download_retry(ticker_map[asset_choice])
        st.session_state['long_hist_data'][asset_choice] = df
        st.success("Data Loaded!")

# Display Plot
if asset_choice in st.session_state['long_hist_data']:
    fig = plot_hurst(asset_choice)
    if fig: st.pyplot(fig)
else:
    st.info("Please select an asset and click 'Fetch Data' in the sidebar.")