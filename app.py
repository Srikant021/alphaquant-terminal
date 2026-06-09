# crypto_alphaquant_with_rolling_metrics_and_explanation.py
# AlphaQuant Terminal — Patched Architecture with Cross-Tab Synthesis Engine

import logging
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

# Optional ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("alphaquant")

st.set_page_config(page_title="AlphaQuant Terminal", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background: #080d12; }
    .metric-box {
        background: linear-gradient(135deg, #0d1520 0%, #111c2b 100%);
        padding: 16px 18px; border-radius: 8px; border-left: 3px solid #0af;
        margin: 5px 0; border-top: 1px solid rgba(0,170,255,0.08);
    }
    .synthesis-card {
        background: linear-gradient(135deg, #09101a 0%, #0d1726 100%);
        border: 1px solid rgba(0, 170, 255, 0.15);
        border-left: 4px solid #0af;
        padding: 20px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
    .section-header {
        font-family: 'Space Mono', monospace; font-size: 13px; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase; color: #0af;
        margin: 24px 0 12px 0; padding-bottom: 6px;
        border-bottom: 1px solid rgba(0,170,255,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TECHNICAL INDICATORS & MATH HELPERS
# ─────────────────────────────────────────────
def compute_rsi(series, period=14):
    if len(series) < period: return pd.Series(50, index=series.index)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs)).fillna(50)


def bollinger_bands(series, period=20, std=2.0):
    if len(series) < period: return series, series, series
    sma = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper_band = sma + (rolling_std * std)
    lower_band = sma - (rolling_std * std)
    return upper_band, sma, lower_band


def compute_atr(df, period=14):
    if len(df) < period: return pd.Series(0, index=df.index)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()


def compute_macd(series, fast=12, slow=26, signal=9):
    if len(series) < slow: return series, series, series
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def hurst_exponent(price_series):
    price = np.asarray(price_series.dropna(), dtype=float)
    n = len(price)
    if n < 35: return 0.5, "Insufficient Data Profile", "low"
    log_prices = np.log(price)
    max_lag = min(n // 2, 200)
    lags = np.unique(np.logspace(1, np.log10(max_lag), num=30).astype(int))
    lags = lags[lags >= 5]
    rs_values, valid_lags = [], []
    for lag in lags:
        n_windows = n // lag
        if n_windows < 2: continue
        rs_window = []
        for i in range(n_windows):
            window = log_prices[i * lag:(i + 1) * lag]
            mean_adj = window - window.mean()
            cumsum = np.cumsum(mean_adj)
            R = cumsum.max() - cumsum.min()
            S = window.std(ddof=1)
            if S > 1e-10: rs_window.append(R / S)
        if len(rs_window) >= 2:
            rs_values.append(np.mean(rs_window))
            valid_lags.append(lag)
    if len(valid_lags) < 4: return 0.5, "Random Walk", "low"
    log_lags = np.log(valid_lags)
    log_rs = np.log(rs_values)
    coeffs = np.polyfit(log_lags, log_rs, 1)
    hurst = float(np.clip(coeffs[0], 0.05, 0.95))
    r2 = 1 - (np.sum((log_rs - np.polyval(coeffs, log_lags)) ** 2) / np.sum((log_rs - np.mean(log_rs)) ** 2)) if np.sum((log_rs - np.mean(log_rs)) ** 2) > 0 else 0
    confidence = "high" if (r2 > 0.95 and len(valid_lags) >= 6) else "medium" if r2 > 0.85 else "low"
    interp = "Strong Trend" if hurst > 0.58 else "Weak Trend" if hurst > 0.53 else "Random Walk" if hurst >= 0.47 else "Weak Mean-Reversion" if hurst >= 0.42 else "Strong Mean-Reversion"
    return hurst, interp, confidence


# ─────────────────────────────────────────────
# SAFE DATA FETCHING & FLATTENER
# ─────────────────────────────────────────────
def _flatten_multiindex(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty: return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data


def _download_with_retry(ticker, period, interval, attempts=3, backoff=1.5):
    for i in range(attempts):
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if data is not None and not data.empty: return data
        except Exception:
            time.sleep(backoff ** i)
    return None


@st.cache_data(ttl=60)
def fetch_data(ticker, period="1y", interval="1d"):
    try:
        raw = _download_with_retry(ticker, period, interval)
        return _flatten_multiindex(raw)
    except Exception as e:
        logger.exception(f"Error fetching {ticker}: {e}")
        return None


# ─────────────────────────────────────────────
# EXTRACTION HELPERS FOR COHESIVE ANALYSIS
# ─────────────────────────────────────────────
def get_volatility_state(ticker):
    """Safely harvests key implied and historical metrics for unified tab interpretation."""
    state = {"ivr": 50.0, "ivp": 50.0, "vrp": 0.0, "vix": 15.0, "hv": 15.0}
    vix_data = fetch_data("^INDIAVIX", period="6mo", interval="1d")
    asset_data = fetch_data(ticker, period="6mo", interval="1d")
    
    if vix_data is not None and len(vix_data) > 10:
        vix_series = vix_data['Close']
        if isinstance(vix_series, pd.DataFrame): vix_series = vix_series.iloc[:, 0]
        c_vix = float(vix_series.iloc[-1])
        h_52w, l_52w = float(vix_series.max()), float(vix_series.min())
        state["vix"] = c_vix
        state["ivr"] = ((c_vix - l_52w) / (h_52w - l_52w + 1e-9)) * 100
        state["ivp"] = ((vix_series < c_vix).sum() / len(vix_series)) * 100
        
        if asset_data is not None and len(asset_data) > 22:
            asset_close = asset_data['Close']
            if isinstance(asset_close, pd.DataFrame): asset_close = asset_close.iloc[:, 0]
            hv = float(np.log(asset_close / asset_close.shift(1)).rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
            state["hv"] = hv
            state["vrp"] = c_vix - hv
    return state


# ─────────────────────────────────────────────
# GRAPHICS ENGINE (PROTECTED AGAINST SCALAR TRANSFORMS)
# ─────────────────────────────────────────────
def plot_candlestick(df, name, timeframe):
    if df is None or df.empty or len(df) < 5: return None
    plot_df = df.tail(100)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.25, 0.75])
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'].iloc[:,0] if isinstance(plot_df['Open'], pd.DataFrame) else plot_df['Open'],
                                 high=plot_df['High'].iloc[:,0] if isinstance(plot_df['High'], pd.DataFrame) else plot_df['High'],
                                 low=plot_df['Low'].iloc[:,0] if isinstance(plot_df['Low'], pd.DataFrame) else plot_df['Low'],
                                 close=plot_df['Close'].iloc[:,0] if isinstance(plot_df['Close'], pd.DataFrame) else plot_df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'].iloc[:,0] if isinstance(plot_df['Volume'], pd.DataFrame) else plot_df['Volume'], name='Volume', marker_color='#00aaff'), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark', height=400, margin=dict(t=10, b=10, l=10, r=10))
    return fig


def plot_india_vix_summary():
    data = fetch_data("^INDIAVIX", period="1y", interval="1d")
    if data is None or data.empty or len(data) < 10: return None
    close_prices = data['Close']
    if isinstance(close_prices, pd.DataFrame): close_prices = close_prices.iloc[:, 0]
    
    current_iv, high_52w, low_52w = float(close_prices.iloc[-1]), float(close_prices.max()), float(close_prices.min())
    ivr = ((current_iv - low_52w) / (high_52w - low_52w + 1e-9)) * 100
    ivp = ((close_prices < current_iv).sum() / len(close_prices)) * 100
    regime = "HIGH VOLATILITY" if ivr > 50 else "LOW VOLATILITY"

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=100)
    ax.plot(close_prices.index, close_prices.values, color='#00FFFF', linewidth=1.5)
    ax.axhline(high_52w, color='red', linestyle='--', alpha=0.5, label=f'52W High: {high_52w:.2f}')
    ax.axhline(low_52w, color='green', linestyle='--', alpha=0.5, label=f'52W Low: {low_52w:.2f}')
    ax.grid(True, color='#2A2A2A', linestyle=':')
    ax.legend(loc='upper right')
    props = {'boxstyle': 'round', 'facecolor': 'black', 'alpha': 0.8, 'edgecolor': '#0af'}
    ax.text(0.02, 0.95, f"IVR: {ivr:.1f}% | IVP: {ivp:.1f}%\nRegime: {regime}", transform=ax.transAxes, bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_vrp(ticker, name):
    nifty_data = fetch_data(ticker, period="6mo", interval="1d")
    vix_data = fetch_data("^INDIAVIX", period="6mo", interval="1d")
    if nifty_data is None or vix_data is None or len(nifty_data) < 25 or len(vix_data) < 25: return None
    
    n_close = nifty_data['Close'].iloc[:, 0] if isinstance(nifty_data['Close'], pd.DataFrame) else nifty_data['Close']
    v_close = vix_data['Close'].iloc[:, 0] if isinstance(vix_data['Close'], pd.DataFrame) else vix_data['Close']
    
    df = pd.DataFrame({'VIX': v_close, 'HV': (np.log(n_close / n_close.shift(1)).rolling(20).std() * np.sqrt(252) * 100)}).dropna()
    if df.empty: return None
    df['VRP'] = df['VIX'] - df['HV']

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), dpi=100, gridspec_kw={'height_ratios': [1.5, 1]})
    ax1.plot(df.index, df['VIX'], color='red', label='India VIX')
    ax1.plot(df.index, df['HV'], color='dodgerblue', label=f'20D HV ({name})')
    ax1.set_title(f'Volatility Risk Premium Variance Spread: {name}', fontsize=11, color='white', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, color='#2A2A2A')
    ax2.bar(df.index, df['VRP'], color=np.where(df['VRP'] > 0, '#00FF00', '#FF3333'), alpha=0.6)
    ax2.axhline(0, color='white', linewidth=0.8)
    ax2.grid(True, color='#2A2A2A')
    plt.tight_layout()
    return fig


def plot_volatility_cone(ticker, name):
    data = fetch_data(ticker, period="3y", interval="1d")
    if data is None or data.empty or len(data) < 260: return None
    close = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
    returns = np.log(close / close.shift(1))
    windows = [10, 20, 30, 60, 90, 120, 252]
    max_vol, min_vol, median_vol, current_vol = [], [], [], []
    for w in windows:
        rolling_vol = returns.rolling(window=w).std() * np.sqrt(252)
        max_vol.append(rolling_vol.max() * 100)
        min_vol.append(rolling_vol.min() * 100)
        median_vol.append(rolling_vol.median() * 100)
        current_vol.append(rolling_vol.dropna().iloc[-1] * 100)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    ax.plot(windows, max_vol, marker='o', color='red', label='Max')
    ax.plot(windows, min_vol, marker='o', color='limegreen', label='Min')
    ax.plot(windows, median_vol, marker='s', color='white', linestyle='--', label='Median')
    ax.plot(windows, current_vol, marker='X', color='yellow', linewidth=2, label='Current')
    ax.fill_between(windows, min_vol, max_vol, color='gray', alpha=0.1)
    ax.set_title(f'Volatility Cone Structure: {name}', fontsize=11, fontweight='bold')
    ax.set_xticks(windows)
    ax.grid(color='gray', linestyle=':', alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig


def plot_expected_move(ticker, name):
    nifty_data = fetch_data(ticker, period="1mo", interval="1d")
    vix_data = fetch_data("^INDIAVIX", period="5d", interval="1d")
    if nifty_data is None or vix_data is None or len(nifty_data) < 5 or len(vix_data) < 1: return None
    n_close = nifty_data['Close'].iloc[:, 0] if isinstance(nifty_data['Close'], pd.DataFrame) else nifty_data['Close']
    v_close = vix_data['Close'].iloc[:, 0] if isinstance(vix_data['Close'], pd.DataFrame) else vix_data['Close']
    
    spot_price = float(n_close.iloc[-1])
    current_vix = float(v_close.iloc[-1])
    expected_move_points = spot_price * ((current_vix / 100) * np.sqrt(1/365))
    upper_bound, lower_bound = spot_price + expected_move_points, spot_price - expected_move_points

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=100)
    recent = n_close.tail(15)
    x_dates = np.arange(len(recent))
    ax.plot(x_dates, recent.values, color='#00FFFF', linewidth=1.5, marker='o')
    tomorrow_x = len(recent)
    ax.scatter(tomorrow_x, upper_bound, color='#00FF00', s=80, marker='^')
    ax.scatter(tomorrow_x, lower_bound, color='#FF3333', s=80, marker='v')
    ax.hlines([upper_bound, lower_bound], xmin=x_dates[-1], xmax=tomorrow_x, colors=['#00FF00', '#FF3333'], linestyles='--')
    ax.set_title(f'{name} Implied Expected Move Range', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    plt.tight_layout()
    return fig


def plot_liquidity_sweep(ticker, name):
    data = fetch_data(ticker, period="5d", interval="15m")
    if data is None or data.empty or len(data) < 25: return None
    close = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
    high = data['High'].iloc[:, 0] if isinstance(data['High'], pd.DataFrame) else data['High']
    low = data['Low'].iloc[:, 0] if isinstance(data['Low'], pd.DataFrame) else data['Low']
    open_p = data['Open'].iloc[:, 0] if isinstance(data['Open'], pd.DataFrame) else data['Open']
    
    prev_high = high.rolling(20).max().shift(1)
    prev_low = low.rolling(20).min().shift(1)
    supply_sweep = (high > prev_high) & (close < prev_high)
    demand_sweep = (low < prev_low) & (close > prev_low)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=100)
    plot_data = data.tail(40).copy()
    plot_idx = np.arange(len(plot_data))
    
    for idx, i in enumerate(plot_data.index):
        color = '#00FF00' if close.loc[i] >= open_p.loc[i] else '#FF3333'
        ax.vlines(plot_idx[idx], low.loc[i], high.loc[i], color=color, linewidth=1)
        ax.bar(plot_idx[idx], abs(close.loc[i] - open_p.loc[i]), bottom=min(open_p.loc[i], close.loc[i]), color=color, width=0.5)
        if supply_sweep.loc[i]: ax.scatter(plot_idx[idx], high.loc[i], marker='v', color='magenta', s=80)
        if demand_sweep.loc[i]: ax.scatter(plot_idx[idx], low.loc[i], marker='^', color='orange', s=80)
            
    ax.set_title(f'{name} Intraday Microstructure Liquidity Sweeps', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    plt.tight_layout()
    return fig


def plot_index_divergence():
    d1 = fetch_data("^NSEI", period="6mo", interval="1d")
    d2 = fetch_data("^NSEBANK", period="6mo", interval="1d")
    if d1 is None or d2 is None or d1.empty or d2.empty: return None
    c1 = d1['Close'].iloc[:,0] if isinstance(d1['Close'], pd.DataFrame) else d1['Close']
    c2 = d2['Close'].iloc[:,0] if isinstance(d2['Close'], pd.DataFrame) else d2['Close']
    
    shared = c1.index.intersection(c2.index)
    n1 = (c1.loc[shared] / c1.loc[shared].iloc[0]) * 100
    n2 = (c2.loc[shared] / c2.loc[shared].iloc[0]) * 100
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    ax.plot(shared, n1, color='#00FFFF', label='Nifty 50')
    ax.plot(shared, n2, color='#FFA500', label='Bank Nifty')
    ax.set_title("Inter-Index Correlation Spread Framework", fontsize=11, fontweight='bold')
    ax.grid(True, color='#2A2A2A')
    ax.legend(loc='upper left')
    plt.tight_layout()
    return fig


def plot_hurst_regime(ticker, name):
    data = fetch_data(ticker, period="1y", interval="1d")
    if data is None or data.empty or len(data) < 50: return None
    close = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
    
    def calc_h(ts):
        if len(ts) < 25: return 0.5
        reg = [np.std(ts.values[l:] - ts.values[:-l]) for l in range(2, 12)]
        return np.polyfit(np.log(range(2, 12)), np.log(reg), 1)[0]
        
    h_series = np.log(close).rolling(45).apply(calc_h, raw=False).dropna()
    if h_series.empty: return None

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    ax.plot(h_series.index, h_series.values, color='#00FFFF', label='Hurst Exponent')
    ax.axhline(0.55, color='#00FF00', linestyle='--', alpha=0.6, label='Trend Frontier')
    ax.axhline(0.45, color='#FF3333', linestyle='--', alpha=0.6, label='Mean-Reversion')
    ax.set_ylim(0.2, 0.8)
    ax.grid(True, color='#2A2A2A')
    ax.legend(loc='lower left')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# FEATURE PIPELINE
# ─────────────────────────────────────────────
def build_ml_features(df):
    feat = pd.DataFrame(index=df.index)
    close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
    feat['rsi'] = compute_rsi(close)
    feat['returns'] = close.pct_change()
    feat['vol_20'] = feat['returns'].rolling(20).std()
    bb_up, _, bb_lo = bollinger_bands(close)
    feat['bb_pos'] = (close - bb_lo) / (bb_up - bb_lo + 1e-9)
    feat['atr'] = compute_atr(df)
    
    vol = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
    feat['vol_ratio'] = vol / vol.rolling(20).mean().replace(0, np.nan)
    macd, sig, _ = compute_macd(close)
    feat['macd_diff'] = macd - sig
    return feat.dropna()


def explain_ml_prediction(model, feat_df, prob_pos):
    latest = feat_df.iloc[-1]
    importances = pd.Series(model.feature_importances_, index=feat_df.columns).sort_values(ascending=False)
    top_feats = importances.head(3).index.tolist()
    supporting, opposing = [], []

    for f in top_feats:
        v = latest.get(f, np.nan)
        if pd.isna(v): continue
        if f == 'rsi':
            if v < 42: supporting.append(f"RSI is structure-oversold ({v:.1f}). Mean reversion imminent.")
            elif v > 58: opposing.append(f"RSI is compression-overbought ({v:.1f}). Overhead boundary overhead.")
        elif f == 'macd_diff':
            if v > 0: supporting.append(f"MACD histogram is expansion-positive ({v:.4f}).")
            else: opposing.append(f"MACD distribution vector is fading ({v:.4f}).")
        elif f == 'bb_pos':
            if v < 0.25: supporting.append(f"Price resting at lower Bollinger Band ({v:.2f}). Channel floor holds.")
            elif v > 0.75: opposing.append(f"Price scanning upper Bollinger variance tier ({v:.2f}). Extension risk.")
    return {"supporting": supporting, "opposing": opposing, "probability": prob_pos}


# ─────────────────────────────────────────────
# CORE EXECUTION ENTRYPOINT
# ─────────────────────────────────────────────
def main():
    st.title("⚡ AlphaQuant Terminal")
    
    st.sidebar.header("Navigation & Asset Matrix")
    market_selection = st.sidebar.radio("Select Target Infrastructure Context", ["Indian Market", "Crypto Assets"])
    
    if market_selection == "Indian Market":
        indian_index = st.sidebar.selectbox("Select Benchmark Index", ["Nifty 50", "Bank Nifty", "Sensex", "Finnifty"])
        index_mapping = {"Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK", "Sensex": "^BSESN", "Finnifty": "^CNXFIN"}
        ticker = index_mapping[indian_index]
        name = indian_index
    else:
        crypto_ticker = st.sidebar.selectbox("Select Derivative Asset Pool", ["BTC-USD", "ETH-USD", "SOL-USD"])
        ticker = crypto_ticker
        name = crypto_ticker.split("-")[0]

    timeframe = st.sidebar.selectbox("Select System Interval Matrix", ["15m", "30m", "1h", "1d"], index=3)
    
    tf_period_map = {
        "15m": {"interval": "15m", "period": "1mo"},
        "30m": {"interval": "30m", "period": "1mo"},
        "1h": {"interval": "60m", "period": "3mo"},
        "1d": {"interval": "1d", "period": "2y"}
    }
    
    selected_tf = tf_period_map[timeframe]
    df_live = fetch_data(ticker, period=selected_tf["period"], interval=selected_tf["interval"])

    tab_vol, tab_struct, tab_ml = st.tabs(["📊 Volatility Engine", "📐 Structural Analysis", "🤖 ML Explanations & Cross-Analysis Matrix"])

    # 1. VOLATILITY PIPELINE
    with tab_vol:
        st.header(f"Volatility Spread Pipeline: {name}")
        if market_selection == "Indian Market":
            col1, col2 = st.columns(2)
            with col1:
                vix_chart = plot_india_vix_summary()
                if vix_chart: st.pyplot(vix_chart)
            with col2:
                vrp_chart = plot_vrp(ticker, name)
                if vrp_chart: st.pyplot(vrp_chart)
            cone_chart = plot_volatility_cone(ticker, name)
            if cone_chart: st.pyplot(cone_chart)
        else:
            v_close = df_live['Close'].iloc[:, 0] if isinstance(df_live['Close'], pd.DataFrame) else df_live['Close']
            c_vol = float(np.log(v_close / v_close.shift(1)).rolling(20).std().iloc[-1] * np.sqrt(252) * 100) if len(v_close) > 22 else 0.0
            st.metric("20-Day Native Asset Volatility", f"{c_vol:.2f}%")

    # 2. STRUCTURAL MICROSTRUCTURE
    with tab_struct:
        st.header("Structural Analysis Engine")
        if df_live is not None and not df_live.empty:
            candle_fig = plot_candlestick(df_live, name, timeframe)
            if candle_fig: st.plotly_chart(candle_fig, use_container_width=True)
            
        if market_selection == "Indian Market":
            col1, col2 = st.columns(2)
            with col1:
                exp_chart = plot_expected_move(ticker, name)
                if exp_chart: st.pyplot(exp_chart)
            with col2:
                sweep_chart = plot_liquidity_sweep(ticker, name)
                if sweep_chart: st.pyplot(sweep_chart)
            
            c1, c2 = st.columns(2)
            with c1:
                div_chart = plot_index_divergence()
                if div_chart: st.pyplot(div_chart)
            with c2:
                hurst_chart = plot_hurst_regime(ticker, name)
                if hurst_chart: st.pyplot(hurst_chart)

    # 3. ML EXPLANATIONS AND UNIFIED SYNTHESIS MATRIX
    with tab_ml:
        st.header(f"Predictive Intelligence & Cross-Tab Synthesis ({timeframe})")
        
        if df_live is None or df_live.empty or len(df_live) < 35:
            st.warning("Insufficient dataframe footprint to activate Machine Learning models.")
            return
            
        features = build_ml_features(df_live)
        c_close = df_live['Close'].iloc[:, 0] if isinstance(df_live['Close'], pd.DataFrame) else df_live['Close']
        target = (c_close.pct_change().shift(-1) > 0).astype(int).loc[features.index]
        common_idx = features.index.intersection(target.index)
        X, y = features.loc[common_idx], target.loc[common_idx]

        if len(X) > 25:
            split = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, _ = y.iloc[:split], y.iloc[split:]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            
            base_rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
            calibrated_model = CalibratedClassifierCV(base_rf, method='sigmoid', cv=3)
            calibrated_model.fit(X_train_s, y_train)
            base_rf.fit(X_train_s, y_train)

            latest_vector = scaler.transform(X.iloc[[-1]])
            prob = float(calibrated_model.predict_proba(latest_vector)[0][1])
            explanation = explain_ml_prediction(base_rf, X, prob)
            
            # ──────────────────────────────────────────────────────────
            # NEW CROSS-TAB ANALYSIS SYNTHESIS ENGINE
            # ──────────────────────────────────────────────────────────
            st.markdown("<div class='section-header'>🔬 Quantitative Cross-Tab Synthesis Executive Summary</div>", unsafe_allow_html=True)
            
            # Gather state variables from earlier modules safely
            vol_metrics = get_volatility_state(ticker) if market_selection == "Indian Market" else {"ivr": 50, "vrp": 0, "vix": 0}
            h_val, h_interp, _ = hurst_exponent(c_close)
            
            # Synthesize trade action vectors based on Options & Technical parameters
            if market_selection == "Indian Market":
                if vol_metrics["ivr"] > 60 and vol_metrics["vrp"] > 2.0:
                    vol_edge_str = f"**HIGH VOLATILITY PREMIUM SPARKED** (IVR: {vol_metrics['ivr']:.1f}%, VRP: {vol_metrics['vrp']:.1f}%). Strategic short premium option configurations (Iron Condors, Strangle credit arrays) hold a historical mathematical execution edge."
                elif vol_metrics["ivr"] < 35:
                    vol_edge_str = f"**COMPRESSED IMPLIED COMPLEX** (IVR: {vol_metrics['ivr']:.1f}%). Option premiums are underpriced relative to statistical variance models. Shift risk metrics toward long exposure or tight debit structures."
                else:
                    vol_edge_str = f"**VOLATILITY SPREAD EQUILIBRIUM** (IVR: {vol_metrics['ivr']:.1f}%). Volatility layers match structural realized trends. Edge must be derived exclusively from directional micro-sweeps."
            else:
                vol_edge_str = "Crypto-derivative metrics are trading inside normal risk channels. Pure structural price parameters take strategy precedence."

            if h_val > 0.55:
                struct_edge_str = f"**PERSISTENT MEMORY EXPANSION** (Hurst: {h_val:.3f} — {h_interp}). The asset exhibits structural continuation memory vectors. Trend-following breakout triggers on higher frames possess velocity confirmation."
            elif h_val < 0.45:
                struct_edge_str = f"**MEAN-REVERTING ELASTIC STATE** (Hurst: {h_val:.3f} — {h_interp}). High structural decay. Breakouts are prone to structural liquidity sweep failures. Deploy range-bound oscillation tactics at the expected move boundaries."
            else:
                struct_edge_str = f"**RANDOM WALK EQUILIBRIUM** (Hurst: {h_val:.3f}). Price matrix has entered efficient geometric noise. Sideline allocations or apply tight scalar delta framing."

            # Render Synthesis Dashboard
            st.markdown(f"""
            <div class='synthesis-card'>
                <h4>📊 Volatility Engine Contextual Overlay</h4>
                <p style='font-size:14px; color:rgba(255,255,255,0.85);'>{vol_edge_str}</p>
                <hr style='border:0; border-top:1px solid rgba(255,255,255,0.1); margin:12px 0;'>
                <h4>📐 Structural & Memory Alignment Matrix</h4>
                <p style='font-size:14px; color:rgba(255,255,255,0.85);'>{struct_edge_str}</p>
                <hr style='border:0; border-top:1px solid rgba(255,255,255,0.1); margin:12px 0;'>
                <h4>🤖 Unified Machine Learning Bias</h4>
                <p style='font-size:14px; color:rgba(255,255,255,0.85);'>The predictive network models next-bar directional distribution at <strong>{prob*100:.1f}% Upside Probability</strong>. This vector aligns as <strong>{"BULLISH ACCELERATION" if prob > 0.54 else "BEARISH ACCELERATION" if prob < 0.46 else "STRUCTURAL MEAN-NEUTRAL CHOP"}</strong> within the current {timeframe} footprint.</p>
            </div>
            """, unsafe_allow_html=True)

            # Traditional Feature metrics layout below
            st.markdown("<div class='section-header'>Explainable AI Rationale Breakdown</div>", unsafe_allow_html=True)
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("### 👍 Alpha Supporting Drivers")
                for item in explanation["supporting"]: st.markdown(f"* {item}")
                if not explanation["supporting"]: st.markdown("_No dominant supporting criteria identified in top-tier feature space weights._")
            with col_m2:
                st.markdown("### 👎 Delta Opposing Vectors")
                for item in explanation["opposing"]: st.markdown(f"* {item}")
                if not explanation["opposing"]: st.markdown("_No material opposing resistance metrics calculated in active state window._")
        else:
            st.warning("Data density profile too small to execute step-forward walk classification systems.")


if __name__ == '__main__':
    main()