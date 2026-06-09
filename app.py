# crypto_alphaquant_with_rolling_metrics_and_explanation.py
# AlphaQuant Terminal — Enhanced Architecture with Multi-Index Tracking and Multi-Timeframe Pipelines

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

# Basic logging to console
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
    .explanation-box {
        background: rgba(0,170,255,0.05); border-left: 3px solid #0af;
        padding: 10px 14px; border-radius: 4px; font-size: 12px; margin: 8px 0;
        color: rgba(255,255,255,0.75);
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
def compute_iv_rank(iv_series, window=252):
    rolling_min = iv_series.rolling(window=window, min_periods=1).min()
    rolling_max = iv_series.rolling(window=window, min_periods=1).max()
    denominator = (rolling_max - rolling_min).replace(0, np.nan)
    ivr = ((iv_series - rolling_min) / denominator) * 100
    return ivr.fillna(0)


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs)).fillna(50)


def bollinger_bands(series, period=20, std=2.0):
    sma = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper_band = sma + (rolling_std * std)
    lower_band = sma - (rolling_std * std)
    return upper_band, sma, lower_band


def compute_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def hurst_exponent(price_series):
    price = np.asarray(price_series.squeeze().dropna(), dtype=float)
    n = len(price)
    if n < 30:
        return 0.5, "Insufficient data", "low"
    log_prices = np.log(price)
    max_lag = min(n // 2, 200)
    lags = np.unique(np.logspace(1, np.log10(max_lag), num=30).astype(int))
    lags = lags[lags >= 5]
    rs_values, valid_lags = [], []
    for lag in lags:
        n_windows = n // lag
        if n_windows < 2:
            continue
        rs_window = []
        for i in range(n_windows):
            window = log_prices[i * lag:(i + 1) * lag]
            mean_adj = window - window.mean()
            cumsum = np.cumsum(mean_adj)
            R = cumsum.max() - cumsum.min()
            S = window.std(ddof=1)
            if S > 1e-10:
                rs_window.append(R / S)
        if len(rs_window) >= 2:
            rs_values.append(np.mean(rs_window))
            valid_lags.append(lag)
    if len(valid_lags) < 4:
        return 0.5, "Insufficient data", "low"
    log_lags = np.log(valid_lags)
    log_rs = np.log(rs_values)
    coeffs = np.polyfit(log_lags, log_rs, 1)
    hurst = float(np.clip(coeffs[0], 0.05, 0.95))
    predicted = np.polyval(coeffs, log_lags)
    ss_res = np.sum((log_rs - predicted) ** 2)
    ss_tot = np.sum((log_rs - np.mean(log_rs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    confidence = "high" if (r2 > 0.95 and len(valid_lags) >= 6) else "medium" if r2 > 0.85 else "low"
    if hurst > 0.58:
        interp = "Strong Trend (Persistent)"
    elif hurst > 0.53:
        interp = "Weak Trend"
    elif hurst >= 0.47:
        interp = "Random Walk"
    elif hurst >= 0.42:
        interp = "Weak Mean-Reversion"
    else:
        interp = "Strong Mean-Reversion"
    return hurst, interp, confidence


# ─────────────────────────────────────────────
# SAFE DATA FETCHING & FLATTENER
# ─────────────────────────────────────────────
def _flatten_multiindex(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data


def _download_with_retry(ticker, period, interval, attempts=3, backoff=1.5):
    last_exc = None
    for i in range(attempts):
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if data is not None and not data.empty:
                return data
        except Exception as e:
            last_exc = e
            time.sleep(backoff ** i)
    raise last_exc


@st.cache_data(ttl=60)
def fetch_data(ticker, period="1y", interval="1d"):
    try:
        raw = _download_with_retry(ticker, period, interval)
        return _flatten_multiindex(raw)
    except Exception as e:
        logger.exception(f"Error fetching {ticker}: {e}")
        return None


# ─────────────────────────────────────────────
# MATPLOTLIB & PLOTLY VISUALIZATION ENGINE
# ─────────────────────────────────────────────
def plot_candlestick(df, name, timeframe):
    if df is None or df.empty or len(df) < 5:
        return None
    plot_df = df.tail(100).copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                        subplot_titles=(f'{name} Price Candlestick Matrix', 'Volume Profile'), row_width=[0.25, 0.75])
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'].squeeze(), high=plot_df['High'].squeeze(),
                                 low=plot_df['Low'].squeeze(), close=plot_df['Close'].squeeze(), name='Price'), row=1, col=1)
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'].squeeze(), name='Volume', marker_color='#00aaff'), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark', height=480, margin=dict(t=30, b=10, l=10, r=10))
    return fig


def plot_india_vix_summary():
    data = fetch_data("^INDIAVIX", period="1y", interval="1d")
    if data is None or data.empty: return None
    close_prices = data['Close'].squeeze()
    current_iv, high_52w, low_52w = float(close_prices.iloc[-1]), float(close_prices.max()), float(close_prices.min())
    ivr = ((current_iv - low_52w) / (high_52w - low_52w)) * 100
    ivp = ((close_prices < current_iv).sum() / len(close_prices)) * 100
    regime, color_theme = ("HIGH VOLATILITY: Net Short Premium", '#00FF00') if ivr > 50 else ("LOW VOLATILITY: Net Long Premium", '#FF3333')

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    ax.plot(close_prices.index, close_prices.values, color='#00FFFF', linewidth=1.5)
    ax.axhline(high_52w, color='red', linestyle='--', alpha=0.5, label=f'52W High: {high_52w:.2f}')
    ax.axhline(low_52w, color='green', linestyle='--', alpha=0.5, label=f'52W Low: {low_52w:.2f}')
    ax.axhline(current_iv, color='white', linestyle='-', linewidth=2, label=f'Current: {current_iv:.2f}')
    ax.set_title('Systemic Implied Volatility Index (India VIX)', fontsize=14, color='white', fontweight='bold')
    ax.grid(True, color='#2A2A2A', linestyle=':')
    ax.legend(loc='upper right', facecolor='black', edgecolor='gray')
    props = {'boxstyle': 'round,pad=0.5', 'facecolor': 'black', 'alpha': 0.8, 'edgecolor': color_theme, 'linewidth': 1.5}
    text_str = f"IV Rank (IVR): {ivr:.1f}%\nIV Percentile (IVP): {ivp:.1f}%\nCurrent VIX: {current_iv:.2f}\n---------------------------\nEdge: {regime}"
    ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_vrp(ticker, name):
    nifty_data = fetch_data(ticker, period="6mo", interval="1d")
    vix_data = fetch_data("^INDIAVIX", period="6mo", interval="1d")
    if nifty_data is None or vix_data is None: return None
    
    df = pd.DataFrame({'VIX': vix_data['Close'].squeeze(), 'HV': (np.log(nifty_data['Close'].squeeze() / nifty_data['Close'].squeeze().shift(1)).rolling(20).std() * np.sqrt(252) * 100)}).dropna()
    if df.empty: return None
    df['VRP'] = df['VIX'] - df['HV']
    current_vrp = float(df['VRP'].iloc[-1])
    color_theme = '#00FF00' if current_vrp > 0 else '#FF3333'

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), dpi=100, gridspec_kw={'height_ratios': [1.5, 1]})
    ax1.plot(df.index, df['VIX'], color='red', linewidth=1.5, label='India VIX (Expected)')
    ax1.plot(df.index, df['HV'], color='dodgerblue', linewidth=1.5, label=f'20-Day HV ({name})')
    ax1.set_title(f'Volatility Risk Premium (VRP) Variance Spread: {name}', fontsize=14, color='white', fontweight='bold')
    ax1.grid(True, color='#2A2A2A', linestyle=':')
    ax1.legend(loc='upper left', facecolor='black', edgecolor='gray')
    ax2.bar(df.index, df['VRP'], color=np.where(df['VRP'] > 0, '#00FF00', '#FF3333'), alpha=0.6, width=1.2)
    ax2.axhline(0, color='white', linewidth=1)
    ax2.grid(True, color='#2A2A2A', linestyle=':')
    
    props = {'boxstyle': 'round,pad=0.5', 'facecolor': 'black', 'alpha': 0.8, 'edgecolor': color_theme, 'linewidth': 1.5}
    text_str = f"VIX (Expected): {df['VIX'].iloc[-1]:.2f}%\n20-Day HV (Actual): {df['HV'].iloc[-1]:.2f}%\nVRP Spread: {current_vrp:+.2f}%"
    ax1.text(0.02, 0.05, text_str, transform=ax1.transAxes, fontsize=10, verticalalignment='bottom', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_volatility_cone(ticker, name):
    data = fetch_data(ticker, period="5y", interval="1d")
    if data is None or data.empty: return None
    data['Returns'] = np.log(data['Close'].squeeze() / data['Close'].squeeze().shift(1))
    windows = [10, 20, 30, 60, 90, 120, 180, 252]
    max_vol, min_vol, median_vol, current_vol = [], [], [], []
    for w in windows:
        rolling_vol = data['Returns'].rolling(window=w).std() * np.sqrt(252)
        max_vol.append(rolling_vol.max() * 100)
        min_vol.append(rolling_vol.min() * 100)
        median_vol.append(rolling_vol.median() * 100)
        current_vol.append(rolling_vol.dropna().iloc[-1] * 100)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 5))
    plt.plot(windows, max_vol, marker='o', color='red', linewidth=1.5, label='Max Vol')
    plt.plot(windows, min_vol, marker='o', color='limegreen', linewidth=1.5, label='Min Vol')
    plt.plot(windows, median_vol, marker='s', color='white', linewidth=1, linestyle='--', label='Median Vol')
    plt.plot(windows, current_vol, marker='X', color='yellow', linewidth=2.5, markersize=8, label='Current Vol')
    plt.fill_between(windows, min_vol, max_vol, color='gray', alpha=0.15)
    plt.title(f'Volatility Cone Structure for {name}', fontsize=14, fontweight='bold', color='white')
    plt.xlabel('Time Window (Trading Days)')
    plt.ylabel('Annualized Volatility (%)')
    plt.xticks(windows)
    plt.grid(color='gray', linestyle=':', alpha=0.4)
    plt.legend(loc='upper right')
    plt.tight_layout()
    return fig


def plot_expected_move(ticker, name):
    nifty_data = fetch_data(ticker, period="1mo", interval="1d")
    vix_data = fetch_data("^INDIAVIX", period="5d", interval="1d")
    if nifty_data is None or vix_data is None: return None
    nifty_close = nifty_data['Close'].squeeze()
    spot_price = float(nifty_close.iloc[-1])
    current_vix = float(vix_data['Close'].squeeze().iloc[-1])
    
    expected_move_points = spot_price * ((current_vix / 100) * np.sqrt(1/365))
    upper_bound, lower_bound = spot_price + expected_move_points, spot_price - expected_move_points

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    recent = nifty_close.tail(15)
    x_dates = np.arange(len(recent))
    ax.plot(x_dates, recent.values, color='#00FFFF', linewidth=2, marker='o')
    tomorrow_x = len(recent)
    ax.hlines(spot_price, xmin=x_dates[-1], xmax=tomorrow_x, color='white', linewidth=1.5)
    ax.scatter(tomorrow_x, upper_bound, color='#00FF00', s=100, marker='^', zorder=5)
    ax.scatter(tomorrow_x, lower_bound, color='#FF3333', s=100, marker='v', zorder=5)
    ax.hlines(upper_bound, xmin=x_dates[-1], xmax=tomorrow_x, color='#00FF00', linestyle='--')
    ax.hlines(lower_bound, xmin=x_dates[-1], xmax=tomorrow_x, color='#FF3333', linestyle='--')
    
    props = {'boxstyle': 'round,pad=0.5', 'facecolor': 'black', 'alpha': 0.8, 'edgecolor': 'white', 'linewidth': 1.5}
    text_str = f"Spot Price: {spot_price:.2f}\nExpected Move: ± {expected_move_points:.1f} pts\n---------------------------\nUpper Boundary: > {upper_bound:.0f}\nLower Boundary: < {lower_bound:.0f}"
    ax.text(0.02, 0.5, text_str, transform=ax.transAxes, fontsize=10, bbox=props, color='white', fontweight='bold')
    ax.set_title(f'{name} Implied Daily Expected Move Range', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    plt.tight_layout()
    return fig


def plot_liquidity_sweep(ticker, name):
    data = fetch_data(ticker, period="5d", interval="15m")
    if data is None or data.empty: return None
    data['Prev_High'] = data['High'].rolling(20).max().shift(1)
    data['Prev_Low'] = data['Low'].rolling(20).min().shift(1)
    data['Supply_Sweep'] = (data['High'] > data['Prev_High']) & (data['Close'] < data['Prev_High'])
    data['Demand_Sweep'] = (data['Low'] < data['Prev_Low']) & (data['Close'] > data['Prev_Low'])
    current_price = float(data['Close'].iloc[-1])
    color_theme = '#FF3333' if data['Supply_Sweep'].iloc[-1] else '#00FF00' if data['Demand_Sweep'].iloc[-1] else '#00FFFF'

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    plot_data = data.tail(45).copy()
    plot_data['Index'] = np.arange(len(plot_data))
    up, down = plot_data[plot_data['Close'] >= plot_data['Open']], plot_data[plot_data['Close'] < plot_data['Open']]
    ax.vlines(up['Index'], up['Low'], up['High'], color='#00FF00', linewidth=1)
    ax.vlines(down['Index'], down['Low'], down['High'], color='#FF3333', linewidth=1)
    ax.bar(up['Index'], (up['Close'] - up['Open']), bottom=up['Open'], color='#00FF00', width=0.5)
    ax.bar(down['Index'], (down['Open'] - down['Close']), bottom=down['Close'], color='#FF3333', width=0.5)
    
    for idx, row in plot_data.iterrows():
        if row['Supply_Sweep']: ax.scatter(row['Index'], row['High'] * 1.0002, marker='v', color='#FF3333', s=100)
        if row['Demand_Sweep']: ax.scatter(row['Index'], row['Low'] * 0.9998, marker='^', color='#00FF00', s=100)
            
    ax.set_title(f'{name} Intraday Microstructure Sweeps (15m Matrix)', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    props = {'boxstyle': 'round,pad=0.5', 'facecolor': 'black', 'alpha': 0.8, 'edgecolor': color_theme, 'linewidth': 1.5}
    ax.text(0.02, 0.05, f"Live Price: {current_price:.2f}\nStatus: Structural Liquidity Scan Complete", transform=ax.transAxes, bbox=props, color='white')
    plt.tight_layout()
    return fig


def plot_index_divergence():
    data = yf.download(["^NSEI", "^NSEBANK"], period="1y", progress=False)
    if data is None or data.empty: return None
    data = _flatten_multiindex(data)['Close'].dropna()
    normalized = (data / data.iloc[0]) * 100
    corr = np.log(data / data.shift(1)).dropna().iloc[:, 0].rolling(20).corr(np.log(data / data.shift(1)).dropna().iloc[:, 1])

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), dpi=100, gridspec_kw={'height_ratios': [1.5, 1]})
    ax1.plot(normalized.index, normalized.iloc[:, 1], color='#00FFFF', label='Nifty 50')
    ax1.plot(normalized.index, normalized.iloc[:, 0], color='#FFA500', label='Bank Nifty')
    ax1.set_title('Inter-Index Correlation Matrix & Decoupling', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, color='#2A2A2A', linestyle=':')
    ax2.plot(corr.index, corr.values, color='white', linewidth=1)
    ax2.axhline(0.5, color='red', linestyle='--')
    ax2.set_ylim(-0.2, 1.1)
    ax2.grid(True, color='#2A2A2A', linestyle=':')
    plt.tight_layout()
    return fig


def plot_hurst_regime(ticker, name):
    data = fetch_data(ticker, period="1y", interval="1d")
    if data is None or data.empty: return None
    close = data['Close'].squeeze()
    
    def calc_h(ts):
        if len(ts) < 20: return np.nan
        arr = ts.values
        reg = [np.std(arr[l:] - arr[:-l]) for l in range(2, 15)]
        return np.polyfit(np.log(range(2, 15)), np.log(reg), 1)[0]
        
    df = pd.DataFrame({'Close': close, 'Hurst': np.log(close).rolling(45).apply(calc_h, raw=False)}).dropna()
    if df.empty: return None
    current_hurst = float(df['Hurst'].iloc[-1])
    color_theme = '#00FF00' if current_hurst > 0.55 else '#FF3333' if current_hurst < 0.45 else '#FFA500'

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), dpi=100, gridspec_kw={'height_ratios': [1.5, 1]})
    ax1.plot(df.index, df['Close'], color='white')
    ax1.set_title(f'{name} Market Efficiency Memory State (Hurst Exponent)', fontsize=14, fontweight='bold')
    ax1.grid(True, color='#2A2A2A', linestyle=':')
    ax2.plot(df.index, df['Hurst'], color='#00FFFF')
    ax2.axhline(0.55, color='#00FF00', linestyle='--')
    ax2.axhline(0.45, color='#FF3333', linestyle='--')
    ax2.set_ylim(0.25, 0.75)
    ax2.grid(True, color='#2A2A2A', linestyle=':')
    props = {'boxstyle': 'round,pad=0.5', 'facecolor': 'black', 'alpha': 0.8, 'edgecolor': color_theme, 'linewidth': 1.5}
    ax1.text(0.02, 0.05, f"Current Hurst Value: {current_hurst:.3f}", transform=ax1.transAxes, bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_crypto_volatility(name, ticker):
    data = fetch_data(ticker, period="1y", interval="1d")
    if data is None or data.empty: return None
    close = data['Close'].squeeze()
    returns = np.log(close / close.shift(1)).dropna()
    c_vol = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    ax.plot(close.index, close.values, color='#00FFFF', linewidth=1.5)
    ax.set_title(f'{name} Derivative Closing Range Timeline', fontsize=14, fontweight='bold')
    ax.grid(True, color='#2A2A2A', linestyle=':')
    props = {'boxstyle': 'round,pad=0.5', 'facecolor': 'black', 'alpha': 0.8, 'edgecolor': '#0af', 'linewidth': 1.5}
    ax.text(0.02, 0.95, f"Current Price: ${close.iloc[-1]:,.2f}\n20-Day Annualized Vol: {c_vol:.2f}%", transform=ax.transAxes, bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# FEATURES GENERATION PIPELINE
# ─────────────────────────────────────────────
def build_ml_features(df):
    feat = pd.DataFrame(index=df.index)
    close = df['Close'].squeeze()
    feat['rsi'] = compute_rsi(close)
    feat['returns'] = close.pct_change()
    feat['vol_20'] = feat['returns'].rolling(20).std()
    bb_up, _, bb_lo = bollinger_bands(close)
    feat['bb_pos'] = (close - bb_lo) / (bb_up - bb_lo + 1e-9)
    feat['atr'] = compute_atr(df)
    feat['vol_ratio'] = df['Volume'].squeeze() / df['Volume'].squeeze().rolling(20).mean().replace(0, np.nan)
    macd, sig, _ = compute_macd(close)
    feat['macd_diff'] = macd - sig
    return feat.dropna()


def explain_ml_prediction(model, feat_df, prob_pos):
    latest = feat_df.iloc[-1]
    importances = pd.Series(model.feature_importances_, index=feat_df.columns).sort_values(ascending=False)
    top_feats = importances.head(4).index.tolist()
    supporting, opposing, neutral = [], [], []

    for f in top_feats:
        v = latest.get(f, np.nan)
        if pd.isna(v): continue
        if f == 'rsi':
            if v < 40: supporting.append(f"RSI is oversold ({v:.1f}), building technical support.")
            elif v > 60: opposing.append(f"RSI is overbought ({v:.1f}), extending overhead resistance.")
            else: neutral.append(f"RSI is mean-neutral ({v:.1f}).")
        elif f == 'macd_diff':
            if v > 0: supporting.append(f"MACD Hist is Positive ({v:.4f}), momentum acceleration detected.")
            else: opposing.append(f"MACD Hist is Negative ({v:.4f}), expansion vector fading.")
        elif f == 'bb_pos':
            if v < 0.3: supporting.append(f"Price compression near Lower BB ({v:.2f}). Upside mean reversion edge.")
            elif v > 0.7: opposing.append(f"Price stretching near Upper BB ({v:.2f}). Downside pull risk.")
            else: neutral.append(f"BB position inside equilibrium channel ({v:.2f}).")
    return {"supporting": supporting, "opposing": opposing, "neutral": neutral, "probability": prob_pos}


# ─────────────────────────────────────────────
# MAIN STREAMLIT APPLICATION INTERFACE
# ─────────────────────────────────────────────
def main():
    st.title("⚡ AlphaQuant Terminal")
    
    st.sidebar.header("Navigation & Asset Matrix")
    market_selection = st.sidebar.radio("Select Target Infrastructure Context", ["Indian Market", "Crypto Assets"])
    
    if market_selection == "Indian Market":
        st.sidebar.markdown("### Institutional Suite")
        indian_index = st.sidebar.selectbox("Select Benchmark Index", ["Nifty 50", "Bank Nifty", "Sensex", "Finnifty"])
        index_mapping = {
            "Nifty 50": "^NSEI",
            "Bank Nifty": "^NSEBANK",
            "Sensex": "^BSESN",
            "Finnifty": "^CNXFIN"
        }
        ticker = index_mapping[indian_index]
        name = indian_index
    else:
        st.sidebar.markdown("### Crypto Liquidity Matrix")
        crypto_ticker = st.sidebar.selectbox("Select Derivative Asset Pool", ["BTC-USD", "ETH-USD", "SOL-USD"])
        ticker = crypto_ticker
        name = crypto_ticker.split("-")[0]

    st.sidebar.markdown("### Structural Timeframe")
    timeframe = st.sidebar.selectbox("Select System Interval Matrix", ["15m", "30m", "1h", "1d"], index=3)
    
    # yfinance compliance dictionary to avoid requesting out-of-bounds data arrays
    tf_period_map = {
        "15m": {"interval": "15m", "period": "1mo"},
        "30m": {"interval": "30m", "period": "1mo"},
        "1h": {"interval": "60m", "period": "3mo"},
        "1d": {"interval": "1d", "period": "2y"}
    }
    
    selected_tf = tf_period_map[timeframe]
    df_live = fetch_data(ticker, period=selected_tf["period"], interval=selected_tf["interval"])

    tab_vol, tab_struct, tab_ml = st.tabs(["📊 Volatility Engine", "📐 Structural Analysis", "🤖 ML Explanations & Timeframe Metrics"])

    # --- Volatility Engine Tab ---
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
            st.markdown("### Long-Term Multi-Day Variance Cone Scaling")
            cone_chart = plot_volatility_cone(ticker, name)
            if cone_chart: st.pyplot(cone_chart)
        else:
            crypto_vol = plot_crypto_volatility(name, ticker)
            if crypto_vol: st.pyplot(crypto_vol)

    # --- Structural Analysis Tab ---
    with tab_struct:
        st.header(f"Market Structure Profiles — Matrix Engine")
        
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
            
            st.markdown("### Multi-Index System Dynamics & Long-Memory Regimes")
            c1, c2 = st.columns(2)
            with c1:
                div_chart = plot_index_divergence()
                if div_chart: st.pyplot(div_chart)
            with c2:
                hurst_chart = plot_hurst_regime(ticker, name)
                if hurt_chart := hurst_chart: st.pyplot(hurt_chart)
        else:
            if df_live is not None:
                h_val, interp, conf = hurst_exponent(df_live['Close'])
                st.metric(f"Asset Memory Factor ({timeframe} Hurst)", f"{h_val:.3f}", delta=interp)

    # --- Machine Learning Tab ---
    with tab_ml:
        st.header(f"Predictive Architecture Framework ({timeframe} Window)")
        if not ML_AVAILABLE:
            st.error("Scikit-Learn Machine Learning Pipeline components unreachable inside container context environment.")
            return

        if df_live is None or df_live.empty:
            st.warning("Feature Generation aborted. Active dataframe matrices returned empty.")
            return

        features = build_ml_features(df_live)
        target = (df_live['Close'].pct_change().shift(-1) > 0).astype(int).loc[features.index]
        common_idx = features.index.intersection(target.index)
        X, y = features.loc[common_idx], target.loc[common_idx]

        if len(X) > 20:
            split = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            base_rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
            calibrated_model = CalibratedClassifierCV(base_rf, method='sigmoid', cv=3)
            calibrated_model.fit(X_train_scaled, y_train)
            base_rf.fit(X_train_scaled, y_train)

            latest_vector = scaler.transform(X.iloc[[-1]])
            prob = calibrated_model.predict_proba(latest_vector)[0][1]
            explanation = explain_ml_prediction(base_rf, X, prob)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric(f"Directional Next-Bar ({timeframe}) Upside Probability", f"{prob * 100:.2f}%")
            with col_m2:
                signal_type = "BULLISH BIAS" if prob > 0.54 else "BEARISH BIAS" if prob < 0.46 else "CONGESTION NEUTRAL"
                st.markdown(f"#### Regime Engine Output Focus: **{signal_type}**")

            st.markdown("<div class='section-header'>Explainable AI Rationale Breakdown</div>", unsafe_allow_html=True)
            c_sup, c_opp = st.columns(2)
            with c_sup:
                st.markdown("### 👍 Alpha Supporting Drivers")
                for item in explanation["supporting"]: st.markdown(f"* {item}")
                if not explanation["supporting"]: st.markdown("_No dynamic supporting criteria tracked in top tier feature weights._")
            with c_opp:
                st.markdown("### 👎 Delta Opposing Vectors")
                for item in explanation["opposing"]: st.markdown(f"* {item}")
                if not explanation["opposing"]: st.markdown("_No negative alpha constraints found in state matrix boundary._")
        else:
            st.warning(f"Row count profile ({len(X)} bars) is too narrow to run step-forward walk classification models for timeframe '{timeframe}'. Choose a smaller step interval or longer data footprint.")


if __name__ == '__main__':
    main()