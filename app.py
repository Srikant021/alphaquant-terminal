# AlphaQuant Terminal — Final Complete Working Version
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os
import calendar

st.set_page_config(page_title="AlphaQuant Terminal", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, .stApp { font-family: 'Inter', sans-serif; background: #0E1117; color: #E0E0E0; }
    [data-testid="stSidebar"] { background: #13161C; border-right: 1px solid #2A2E39; }
    .metric-card { background: #1A1D24; border: 1px solid #2A2E39; border-radius: 8px; padding: 12px 18px; }
    .metric-card .label { font-size: 0.75rem; color: #A0A7B8; }
    .metric-card .value { font-size: 1.3rem; font-weight: 700; color: #FFFFFF; }
    .stButton>button { background: #2A3A5C; color: white; border: none; border-radius: 4px; }
    .explanation-box { background: #1A1D24; border: 1px solid #2A2E39; border-radius: 8px; padding: 15px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'live_mode' not in st.session_state:
    st.session_state['live_mode'] = False
if 'refresh_interval' not in st.session_state:
    st.session_state['refresh_interval'] = 120
if 'selected_market' not in st.session_state:
    st.session_state['selected_market'] = "Crypto"
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = 'Dashboard'

# Technical Indicators
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bbands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_hurst_exponent(ts):
    if len(ts) < 20:
        return np.nan
    lags = range(2, min(20, len(ts) // 5))
    if len(lags) < 3:
        return np.nan
    try:
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        return np.polyfit(np.log(list(lags)), np.log(tau), 1)[0] * 2.0
    except:
        return np.nan

def calculate_parkinson_volatility(high_px, low_px, periods_per_year=252):
    if len(high_px) != len(low_px) or len(high_px) < 2:
        return 0.0
    log_hl = (np.log(high_px / low_px) ** 2)
    n = len(log_hl)
    variance = log_hl.sum() / (4 * n * np.log(2))
    return np.sqrt(variance * periods_per_year) * 100

def yf_download_retry(*args, max_retries=2, **kwargs):
    for _ in range(max_retries):
        try:
            d = yf.download(*args, progress=False, **kwargs)
            if d is not None and not d.empty:
                return d
        except:
            pass
        time.sleep(2)
    return pd.DataFrame()

def flatten_df(df_raw):
    if isinstance(df_raw.columns, pd.MultiIndex):
        df = df_raw.copy()
        df.columns = df_raw.columns.get_level_values(0)
        return df
    return df_raw

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_long_hist(ticker):
    raw = yf_download_retry(ticker, period="2y")
    if raw.empty:
        return pd.DataFrame()
    return flatten_df(raw)

@st.cache_data(ttl=300, show_spinner=False)
def live_price(ticker):
    raw = yf_download_retry(ticker, period="2d")
    if raw.empty:
        return None
    df = flatten_df(raw)
    if len(df) < 2:
        return None
    last = float(df['Close'].iloc[-1])
    prev = float(df['Close'].iloc[-2])
    return {
        'spot': last,
        'prev_close': prev,
        'change': last - prev,
        'pct': ((last - prev) / prev) * 100 if prev else 0,
        'ts': datetime.now().strftime('%H:%M:%S')
    }

def compute_iv_rank_percentile(close_px, window=20):
    if len(close_px) < window:
        return 50.0, 50.0
    log_ret = np.log(close_px / close_px.shift(1)).dropna()
    rv = log_ret.rolling(window).std() * np.sqrt(252) * 100
    cur = rv.iloc[-1] if not rv.empty else 0
    if rv.empty or np.isnan(cur):
        return 50.0, 50.0
    vmin, vmax = rv.min(), rv.max()
    ivr = ((cur - vmin) / (vmax - vmin)) * 100 if vmax != vmin else 50.0
    ivp = (rv < cur).sum() / len(rv) * 100
    return ivr, ivp

@st.cache_data(ttl=300)
def get_india_vix(period="5d"):
    v = yf_download_retry("^INDIAVIX", period=period)
    if v.empty:
        return None
    return v['Close'].squeeze()

# Chart functions
def chart_correlation():
    selected_market = st.session_state['selected_market']
    if selected_market == "Crypto":
        corr_tickers = ['BTC-USD', 'ETH-USD']
        names = ("Bitcoin", "Ethereum")
    else:
        corr_tickers = ['^NSEI', '^NSEBANK']
        names = ("Nifty 50", "Bank Nifty")
    
    corr_data = yf_download_retry(corr_tickers, period="1y")['Close']
    if corr_data.empty:
        return None
    
    df = corr_data.dropna()
    df.columns = names
    norm = df / df.iloc[0] * 100
    
    log_ret = np.log(df / df.shift(1)).dropna()
    roll_corr = log_ret[names[0]].rolling(20).corr(log_ret[names[1]])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(norm.index, norm[names[0]], label=names[0])
    ax1.plot(norm.index, norm[names[1]], label=names[1])
    ax1.legend()
    ax1.set_title("Price Correlation")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(roll_corr.index, roll_corr, color='white')
    ax2.axhline(0.8, color='green', linestyle='--')
    ax2.axhline(0.5, color='red', linestyle='--')
    ax2.set_ylim(-0.2, 1.1)
    ax2.set_ylabel("Correlation")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def chart_expected_move(asset_spot, garch_vol, trading_days):
    recent_close = hist_data['Close'].squeeze().tail(20)
    daily_vol = garch_vol / 100 / np.sqrt(trading_days)
    dm = asset_spot * daily_vol
    wm = dm * np.sqrt(7)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(recent_close.index, recent_close.values, color='white', linewidth=2)
    
    last_idx = recent_close.index[-1]
    next_day = last_idx + pd.Timedelta(days=1)
    next_week = last_idx + pd.Timedelta(days=7)
    
    ax.hlines(asset_spot + dm, last_idx, next_day, colors='cyan', linestyles='--', linewidth=2, label='Daily Range')
    ax.hlines(asset_spot - dm, last_idx, next_day, colors='cyan', linestyles='--', linewidth=2)
    ax.hlines(asset_spot + wm, last_idx, next_week, colors='orange', linestyles='--', linewidth=2, label='Weekly Range')
    ax.hlines(asset_spot - wm, last_idx, next_week, colors='orange', linestyles='--', linewidth=2)
    
    ax.legend()
    ax.set_title("Expected Moves")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def chart_hurst(hist_data):
    close = hist_data['Close'].squeeze()
    log_p = np.log(close)
    
    def hurst_calc(x):
        return calculate_hurst_exponent(x) if len(x) >= 20 else np.nan
    
    hurst_series = log_p.rolling(60).apply(hurst_calc, raw=False)
    df = pd.DataFrame({'Close': close, 'Hurst': hurst_series}).dropna()
    
    if df.empty:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(df.index, df['Close'], color='white', linewidth=1.5)
    ax1.set_title("Price")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df.index, df['Hurst'], color='cyan', linewidth=2)
    ax2.axhline(0.55, color='green', linestyle='--', linewidth=1.5, label='Trending')
    ax2.axhline(0.45, color='red', linestyle='--', linewidth=1.5, label='Mean-Reverting')
    ax2.axhline(0.50, color='gray', linestyle='-', alpha=0.5)
    ax2.set_ylim(0.3, 0.7)
    ax2.set_ylabel("Hurst Exponent")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def chart_ivr(hist_data, trading_days):
    close = hist_data['Close'].squeeze()
    log_ret = np.log(close / close.shift(1)).dropna()
    rolling_vol = log_ret.rolling(20).std() * np.sqrt(trading_days) * 100
    vol_series = rolling_vol.dropna()
    
    if vol_series.empty:
        return None
    
    cur = vol_series.iloc[-1]
    vmin, vmax = vol_series.min(), vol_series.max()
    ivr = ((cur - vmin) / (vmax - vmin)) * 100 if vmax != vmin else 50
    ivp = (vol_series < cur).sum() / len(vol_series) * 100
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(vol_series.index, vol_series, color='cyan', linewidth=1.5)
    ax.axhline(vmax, color='red', linestyle='--', alpha=0.5, label=f'High: {vmax:.1f}%')
    ax.axhline(vmin, color='green', linestyle='--', alpha=0.5, label=f'Low: {vmin:.1f}%')
    ax.axhline(cur, color='white', linestyle='-', linewidth=2, label=f'Current: {cur:.1f}%')
    ax.set_title(f"IV Rank: {ivr:.0f}% | IV Percentile: {ivp:.0f}%")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def chart_liquidity(ticker):
    intra = yf_download_retry(ticker, period="5d", interval="30m")
    if intra.empty:
        return None
    intra = flatten_df(intra).tail(60)
    df = intra.copy()
    
    df['Prev_High'] = df['High'].rolling(20).max().shift(1)
    df['Prev_Low'] = df['Low'].rolling(20).min().shift(1)
    df['Supply'] = (df['High'] > df['Prev_High']) & (df['Close'] < df['Prev_High'])
    df['Demand'] = (df['Low'] < df['Prev_Low']) & (df['Close'] > df['Prev_Low'])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Close'], color='white', linewidth=1.5)
    
    supply_idx = df.index[df['Supply']]
    demand_idx = df.index[df['Demand']]
    ax.scatter(supply_idx, df['High'][df['Supply']] + 10, color='red', marker='v', s=100, label='Supply Sweep')
    ax.scatter(demand_idx, df['Low'][df['Demand']] - 10, color='green', marker='^', s=100, label='Demand Sweep')
    
    ax.legend()
    ax.set_title("Liquidity Sweeps")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def chart_parkinson(hist_data, trading_days):
    high = hist_data['High'].squeeze().tail(60)
    low = hist_data['Low'].squeeze().tail(60)
    park_val = calculate_parkinson_volatility(high, low, trading_days)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(['Parkinson Volatility'], [park_val], color='orange', width=0.5)
    ax.set_ylabel('%')
    ax.set_title(f"Parkinson Volatility = {park_val:.1f}%")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def chart_cone(hist_data, trading_days):
    close = hist_data['Close'].squeeze()
    log_ret = np.log(close / close.shift(1)).dropna()
    windows = [10, 20, 30, 60, 90, 120, 180, 252]
    max_v, min_v, med_v, cur_v = [], [], [], []
    
    for w in windows:
        rv = log_ret.rolling(w).std() * np.sqrt(trading_days) * 100
        if not rv.dropna().empty:
            max_v.append(rv.max())
            min_v.append(rv.min())
            med_v.append(rv.median())
            cur_v.append(rv.iloc[-1])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(windows, max_v, 'o-', color='red', linewidth=2, markersize=8, label='Maximum')
    ax.plot(windows, min_v, 'o-', color='green', linewidth=2, markersize=8, label='Minimum')
    ax.plot(windows, med_v, 's--', color='white', linewidth=2, markersize=6, label='Median')
    ax.plot(windows, cur_v, 'X-', color='yellow', linewidth=3, markersize=12, label='Current')
    ax.fill_between(windows, min_v, max_v, color='gray', alpha=0.2)
    ax.legend()
    ax.set_xlabel("Window (Trading Days)")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_title("Volatility Cone")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

EXPLANATIONS = {
    "Correlation": "**Correlation Analysis** measures how closely two assets move together. Above 0.8 = high correlation. Below 0.5 = low correlation.",
    "Expected Move": "**Expected Move** projects tomorrow's and next week's likely price range. 68% probability price stays within these lines.",
    "Hurst Exponent": "**Hurst Exponent** classifies market regime. H > 0.55 = Trending. H < 0.45 = Mean-Reverting.",
    "IV Rank & IV Percentile": "**IV Rank** tells if options are expensive or cheap. >65% = expensive. <30% = cheap.",
    "Liquidity Detector": "**Liquidity Sweeps** detect when big players absorb retail orders. Red = bearish. Green = bullish.",
    "Parkinson Volatility": "**Parkinson Volatility** estimates intraday volatility using High-Low range.",
    "Volatility Cone": "**Volatility Cone** shows current vol vs historical ranges. Yellow = current position."
}

# ───── SIDEBAR ─────
with st.sidebar:
    st.markdown("## AlphaQuant Terminal")
    
    market_type = st.radio("Market", ["Crypto", "Indian Market"], horizontal=True,
                           index=0 if st.session_state['selected_market'] == 'Crypto' else 1)
    if market_type != st.session_state['selected_market']:
        st.session_state['selected_market'] = market_type
        st.cache_data.clear()
        st.rerun()
    
    selected_market = st.session_state['selected_market']
    
    if selected_market == "Crypto":
        tickers = {'Bitcoin': 'BTC-USD', 'Ethereum': 'ETH-USD', 'Dogecoin': 'DOGE-USD', 'XRP': 'XRP-USD'}
        trading_days = 365
        currency = "$"
    else:
        tickers = {'Nifty 50': '^NSEI', 'Sensex': '^BSESN', 'Bank Nifty': '^NSEBANK'}
        trading_days = 252
        currency = "₹"
    
    asset_choice = st.selectbox("Asset", list(tickers.keys()))
    ticker = tickers[asset_choice]
    
    st.markdown("---")
    tab_options = ["Dashboard", "Technical"]
    active_tab = st.radio("Navigate", tab_options, index=tab_options.index(st.session_state['active_tab']))
    if active_tab != st.session_state['active_tab']:
        st.session_state['active_tab'] = active_tab
        st.rerun()
    
    st.markdown("---")
    st.session_state['live_mode'] = st.checkbox("Live Mode", value=st.session_state['live_mode'])
    
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

# ───── DATA LOADING ─────
hist_data = fetch_long_hist(ticker)

lp = live_price(ticker)
if lp is None and not hist_data.empty:
    close = hist_data['Close'].squeeze()
    if len(close) >= 2:
        spot = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        lp = {'spot': spot, 'prev_close': prev, 'change': spot - prev,
              'pct': ((spot - prev) / prev) * 100 if prev else 0, 'ts': 'hist'}
    else:
        lp = {'spot': 0, 'prev_close': 0, 'change': 0, 'pct': 0, 'ts': 'unavailable'}

asset_spot = lp['spot']
garch_vol = 80  # placeholder for GARCH

# Calculate metrics if data available
park_vol = None
ivr_val = None
ivp_val = None

if not hist_data.empty and all(c in hist_data.columns for c in ['High', 'Low', 'Close']):
    high_series = hist_data['High'].squeeze().tail(60)
    low_series = hist_data['Low'].squeeze().tail(60)
    close_series = hist_data['Close'].squeeze().tail(60)
    park_vol = calculate_parkinson_volatility(high_series, low_series, trading_days)
    ivr_val, ivp_val = compute_iv_rank_percentile(close_series)

# ───── ROUTING ─────
if st.session_state['active_tab'] == "Dashboard":
    st.title("Dashboard")
    
    tf = st.radio("Chart Timeframe", ["1D", "5m", "15m", "1h"], horizontal=True)
    period_map = {"1D": ("6mo", "1d"), "5m": ("5d", "5m"), "15m": ("5d", "15m"), "1h": ("1mo", "1h")}
    period, interval = period_map[tf]
    
    df_chart = yf_download_retry(ticker, period=period, interval=interval)
    if not df_chart.empty:
        df_chart = flatten_df(df_chart).tail(60 if tf == "1D" else 100)
        bb_upper, bb_mid, bb_lower = compute_bbands(df_chart['Close'])
        
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(go.Candlestick(
            x=df_chart.index,
            open=df_chart['Open'],
            high=df_chart['High'],
            low=df_chart['Low'],
            close=df_chart['Close'],
            name='Price'
        ))
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=bb_upper,
            line=dict(color='gray', width=1, dash='dot'),
            name='BB Upper'
        ))
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=bb_lower,
            line=dict(color='gray', width=1, dash='dot'),
            name='BB Lower'
        ))
        fig.update_layout(template='plotly_dark', height=500, xaxis_rangeslider_visible=False, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="label">Spot Price</div><div class="value">{currency}{asset_spot:,.2f}</div></div>', unsafe_allow_html=True)
    with col2:
        pct_change = lp['pct'] if lp else 0
        color = "#00FF00" if pct_change >= 0 else "#FF3333"
        st.markdown(f'<div class="metric-card"><div class="label">Daily Change</div><div class="value" style="color:{color}">{pct_change:+.2f}%</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="label">Parkinson Vol</div><div class="value">{park_vol:.1f}%</div></div>' if park_vol else '<div class="metric-card"><div class="label">Parkinson Vol</div><div class="value">N/A</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="label">IV Rank</div><div class="value">{ivr_val:.0f}%</div></div>' if ivr_val else '<div class="metric-card"><div class="label">IV Rank</div><div class="value">N/A</div></div>', unsafe_allow_html=True)

elif st.session_state['active_tab'] == "Technical":
    st.title("Full Technical Analysis")
    
    modules = [
        ("Correlation", chart_correlation, "Correlation"),
        ("Expected Move", lambda: chart_expected_move(asset_spot, garch_vol, trading_days), "Expected Move"),
        ("Hurst Exponent", lambda: chart_hurst(hist_data), "Hurst Exponent"),
        ("IV Rank & IV Percentile", lambda: chart_ivr(hist_data, trading_days), "IV Rank & IV Percentile"),
        ("Liquidity Detector", lambda: chart_liquidity(ticker), "Liquidity Detector"),
        ("Parkinson Volatility", lambda: chart_parkinson(hist_data, trading_days), "Parkinson Volatility"),
        ("Volatility Cone", lambda: chart_cone(hist_data, trading_days), "Volatility Cone")
    ]
    
    for i in range(0, len(modules), 2):
        cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx < len(modules):
                name, func, key = modules[idx]
                with cols[j]:
                    with st.expander(f"{name}", expanded=False):
                        try:
                            fig = func()
                            if fig:
                                st.pyplot(fig)
                                st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
                                st.markdown(EXPLANATIONS.get(key, ""))
                                st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.warning("Data unavailable")
                        except Exception as e:
                            st.error(f"Error generating chart: {e}")

if st.session_state['live_mode']:
    time.sleep(st.session_state['refresh_interval'])
    st.rerun()