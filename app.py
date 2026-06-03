# AlphaQuant Terminal - Clean Working Version
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="AlphaQuant Terminal", layout="wide", initial_sidebar_state="expanded")

# Simple Hurst Exponent - Reliable Implementation
def hurst_exponent(price_series):
    """Calculate Hurst exponent using rescaled range analysis."""
    # Convert to numpy and remove NaN
    price = np.array(price_series.dropna())
    if len(price) < 20:
        return 0.5
    
    # Use log returns for better numerical stability
    log_prices = np.log(price)
    
    # Calculate for different lags
    max_lag = min(len(log_prices) // 4, 100)
    if max_lag < 5:
        return 0.5
    
    lags = range(5, max_lag, 5)
    if len(lags) < 2:
        return 0.5
    
    rs_values = []
    for lag in lags:
        # Number of windows
        n_windows = len(log_prices) // lag
        if n_windows < 1:
            continue
        
        rs_window = []
        for i in range(n_windows):
            window = log_prices[i*lag:(i+1)*lag]
            if len(window) < 2:
                continue
            # Mean-centered
            mean_adj = window - np.mean(window)
            cumsum = np.cumsum(mean_adj)
            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(window)
            if S > 0:
                rs_window.append(R / S)
        
        if rs_window:
            rs_values.append(np.mean(rs_window))
    
    if len(rs_values) < 2:
        return 0.5
    
    # Fit log-log regression
    log_lags = np.log([lags[i] for i in range(len(rs_values))])
    log_rs = np.log(rs_values)
    hurst = np.polyfit(log_lags, log_rs, 1)[0]
    
    # Clip to reasonable range
    return np.clip(hurst, 0.2, 0.8)

# Simple Technical Indicators
def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(close, period=20, std=2):
    sma = close.rolling(window=period).mean()
    std_dev = close.rolling(window=period).std()
    return sma + (std * std_dev), sma, sma - (std * std_dev)

@st.cache_data(ttl=300)
def fetch_data(ticker, period="1y", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return None

@st.cache_data(ttl=60)
def get_live_price(ticker):
    data = fetch_data(ticker, period="2d", interval="1d")
    if data is None or len(data) < 2:
        return None
    last = float(data['Close'].iloc[-1])
    prev = float(data['Close'].iloc[-2])
    return {
        'price': last,
        'change': last - prev,
        'pct': ((last - prev) / prev) * 100 if prev else 0
    }

def compute_parkinson_vol(high, low, periods=252):
    """Parkinson volatility estimator."""
    high = np.array(high.dropna())
    low = np.array(low.dropna())
    if len(high) < 2 or len(low) < 2:
        return 0
    log_hl = (np.log(high / low) ** 2)
    variance = log_hl.sum() / (4 * len(log_hl) * np.log(2))
    return np.sqrt(variance * periods) * 100

def compute_iv_rank(close, window=20):
    """IV Rank and IV Percentile."""
    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) < window:
        return 50, 50
    hv = log_ret.rolling(window).std() * np.sqrt(252) * 100
    hv = hv.dropna()
    if hv.empty:
        return 50, 50
    current = hv.iloc[-1]
    ivr = (current - hv.min()) / (hv.max() - hv.min()) * 100 if hv.max() != hv.min() else 50
    ivp = (hv < current).sum() / len(hv) * 100
    return ivr, ivp

# Chart Functions
def plot_correlation(ticker1, ticker2, name1, name2):
    data1 = fetch_data(ticker1, period="1y")
    data2 = fetch_data(ticker2, period="1y")
    if data1 is None or data2 is None:
        return None
    
    # Align data
    merged = pd.DataFrame({
        name1: data1['Close'],
        name2: data2['Close']
    }).dropna()
    
    if len(merged) < 20:
        return None
    
    # Normalize
    norm = merged / merged.iloc[0] * 100
    
    # Rolling correlation
    log_ret = np.log(merged / merged.shift(1)).dropna()
    corr = log_ret[name1].rolling(20).corr(log_ret[name2])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(norm.index, norm[name1], label=name1, linewidth=1.5)
    ax1.plot(norm.index, norm[name2], label=name2, linewidth=1.5)
    ax1.legend()
    ax1.set_title("Relative Performance (Normalized to 100)")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(corr.index, corr, color='cyan', linewidth=1.5)
    ax2.axhline(0.8, color='green', linestyle='--', alpha=0.7, label='High Correlation')
    ax2.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Low Correlation')
    ax2.set_ylim(-0.2, 1.1)
    ax2.set_ylabel("20-Day Correlation")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_hurst(close):
    if len(close) < 100:
        return None
    
    # Calculate rolling Hurst (60-day window, 30-day step)
    hurst_values = []
    dates = []
    
    for i in range(60, len(close), 10):
        window = close.iloc[i-60:i]
        if len(window) >= 30:
            hurst = hurst_exponent(window)
            hurst_values.append(hurst)
            dates.append(close.index[i-1])
    
    if len(hurst_values) < 5:
        return None
    
    # Current Hurst
    current_hurst = hurst_exponent(close.tail(100))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(close.index, close.values, color='white', linewidth=1.5)
    ax1.set_title("Price")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(dates, hurst_values, color='cyan', linewidth=2, marker='o', markersize=4)
    ax2.axhline(0.55, color='green', linestyle='--', linewidth=1.5, label='Trending (>0.55)')
    ax2.axhline(0.45, color='red', linestyle='--', linewidth=1.5, label='Mean-Reverting (<0.45)')
    ax2.axhline(0.50, color='gray', linestyle='-', alpha=0.5)
    ax2.set_ylim(0.2, 0.8)
    ax2.set_ylabel("Hurst Exponent")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add current value annotation
    ax2.annotate(f'Current: {current_hurst:.2f}', 
                 xy=(dates[-1], hurst_values[-1]),
                 xytext=(10, 10), textcoords='offset points',
                 fontsize=10, color='yellow')
    
    plt.tight_layout()
    return fig

def plot_iv_rank(close, trading_days=252):
    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) < 30:
        return None
    
    hv = log_ret.rolling(20).std() * np.sqrt(trading_days) * 100
    hv = hv.dropna()
    
    if len(hv) < 20:
        return None
    
    current = hv.iloc[-1]
    ivr = (current - hv.min()) / (hv.max() - hv.min()) * 100 if hv.max() != hv.min() else 50
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hv.index, hv.values, color='cyan', linewidth=1.5)
    ax.axhline(hv.max(), color='red', linestyle='--', alpha=0.5, label=f'52W High: {hv.max():.1f}%')
    ax.axhline(hv.min(), color='green', linestyle='--', alpha=0.5, label=f'52W Low: {hv.min():.1f}%')
    ax.axhline(current, color='white', linestyle='-', linewidth=2, label=f'Current: {current:.1f}%')
    ax.set_title(f"IV Rank: {ivr:.0f}% | IV Percentile: {(hv < current).sum() / len(hv) * 100:.0f}%")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_parkinson(high, low, trading_days=252):
    park = compute_parkinson_vol(high.tail(60), low.tail(60), trading_days)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Parkinson Volatility'], [park], color='orange', width=0.5)
    ax.set_ylabel('%')
    ax.set_title(f"Parkinson Volatility = {park:.1f}%")
    ax.grid(True, alpha=0.3)
    return fig

def plot_volatility_cone(close, trading_days=252):
    log_ret = np.log(close / close.shift(1)).dropna()
    windows = [10, 20, 30, 60, 90, 120, 180]
    
    max_vol, min_vol, med_vol, cur_vol = [], [], [], []
    
    for w in windows:
        rv = log_ret.rolling(w).std() * np.sqrt(trading_days) * 100
        rv = rv.dropna()
        if len(rv) > 0:
            max_vol.append(rv.max())
            min_vol.append(rv.min())
            med_vol.append(rv.median())
            cur_vol.append(rv.iloc[-1])
    
    if len(max_vol) < 3:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(windows[:len(max_vol)], max_vol, 'o-', color='red', linewidth=2, markersize=6, label='Maximum')
    ax.plot(windows[:len(min_vol)], min_vol, 'o-', color='green', linewidth=2, markersize=6, label='Minimum')
    ax.plot(windows[:len(med_vol)], med_vol, 's--', color='white', linewidth=1.5, markersize=5, label='Median')
    ax.plot(windows[:len(cur_vol)], cur_vol, 'X-', color='yellow', linewidth=2, markersize=8, label='Current')
    ax.fill_between(windows[:len(max_vol)], min_vol, max_vol, color='gray', alpha=0.2)
    ax.legend()
    ax.set_xlabel("Window (Days)")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_title("Volatility Cone")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_expected_move(spot, implied_vol, trading_days):
    daily_move = spot * (implied_vol / 100) / np.sqrt(trading_days)
    weekly_move = daily_move * np.sqrt(5)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    categories = ['Spot', 'Day Range (±)', 'Week Range (±)']
    values = [spot, daily_move, weekly_move]
    colors = ['white', 'cyan', 'orange']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7)
    ax.set_ylabel(f"Price ({currency})")
    ax.set_title(f"Expected Move (IV: {implied_vol:.1f}%)")
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (spot*0.01),
                f'{val:,.0f}', ha='center', va='bottom', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_liquidity_sweeps(data):
    if data is None or len(data) < 30:
        return None
    
    df = data.tail(60).copy()
    df['Prev_High'] = df['High'].rolling(20).max().shift(1)
    df['Prev_Low'] = df['Low'].rolling(20).min().shift(1)
    df['Supply'] = (df['High'] > df['Prev_High']) & (df['Close'] < df['Prev_High'])
    df['Demand'] = (df['Low'] < df['Prev_Low']) & (df['Close'] > df['Prev_Low'])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Close'], color='white', linewidth=1.5)
    
    supply = df[df['Supply']]
    demand = df[df['Demand']]
    
    if not supply.empty:
        ax.scatter(supply.index, supply['High'] + (supply['High'] * 0.002), 
                  color='red', marker='v', s=80, label='Supply Sweep', zorder=5)
    if not demand.empty:
        ax.scatter(demand.index, demand['Low'] - (demand['Low'] * 0.002), 
                  color='green', marker='^', s=80, label='Demand Sweep', zorder=5)
    
    ax.legend()
    ax.set_title("Liquidity Sweeps (15-min)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_open_interest(spot):
    """Simulated OI profile (demo)."""
    step = 500 if spot > 10000 else 50
    base = round(spot / step) * step
    strikes = np.arange(base - 8*step, base + 9*step, step)
    
    # Simulate OI - in production, this would come from broker API
    np.random.seed(int(spot) % 1000)
    calls = np.random.randint(10, 80, len(strikes)) * 50000
    puts = np.random.randint(10, 80, len(strikes)) * 50000
    
    # Calculate max pain
    pain = {}
    for s in strikes:
        pain[s] = np.sum(np.maximum(0, s - strikes) * calls + np.maximum(0, strikes - s) * puts)
    max_pain = min(pain, key=pain.get)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(strikes, calls/1e5, color='red', alpha=0.7, label='Call OI')
    ax.barh(strikes, -puts/1e5, color='green', alpha=0.7, label='Put OI')
    ax.axhline(spot, color='cyan', linewidth=2, label=f'Spot: {spot:,.0f}')
    ax.axhline(max_pain, color='white', linestyle='--', linewidth=2, label=f'Max Pain: {max_pain:,.0f}')
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel("Open Interest (Lakhs)")
    ax.set_ylabel("Strike Price")
    ax.set_title("Open Interest Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# Explanations
EXPLANATIONS = {
    "Correlation": "**Correlation Analysis** - Measures how closely two assets move together. High correlation (>0.8) suggests they're moving in sync; low correlation (<0.5) suggests independent movement.",
    "Hurst Exponent": "**Hurst Exponent** - Classifies market behavior. >0.55 = trending (use momentum strategies), <0.45 = mean-reverting (fade breakouts), ~0.5 = random walk.",
    "IV Rank": "**IV Rank & Percentile** - IVR >65% means options are expensive (sell premium). IVR <30% means options are cheap (buy premium).",
    "Parkinson": "**Parkinson Volatility** - Uses intraday high-low range for more efficient volatility estimation than close-to-close.",
    "Volatility Cone": "**Volatility Cone** - Shows where current volatility stands relative to historical ranges. Yellow X marks current position.",
    "Expected Move": "**Expected Move** - Shows projected price range. Daily range (cyan) has ~68% probability; weekly range (orange) has ~68% probability.",
    "Liquidity": "**Liquidity Sweeps** - Red markers (supply) show price piercing previous highs and closing lower. Green markers (demand) show price piercing lows and closing higher.",
    "OI Profile": "**Open Interest Profile** - Shows concentration of option positions. Price often gravitates toward Max Pain (white line)."
}

# ========== MAIN APP ==========
# Sidebar
with st.sidebar:
    st.markdown("## AlphaQuant Terminal")
    
    market = st.radio("Market", ["Crypto", "Indian Market"], horizontal=True)
    
    if market == "Crypto":
        assets = {'Bitcoin': 'BTC-USD', 'Ethereum': 'ETH-USD', 'Dogecoin': 'DOGE-USD', 'XRP': 'XRP-USD'}
        trading_days = 365
        currency = "$"
    else:
        assets = {'Nifty 50': '^NSEI', 'Sensex': '^BSESN', 'Bank Nifty': '^NSEBANK'}
        trading_days = 252
        currency = "₹"
    
    selected_asset = st.selectbox("Asset", list(assets.keys()))
    ticker = assets[selected_asset]
    
    st.markdown("---")
    tab = st.radio("View", ["Dashboard", "Technical Analysis"])
    st.markdown("---")
    
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Load data
hist = fetch_data(ticker, period="2y")
live = get_live_price(ticker)

if hist is None or hist.empty:
    st.error("Unable to load data. Please try again.")
    st.stop()

# Current price
if live:
    spot = live['price']
    change_pct = live['pct']
else:
    spot = float(hist['Close'].iloc[-1])
    change_pct = 0

# Calculate metrics
close_series = hist['Close'].squeeze()
high_series = hist['High'].squeeze()
low_series = hist['Low'].squeeze()

iv_rank, iv_percentile = compute_iv_rank(close_series, 20)
parkinson = compute_parkinson_vol(high_series, low_series, trading_days)
hurst_current = hurst_exponent(close_series.tail(200))

# Use VIX-like placeholder for implied volatility
implied_vol = 25 if market == "Crypto" else 18

# Dashboard Tab
if tab == "Dashboard":
    st.title("Dashboard")
    
    # Price chart
    tf = st.radio("Chart Timeframe", ["1D", "1h", "15m"], horizontal=True)
    tf_map = {"1D": ("1y", "1d"), "1h": ("1mo", "1h"), "15m": ("5d", "15m")}
    period, interval = tf_map[tf]
    
    chart_data = fetch_data(ticker, period=period, interval=interval)
    if chart_data is not None and not chart_data.empty:
        bb_upper, bb_mid, bb_lower = bollinger_bands(chart_data['Close'])
        
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(go.Candlestick(
            x=chart_data.index,
            open=chart_data['Open'],
            high=chart_data['High'],
            low=chart_data['Low'],
            close=chart_data['Close'],
            name='Price'
        ))
        fig.add_trace(go.Scatter(x=chart_data.index, y=bb_upper, line=dict(color='gray', width=1, dash='dot'), name='BB Upper'))
        fig.add_trace(go.Scatter(x=chart_data.index, y=bb_lower, line=dict(color='gray', width=1, dash='dot'), name='BB Lower'))
        fig.update_layout(template='plotly_dark', height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        color = "#00FF00" if change_pct >= 0 else "#FF3333"
        st.metric("Spot Price", f"{currency}{spot:,.2f}", f"{change_pct:+.2f}%")
    with col2:
        st.metric("Parkinson Vol", f"{parkinson:.1f}%")
    with col3:
        st.metric("IV Rank", f"{iv_rank:.0f}%")
    with col4:
        regime = "Trending" if hurst_current > 0.55 else ("Mean-Reverting" if hurst_current < 0.45 else "Random Walk")
        st.metric("Market Regime", regime, f"H={hurst_current:.2f}")

# Technical Analysis Tab
else:
    st.title("Technical Analysis")
    
    # Row 1: Correlation
    with st.expander("Correlation Analysis", expanded=False):
        if market == "Crypto":
            fig = plot_correlation('BTC-USD', 'ETH-USD', 'Bitcoin', 'Ethereum')
        else:
            fig = plot_correlation('^NSEI', '^NSEBANK', 'Nifty 50', 'Bank Nifty')
        if fig:
            st.pyplot(fig)
            st.markdown(f'<div class="explanation-box">{EXPLANATIONS["Correlation"]}</div>', unsafe_allow_html=True)
        else:
            st.warning("Correlation data unavailable")
    
    # Row 2: Hurst and IV Rank
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Hurst Exponent", expanded=False):
            fig = plot_hurst(close_series)
            if fig:
                st.pyplot(fig)
                st.markdown(f'<div class="explanation-box">{EXPLANATIONS["Hurst Exponent"]}</div>', unsafe_allow_html=True)
            else:
                st.warning("Need more data for Hurst calculation (min 100 days)")
    
    with col2:
        with st.expander("IV Rank & IV Percentile", expanded=False):
            fig = plot_iv_rank(close_series, trading_days)
            if fig:
                st.pyplot(fig)
                st.markdown(f'<div class="explanation-box">{EXPLANATIONS["IV Rank"]}</div>', unsafe_allow_html=True)
            else:
                st.warning("IV data unavailable")
    
    # Row 3: Parkinson and Volatility Cone
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Parkinson Volatility", expanded=False):
            fig = plot_parkinson(high_series, low_series, trading_days)
            st.pyplot(fig)
            st.markdown(f'<div class="explanation-box">{EXPLANATIONS["Parkinson"]}</div>', unsafe_allow_html=True)
    
    with col2:
        with st.expander("Volatility Cone", expanded=False):
            fig = plot_volatility_cone(close_series, trading_days)
            if fig:
                st.pyplot(fig)
                st.markdown(f'<div class="explanation-box">{EXPLANATIONS["Volatility Cone"]}</div>', unsafe_allow_html=True)
            else:
                st.warning("Need more data for volatility cone")
    
    # Row 4: Expected Move and Liquidity
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Expected Move", expanded=False):
            fig = plot_expected_move(spot, implied_vol, trading_days)
            st.pyplot(fig)
            st.markdown(f'<div class="explanation-box">{EXPLANATIONS["Expected Move"]}</div>', unsafe_allow_html=True)
    
    with col2:
        with st.expander("Liquidity Detector (15-min)", expanded=False):
            intraday = fetch_data(ticker, period="5d", interval="15m")
            fig = plot_liquidity_sweeps(intraday)
            if fig:
                st.pyplot(fig)
                st.markdown(f'<div class="explanation-box">{EXPLANATIONS["Liquidity"]}</div>', unsafe_allow_html=True)
            else:
                st.warning("Insufficient intraday data")
    
    # Row 5: OI Profile (simulated)
    with st.expander("Open Interest Profile", expanded=False):
        fig = plot_open_interest(spot)
        st.pyplot(fig)
        st.markdown(f'<div class="explanation-box">{EXPLANATIONS["OI Profile"]}</div>', unsafe_allow_html=True)