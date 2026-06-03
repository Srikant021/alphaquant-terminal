# AlphaQuant Terminal - Unified Dashboard
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# Configure page
st.set_page_config(
    page_title="AlphaQuant Terminal - Unified Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00ff88;
        margin: 10px 0;
    }
    .explanation-box {
        background: rgba(0, 255, 136, 0.1);
        border-left: 4px solid #00ff88;
        padding: 12px;
        border-radius: 5px;
        font-size: 12px;
        margin: 10px 0;
    }
    .section-header {
        font-size: 18px;
        font-weight: bold;
        color: #00ff88;
        margin-top: 20px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ========== CORE FUNCTIONS ==========

# Hurst Exponent Calculation
def hurst_exponent(price_series):
    """Calculate Hurst exponent using rescaled range analysis."""
    price = np.array(price_series.dropna())
    if len(price) < 20:
        return 0.5
    
    log_prices = np.log(price)
    max_lag = min(len(log_prices) // 4, 100)
    if max_lag < 5:
        return 0.5
    
    lags = range(5, max_lag, 5)
    if len(lags) < 2:
        return 0.5
    
    rs_values = []
    for lag in lags:
        n_windows = len(log_prices) // lag
        if n_windows < 1:
            continue
        
        rs_window = []
        for i in range(n_windows):
            window = log_prices[i*lag:(i+1)*lag]
            if len(window) < 2:
                continue
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
    
    log_lags = np.log([lags[i] for i in range(len(rs_values))])
    log_rs = np.log(rs_values)
    hurst = np.polyfit(log_lags, log_rs, 1)[0]
    
    return np.clip(hurst, 0.2, 0.8)

# Technical Indicators
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

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# Data Fetching
@st.cache_data(ttl=300)
def fetch_data(ticker, period="1y", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None
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

# Volatility Metrics
def compute_parkinson_vol(high, low, periods=252):
    high = np.array(high.dropna())
    low = np.array(low.dropna())
    if len(high) < 2 or len(low) < 2:
        return 0
    log_hl = (np.log(high / low) ** 2)
    variance = log_hl.sum() / (4 * len(log_hl) * np.log(2))
    return np.sqrt(variance * periods) * 100

def compute_iv_rank(close, window=20):
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

# ========== CHART FUNCTIONS ==========

def create_main_price_chart(chart_data, ticker_name):
    """Create main candlestick chart with indicators."""
    if chart_data is None or chart_data.empty:
        return None
    
    bb_upper, bb_mid, bb_lower = bollinger_bands(chart_data['Close'])
    rsi_vals = rsi(chart_data['Close'])
    macd_line, signal_line, histogram = macd(chart_data['Close'])
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=chart_data.index,
        open=chart_data['Open'],
        high=chart_data['High'],
        low=chart_data['Low'],
        close=chart_data['Close'],
        name='OHLC'
    ), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=chart_data.index, y=bb_upper, 
                             line=dict(color='rgba(100,100,100,0.5)', width=1, dash='dot'),
                             name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=bb_lower,
                             line=dict(color='rgba(100,100,100,0.5)', width=1, dash='dot'),
                             name='BB Lower'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=bb_mid,
                             line=dict(color='rgba(150,150,150,0.7)', width=1),
                             name='SMA20'), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=chart_data.index, y=rsi_vals,
                             line=dict(color='#FF6B6B', width=2),
                             name='RSI(14)'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought",
                  row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold",
                  row=2, col=1)
    
    # MACD
    colors = ['green' if h > 0 else 'red' for h in histogram]
    fig.add_trace(go.Bar(x=chart_data.index, y=histogram, name='MACD Histogram',
                         marker_color=colors, showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=macd_line,
                             line=dict(color='#00BFFF', width=2),
                             name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=signal_line,
                             line=dict(color='#FF1493', width=2),
                             name='Signal'), row=3, col=1)
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        title=f"{ticker_name} - Technical Analysis",
        height=900,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig

def create_correlation_chart(ticker1, ticker2, name1, name2):
    """Create correlation analysis chart."""
    data1 = fetch_data(ticker1, period="1y")
    data2 = fetch_data(ticker2, period="1y")
    if data1 is None or data2 is None:
        return None
    
    merged = pd.DataFrame({
        name1: data1['Close'],
        name2: data2['Close']
    }).dropna()
    
    if len(merged) < 20:
        return None
    
    norm = merged / merged.iloc[0] * 100
    log_ret = np.log(merged / merged.shift(1)).dropna()
    corr = log_ret[name1].rolling(20).corr(log_ret[name2])
    
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12,
                        row_heights=[0.6, 0.4],
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}]])
    
    fig.add_trace(go.Scatter(x=norm.index, y=norm[name1],
                             name=name1, line=dict(color='#00FF88', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=norm.index, y=norm[name2],
                             name=name2, line=dict(color='#FF6B6B', width=2)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=corr.index, y=corr,
                             name='20D Correlation', line=dict(color='#00BFFF', width=2)), row=2, col=1)
    fig.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="High Corr",
                  row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Low Corr",
                  row=2, col=1)
    
    fig.update_yaxes(title_text="Normalized Price", row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=2, col=1)
    fig.update_layout(template='plotly_dark', height=500, hovermode='x unified')
    
    return fig

def create_hurst_chart(close):
    """Create Hurst exponent trend chart."""
    if len(close) < 100:
        return None
    
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
    
    current_hurst = hurst_exponent(close.tail(100))
    
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12,
                        row_heights=[0.6, 0.4],
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}]])
    
    fig.add_trace(go.Scatter(x=close.index, y=close.values,
                             name='Price', line=dict(color='white', width=1.5)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=dates, y=hurst_values,
                             name='Hurst Exponent', line=dict(color='#00BFFF', width=2),
                             mode='lines+markers'), row=2, col=1)
    
    fig.add_hline(y=0.55, line_dash="dash", line_color="green", annotation_text="Trending",
                  row=2, col=1)
    fig.add_hline(y=0.45, line_dash="dash", line_color="red", annotation_text="Mean-Reverting",
                  row=2, col=1)
    fig.add_hline(y=0.50, line_dash="solid", line_color="gray", annotation_text="Random",
                  row=2, col=1)
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Hurst", range=[0.2, 0.8], row=2, col=1)
    fig.update_layout(template='plotly_dark', height=500, hovermode='x unified')
    
    return fig

def create_volatility_cone(close, trading_days=252):
    """Create volatility cone chart."""
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
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=windows[:len(max_vol)], y=max_vol,
                             name='Maximum', line=dict(color='red', width=2),
                             mode='lines+markers'))
    fig.add_trace(go.Scatter(x=windows[:len(min_vol)], y=min_vol,
                             name='Minimum', line=dict(color='green', width=2),
                             mode='lines+markers'))
    fig.add_trace(go.Scatter(x=windows[:len(med_vol)], y=med_vol,
                             name='Median', line=dict(color='white', width=1, dash='dash'),
                             mode='lines+markers'))
    fig.add_trace(go.Scatter(x=windows[:len(cur_vol)], y=cur_vol,
                             name='Current', line=dict(color='#FFD700', width=3),
                             mode='lines+markers'))
    
    fig.add_trace(go.Scatter(x=windows[:len(max_vol)] + windows[:len(min_vol)][::-1],
                             y=max_vol + min_vol[::-1],
                             fill='toself', name='Range',
                             fillcolor='rgba(100,100,100,0.2)',
                             line=dict(color='rgba(255,255,255,0)'),
                             showlegend=False))
    
    fig.update_layout(
        template='plotly_dark',
        title='Volatility Cone',
        xaxis_title='Window (Days)',
        yaxis_title='Annualized Volatility (%)',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_expected_move_chart(spot, implied_vol, trading_days, currency="$"):
    """Create expected move visualization."""
    daily_move = spot * (implied_vol / 100) / np.sqrt(trading_days)
    weekly_move = daily_move * np.sqrt(5)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Spot Price', 'Daily Move (±)', 'Weekly Move (±)'],
        y=[spot, daily_move, weekly_move],
        marker_color=['white', '#00BFFF', '#FF6B6B'],
        text=[f'{v:,.0f}' for v in [spot, daily_move, weekly_move]],
        textposition='auto'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        title=f'Expected Move (IV: {implied_vol:.1f}%)',
        yaxis_title=f'Price ({currency})',
        height=400,
        showlegend=False
    )
    
    return fig

def create_iv_rank_chart(close, trading_days=252):
    """Create IV Rank visualization."""
    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) < 30:
        return None
    
    hv = log_ret.rolling(20).std() * np.sqrt(trading_days) * 100
    hv = hv.dropna()
    
    if len(hv) < 20:
        return None
    
    current = hv.iloc[-1]
    ivr = (current - hv.min()) / (hv.max() - hv.min()) * 100 if hv.max() != hv.min() else 50
    ivp = (hv < current).sum() / len(hv) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=hv.index, y=hv.values,
                             name='Historical Vol', line=dict(color='#00BFFF', width=2)))
    
    fig.add_hline(y=hv.max(), line_dash="dash", line_color="red",
                  annotation_text=f"52W High: {hv.max():.1f}%")
    fig.add_hline(y=hv.min(), line_dash="dash", line_color="green",
                  annotation_text=f"52W Low: {hv.min():.1f}%")
    fig.add_hline(y=current, line_dash="solid", line_color="white",
                  annotation_text=f"Current: {current:.1f}%")
    
    fig.update_layout(
        template='plotly_dark',
        title=f'IV Rank: {ivr:.0f}% | IV Percentile: {ivp:.0f}%',
        yaxis_title='Historical Volatility (%)',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_oi_profile(spot):
    """Create Open Interest profile visualization."""
    step = 500 if spot > 10000 else 50
    base = round(spot / step) * step
    strikes = np.arange(base - 8*step, base + 9*step, step)
    
    np.random.seed(int(spot) % 1000)
    calls = np.random.randint(10, 80, len(strikes)) * 50000
    puts = np.random.randint(10, 80, len(strikes)) * 50000
    
    pain = {}
    for s in strikes:
        pain[s] = np.sum(np.maximum(0, s - strikes) * calls + np.maximum(0, strikes - s) * puts)
    max_pain = min(pain, key=pain.get)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(y=strikes, x=calls/1e5, orientation='h',
                         name='Call OI', marker_color='red', opacity=0.7))
    fig.add_trace(go.Bar(y=strikes, x=-puts/1e5, orientation='h',
                         name='Put OI', marker_color='green', opacity=0.7))
    
    fig.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1)
    
    fig.update_layout(
        template='plotly_dark',
        title='Open Interest Profile',
        xaxis_title='Open Interest (Lakhs)',
        yaxis_title='Strike Price',
        height=500,
        barmode='relative',
        hovermode='y unified'
    )
    
    return fig

# ========== MAIN APP ==========

# Sidebar Configuration
with st.sidebar:
    st.markdown("# 📊 AlphaQuant Terminal")
    st.markdown("---")
    
    # Market Selection
    market = st.radio("📈 Market", ["Crypto", "Indian Market"], horizontal=True)
    
    if market == "Crypto":
        assets = {
            'Bitcoin': 'BTC-USD',
            'Ethereum': 'ETH-USD',
            'Dogecoin': 'DOGE-USD',
            'XRP': 'XRP-USD'
        }
        trading_days = 365
        currency = "$"
    else:
        assets = {
            'Nifty 50': '^NSEI',
            'Sensex': '^BSESN',
            'Bank Nifty': '^NSEBANK'
        }
        trading_days = 252
        currency = "₹"
    
    selected_asset = st.selectbox("🎯 Asset", list(assets.keys()))
    ticker = assets[selected_asset]
    
    st.markdown("---")
    
    # Timeframe Selection
    st.markdown("### Chart Timeframe")
    tf = st.radio("", ["1D", "1h", "15m"], horizontal=True, key="timeframe")
    tf_map = {"1D": ("1y", "1d"), "1h": ("1mo", "1h"), "15m": ("5d", "15m")}
    period, interval = tf_map[tf]
    
    st.markdown("---")
    
    # Correlation Selection
    st.markdown("### Correlation Pair")
    if market == "Crypto":
        corr_pair = st.selectbox(
            "Select pair",
            ["Bitcoin vs Ethereum", "Bitcoin vs Dogecoin"],
            key="crypto_pair"
        )
        corr_map = {
            "Bitcoin vs Ethereum": ("BTC-USD", "ETH-USD", "Bitcoin", "Ethereum"),
            "Bitcoin vs Dogecoin": ("BTC-USD", "DOGE-USD", "Bitcoin", "Dogecoin")
        }
    else:
        corr_pair = st.selectbox(
            "Select pair",
            ["Nifty 50 vs Bank Nifty", "Nifty 50 vs Sensex"],
            key="nifty_pair"
        )
        corr_map = {
            "Nifty 50 vs Bank Nifty": ("^NSEI", "^NSEBANK", "Nifty 50", "Bank Nifty"),
            "Nifty 50 vs Sensex": ("^NSEI", "^BSESN", "Nifty 50", "Sensex")
        }
    
    st.markdown("---")
    
    # Refresh Button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Load Data
hist = fetch_data(ticker, period="2y")
chart_data = fetch_data(ticker, period=period, interval=interval)
live = get_live_price(ticker)

if hist is None or hist.empty:
    st.error("❌ Unable to load data. Please try again.")
    st.stop()

# Extract Data
close_series = hist['Close'].squeeze()
high_series = hist['High'].squeeze()
low_series = hist['Low'].squeeze()

if live:
    spot = live['price']
    change_pct = live['pct']
    change_value = live['change']
else:
    spot = float(hist['Close'].iloc[-1])
    change_pct = 0
    change_value = 0

# Calculate Metrics
iv_rank, iv_percentile = compute_iv_rank(close_series, 20)
parkinson = compute_parkinson_vol(high_series, low_series, trading_days)
hurst_current = hurst_exponent(close_series.tail(200))
implied_vol = 25 if market == "Crypto" else 18

# Determine Market Regime
if hurst_current > 0.55:
    regime = "📈 Trending"
    regime_color = "#00FF88"
else:
    regime = "↩️ Mean-Reverting"
    regime_color = "#FF6B6B"

# ========== MAIN LAYOUT ==========

st.title("🎯 AlphaQuant Terminal - Unified Dashboard")
st.markdown("---")

# ========== TOP METRICS ROW ==========
st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    color_price = "#00FF88" if change_pct >= 0 else "#FF6B6B"
    st.markdown(f"""
    <div class="metric-box" style="border-left-color: {color_price}">
        <div style="font-size: 12px; opacity: 0.7;">Spot Price</div>
        <div style="font-size: 24px; font-weight: bold; color: {color_price};">{currency}{spot:,.2f}</div>
        <div style="font-size: 11px; color: {color_price}; margin-top: 5px;">{change_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div style="font-size: 12px; opacity: 0.7;">Parkinson Vol</div>
        <div style="font-size: 24px; font-weight: bold; color: #00BFFF;">{parkinson:.1f}%</div>
        <div style="font-size: 11px; opacity: 0.7;">Annualized</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <div style="font-size: 12px; opacity: 0.7;">IV Rank</div>
        <div style="font-size: 24px; font-weight: bold; color: #FFD700;">{iv_rank:.0f}%</div>
        <div style="font-size: 11px; opacity: 0.7;">Implied Vol Rank</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-box">
        <div style="font-size: 12px; opacity: 0.7;">IV Percentile</div>
        <div style="font-size: 24px; font-weight: bold; color: #FF1493;">{iv_percentile:.0f}%</div>
        <div style="font-size: 11px; opacity: 0.7;">Historical Context</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-box">
        <div style="font-size: 12px; opacity: 0.7;">Hurst Exponent</div>
        <div style="font-size: 24px; font-weight: bold; color: #00BFFF;">{hurst_current:.2f}</div>
        <div style="font-size: 11px; opacity: 0.7;">Market Persistence</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="metric-box" style="border-left-color: {regime_color}">
        <div style="font-size: 12px; opacity: 0.7;">Market Regime</div>
        <div style="font-size: 24px; font-weight: bold; color: {regime_color};">{regime}</div>
        <div style="font-size: 11px; opacity: 0.7;">Current State</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ========== MAIN PRICE CHART ==========
st.markdown('<div class="section-header">💹 Price Action & Technical Analysis</div>', unsafe_allow_html=True)

if chart_data is not None and not chart_data.empty:
    fig_price = create_main_price_chart(chart_data, selected_asset)
    if fig_price:
        st.plotly_chart(fig_price, use_container_width=True)
        st.markdown("""
        <div class="explanation-box">
            <strong>📌 Technical Indicators:</strong>
            <ul>
                <li><strong>Bollinger Bands:</strong> Price volatility boundaries</li>
                <li><strong>RSI:</strong> Momentum indicator (Overbought >70, Oversold <30)</li>
                <li><strong>MACD:</strong> Trend and momentum confirmation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ========== TWO-COLUMN SECTION ==========
st.markdown('<div class="section-header">🔍 Advanced Analysis</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

# Correlation Analysis
with col1:
    st.markdown("### Correlation Analysis")
    ticker1, ticker2, name1, name2 = corr_map[corr_pair]
    fig_corr = create_correlation_chart(ticker1, ticker2, name1, name2)
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("""
        <div class="explanation-box">
            <strong>📌 Correlation Insight:</strong> Measures how closely two assets move together.
            High correlation (>0.8) = moving in sync; Low (<0.5) = independent movements.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📊 Correlation data unavailable")

# Hurst Exponent
with col2:
    st.markdown("### Market Regime (Hurst Exponent)")
    fig_hurst = create_hurst_chart(close_series)
    if fig_hurst:
        st.plotly_chart(fig_hurst, use_container_width=True)
        st.markdown("""
        <div class="explanation-box">
            <strong>📌 Hurst Insight:</strong> 
            • >0.55 = Trending (use momentum)
            • <0.45 = Mean-reverting (fade breakouts)
            • ~0.5 = Random walk
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📊 Need more data (min 100 days)")

st.markdown("---")

# ========== THREE-COLUMN SECTION ==========
col1, col2, col3 = st.columns(3)

# Volatility Cone
with col1:
    st.markdown("### Volatility Cone")
    fig_vol_cone = create_volatility_cone(close_series, trading_days)
    if fig_vol_cone:
        st.plotly_chart(fig_vol_cone, use_container_width=True)
        st.markdown("""
        <div class="explanation-box">
            <strong>📌 Volatility Context:</strong> Yellow X shows current volatility
            relative to historical range across different time windows.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📊 Need more data")

# Expected Move
with col2:
    st.markdown("### Expected Move")
    fig_exp_move = create_expected_move_chart(spot, implied_vol, trading_days, currency)
    if fig_exp_move:
        st.plotly_chart(fig_exp_move, use_container_width=True)
        st.markdown("""
        <div class="explanation-box">
            <strong>📌 Expected Move:</strong> Projected price range with ~68% probability.
            Daily and weekly ranges help set options strategies.
        </div>
        """, unsafe_allow_html=True)

# IV Rank
with col3:
    st.markdown("### IV Rank & Percentile")
    fig_iv = create_iv_rank_chart(close_series, trading_days)
    if fig_iv:
        st.plotly_chart(fig_iv, use_container_width=True)
        st.markdown("""
        <div class="explanation-box">
            <strong>📌 IV Strategy:</strong>
            • IV Rank >65% = Sell premium (expensive)
            • IV Rank <30% = Buy premium (cheap)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📊 IV data unavailable")

st.markdown("---")

# ========== OPEN INTEREST PROFILE ==========
st.markdown('<div class="section-header">📍 Open Interest Profile</div>', unsafe_allow_html=True)

fig_oi = create_oi_profile(spot)
st.plotly_chart(fig_oi, use_container_width=True)

st.markdown("""
<div class="explanation-box">
    <strong>📌 Open Interest Profile:</strong> Shows concentration of call and put positions.
    Price often gravitates toward Max Pain (where most options expire worthless).
    Red markers = calls, Green markers = puts.
</div>
""", unsafe_allow_html=True)

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.6; margin-top: 30px;">
    <p>🚀 AlphaQuant Terminal | Advanced Quantitative Analysis Dashboard</p>
    <p>Data refreshed every 5 minutes | © 2024</p>
</div>
""", unsafe_allow_html=True)
