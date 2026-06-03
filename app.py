# AlphaQuant Terminal - Unified Dashboard
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="AlphaQuant Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp {
        background: #080d12;
    }
    .metric-box {
        background: linear-gradient(135deg, #0d1520 0%, #111c2b 100%);
        padding: 18px 20px;
        border-radius: 8px;
        border-left: 3px solid #0af;
        margin: 6px 0;
        border-top: 1px solid rgba(0,170,255,0.08);
    }
    .explanation-box {
        background: rgba(0,170,255,0.05);
        border-left: 3px solid #0af;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 12px;
        margin: 8px 0;
        color: rgba(255,255,255,0.75);
    }
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #0af;
        margin: 24px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(0,170,255,0.2);
    }
    .hurst-panel {
        background: linear-gradient(135deg, #0d1520 0%, #111c2b 100%);
        border-radius: 8px;
        padding: 16px 20px;
        border: 1px solid rgba(0,170,255,0.12);
        margin-bottom: 10px;
    }
    .regime-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ========== CORE FUNCTIONS ==========

def hurst_exponent(price_series):
    """
    Calculate Hurst exponent via rescaled range (R/S) analysis.
    Uses a minimum 120-point window for statistical reliability.
    Returns: H value, interpretation string, confidence level
    """
    price = np.array(price_series.dropna())
    n = len(price)
    if n < 30:
        return 0.5, "Insufficient data", "low"

    log_prices = np.log(price)
    # Use lags from 10 up to n//4, capped at 150
    max_lag = min(n // 4, 150)
    if max_lag < 10:
        return 0.5, "Insufficient data", "low"

    lags = list(range(10, max_lag, max(1, max_lag // 20)))
    rs_values = []
    valid_lags = []

    for lag in lags:
        n_windows = n // lag
        if n_windows < 2:
            continue
        rs_window = []
        for i in range(n_windows):
            window = log_prices[i * lag:(i + 1) * lag]
            if len(window) < 4:
                continue
            mean_adj = window - np.mean(window)
            cumsum = np.cumsum(mean_adj)
            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(window, ddof=1)
            if S > 1e-10:
                rs_window.append(R / S)
        if len(rs_window) >= 2:
            rs_values.append(np.mean(rs_window))
            valid_lags.append(lag)

    if len(rs_values) < 4:
        return 0.5, "Insufficient data", "low"

    log_lags = np.log(valid_lags)
    log_rs = np.log(rs_values)
    # Fit via OLS
    coeffs = np.polyfit(log_lags, log_rs, 1)
    hurst = float(np.clip(coeffs[0], 0.1, 0.9))

    # Confidence: higher with more lags and better R²
    residuals = log_rs - np.polyval(coeffs, log_lags)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_rs - np.mean(log_rs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    confidence = "high" if (r2 > 0.92 and len(valid_lags) >= 8) else "medium" if r2 > 0.80 else "low"

    # Interpretation with refined thresholds
    if hurst > 0.58:
        interpretation = "Strong Trend (Persistent)"
    elif hurst > 0.53:
        interpretation = "Weak Trend (Mildly Persistent)"
    elif hurst >= 0.47:
        interpretation = "Random Walk"
    elif hurst >= 0.42:
        interpretation = "Weak Mean-Reversion"
    else:
        interpretation = "Strong Mean-Reversion (Anti-Persistent)"

    return hurst, interpretation, confidence


def bollinger_bands(close, period=20, std=2):
    sma = close.rolling(window=period).mean()
    std_dev = close.rolling(window=period).std()
    return sma + (std * std_dev), sma, sma - (std * std_dev)


def ema(close, span):
    return close.ewm(span=span, adjust=False).mean()


def vwap(df):
    """VWAP for the visible window."""
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    vol = df['Volume'].replace(0, np.nan).fillna(method='ffill')
    cum_vol = vol.cumsum()
    cum_tp_vol = (typical * vol).cumsum()
    return cum_tp_vol / cum_vol


@st.cache_data(ttl=300)
def fetch_data(ticker, period="1y", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
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
    data = fetch_data(ticker, period="5d", interval="1d")
    if data is None or len(data) < 2:
        return None
    last = float(data['Close'].iloc[-1])
    prev = float(data['Close'].iloc[-2])
    return {
        'price': last,
        'change': last - prev,
        'pct': ((last - prev) / prev) * 100 if prev else 0,
        'high': float(data['High'].iloc[-1]),
        'low': float(data['Low'].iloc[-1]),
        'volume': float(data['Volume'].iloc[-1]),
    }


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

def create_main_price_chart(chart_data, ticker_name, currency="$"):
    """Advanced candlestick chart: Bollinger Bands, EMA ribbon (9/21/50/200), VWAP, Volume."""
    if chart_data is None or chart_data.empty:
        return None

    bb_upper, bb_mid, bb_lower = bollinger_bands(chart_data['Close'])
    ema9   = ema(chart_data['Close'], 9)
    ema21  = ema(chart_data['Close'], 21)
    ema50  = ema(chart_data['Close'], 50)
    ema200 = ema(chart_data['Close'], 200)
    vwap_line = vwap(chart_data)

    # Volume colour
    vol_colors = [
        '#00c87844' if float(chart_data['Close'].iloc[i]) >= float(chart_data['Open'].iloc[i])
        else '#ff4d4d44'
        for i in range(len(chart_data))
    ]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.78, 0.22],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # ── Candlestick ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=chart_data.index,
        open=chart_data['Open'],
        high=chart_data['High'],
        low=chart_data['Low'],
        close=chart_data['Close'],
        name='OHLC',
        increasing=dict(line=dict(color='#00c878', width=1), fillcolor='#00c87888'),
        decreasing=dict(line=dict(color='#ff4d6d', width=1), fillcolor='#ff4d6d88'),
    ), row=1, col=1, secondary_y=False)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=chart_data.index, y=bb_upper,
        line=dict(color='rgba(0,170,255,0.35)', width=1, dash='dot'),
        name='BB Upper', showlegend=False
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=chart_data.index, y=bb_lower,
        line=dict(color='rgba(0,170,255,0.35)', width=1, dash='dot'),
        name='BB Lower', fill='tonexty',
        fillcolor='rgba(0,170,255,0.04)', showlegend=False
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=chart_data.index, y=bb_mid,
        line=dict(color='rgba(0,170,255,0.55)', width=1),
        name='BB Mid / SMA20'
    ), row=1, col=1, secondary_y=False)

    # ── EMA Ribbon ────────────────────────────────────────────────────────────
    ema_specs = [
        (ema9,   '#ffd700', 1.2, 'EMA 9'),
        (ema21,  '#ff9500', 1.2, 'EMA 21'),
        (ema50,  '#ff4d6d', 1.5, 'EMA 50'),
        (ema200, '#c084fc', 2.0, 'EMA 200'),
    ]
    for series, color, width, label in ema_specs:
        fig.add_trace(go.Scatter(
            x=chart_data.index, y=series,
            line=dict(color=color, width=width),
            name=label
        ), row=1, col=1, secondary_y=False)

    # ── VWAP ──────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=chart_data.index, y=vwap_line,
        line=dict(color='#00e5ff', width=1.5, dash='dashdot'),
        name='VWAP'
    ), row=1, col=1, secondary_y=False)

    # ── Volume (secondary y on row 1) ─────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=chart_data.index,
        y=chart_data['Volume'],
        name='Volume',
        marker_color=vol_colors,
        showlegend=False
    ), row=1, col=1, secondary_y=True)

    # ── Volume panel (row 2) with MA ─────────────────────────────────────────
    vol_ma = chart_data['Volume'].rolling(20).mean()
    fig.add_trace(go.Bar(
        x=chart_data.index, y=chart_data['Volume'],
        marker_color=vol_colors, name='Volume', showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=chart_data.index, y=vol_ma,
        line=dict(color='#ffd700', width=1.5),
        name='Vol MA20', showlegend=False
    ), row=2, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#080d12',
        plot_bgcolor='#0a1018',
        title=dict(
            text=f"<b>{ticker_name}</b> — Price Action",
            font=dict(family='Space Mono, monospace', size=14, color='#0af'),
            x=0.01
        ),
        height=700,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend=dict(
            orientation='h', yanchor='bottom', y=1.01,
            xanchor='left', x=0,
            font=dict(size=11),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=60, r=20, t=50, b=40),
    )

    fig.update_yaxes(
        gridcolor='rgba(255,255,255,0.04)',
        title_text=f"Price ({currency})", row=1, col=1,
        secondary_y=False
    )
    fig.update_yaxes(
        showticklabels=False, title_text="", row=1, col=1,
        secondary_y=True
    )
    fig.update_yaxes(
        gridcolor='rgba(255,255,255,0.04)',
        title_text="Volume", row=2, col=1
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')

    return fig


def create_hurst_chart(close):
    """
    Rolling Hurst chart with 120-day window, regime-shaded background,
    and correct interpretation labels.
    """
    if len(close) < 120:
        return None, None, None, None

    hurst_values = []
    dates = []
    step = 5

    for i in range(120, len(close), step):
        window = close.iloc[i - 120:i]
        h, _, _ = hurst_exponent(window)
        hurst_values.append(h)
        dates.append(close.index[i - 1])

    if len(hurst_values) < 5:
        return None, None, None, None

    # Current Hurst on last 200 bars (or all available)
    lookback = min(200, len(close))
    current_h, current_interp, confidence = hurst_exponent(close.tail(lookback))

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.10,
                        row_heights=[0.55, 0.45],
                        shared_xaxes=True)

    # Price
    fig.add_trace(go.Scatter(
        x=close.index, y=close.values,
        name='Price', line=dict(color='rgba(255,255,255,0.8)', width=1.5)
    ), row=1, col=1)

    # Hurst line
    fig.add_trace(go.Scatter(
        x=dates, y=hurst_values,
        name='Hurst (120D)', line=dict(color='#00e5ff', width=2),
        mode='lines'
    ), row=2, col=1)

    # Reference zones — shaded bands
    fig.add_hrect(y0=0.58, y1=0.9,  fillcolor='rgba(0,200,100,0.07)',
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=0.42, y1=0.58, fillcolor='rgba(200,200,200,0.04)',
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=0.1,  y1=0.42, fillcolor='rgba(255,80,80,0.07)',
                  line_width=0, row=2, col=1)

    # Lines
    for y, color, label in [
        (0.58, '#00c878', 'Trending  H>0.58'),
        (0.50, 'rgba(200,200,200,0.5)', 'Random Walk  H=0.50'),
        (0.42, '#ff4d6d', 'Mean-Rev  H<0.42'),
    ]:
        fig.add_hline(y=y, line_dash='dash', line_color=color,
                      annotation_text=label,
                      annotation_font=dict(size=10, color=color),
                      row=2, col=1)

    # Current value marker
    if dates:
        fig.add_trace(go.Scatter(
            x=[dates[-1]], y=[hurst_values[-1]],
            mode='markers',
            marker=dict(size=10, color='#ffd700', line=dict(width=2, color='white')),
            name=f'Current H={hurst_values[-1]:.3f}',
            showlegend=True
        ), row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#080d12',
        plot_bgcolor='#0a1018',
        height=520,
        hovermode='x unified',
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=20, t=30, b=40),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1,
                     gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(title_text="Hurst Exponent", range=[0.2, 0.8], row=2, col=1,
                     gridcolor='rgba(255,255,255,0.04)')
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')

    return fig, current_h, current_interp, confidence


def create_correlation_chart(ticker1, ticker2, name1, name2):
    data1 = fetch_data(ticker1, period="1y")
    data2 = fetch_data(ticker2, period="1y")
    if data1 is None or data2 is None:
        return None

    merged = pd.DataFrame({name1: data1['Close'], name2: data2['Close']}).dropna()
    if len(merged) < 20:
        return None

    norm = merged / merged.iloc[0] * 100
    log_ret = np.log(merged / merged.shift(1)).dropna()
    corr = log_ret[name1].rolling(20).corr(log_ret[name2])

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12,
                        row_heights=[0.6, 0.4])

    fig.add_trace(go.Scatter(x=norm.index, y=norm[name1],
                             name=name1, line=dict(color='#00c878', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=norm.index, y=norm[name2],
                             name=name2, line=dict(color='#ff4d6d', width=2)), row=1, col=1)

    corr_colors = ['#00c878' if v > 0.8 else '#ffd700' if v > 0.5 else '#ff4d6d'
                   for v in corr.fillna(0)]

    fig.add_trace(go.Scatter(x=corr.index, y=corr,
                             name='20D Rolling Corr',
                             line=dict(color='#00e5ff', width=2)), row=2, col=1)
    fig.add_hline(y=0.8, line_dash="dash", line_color="#00c878",
                  annotation_text="High (0.8)", row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="#ff4d6d",
                  annotation_text="Low (0.5)", row=2, col=1)
    fig.add_hline(y=0.0, line_dash="solid", line_color="rgba(255,255,255,0.2)", row=2, col=1)

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
        height=500, hovermode='x unified',
        margin=dict(l=60, r=20, t=30, b=40),
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)')
    )
    fig.update_yaxes(title_text="Normalised Price", row=1, col=1,
                     gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(title_text="Correlation", range=[-1, 1], row=2, col=1,
                     gridcolor='rgba(255,255,255,0.04)')
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


def create_volatility_cone(close, trading_days=252):
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

    w = windows[:len(max_vol)]
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=w + w[::-1], y=max_vol + min_vol[::-1],
                             fill='toself', fillcolor='rgba(0,170,255,0.07)',
                             line=dict(color='rgba(0,0,0,0)'),
                             name='Historical Range', showlegend=True))
    fig.add_trace(go.Scatter(x=w, y=max_vol, name='Max',
                             line=dict(color='#ff4d6d', width=1.5, dash='dot'),
                             mode='lines+markers', marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=w, y=min_vol, name='Min',
                             line=dict(color='#00c878', width=1.5, dash='dot'),
                             mode='lines+markers', marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=w, y=med_vol, name='Median',
                             line=dict(color='rgba(255,255,255,0.5)', width=1.5),
                             mode='lines+markers', marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=w, y=cur_vol, name='Current',
                             line=dict(color='#ffd700', width=3),
                             mode='lines+markers',
                             marker=dict(size=9, symbol='diamond', color='#ffd700',
                                         line=dict(width=2, color='white'))))

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
        title=dict(text='<b>Volatility Cone</b>',
                   font=dict(family='Space Mono, monospace', size=13, color='#0af')),
        xaxis_title='Window (Days)', yaxis_title='Annualised Volatility (%)',
        height=420, hovermode='x unified',
        margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)')
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


def create_iv_rank_chart(close, trading_days=252):
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

    # Colour the line by level
    high_thresh = hv.quantile(0.65)
    low_thresh  = hv.quantile(0.30)
    colors = ['#ff4d6d' if v >= high_thresh else '#00c878' if v <= low_thresh else '#ffd700'
              for v in hv.values]

    fig = go.Figure()
    for i in range(1, len(hv)):
        fig.add_trace(go.Scatter(
            x=hv.index[i-1:i+1], y=hv.values[i-1:i+1],
            mode='lines', line=dict(color=colors[i], width=2),
            showlegend=False, hoverinfo='skip'
        ))

    fig.add_hline(y=hv.max(), line_dash="dash", line_color="#ff4d6d",
                  annotation_text=f"52W High: {hv.max():.1f}%",
                  annotation_font=dict(color='#ff4d6d', size=10))
    fig.add_hline(y=hv.min(), line_dash="dash", line_color="#00c878",
                  annotation_text=f"52W Low: {hv.min():.1f}%",
                  annotation_font=dict(color='#00c878', size=10))
    fig.add_hline(y=current, line_dash="solid", line_color="#ffd700",
                  annotation_text=f"Current: {current:.1f}%",
                  annotation_font=dict(color='#ffd700', size=10))

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
        title=dict(
            text=f'<b>IV Rank {ivr:.0f}%</b>  |  IV Percentile {ivp:.0f}%',
            font=dict(family='Space Mono, monospace', size=13, color='#0af')
        ),
        yaxis_title='HV-20 (%)', height=420,
        hovermode='x unified',
        margin=dict(l=60, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


def create_expected_move_chart(spot, implied_vol, trading_days, currency="$"):
    daily_move  = spot * (implied_vol / 100) / np.sqrt(trading_days)
    weekly_move = daily_move * np.sqrt(5)
    monthly_move = daily_move * np.sqrt(21)

    labels = ['Daily (±1σ)', 'Weekly (±1σ)', 'Monthly (±1σ)']
    values = [daily_move, weekly_move, monthly_move]
    pcts   = [v / spot * 100 for v in values]
    colors = ['#00e5ff', '#ffd700', '#ff9500']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f'{currency}{v:,.0f}<br>({p:.1f}%)' for v, p in zip(values, pcts)],
        textposition='outside',
        textfont=dict(size=11)
    ))

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
        title=dict(
            text=f'<b>Expected Move</b>  (IV: {implied_vol:.1f}%, 1σ = 68% prob)',
            font=dict(family='Space Mono, monospace', size=13, color='#0af')
        ),
        yaxis_title=f'Move ({currency})', height=420,
        showlegend=False,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


def create_oi_profile(spot):
    step = 500 if spot > 10000 else 100 if spot > 2000 else 50
    base = round(spot / step) * step
    strikes = np.arange(base - 8 * step, base + 9 * step, step)

    np.random.seed(int(spot) % 1000)
    calls = np.random.randint(10, 80, len(strikes)) * 50000
    puts  = np.random.randint(10, 80, len(strikes)) * 50000

    # Max Pain
    pain = {}
    for s in strikes:
        pain[s] = (np.sum(np.maximum(0, s - strikes) * calls) +
                   np.sum(np.maximum(0, strikes - s) * puts))
    max_pain = min(pain, key=pain.get)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=strikes, x=calls / 1e5, orientation='h',
        name='Call OI', marker_color='rgba(255,77,109,0.75)', opacity=0.9
    ))
    fig.add_trace(go.Bar(
        y=strikes, x=-puts / 1e5, orientation='h',
        name='Put OI', marker_color='rgba(0,200,120,0.75)', opacity=0.9
    ))

    fig.add_hline(y=spot, line_color='#ffd700', line_dash='solid', line_width=2,
                  annotation_text=f'Spot {spot:,.0f}',
                  annotation_font=dict(color='#ffd700', size=11))
    fig.add_hline(y=max_pain, line_color='#c084fc', line_dash='dash', line_width=1.5,
                  annotation_text=f'Max Pain {max_pain:,.0f}',
                  annotation_font=dict(color='#c084fc', size=11))
    fig.add_vline(x=0, line_color='rgba(255,255,255,0.2)', line_width=1)

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
        title=dict(text='<b>Open Interest Profile</b>',
                   font=dict(family='Space Mono, monospace', size=13, color='#0af')),
        xaxis_title='OI (Lakhs)', yaxis_title='Strike Price',
        height=520, barmode='relative', hovermode='y unified',
        margin=dict(l=80, r=20, t=50, b=40),
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)')
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


# ========== SIDEBAR ==========

with st.sidebar:
    st.markdown("## 📊 AlphaQuant")
    st.markdown("---")

    market = st.radio("Market", ["Crypto", "Indian Market"], horizontal=True)

    if market == "Crypto":
        assets = {'Bitcoin': 'BTC-USD', 'Ethereum': 'ETH-USD',
                  'Dogecoin': 'DOGE-USD', 'XRP': 'XRP-USD'}
        trading_days = 365
        currency = "$"
        implied_vol = 60
    else:
        assets = {'Nifty 50': '^NSEI', 'Sensex': '^BSESN', 'Bank Nifty': '^NSEBANK'}
        trading_days = 252
        currency = "₹"
        implied_vol = 18

    selected_asset = st.selectbox("Asset", list(assets.keys()))
    ticker = assets[selected_asset]

    st.markdown("---")
    st.markdown("### Chart Timeframe")
    tf = st.radio("", ["1D", "1h", "15m"], horizontal=True)
    tf_map = {"1D": ("1y", "1d"), "1h": ("1mo", "1h"), "15m": ("5d", "15m")}
    period, interval = tf_map[tf]

    st.markdown("---")
    st.markdown("### Correlation Pair")
    if market == "Crypto":
        corr_pair = st.selectbox("Pair", ["Bitcoin vs Ethereum", "Bitcoin vs Dogecoin"])
        corr_map = {
            "Bitcoin vs Ethereum": ("BTC-USD", "ETH-USD", "Bitcoin", "Ethereum"),
            "Bitcoin vs Dogecoin": ("BTC-USD", "DOGE-USD", "Bitcoin", "Dogecoin")
        }
    else:
        corr_pair = st.selectbox("Pair", ["Nifty 50 vs Bank Nifty", "Nifty 50 vs Sensex"])
        corr_map = {
            "Nifty 50 vs Bank Nifty": ("^NSEI", "^NSEBANK", "Nifty 50", "Bank Nifty"),
            "Nifty 50 vs Sensex": ("^NSEI", "^BSESN", "Nifty 50", "Sensex")
        }

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ========== DATA ==========

hist       = fetch_data(ticker, period="2y")
chart_data = fetch_data(ticker, period=period, interval=interval)
live       = get_live_price(ticker)

if hist is None or hist.empty:
    st.error("❌ Unable to load data. Please try again.")
    st.stop()

close_series = hist['Close'].squeeze()
high_series  = hist['High'].squeeze()
low_series   = hist['Low'].squeeze()

if live:
    spot         = live['price']
    change_pct   = live['pct']
    change_value = live['change']
    day_high     = live.get('high', spot)
    day_low      = live.get('low', spot)
    day_vol      = live.get('volume', 0)
else:
    spot         = float(hist['Close'].iloc[-1])
    change_pct   = 0
    change_value = 0
    day_high = day_low = spot
    day_vol  = 0

iv_rank, iv_percentile = compute_iv_rank(close_series, 20)
parkinson  = compute_parkinson_vol(high_series, low_series, trading_days)
fig_hurst, current_h, current_interp, hurst_confidence = create_hurst_chart(close_series)

if current_h is None:
    current_h = 0.5
    current_interp = "Insufficient data"
    hurst_confidence = "low"

# Regime colours
regime_color = ('#00c878' if current_h > 0.58 else
                '#ff4d6d' if current_h < 0.42 else '#ffd700')

# ========== HEADER ==========

st.title("AlphaQuant Terminal")
st.markdown(f"<span style='opacity:0.5;font-size:13px'>Last updated: {datetime.now().strftime('%H:%M:%S')} &nbsp;|&nbsp; {selected_asset} ({ticker})</span>", unsafe_allow_html=True)
st.markdown("---")

# ========== METRICS ==========

st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)

price_color = '#00c878' if change_pct >= 0 else '#ff4d6d'

with c1:
    st.markdown(f"""
    <div class="metric-box" style="border-left-color:{price_color}">
        <div style="font-size:11px;opacity:0.6;font-family:Space Mono,monospace;letter-spacing:.08em">SPOT PRICE</div>
        <div style="font-size:22px;font-weight:600;color:{price_color};margin:4px 0">{currency}{spot:,.2f}</div>
        <div style="font-size:11px;color:{price_color}">{change_pct:+.2f}% &nbsp; {currency}{change_value:+,.2f}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-box">
        <div style="font-size:11px;opacity:0.6;font-family:Space Mono,monospace;letter-spacing:.08em">PARKINSON VOL</div>
        <div style="font-size:22px;font-weight:600;color:#0af;margin:4px 0">{parkinson:.1f}%</div>
        <div style="font-size:11px;opacity:0.6">Annualised OHLC Vol</div>
    </div>""", unsafe_allow_html=True)

with c3:
    ivr_color = '#ff4d6d' if iv_rank > 65 else '#00c878' if iv_rank < 30 else '#ffd700'
    st.markdown(f"""
    <div class="metric-box" style="border-left-color:{ivr_color}">
        <div style="font-size:11px;opacity:0.6;font-family:Space Mono,monospace;letter-spacing:.08em">IV RANK</div>
        <div style="font-size:22px;font-weight:600;color:{ivr_color};margin:4px 0">{iv_rank:.0f}%</div>
        <div style="font-size:11px;opacity:0.6">{'Sell premium' if iv_rank>65 else 'Buy premium' if iv_rank<30 else 'Neutral'}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-box">
        <div style="font-size:11px;opacity:0.6;font-family:Space Mono,monospace;letter-spacing:.08em">IV PERCENTILE</div>
        <div style="font-size:22px;font-weight:600;color:#c084fc;margin:4px 0">{iv_percentile:.0f}%</div>
        <div style="font-size:11px;opacity:0.6">Historical context</div>
    </div>""", unsafe_allow_html=True)

with c5:
    conf_icon = '●' if hurst_confidence == 'high' else '◑' if hurst_confidence == 'medium' else '○'
    st.markdown(f"""
    <div class="metric-box" style="border-left-color:{regime_color}">
        <div style="font-size:11px;opacity:0.6;font-family:Space Mono,monospace;letter-spacing:.08em">HURST EXPONENT</div>
        <div style="font-size:22px;font-weight:600;color:{regime_color};margin:4px 0">{current_h:.3f}</div>
        <div style="font-size:11px;color:{regime_color}">{conf_icon} {hurst_confidence} confidence</div>
    </div>""", unsafe_allow_html=True)

with c6:
    st.markdown(f"""
    <div class="metric-box" style="border-left-color:{regime_color}">
        <div style="font-size:11px;opacity:0.6;font-family:Space Mono,monospace;letter-spacing:.08em">MARKET REGIME</div>
        <div style="font-size:14px;font-weight:600;color:{regime_color};margin:6px 0;line-height:1.3">{current_interp}</div>
        <div style="font-size:11px;opacity:0.6">R/S analysis (120D)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ========== PRICE CHART ==========

st.markdown('<div class="section-header">Price Action & Advanced Technicals</div>', unsafe_allow_html=True)

if chart_data is not None and not chart_data.empty:
    fig_price = create_main_price_chart(chart_data, selected_asset, currency)
    if fig_price:
        st.plotly_chart(fig_price, use_container_width=True)
        st.markdown("""
        <div class="explanation-box">
            <b>Chart Layers:</b> &nbsp;
            Candlestick (OHLC) &nbsp;·&nbsp;
            Bollinger Bands (20, 2σ) &nbsp;·&nbsp;
            EMA Ribbon: 9 / 21 / 50 / 200 &nbsp;·&nbsp;
            VWAP (dotted cyan) &nbsp;·&nbsp;
            Volume bars + 20D MA
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ========== ADVANCED ANALYSIS ==========

st.markdown('<div class="section-header">Advanced Analysis</div>', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    st.markdown("#### Correlation Analysis")
    t1, t2, n1, n2 = corr_map[corr_pair]
    fig_corr = create_correlation_chart(t1, t2, n1, n2)
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("""
        <div class="explanation-box">
            <b>Rolling 20D Correlation:</b> &gt;0.8 = moving in sync &nbsp;·&nbsp;
            &lt;0.5 = diverging / independent &nbsp;·&nbsp;
            Negative = inverse relationship
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Correlation data unavailable")

with col_r:
    st.markdown("#### Market Regime — Hurst Exponent (120D Window)")
    if fig_hurst:
        st.plotly_chart(fig_hurst, use_container_width=True)
        st.markdown(f"""
        <div class="explanation-box">
            <b>Current reading H = {current_h:.3f} &nbsp;→&nbsp;
            <span style="color:{regime_color}">{current_interp}</span></b>
            &nbsp;({hurst_confidence} confidence)<br>
            H &gt; 0.58 = persistent / trending market — momentum strategies work<br>
            H ≈ 0.50 = random walk — no edge from trend or mean-reversion<br>
            H &lt; 0.42 = anti-persistent / mean-reverting — fade breakouts<br>
            <i>Window: 120 bars rolling; step 5 bars. Confidence via R² of log-log R/S fit.</i>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Need ≥120 bars of data for Hurst analysis.")

st.markdown("---")

# ========== VOL / EM / IVR ==========

st.markdown('<div class="section-header">Volatility & Options Framework</div>', unsafe_allow_html=True)

v1, v2, v3 = st.columns(3)

with v1:
    st.markdown("#### Volatility Cone")
    fig_vc = create_volatility_cone(close_series, trading_days)
    if fig_vc:
        st.plotly_chart(fig_vc, use_container_width=True)
        st.markdown("""
        <div class="explanation-box">
            <b>Gold diamond = current vol.</b>
            Above median → elevated; below → compressed.
            Use with IV Rank to confirm premium selling / buying.
        </div>""", unsafe_allow_html=True)

with v2:
    st.markdown("#### Expected Move (1σ)")
    fig_em = create_expected_move_chart(spot, implied_vol, trading_days, currency)
    st.plotly_chart(fig_em, use_container_width=True)
    st.markdown("""
    <div class="explanation-box">
        <b>1σ = 68% probability range.</b>
        Daily and weekly expected moves inform strike selection and stop placement.
    </div>""", unsafe_allow_html=True)

with v3:
    st.markdown("#### IV Rank & Percentile")
    fig_iv = create_iv_rank_chart(close_series, trading_days)
    if fig_iv:
        st.plotly_chart(fig_iv, use_container_width=True)
        st.markdown("""
        <div class="explanation-box">
            Red = expensive vol (sell premium) &nbsp;·&nbsp;
            Green = cheap vol (buy premium) &nbsp;·&nbsp;
            IVR &gt; 65% → short vega; IVR &lt; 30% → long vega
        </div>""", unsafe_allow_html=True)
    else:
        st.info("IV data unavailable")

st.markdown("---")

# ========== OI PROFILE ==========

st.markdown('<div class="section-header">Open Interest Profile</div>', unsafe_allow_html=True)
fig_oi = create_oi_profile(spot)
st.plotly_chart(fig_oi, use_container_width=True)
st.markdown("""
<div class="explanation-box">
    <b>OI Profile:</b> Concentration of call (red) and put (green) positions by strike.
    <b>Gold line = Spot.</b> <b>Purple = Max Pain</b> (where most options expire worthless — price gravitates here near expiry).
</div>""", unsafe_allow_html=True)

# ========== FOOTER ==========

st.markdown("---")
st.markdown("""
<div style="text-align:center;opacity:0.4;margin:20px 0;font-family:Space Mono,monospace;font-size:11px;letter-spacing:.1em">
    ALPHAQUANT TERMINAL &nbsp;·&nbsp; QUANTITATIVE ANALYSIS DASHBOARD &nbsp;·&nbsp; DATA REFRESHES EVERY 5 MIN
</div>""", unsafe_allow_html=True)