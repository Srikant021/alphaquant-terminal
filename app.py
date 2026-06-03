# AlphaQuant Terminal - Unified Dashboard
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

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
    .sweep-badge {
        display: inline-block; padding: 5px 14px; border-radius: 20px;
        font-family: 'Space Mono', monospace; font-size: 11px; font-weight: 700;
        letter-spacing: 0.08em; text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CORE MATH
# ============================================================

def hurst_exponent(price_series):
    price = np.array(price_series.dropna())
    n = len(price)
    if n < 30:
        return 0.5, "Insufficient data", "low"
    log_prices = np.log(price)
    max_lag = min(n // 4, 150)
    if max_lag < 10:
        return 0.5, "Insufficient data", "low"
    lags = list(range(10, max_lag, max(1, max_lag // 20)))
    rs_values, valid_lags = [], []
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
    log_rs   = np.log(rs_values)
    coeffs   = np.polyfit(log_lags, log_rs, 1)
    hurst    = float(np.clip(coeffs[0], 0.1, 0.9))
    residuals = log_rs - np.polyval(coeffs, log_lags)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_rs - np.mean(log_rs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    confidence = "high" if (r2 > 0.92 and len(valid_lags) >= 8) else "medium" if r2 > 0.80 else "low"
    if hurst > 0.58:
        interp = "Strong Trend (Persistent)"
    elif hurst > 0.53:
        interp = "Weak Trend (Mildly Persistent)"
    elif hurst >= 0.47:
        interp = "Random Walk"
    elif hurst >= 0.42:
        interp = "Weak Mean-Reversion"
    else:
        interp = "Strong Mean-Reversion (Anti-Persistent)"
    return hurst, interp, confidence


def bollinger_bands(close, period=20, std=2):
    sma = close.rolling(window=period).mean()
    std_dev = close.rolling(window=period).std()
    return sma + (std * std_dev), sma, sma - (std * std_dev)


def vwap(df):
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    vol = df['Volume'].replace(0, np.nan).ffill()
    cum_vol = vol.cumsum()
    cum_tp_vol = (typical * vol).cumsum()
    return cum_tp_vol / cum_vol


def compute_liquidity_sweeps(df, window=20):
    """Detect Supply/Demand liquidity sweeps."""
    df = df.copy()
    df['Prev_High'] = df['High'].rolling(window=window).max().shift(1)
    df['Prev_Low']  = df['Low'].rolling(window=window).min().shift(1)
    # Supply Sweep: wicked above structural high but closed below it
    df['Supply_Sweep'] = (df['High'] > df['Prev_High']) & (df['Close'] < df['Prev_High'])
    # Demand Sweep: wicked below structural low but closed above it
    df['Demand_Sweep'] = (df['Low'] < df['Prev_Low'])  & (df['Close'] > df['Prev_Low'])
    return df


def compute_parkinson_vol(high, low, periods=252):
    high = np.array(high.dropna())
    low  = np.array(low.dropna())
    if len(high) < 2 or len(low) < 2:
        return 0
    log_hl   = (np.log(high / low) ** 2)
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


# ============================================================
# DATA FETCHING
# ============================================================

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
        'price': last, 'change': last - prev,
        'pct': ((last - prev) / prev) * 100 if prev else 0,
        'high': float(data['High'].iloc[-1]),
        'low':  float(data['Low'].iloc[-1]),
        'volume': float(data['Volume'].iloc[-1]),
    }


# ============================================================
# CHART FUNCTIONS
# ============================================================

def create_price_chart(chart_data, ticker_name, currency, show_sweeps=True, sweep_window=20):
    """Candlestick + BB + VWAP + Volume + optional Liquidity Sweeps."""
    if chart_data is None or chart_data.empty:
        return None

    df = compute_liquidity_sweeps(chart_data, window=sweep_window) if show_sweeps else chart_data.copy()

    bb_upper, bb_mid, bb_lower = bollinger_bands(df['Close'])
    vwap_line = vwap(df)

    vol_colors = [
        'rgba(0,200,120,0.27)' if float(df['Close'].iloc[i]) >= float(df['Open'].iloc[i])
        else 'rgba(255,77,77,0.27)'
        for i in range(len(df))
    ]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=[0.78, 0.22],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='OHLC',
        increasing=dict(line=dict(color='#00c878', width=1)),
        decreasing=dict(line=dict(color='#ff4d6d', width=1)),
    ), row=1, col=1, secondary_y=False)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=bb_upper,
        line=dict(color='rgba(0,170,255,0.30)', width=1, dash='dot'),
        name='BB Upper', showlegend=False), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=bb_lower,
        line=dict(color='rgba(0,170,255,0.30)', width=1, dash='dot'),
        name='BB Lower', fill='tonexty', fillcolor='rgba(0,170,255,0.04)',
        showlegend=False), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=bb_mid,
        line=dict(color='rgba(0,170,255,0.55)', width=1),
        name='BB Mid / SMA20'), row=1, col=1, secondary_y=False)

    # VWAP
    fig.add_trace(go.Scatter(x=df.index, y=vwap_line,
        line=dict(color='#00e5ff', width=1.5, dash='dashdot'),
        name='VWAP'), row=1, col=1, secondary_y=False)

    # ── Liquidity Sweeps ─────────────────────────────────────────────────────
    if show_sweeps and 'Supply_Sweep' in df.columns:
        supply_rows = df[df['Supply_Sweep'] == True]
        demand_rows = df[df['Demand_Sweep'] == True]

        # Structural level lines (recent ones only — last 5 each to avoid clutter)
        plotted_supply_levels, plotted_demand_levels = set(), set()

        for idx, row in supply_rows.tail(5).iterrows():
            level = round(float(row['Prev_High']), 2)
            if level not in plotted_supply_levels:
                fig.add_hline(y=level,
                    line=dict(color='rgba(255,77,77,0.45)', width=1, dash='dash'),
                    row=1, col=1)
                plotted_supply_levels.add(level)

        for idx, row in demand_rows.tail(5).iterrows():
            level = round(float(row['Prev_Low']), 2)
            if level not in plotted_demand_levels:
                fig.add_hline(y=level,
                    line=dict(color='rgba(0,200,120,0.45)', width=1, dash='dash'),
                    row=1, col=1)
                plotted_demand_levels.add(level)

        # Supply sweep markers (▼ above candle high)
        if not supply_rows.empty:
            price_range = float(df['High'].max() - df['Low'].min())
            offset = price_range * 0.008
            fig.add_trace(go.Scatter(
                x=supply_rows.index,
                y=supply_rows['High'] + offset,
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='#ff4d6d',
                            line=dict(width=1, color='#ff0000')),
                name='Supply Sweep',
                hovertemplate='Supply Sweep<br>High: %{customdata:.2f}<extra></extra>',
                customdata=supply_rows['High'],
            ), row=1, col=1, secondary_y=False)

        # Demand sweep markers (▲ below candle low)
        if not demand_rows.empty:
            price_range = float(df['High'].max() - df['Low'].min())
            offset = price_range * 0.008
            fig.add_trace(go.Scatter(
                x=demand_rows.index,
                y=demand_rows['Low'] - offset,
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='#00c878',
                            line=dict(width=1, color='#00ff88')),
                name='Demand Sweep',
                hovertemplate='Demand Sweep<br>Low: %{customdata:.2f}<extra></extra>',
                customdata=demand_rows['Low'],
            ), row=1, col=1, secondary_y=False)

    # Volume (ghost bars on price panel)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'],
        marker_color=vol_colors, showlegend=False, name='Volume'),
        row=1, col=1, secondary_y=True)

    # Volume panel
    vol_ma = df['Volume'].rolling(20).mean()
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'],
        marker_color=vol_colors, name='Volume', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=vol_ma,
        line=dict(color='#ffd700', width=1.5),
        name='Vol MA20', showlegend=False), row=2, col=1)

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
        title=dict(text=f"<b>{ticker_name}</b> — Price Action",
                   font=dict(family='Space Mono, monospace', size=14, color='#0af'), x=0.01),
        height=700, xaxis_rangeslider_visible=False, hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0,
                    font=dict(size=11), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=20, t=50, b=40),
    )
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)', title_text=f"Price ({currency})",
                     row=1, col=1, secondary_y=False)
    fig.update_yaxes(showticklabels=False, title_text="", row=1, col=1, secondary_y=True)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)', title_text="Volume", row=2, col=1)
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


def create_hurst_yearly_chart(daily_1y):
    """
    Hurst on 1 year of daily data: rolling 120-day window plotted over full year.
    Also returns current H, interp, confidence.
    """
    close = daily_1y['Close'].squeeze()
    if len(close) < 120:
        return None, 0.5, "Insufficient data", "low"

    hurst_vals, dates = [], []
    for i in range(120, len(close), 5):
        h, _, _ = hurst_exponent(close.iloc[i - 120:i])
        hurst_vals.append(h)
        dates.append(close.index[i - 1])

    current_h, current_interp, confidence = hurst_exponent(close.tail(min(252, len(close))))

    if not hurst_vals:
        return None, current_h, current_interp, confidence

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.10, row_heights=[0.50, 0.50])

    fig.add_trace(go.Scatter(x=close.index, y=close.values,
        name='Price (1Y Daily)', line=dict(color='rgba(255,255,255,0.75)', width=1.5)),
        row=1, col=1)

    # Hurst line — colour-coded by regime
    colors_h = ['#00c878' if h > 0.58 else '#ff4d6d' if h < 0.42 else '#ffd700'
                for h in hurst_vals]
    for i in range(1, len(hurst_vals)):
        fig.add_trace(go.Scatter(
            x=dates[i-1:i+1], y=hurst_vals[i-1:i+1],
            mode='lines', line=dict(color=colors_h[i], width=2),
            showlegend=False, hoverinfo='skip'), row=2, col=1)

    # Zone shading
    fig.add_hrect(y0=0.58, y1=0.90, fillcolor='rgba(0,200,100,0.06)', line_width=0, row=2, col=1)
    fig.add_hrect(y0=0.42, y1=0.58, fillcolor='rgba(200,200,200,0.03)', line_width=0, row=2, col=1)
    fig.add_hrect(y0=0.10, y1=0.42, fillcolor='rgba(255,80,80,0.06)',  line_width=0, row=2, col=1)

    for y, color, label in [
        (0.58, '#00c878', 'Trending  H>0.58'),
        (0.50, 'rgba(200,200,200,0.4)', 'Random Walk'),
        (0.42, '#ff4d6d', 'Mean-Rev  H<0.42'),
    ]:
        fig.add_hline(y=y, line_dash='dash', line_color=color,
                      annotation_text=label,
                      annotation_font=dict(size=10, color=color), row=2, col=1)

    # Current marker
    fig.add_trace(go.Scatter(
        x=[dates[-1]], y=[hurst_vals[-1]], mode='markers',
        marker=dict(size=11, color='#ffd700', symbol='diamond',
                    line=dict(width=2, color='white')),
        name=f'Current H={hurst_vals[-1]:.3f}'), row=2, col=1)

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
        height=540, hovermode='x unified',
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=20, t=30, b=40),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(title_text="Hurst (120D rolling)", range=[0.2, 0.8], row=2, col=1,
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
    norm    = merged / merged.iloc[0] * 100
    log_ret = np.log(merged / merged.shift(1)).dropna()
    corr    = log_ret[name1].rolling(20).corr(log_ret[name2])

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12, row_heights=[0.6, 0.4])
    fig.add_trace(go.Scatter(x=norm.index, y=norm[name1], name=name1,
        line=dict(color='#00c878', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=norm.index, y=norm[name2], name=name2,
        line=dict(color='#ff4d6d', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=corr.index, y=corr, name='20D Rolling Corr',
        line=dict(color='#00e5ff', width=2)), row=2, col=1)
    fig.add_hline(y=0.8, line_dash="dash", line_color="#00c878",
        annotation_text="High (0.8)", row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="#ff4d6d",
        annotation_text="Low (0.5)", row=2, col=1)
    fig.add_hline(y=0.0, line_dash="solid", line_color="rgba(255,255,255,0.2)", row=2, col=1)
    fig.update_layout(template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
        height=500, hovermode='x unified', margin=dict(l=60, r=20, t=30, b=40),
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
    fig.update_yaxes(title_text="Normalised Price", row=1, col=1, gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(title_text="Correlation", range=[-1, 1], row=2, col=1, gridcolor='rgba(255,255,255,0.04)')
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
            max_vol.append(rv.max()); min_vol.append(rv.min())
            med_vol.append(rv.median()); cur_vol.append(rv.iloc[-1])
    if len(max_vol) < 3:
        return None
    w = windows[:len(max_vol)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w + w[::-1], y=max_vol + min_vol[::-1],
        fill='toself', fillcolor='rgba(0,170,255,0.07)', line=dict(color='rgba(0,0,0,0)'),
        name='Historical Range'))
    fig.add_trace(go.Scatter(x=w, y=max_vol, name='Max',
        line=dict(color='#ff4d6d', width=1.5, dash='dot'), mode='lines+markers', marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=w, y=min_vol, name='Min',
        line=dict(color='#00c878', width=1.5, dash='dot'), mode='lines+markers', marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=w, y=med_vol, name='Median',
        line=dict(color='rgba(255,255,255,0.5)', width=1.5), mode='lines+markers', marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=w, y=cur_vol, name='Current',
        line=dict(color='#ffd700', width=3), mode='lines+markers',
        marker=dict(size=9, symbol='diamond', color='#ffd700', line=dict(width=2, color='white'))))
    fig.update_layout(template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
        title=dict(text='<b>Volatility Cone</b>', font=dict(family='Space Mono, monospace', size=13, color='#0af')),
        xaxis_title='Window (Days)', yaxis_title='Annualised Volatility (%)',
        height=420, hovermode='x unified', margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
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
    current    = hv.iloc[-1]
    ivr = (current - hv.min()) / (hv.max() - hv.min()) * 100 if hv.max() != hv.min() else 50
    ivp = (hv < current).sum() / len(hv) * 100
    high_thresh = hv.quantile(0.65)
    low_thresh  = hv.quantile(0.30)
    colors = ['#ff4d6d' if v >= high_thresh else '#00c878' if v <= low_thresh else '#ffd700'
              for v in hv.values]
    fig = go.Figure()
    for i in range(1, len(hv)):
        fig.add_trace(go.Scatter(x=hv.index[i-1:i+1], y=hv.values[i-1:i+1],
            mode='lines', line=dict(color=colors[i], width=2),
            showlegend=False, hoverinfo='skip'))
    for y, color, label in [
        (hv.max(), '#ff4d6d', f"52W High: {hv.max():.1f}%"),
        (hv.min(), '#00c878', f"52W Low: {hv.min():.1f}%"),
        (current,  '#ffd700', f"Current: {current:.1f}%"),
    ]:
        fig.add_hline(y=y, line_dash="dash", line_color=color,
            annotation_text=label, annotation_font=dict(color=color, size=10))
    fig.update_layout(template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
        title=dict(text=f'<b>IV Rank {ivr:.0f}%</b>  |  IV Percentile {ivp:.0f}%',
                   font=dict(family='Space Mono, monospace', size=13, color='#0af')),
        yaxis_title='HV-20 (%)', height=420, hovermode='x unified',
        margin=dict(l=60, r=20, t=50, b=40))
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


def create_expected_move_chart(spot, implied_vol, trading_days, currency="$"):
    daily_move   = spot * (implied_vol / 100) / np.sqrt(trading_days)
    weekly_move  = daily_move * np.sqrt(5)
    monthly_move = daily_move * np.sqrt(21)
    labels = ['Daily (±1σ)', 'Weekly (±1σ)', 'Monthly (±1σ)']
    values = [daily_move, weekly_move, monthly_move]
    pcts   = [v / spot * 100 for v in values]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values,
        marker_color=['#00e5ff', '#ffd700', '#ff9500'],
        text=[f'{currency}{v:,.0f}<br>({p:.1f}%)' for v, p in zip(values, pcts)],
        textposition='outside', textfont=dict(size=11)))
    fig.update_layout(template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
        title=dict(text=f'<b>Expected Move</b>  (IV: {implied_vol:.1f}%, 1σ = 68% prob)',
                   font=dict(family='Space Mono, monospace', size=13, color='#0af')),
        yaxis_title=f'Move ({currency})', height=420, showlegend=False,
        margin=dict(l=60, r=20, t=60, b=40))
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


def create_oi_profile(spot):
    step  = 500 if spot > 10000 else 100 if spot > 2000 else 50
    base  = round(spot / step) * step
    strikes = np.arange(base - 8 * step, base + 9 * step, step)
    np.random.seed(int(spot) % 1000)
    calls = np.random.randint(10, 80, len(strikes)) * 50000
    puts  = np.random.randint(10, 80, len(strikes)) * 50000
    pain  = {s: (np.sum(np.maximum(0, s - strikes) * calls) +
                 np.sum(np.maximum(0, strikes - s) * puts)) for s in strikes}
    max_pain = min(pain, key=pain.get)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=strikes, x=calls / 1e5, orientation='h',
        name='Call OI', marker_color='rgba(255,77,109,0.75)'))
    fig.add_trace(go.Bar(y=strikes, x=-puts / 1e5, orientation='h',
        name='Put OI', marker_color='rgba(0,200,120,0.75)'))
    fig.add_hline(y=spot, line_color='#ffd700', line_dash='solid', line_width=2,
        annotation_text=f'Spot {spot:,.0f}', annotation_font=dict(color='#ffd700', size=11))
    fig.add_hline(y=max_pain, line_color='#c084fc', line_dash='dash', line_width=1.5,
        annotation_text=f'Max Pain {max_pain:,.0f}', annotation_font=dict(color='#c084fc', size=11))
    fig.add_vline(x=0, line_color='rgba(255,255,255,0.2)', line_width=1)
    fig.update_layout(template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
        title=dict(text='<b>Open Interest Profile</b>',
                   font=dict(family='Space Mono, monospace', size=13, color='#0af')),
        xaxis_title='OI (Lakhs)', yaxis_title='Strike Price',
        height=520, barmode='relative', hovermode='y unified',
        margin=dict(l=80, r=20, t=50, b=40),
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


# ============================================================
# SIDEBAR
# ============================================================

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
    st.markdown("### Liquidity Sweep Settings")
    show_sweeps   = st.toggle("Show Liquidity Sweeps", value=True)
    sweep_window  = st.slider("Sweep Detection Window", min_value=10, max_value=50, value=20, step=5)

    st.markdown("---")
    st.markdown("### Correlation Pair")
    if market == "Crypto":
        corr_pair = st.selectbox("Pair", ["Bitcoin vs Ethereum", "Bitcoin vs Dogecoin"])
        corr_map = {
            "Bitcoin vs Ethereum": ("BTC-USD", "ETH-USD", "Bitcoin", "Ethereum"),
            "Bitcoin vs Dogecoin": ("BTC-USD", "DOGE-USD", "Bitcoin", "Dogecoin"),
        }
    else:
        corr_pair = st.selectbox("Pair", ["Nifty 50 vs Bank Nifty", "Nifty 50 vs Sensex"])
        corr_map = {
            "Nifty 50 vs Bank Nifty": ("^NSEI", "^NSEBANK", "Nifty 50", "Bank Nifty"),
            "Nifty 50 vs Sensex":     ("^NSEI", "^BSESN",   "Nifty 50", "Sensex"),
        }

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ============================================================
# DATA LOAD
# ============================================================

# Always fetch 1-year daily for Hurst + metrics
hist_1y  = fetch_data(ticker, period="1y",  interval="1d")
hist_2y  = fetch_data(ticker, period="2y",  interval="1d")
live     = get_live_price(ticker)

if hist_1y is None or hist_1y.empty:
    st.error("❌ Unable to load data. Please try again.")
    st.stop()

close_1y = hist_1y['Close'].squeeze()
close_2y = hist_2y['Close'].squeeze() if hist_2y is not None else close_1y
high_1y  = hist_1y['High'].squeeze()
low_1y   = hist_1y['Low'].squeeze()

if live:
    spot = live['price']; change_pct = live['pct']; change_value = live['change']
else:
    spot = float(hist_1y['Close'].iloc[-1]); change_pct = 0; change_value = 0

iv_rank, iv_percentile = compute_iv_rank(close_1y, 20)
parkinson = compute_parkinson_vol(high_1y, low_1y, trading_days)

# Hurst on 1-year daily
hurst_result = create_hurst_yearly_chart(hist_1y)
if isinstance(hurst_result, tuple) and len(hurst_result) == 4:
    fig_hurst, current_h, current_interp, hurst_confidence = hurst_result
else:
    fig_hurst = None; current_h = 0.5; current_interp = "Insufficient data"; hurst_confidence = "low"

regime_color = '#00c878' if current_h > 0.58 else '#ff4d6d' if current_h < 0.42 else '#ffd700'

# Liquidity sweep summary on 1Y daily
liq_df = compute_liquidity_sweeps(hist_1y, window=sweep_window)
last_supply = liq_df['Supply_Sweep'].iloc[-1] if 'Supply_Sweep' in liq_df.columns else False
last_demand = liq_df['Demand_Sweep'].iloc[-1] if 'Demand_Sweep' in liq_df.columns else False
total_supply = int(liq_df['Supply_Sweep'].sum()) if 'Supply_Sweep' in liq_df.columns else 0
total_demand = int(liq_df['Demand_Sweep'].sum()) if 'Demand_Sweep' in liq_df.columns else 0

if last_supply:
    sweep_regime      = "SUPPLY SWEEP ACTIVE"
    sweep_desc        = "Failed breakout / Institutional absorption at highs"
    sweep_badge_color = "#ff4d6d"
elif last_demand:
    sweep_regime      = "DEMAND SWEEP ACTIVE"
    sweep_desc        = "Failed breakdown / Institutional absorption at lows"
    sweep_badge_color = "#00c878"
else:
    sweep_regime      = "PRICE DISCOVERY PHASE"
    sweep_desc        = "Trading inside established structural bounds"
    sweep_badge_color = "#00e5ff"


# ============================================================
# HEADER + METRICS
# ============================================================

st.title("AlphaQuant Terminal")
st.markdown(f"<span style='opacity:0.45;font-size:12px;font-family:Space Mono,monospace'>"
            f"LIVE · {selected_asset} ({ticker}) · {datetime.now().strftime('%H:%M:%S IST')}"
            f"</span>", unsafe_allow_html=True)
st.markdown("---")

st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
price_color = '#00c878' if change_pct >= 0 else '#ff4d6d'

with c1:
    st.markdown(f"""<div class="metric-box" style="border-left-color:{price_color}">
        <div style="font-size:10px;opacity:0.6;font-family:Space Mono,monospace">SPOT PRICE</div>
        <div style="font-size:20px;font-weight:600;color:{price_color};margin:3px 0">{currency}{spot:,.2f}</div>
        <div style="font-size:10px;color:{price_color}">{change_pct:+.2f}%</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""<div class="metric-box">
        <div style="font-size:10px;opacity:0.6;font-family:Space Mono,monospace">PARKINSON VOL</div>
        <div style="font-size:20px;font-weight:600;color:#0af;margin:3px 0">{parkinson:.1f}%</div>
        <div style="font-size:10px;opacity:0.6">Annualised</div>
    </div>""", unsafe_allow_html=True)

ivr_color = '#ff4d6d' if iv_rank > 65 else '#00c878' if iv_rank < 30 else '#ffd700'
with c3:
    st.markdown(f"""<div class="metric-box" style="border-left-color:{ivr_color}">
        <div style="font-size:10px;opacity:0.6;font-family:Space Mono,monospace">IV RANK</div>
        <div style="font-size:20px;font-weight:600;color:{ivr_color};margin:3px 0">{iv_rank:.0f}%</div>
        <div style="font-size:10px;opacity:0.6">{'Sell' if iv_rank>65 else 'Buy' if iv_rank<30 else 'Neutral'} premium</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""<div class="metric-box">
        <div style="font-size:10px;opacity:0.6;font-family:Space Mono,monospace">IV PERCENTILE</div>
        <div style="font-size:20px;font-weight:600;color:#c084fc;margin:3px 0">{iv_percentile:.0f}%</div>
        <div style="font-size:10px;opacity:0.6">Historical ctx</div>
    </div>""", unsafe_allow_html=True)

conf_icon = '●' if hurst_confidence == 'high' else '◑' if hurst_confidence == 'medium' else '○'
with c5:
    st.markdown(f"""<div class="metric-box" style="border-left-color:{regime_color}">
        <div style="font-size:10px;opacity:0.6;font-family:Space Mono,monospace">HURST (1Y)</div>
        <div style="font-size:20px;font-weight:600;color:{regime_color};margin:3px 0">{current_h:.3f}</div>
        <div style="font-size:10px;color:{regime_color}">{conf_icon} {hurst_confidence}</div>
    </div>""", unsafe_allow_html=True)

with c6:
    st.markdown(f"""<div class="metric-box" style="border-left-color:{regime_color}">
        <div style="font-size:10px;opacity:0.6;font-family:Space Mono,monospace">MARKET REGIME</div>
        <div style="font-size:13px;font-weight:600;color:{regime_color};margin:3px 0;line-height:1.3">{current_interp}</div>
        <div style="font-size:10px;opacity:0.6">R/S (120D window)</div>
    </div>""", unsafe_allow_html=True)

with c7:
    st.markdown(f"""<div class="metric-box" style="border-left-color:{sweep_badge_color}">
        <div style="font-size:10px;opacity:0.6;font-family:Space Mono,monospace">LIQUIDITY</div>
        <div style="font-size:11px;font-weight:700;color:{sweep_badge_color};margin:3px 0;line-height:1.4">{sweep_regime}</div>
        <div style="font-size:10px;opacity:0.6">{total_supply}↓ supply / {total_demand}↑ demand (1Y)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# MULTI-TIMEFRAME PRICE CHART
# ============================================================

st.markdown('<div class="section-header">Price Action — Multi-Timeframe</div>', unsafe_allow_html=True)

TF_MAP = {
    "15m  (5D)":  ("5d",  "15m"),
    "1h   (1M)":  ("1mo", "1h"),
    "4h   (3M)":  ("3mo", "60m"),   # yfinance uses 60m for 4h equivalent
    "1D   (1Y)":  ("1y",  "1d"),
    "1W   (5Y)":  ("5y",  "1wk"),
}

tab_labels = list(TF_MAP.keys())
tabs = st.tabs(tab_labels)

for tab, label in zip(tabs, tab_labels):
    with tab:
        period_tf, interval_tf = TF_MAP[label]
        tf_data = fetch_data(ticker, period=period_tf, interval=interval_tf)
        if tf_data is not None and not tf_data.empty:
            fig_price = create_price_chart(
                tf_data, selected_asset, currency,
                show_sweeps=show_sweeps, sweep_window=sweep_window
            )
            if fig_price:
                st.plotly_chart(fig_price, use_container_width=True)
                sweep_note = " · **▼ Red triangle** = Supply Sweep (failed breakout) · **▲ Green triangle** = Demand Sweep (failed breakdown) · Dashed lines = structural levels" if show_sweeps else ""
                st.markdown(f"""<div class="explanation-box">
                    <b>Chart layers:</b> Candlestick · Bollinger Bands (20, 2σ) · VWAP · Volume + 20-bar MA{sweep_note}
                </div>""", unsafe_allow_html=True)
        else:
            st.info(f"No data available for {label} timeframe.")

st.markdown("---")

# ============================================================
# LIQUIDITY SWEEP SECTION
# ============================================================

st.markdown('<div class="section-header">Liquidity Sweep Analysis (1Y Daily)</div>', unsafe_allow_html=True)

lsw_col1, lsw_col2, lsw_col3 = st.columns([1, 1, 2])

with lsw_col1:
    st.markdown(f"""
    <div class="metric-box" style="border-left-color:{sweep_badge_color};padding:20px">
        <div style="font-size:11px;opacity:0.6;font-family:Space Mono,monospace;margin-bottom:8px">CURRENT MICROSTRUCTURE</div>
        <div style="font-size:15px;font-weight:700;color:{sweep_badge_color};margin-bottom:6px">{sweep_regime}</div>
        <div style="font-size:11px;color:rgba(255,255,255,0.65);line-height:1.5">{sweep_desc}</div>
    </div>""", unsafe_allow_html=True)

with lsw_col2:
    recent_supply = liq_df[liq_df['Supply_Sweep']].tail(3)
    recent_demand = liq_df[liq_df['Demand_Sweep']].tail(3)
    supply_levels = [f"{float(r['Prev_High']):.2f}" for _, r in recent_supply.iterrows()]
    demand_levels = [f"{float(r['Prev_Low']):.2f}" for _, r in recent_demand.iterrows()]

    st.markdown(f"""
    <div class="metric-box" style="padding:20px">
        <div style="font-size:11px;opacity:0.6;font-family:Space Mono,monospace;margin-bottom:8px">RECENT SWEPT LEVELS</div>
        <div style="font-size:11px;color:#ff4d6d;margin-bottom:4px">
            <b>Supply swept:</b><br>{'  ·  '.join(supply_levels) if supply_levels else 'None in window'}
        </div>
        <div style="font-size:11px;color:#00c878;margin-top:8px">
            <b>Demand swept:</b><br>{'  ·  '.join(demand_levels) if demand_levels else 'None in window'}
        </div>
        <div style="font-size:10px;opacity:0.5;margin-top:8px">Detection window: {sweep_window} bars</div>
    </div>""", unsafe_allow_html=True)

with lsw_col3:
    # Liquidity sweep count over time (bar chart)
    liq_monthly = liq_df[['Supply_Sweep', 'Demand_Sweep']].resample('ME').sum()
    if not liq_monthly.empty:
        fig_liq = go.Figure()
        fig_liq.add_trace(go.Bar(
            x=liq_monthly.index, y=liq_monthly['Supply_Sweep'],
            name='Supply Sweeps', marker_color='rgba(255,77,109,0.75)'))
        fig_liq.add_trace(go.Bar(
            x=liq_monthly.index, y=liq_monthly['Demand_Sweep'],
            name='Demand Sweeps', marker_color='rgba(0,200,120,0.75)'))
        fig_liq.update_layout(
            template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
            title=dict(text='<b>Monthly Sweep Frequency (1Y)</b>',
                       font=dict(family='Space Mono, monospace', size=12, color='#0af')),
            height=220, barmode='group', hovermode='x unified',
            margin=dict(l=40, r=20, t=40, b=30),
            legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)', orientation='h'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
        )
        st.plotly_chart(fig_liq, use_container_width=True)

st.markdown("""<div class="explanation-box">
    <b>Liquidity Sweep Logic:</b>
    A <span style="color:#ff4d6d"><b>Supply Sweep</b></span> occurs when price wicks above the rolling structural high but closes back below it
    — indicating institutional absorption / failed breakout (bearish).
    A <span style="color:#00c878"><b>Demand Sweep</b></span> occurs when price wicks below the rolling structural low but closes back above it
    — indicating smart-money accumulation / failed breakdown (bullish).
    These are the same signals used in ICT/SMC methodology.
</div>""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# HURST EXPONENT (1-YEAR DAILY)
# ============================================================

st.markdown('<div class="section-header">Market Regime — Hurst Exponent (1 Year Daily)</div>', unsafe_allow_html=True)

if fig_hurst:
    st.plotly_chart(fig_hurst, use_container_width=True)
    st.markdown(f"""<div class="explanation-box">
        <b>Current H = {current_h:.3f} → <span style="color:{regime_color}">{current_interp}</span></b>
        &nbsp;({hurst_confidence} confidence) &nbsp;·&nbsp; Computed on 1Y of daily closes, rolling 120-bar window<br>
        H &gt; 0.58 = persistent / trending — momentum strategies have statistical edge<br>
        H ≈ 0.50 = random walk — no directional edge<br>
        H &lt; 0.42 = anti-persistent / mean-reverting — fade breakouts, sell moves into resistance<br>
        <i>Confidence = R² of log-log R/S regression. Use High/Medium signals only.</i>
    </div>""", unsafe_allow_html=True)
else:
    st.info("Need ≥120 bars of daily data for Hurst analysis.")

st.markdown("---")

# ============================================================
# CORRELATION + ADVANCED ANALYSIS
# ============================================================

st.markdown('<div class="section-header">Advanced Analysis</div>', unsafe_allow_html=True)

col_l, col_r = st.columns(2)
with col_l:
    st.markdown("#### Correlation Analysis")
    t1, t2, n1, n2 = corr_map[corr_pair]
    fig_corr = create_correlation_chart(t1, t2, n1, n2)
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("""<div class="explanation-box">
            <b>Rolling 20D Correlation:</b> &gt;0.8 = in sync · &lt;0.5 = diverging · Negative = inverse
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Correlation data unavailable")

with col_r:
    st.markdown("#### IV Rank & Percentile")
    fig_iv = create_iv_rank_chart(close_1y, trading_days)
    if fig_iv:
        st.plotly_chart(fig_iv, use_container_width=True)
        st.markdown("""<div class="explanation-box">
            Red = expensive vol (sell premium) · Green = cheap vol (buy premium) ·
            IVR &gt; 65% → short vega; IVR &lt; 30% → long vega
        </div>""", unsafe_allow_html=True)
    else:
        st.info("IV data unavailable")

st.markdown("---")

# ============================================================
# VOL CONE + EXPECTED MOVE
# ============================================================

st.markdown('<div class="section-header">Volatility & Options Framework</div>', unsafe_allow_html=True)

v1, v2 = st.columns(2)
with v1:
    st.markdown("#### Volatility Cone")
    fig_vc = create_volatility_cone(close_1y, trading_days)
    if fig_vc:
        st.plotly_chart(fig_vc, use_container_width=True)
        st.markdown("""<div class="explanation-box">
            <b>Gold diamond = current vol.</b> Above median → elevated; below → compressed.
        </div>""", unsafe_allow_html=True)

with v2:
    st.markdown("#### Expected Move (1σ, 68% probability)")
    fig_em = create_expected_move_chart(spot, implied_vol, trading_days, currency)
    st.plotly_chart(fig_em, use_container_width=True)
    st.markdown("""<div class="explanation-box">
        <b>1σ range</b> covers ~68% of expected outcomes. Daily/weekly ranges help set stops and option strikes.
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# OI PROFILE
# ============================================================

st.markdown('<div class="section-header">Open Interest Profile</div>', unsafe_allow_html=True)
fig_oi = create_oi_profile(spot)
st.plotly_chart(fig_oi, use_container_width=True)
st.markdown("""<div class="explanation-box">
    <b>OI Profile:</b> Call (red) vs Put (green) open interest by strike.
    <b>Gold = Spot · Purple = Max Pain</b> — price gravitates toward max pain near expiry.
</div>""", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""<div style="text-align:center;opacity:0.35;margin:16px 0;
    font-family:Space Mono,monospace;font-size:10px;letter-spacing:.12em">
    ALPHAQUANT TERMINAL &nbsp;·&nbsp; QUANTITATIVE ANALYSIS &nbsp;·&nbsp; DATA REFRESHES EVERY 5 MIN
</div>""", unsafe_allow_html=True)