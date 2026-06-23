import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import timedelta
import requests
import time
import scipy.stats as si
import concurrent.futures
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ========================= CONFIG & THEME =========================
st.set_page_config(
    page_title="AlphaQuant Master Terminal v10.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHART_THEME = {
    "template": "plotly_dark",
    "primary": "#67e8f9",     # Cyan
    "secondary": "#fbbf24",   # Amber
    "bullish": "#22c55e",     # Green
    "bearish": "#ef4444",     # Red
    "neutral": "white",
    "watermark": "gray"
}

st.markdown("""
    <style>
        .stApp { background-color: #0B0F19; color: #FFFFFF; }
        .block-container { padding-top: 2rem; padding-bottom: 5rem; }
        div[data-testid="stMetricValue"] { color: #67e8f9; font-family: 'Courier New', monospace; font-weight: bold; }
        hr { margin-top: 3em; margin-bottom: 3em; border-color: #1F2A45; }
    </style>
""", unsafe_allow_html=True)

INDIAN_ASSETS = {
    "Nifty 50 (Index)": "^NSEI",
    "Bank Nifty (Index)": "^NSEBANK",
    "Reliance Ind (Stock)": "RELIANCE.NS",
    "HDFC Bank (Stock)": "HDFCBANK.NS"
}

CRYPTO_ASSETS = {
    "Bitcoin (BTC)": "BTCUSD",
    "Ethereum (ETH)": "ETHUSD",
    "Solana (SOL)": "SOLUSD"
}

# ========================= DATA INGESTION ENGINE =========================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, period: str = "1y", interval: str = "1d", is_crypto: bool = False):
    if not is_crypto:
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if data is None or data.empty: return None
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            return data
        except Exception:
            return None
    else:
        try:
            days_map = {"5d": 5, "1mo": 30, "30d": 30, "60d": 60, "6mo": 180, "1y": 365, "730d": 730, "2y": 730, "max": 1000}
            days_back = days_map.get(period, 365)
            res_map = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
            resolution = res_map.get(interval, "1d")
            
            end_time = int(time.time())
            start_time = end_time - (days_back * 24 * 60 * 60)
            
            url = "https://api.delta.exchange/v2/history/candles"
            params = {"symbol": ticker, "resolution": resolution, "start": start_time, "end": end_time}
            
            response = requests.get(url, params=params, timeout=10)
            res_data = response.json()
            if not res_data.get('success'): return None
                
            df = pd.DataFrame(res_data['result'])
            if df.empty: return None
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']: df[col] = df[col].astype(float)
            return df.sort_index()
        except Exception:
            return None

def get_vix_data(asset_class, ticker, period="1y", is_crypto=False):
    if asset_class == "Indian Equities":
        return fetch_data("^INDIAVIX", period=period, is_crypto=False)
    else:
        data = fetch_data(ticker, period="2y", is_crypto=True) 
        if data is None: return None
        ret = np.log(data['Close'] / data['Close'].shift(1))
        synth_vix = ret.rolling(30).std() * np.sqrt(365) * 100
        vix_df = data[['Close']].copy()
        vix_df['Close'] = synth_vix
        vix_df = vix_df.dropna()
        days_map = {"5d": 5, "6mo": 180, "1y": 365}
        return vix_df.tail(days_map.get(period, 365))

def get_scalar(series):
    if isinstance(series, pd.Series):
        if series.empty: return 0.0
        val = series.iloc[-1]
    else:
        val = series
    if hasattr(val, 'item'): return float(val.item())
    return float(val)

def add_watermark(fig):
    fig.add_annotation(text="AlphaQuant Terminal v10.0", xref="paper", yref="paper", x=0.99, y=0.01, showarrow=False, font=dict(size=10, color=CHART_THEME["watermark"]), opacity=0.6)

# ========================= 0. REAL-TIME CHART & VWAP =========================
def calc_vwap(df, timeframe):
    df = df.copy()
    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        df['VWAP'] = np.nan
        return df
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['Typical_Price'] * df['Volume']
    grouper = df.index.date if timeframe in ['15m', '1h', '4h'] else df.index.to_period('M')
    df['Cum_Vol'] = df.groupby(grouper)['Volume'].cumsum()
    df['Cum_VP'] = df.groupby(grouper)['VP'].cumsum()
    df['VWAP'] = df['Cum_VP'] / df['Cum_Vol']
    return df

def render_realtime_chart(selected_name, ticker, is_crypto):
    st.subheader(f"⚡ Real-Time Price Action & Volume Profile")
    timeframe = st.radio("Select Timeframe", ["15m", "1h", "4h", "1d"], index=1, horizontal=True, label_visibility="collapsed")
    
    period, interval = ("60d", "15m") if timeframe == "15m" else ("730d", "1h") if timeframe in ["1h", "4h"] else ("2y", "1d")
        
    data = fetch_data(ticker, period=period, interval=interval, is_crypto=is_crypto)
    if data is None or data.empty: return st.warning(f"Real-time data unavailable.")
    
    if timeframe == "4h": data = data.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    data = data.tail(300 if timeframe in ["15m", "1h", "4h"] else 252)
    data = calc_vwap(data, timeframe)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price', increasing_line_color=CHART_THEME['bullish'], decreasing_line_color=CHART_THEME['bearish']), row=1, col=1)
    
    if 'VWAP' in data.columns and not data['VWAP'].isna().all():
        fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='Anchored VWAP', line=dict(color=CHART_THEME['secondary'], width=2, dash='dot')), row=1, col=1)
    if 'Volume' in data.columns and not (data['Volume'] == 0).all():
        colors = [CHART_THEME['bullish'] if row['Close'] >= row['Open'] else CHART_THEME['bearish'] for _, row in data.iterrows()]
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors, opacity=0.8), row=2, col=1)
        
    fig.update_layout(template=CHART_THEME['template'], height=600, xaxis_rangeslider_visible=False, hovermode='x unified')
    if timeframe in ['15m', '1h', '4h']: fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])]) 
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

# ========================= 1. IVR & IVP =========================
def render_volatility_metrics(asset_class, ticker, is_crypto):
    st.markdown("### 1. Implied Volatility Rank (IVR)")
    vix_name = "India VIX" if asset_class == "Indian Equities" else "Synthetic IV (30D HV)"
    data = get_vix_data(asset_class, ticker, period="1y", is_crypto=is_crypto)
    if data is None: return st.warning(f"{vix_name} data unavailable.")
    
    close = data['Close']
    current_iv, high_52w, low_52w = get_scalar(close), get_scalar(close.max()), get_scalar(close.min())
    denom = (high_52w - low_52w)
    ivr = ((current_iv - low_52w) / denom * 100) if denom != 0 else 0.0
    ivp = (close[close < current_iv].count() / len(close)) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, mode='lines', name=vix_name, line=dict(color=CHART_THEME["primary"], width=2.5)))
    fig.add_hline(y=high_52w, line_dash="dash", line_color=CHART_THEME["bearish"])
    fig.add_hline(y=low_52w, line_dash="dash", line_color=CHART_THEME["bullish"])
    fig.add_hline(y=current_iv, line_color=CHART_THEME["neutral"], line_width=3)
    fig.update_layout(template=CHART_THEME["template"], height=400, margin=dict(t=10, b=10))
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns(2)
    c1.metric("IV Rank (IVR)", f"{ivr:.1f}%")
    c2.metric("IV Percentile (IVP)", f"{ivp:.1f}%")

# ========================= 2. Expected Move =========================
def render_expected_move(selected_name, ticker, asset_class, currency, trading_days, is_crypto):
    st.markdown("### 2. Expected Move (1σ)")
    asset_data = fetch_data(ticker, period="1mo", is_crypto=is_crypto)
    vix = get_vix_data(asset_class, ticker, period="5d", is_crypto=is_crypto)
    if asset_data is None or vix is None: return st.warning("Data fetch failed for Expected Move.")

    spot = get_scalar(asset_data['Close'])
    current_vix = get_scalar(vix['Close'])
    daily_vol = (current_vix / 100) * np.sqrt(1 / trading_days)
    exp_move = spot * daily_vol

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=asset_data.index, y=asset_data['Close'], mode='lines', name=selected_name, line=dict(color=CHART_THEME["primary"], width=3)))
    tomorrow = asset_data.index[-1] + timedelta(days=1)
    fig.add_trace(go.Scatter(x=[asset_data.index[-1], tomorrow], y=[spot, spot], mode='lines', name='Spot', line=dict(color=CHART_THEME["neutral"], dash='dash')))
    fig.add_trace(go.Scatter(x=[tomorrow], y=[spot + exp_move], mode='markers', name='+1σ', marker=dict(color=CHART_THEME["bullish"], size=16, symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=[tomorrow], y=[spot - exp_move], mode='markers', name='-1σ', marker=dict(color=CHART_THEME["bearish"], size=16, symbol='triangle-down')))
    fig.update_layout(template=CHART_THEME["template"], height=400, margin=dict(t=10, b=10))
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Spot Price", f"{currency}{spot:,.2f}")
    c2.metric("Daily Implied Move", f"± {currency}{exp_move:,.1f}")

# ========================= 3. Index Divergence =========================
def render_index_divergence(div1, div2, name1, name2, currency, is_crypto):
    st.markdown(f"### 3. Systemic Divergence ({name1} vs {name2})")
    d1 = fetch_data(div1, period="1y", is_crypto=is_crypto)
    d2 = fetch_data(div2, period="1y", is_crypto=is_crypto)
    if d1 is None or d2 is None: return st.warning("Divergence data unavailable.")

    data = pd.merge(d1['Close'].to_frame(name1), d2['Close'].to_frame(name2), left_index=True, right_index=True).dropna()
    normalized = (data / data.iloc[0]) * 100
    rolling_corr = np.log(data / data.shift(1)).dropna()[name1].rolling(20).corr(np.log(data / data.shift(1)).dropna()[name2]).dropna()
    
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[name1], name=name1, line=dict(color=CHART_THEME["primary"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[name2], name=name2, line=dict(color=CHART_THEME["secondary"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, name='20D Corr', line=dict(color=CHART_THEME["neutral"])), row=2, col=1)
    fig.update_layout(template=CHART_THEME["template"], height=450, margin=dict(t=10, b=10))
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

# ========================= 4. Volatility Cone =========================
def render_volatility_cone(selected_name, ticker, trading_days, is_crypto):
    st.markdown("### 4. Volatility Term Structure (Cone)")
    data = fetch_data(ticker, period="max", is_crypto=is_crypto)
    if data is None: return st.warning("Data unavailable.")

    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    stats = [{'window': w, 'max': float((returns.rolling(w).std() * np.sqrt(trading_days) * 100).max()), 'min': float((returns.rolling(w).std() * np.sqrt(trading_days) * 100).min()), 'median': float((returns.rolling(w).std() * np.sqrt(trading_days) * 100).median()), 'current': float((returns.rolling(w).std() * np.sqrt(trading_days) * 100).iloc[-1])} for w in [10, 20, 30, 60, 90, 120, 180, trading_days]]
    df_stats = pd.DataFrame(stats)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_stats['window'], y=df_stats['max'], name='Max Vol', mode='lines+markers', line=dict(color=CHART_THEME["bearish"])))
    fig.add_trace(go.Scatter(x=df_stats['window'], y=df_stats['min'], name='Min Vol', mode='lines+markers', line=dict(color=CHART_THEME["bullish"])))
    fig.add_trace(go.Scatter(x=df_stats['window'], y=df_stats['median'], name='Median Vol', mode='lines+markers', line=dict(color=CHART_THEME["neutral"], dash='dash')))
    fig.add_trace(go.Scatter(x=df_stats['window'], y=df_stats['current'], name='Current Vol', mode='lines+markers', line=dict(color=CHART_THEME["primary"], width=4)))
    fig.update_layout(template=CHART_THEME["template"], height=400, margin=dict(t=10, b=10))
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

# ========================= 5. Volatility Risk Premium =========================
def render_vrp(selected_name, ticker, asset_class, trading_days, is_crypto):
    st.markdown("### 5. Volatility Risk Premium (VRP)")
    main_data = fetch_data(ticker, period="6mo", is_crypto=is_crypto)
    vix = get_vix_data(asset_class, ticker, period="6mo", is_crypto=is_crypto)
    if main_data is None or vix is None: return st.warning("Data unavailable.")

    hv = np.log(main_data['Close'] / main_data['Close'].shift(1)).rolling(20).std() * np.sqrt(trading_days) * 100
    df = pd.merge(vix['Close'].to_frame('VIX'), hv.to_frame('HV'), left_index=True, right_index=True).dropna()
    df['VRP'] = df['VIX'] - df['HV']

    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['VIX'], name='Implied Vol', line=dict(color=CHART_THEME["primary"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['HV'], name='Realized Vol', line=dict(color=CHART_THEME["secondary"])), row=1, col=1)
    colors = np.where(df['VRP'] > 0, CHART_THEME["bullish"], CHART_THEME["bearish"])
    fig.add_trace(go.Bar(x=df.index, y=df['VRP'], name='VRP Spread', marker_color=colors), row=2, col=1)
    fig.update_layout(template=CHART_THEME["template"], height=450, margin=dict(t=10, b=10))
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

# ========================= 6. Hurst Regime =========================
def render_hurst_regime(selected_name, ticker, is_crypto):
    st.markdown("### 6. Hurst Exponent (Regime)")
    data = fetch_data(ticker, period="1y", is_crypto=is_crypto)
    if data is None: return st.warning("Data unavailable.")

    def calculate_hurst(ts):
        if len(ts) < 20: return np.nan
        lags = range(2, 20)
        reg_val = [np.std(ts.values[lag:] - ts.values[:-lag]) for lag in lags]
        return np.polyfit(np.log(lags), np.log(reg_val), 1)[0]

    log_prices = np.log(data['Close'])
    df = pd.DataFrame({'Close': data['Close'], 'Hurst': log_prices.rolling(window=60).apply(calculate_hurst, raw=False)}).dropna()

    fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35], shared_xaxes=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color=CHART_THEME["neutral"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Hurst'], name='Hurst', line=dict(color=CHART_THEME["primary"])), row=2, col=1)
    fig.update_layout(template=CHART_THEME["template"], height=400, margin=dict(t=10, b=10))
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

# ========================= 7. Liquidity Sweeps =========================
def render_liquidity_sweep(selected_name, ticker, is_crypto):
    st.markdown("### 7. Intraday Liquidity Sweeps (15m)")
    data = fetch_data(ticker, period="30d", interval="15m", is_crypto=is_crypto)
    if data is None: return st.warning("Intraday data lookup failed.")

    df = data.copy()
    df['Prev_High'], df['Prev_Low'] = df['High'].rolling(20).max().shift(1), df['Low'].rolling(20).min().shift(1)
    df['Supply_Sweep'] = (df['High'] > df['Prev_High']) & (df['Close'] < df['Prev_High'])
    df['Demand_Sweep'] = (df['Low'] < df['Prev_Low']) & (df['Close'] > df['Prev_Low'])
    plot_df = df.tail(100).reset_index()

    fig = go.Figure(data=[go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='Candles')])
    supply_idx, demand_idx = plot_df[plot_df['Supply_Sweep']].index, plot_df[plot_df['Demand_Sweep']].index

    if not supply_idx.empty: fig.add_trace(go.Scatter(x=supply_idx, y=plot_df.loc[supply_idx, 'High']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=14, color=CHART_THEME["bearish"]), name='Supply Trap'))
    if not demand_idx.empty: fig.add_trace(go.Scatter(x=demand_idx, y=plot_df.loc[demand_idx, 'Low']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=14, color=CHART_THEME["bullish"]), name='Demand Trap'))
    
    fig.update_layout(template=CHART_THEME["template"], height=450, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10))
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

# ========================= 8. Advanced Volatility =========================
def render_advanced_volatility(selected_name, ticker, trading_days, is_crypto):
    st.markdown("### 8. Yang-Zhang Volatility (Overnight + Intraday)")
    data = fetch_data(ticker, period="1y", is_crypto=is_crypto)
    if data is None: return st.warning("Data unavailable.")

    df = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
    o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
    N = len(o)
    
    vol_o, vol_c, vol_rs = np.log(o / c.shift(1)).std()**2, np.log(c / o).std()**2, ((np.log(h/o) * np.log(h/c)) + (np.log(l/o) * np.log(l/c))).mean()
    k = 0.34 / (1.34 + (N + 1) / (N - 1)) if N > 1 else 0
    yz_vol = np.sqrt(vol_o + k * vol_c + (1 - k) * vol_rs) * np.sqrt(trading_days) * 100
    c2c_vol = np.log(c / c.shift(1)).std() * np.sqrt(trading_days) * 100
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Yang-Zhang True Volatility", f"{yz_vol:.2f}%")
    c2.metric("Basic Close-to-Close Vol", f"{c2c_vol:.2f}%")
    c3.metric("Hidden Gap Risk", f"{yz_vol - c2c_vol:+.2f}%")

# ========================= 9. Market Synthesis =========================
def render_market_synthesis(selected_name, ticker, asset_class, div1, div2, div1_name, div2_name, currency, trading_days, is_crypto):
    st.subheader(f"🧠 9. Multi-Timeframe Market Synthesis")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_vix = executor.submit(get_vix_data, asset_class, ticker, "6mo", is_crypto)
        f_daily = executor.submit(fetch_data, ticker, "1y", "1d", is_crypto)
        f_div1 = executor.submit(fetch_data, div1, "1y", "1d", is_crypto)
        f_div2 = executor.submit(fetch_data, div2, "1y", "1d", is_crypto)

        vix_data, daily_data, d1_data, d2_data = f_vix.result(), f_daily.result(), f_div1.result(), f_div2.result()

    if any(d is None for d in [vix_data, daily_data, d1_data, d2_data]): return st.warning("Synthesis data incomplete.")

    current_vix = get_scalar(vix_data['Close'])
    ivr = ((current_vix - get_scalar(vix_data['Close'].min())) / (get_scalar(vix_data['Close'].max()) - get_scalar(vix_data['Close'].min())) * 100) if (get_scalar(vix_data['Close'].max()) - get_scalar(vix_data['Close'].min())) != 0 else 0
    vrp_val = current_vix - (np.log(daily_data['Close'] / daily_data['Close'].shift(1)).rolling(20).std() * np.sqrt(trading_days) * 100).iloc[-1]
    
    data_div = pd.merge(d1_data['Close'].to_frame('A'), d2_data['Close'].to_frame('B'), left_index=True, right_index=True).dropna()
    corr_val = float(np.log(data_div / data_div.shift(1)).dropna()['A'].rolling(20).corr(np.log(data_div / data_div.shift(1)).dropna()['B']).iloc[-1])

    c1, c2, c3 = st.columns(3)
    c1.metric("IV Rank (Options Cost)", f"{ivr:.1f}%")
    c2.metric("VRP (Sellers Edge)", f"{vrp_val:+.2f}%")
    c3.metric("Systemic Correlation", f"{corr_val:.2f}")

# ========================= 10. FULL OPTION GREEKS ENGINE =========================
def bs_greeks(S, K, T_days, r_pct, sigma_pct):
    T, r, sigma = T_days / 365.0, r_pct / 100.0, sigma_pct / 100.0
    if T <= 0: T = 1e-5 
    if sigma <= 0: sigma = 1e-5
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return {
        'Call': {
            'Price': S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2), 
            'Delta': si.norm.cdf(d1), 
            'Gamma': si.norm.pdf(d1) / (S * sigma * np.sqrt(T)),
            'Theta': (-S * si.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2)) / 365,
            'Vega': S * si.norm.pdf(d1) * np.sqrt(T) / 100,
            'Rho': K * T * np.exp(-r * T) * si.norm.cdf(d2) / 100
        },
        'Put': {
            'Price': K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1), 
            'Delta': si.norm.cdf(d1) - 1, 
            'Gamma': si.norm.pdf(d1) / (S * sigma * np.sqrt(T)),
            'Theta': (-S * si.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2)) / 365,
            'Vega': S * si.norm.pdf(d1) * np.sqrt(T) / 100,
            'Rho': -K * T * np.exp(-r * T) * si.norm.cdf(-d2) / 100
        }
    }

def render_options_greeks(selected_name, ticker, asset_class, is_crypto):
    st.subheader(f"🧪 10. Advanced Options Pricing & Greeks")
    asset_data = fetch_data(ticker, period="5d", is_crypto=is_crypto)
    vix = get_vix_data(asset_class, ticker, period="5d", is_crypto=is_crypto)
    if asset_data is None: return st.warning("Spot price data unavailable.")
    
    live_spot = get_scalar(asset_data['Close'])
    live_iv = get_scalar(vix['Close']) if vix is not None else 30.0
    default_rate = 7.0 if asset_class == "Indian Equities" else 5.0

    st.markdown("### 🎛️ Simulation Parameters")
    c1, c2, c3, c4 = st.columns(4)
    with c1: spot = st.number_input("Spot Price", value=float(live_spot), step=10.0)
    with c2: strike = st.number_input("Strike Price (K)", value=float(round(live_spot/100)*100), step=100.0)
    with c3: dte = st.number_input("Days to Expiry (DTE)", value=7, min_value=0, max_value=1000)
    with c4: iv = st.number_input("Implied Volatility (%)", value=float(live_iv), step=1.0)

    greeks = bs_greeks(spot, strike, dte, default_rate, iv)
    
    st.markdown(f"### 📊 Black-Scholes Output | Strike: **{strike:,.2f}** | DTE: **{dte}**")
    
    c1, c2 = st.columns(2)
    def render_greek_card(title, data, color):
        st.markdown(f"#### <span style='color:{color}'>{title}</span>", unsafe_allow_html=True)
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Theoretical Price", f"${data['Price']:.2f}")
        rc2.metric("Delta (Direction)", f"{data['Delta']:.4f}")
        rc3.metric("Gamma (Acceleration)", f"{data['Gamma']:.4f}")
        rc4, rc5, rc6 = st.columns(3)
        rc4.metric("Theta (Time Decay)", f"{data['Theta']:.2f} / day")
        rc5.metric("Vega (Vol Sensitivity)", f"{data['Vega']:.4f}")
        rc6.metric("Rho (Rate Sensitivity)", f"{data['Rho']:.4f}")
        st.markdown("<br>", unsafe_allow_html=True)
        
    with c1: render_greek_card("CALL OPTION", greeks['Call'], CHART_THEME["bullish"])
    with c2: render_greek_card("PUT OPTION", greeks['Put'], CHART_THEME["bearish"])

    # --- DYNAMIC QUANT RISK INTERPRETATION ---
    st.markdown("### 🧠 AI Options Analytics & Risk Breakdown")
    
    pct_from_strike = ((spot - strike) / strike) * 100
    call_delta = greeks['Call']['Delta']
    put_delta = greeks['Put']['Delta']
    gamma_val = greeks['Call']['Gamma']
    theta_call = greeks['Call']['Theta']
    vega_val = greeks['Call']['Vega']
    
    if abs(pct_from_strike) < 1.0:
        moneyness_status = "At-The-Money (ATM)"
        moneyness_desc = "The spot price is sitting directly on the strike. This is the zone of maximum uncertainty, where premium value is purely extrinsic time value, and structural risk metrics fluctuate at their most aggressive rates."
    elif spot > strike:
        moneyness_status = "Call is In-The-Money (ITM) / Put is Out-of-the-Money (OTM)"
        moneyness_desc = f"The underlying spot price is trading **{abs(pct_from_strike):.2f}% above** the strike selection. The Call option possesses intrinsic value, while the Put option is purely extrinsic paper value waiting to expire worthless if conditions hold."
    else:
        moneyness_status = "Call is Out-of-the-Money (OTM) / Put is In-The-Money (ITM)"
        moneyness_desc = f"The underlying spot price is trading **{abs(pct_from_strike):.2f}% below** the strike selection. The Put option possesses intrinsic value, while the Call option relies entirely on speculative extrinsic value."

    prob_call_itm = call_delta * 100
    prob_put_itm = abs(put_delta) * 100
    delta_text = f"The **Call Delta ({call_delta:.4f})** indicates that for every 1-point gain in {selected_name}, the Call premium will theoretically gain {call_delta:.2f} points. Mechanically, the market assigns an estimated **{prob_call_itm:.1f}% theoretical probability** of this Call expiring in-the-money."

    if dte <= 7 and abs(pct_from_strike) < 2.0:
        gamma_text = f"**CRITICAL GAMMA RISK:** Gamma is highly concentrated at **{gamma_val:.6f}**. Because the option is close to expiration ({dte} DTE) and near-the-money, the Deltas will swing violently with minor price moves."
    else:
        gamma_text = f"**STABLE GAMMA PROFILE:** Gamma is measured at a mild **{gamma_val:.6f}**. Delta adjustments will remain gradual and predictable, minimizing the risk of rapid delta-flips."

    call_decay_pct = (abs(theta_call) / greeks['Call']['Price'] * 100) if greeks['Call']['Price'] > 0 else 0
    if dte <= 5:
        theta_text = f"**EXPONENTIAL TIME DECAY:** The Call option is shedding **{abs(theta_call):.2f} points per day**, roughly **{call_decay_pct:.1f}%** of its total value every 24 hours. The clock is a weapon for option sellers right now."
    else:
        theta_text = f"**LINEAR TIME DECAY:** Theta decay is functioning linearly at **{abs(theta_call):.2f} points per day**."

    implied_move_impact = vega_val * 1.0
    vega_text = f"**VOLATILITY SENSITIVITY:** Vega stands at **{vega_val:.4f}**. If structural marketplace volatility drops by a mere 1% (Vol Crush), the contract values will automatically contract by **{implied_move_impact:.2f} points**."

    st.info(f"""
    🌐 **Moneyness Matrix:** This option cluster is currently **{moneyness_status}**. {moneyness_desc}  
    🎯 **Direction & Probability (Delta):** {delta_text}  
    ⚡ **Acceleration Risk (Gamma):** {gamma_text}  
    ⏳ **The Clock (Theta):** {theta_text}  
    🌊 **Implied Risk Pricing (Vega):** {vega_text}
    """)

    # --- EXPOSURE CURVES & VOLATILITY SURFACE ---
    st.markdown("### 📈 Exposure Curves & Volatility Surface")
    col_curve, col_surf = st.columns(2)
    
    spot_range = np.linspace(spot * 0.85, spot * 1.15, 100)
    call_deltas = [bs_greeks(s, strike, dte, default_rate, iv)['Call']['Delta'] for s in spot_range]
    put_deltas = [bs_greeks(s, strike, dte, default_rate, iv)['Put']['Delta'] for s in spot_range]
    
    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(x=spot_range, y=call_deltas, name='Call Delta', line=dict(color=CHART_THEME['bullish'], width=3)))
    fig_delta.add_trace(go.Scatter(x=spot_range, y=put_deltas, name='Put Delta', line=dict(color=CHART_THEME['bearish'], width=3)))
    fig_delta.add_vline(x=spot, line_dash="dash", line_color=CHART_THEME['neutral'], annotation_text="Current Spot")
    fig_delta.add_vline(x=strike, line_dash="dot", line_color=CHART_THEME['secondary'], annotation_text="Strike Price")
    fig_delta.update_layout(title="Option Delta Exposure Curve", template=CHART_THEME['template'], height=450, hovermode='x unified', margin=dict(l=10, r=10, t=40, b=10))
    add_watermark(fig_delta)
    with col_curve:
        st.plotly_chart(fig_delta, use_container_width=True)

    strikes_grid = np.linspace(spot * 0.8, spot * 1.2, 30)
    dtes_grid = np.array([7, 14, 30, 60, 90, 120, 180, 252, 365])
    K_mesh, T_mesh = np.meshgrid(strikes_grid, dtes_grid)
    moneyness = K_mesh / spot
    simulated_iv_surface = iv + (iv * 2.0) * (moneyness - 1)**2 - (iv * 0.5) * (moneyness - 1) + (10 / np.sqrt(T_mesh))
    
    fig_surface = go.Figure(data=[go.Surface(z=simulated_iv_surface, x=K_mesh, y=T_mesh, colorscale='Inferno')])
    fig_surface.update_layout(title='3D Implied Volatility Surface (Simulated)', scene=dict(xaxis_title='Strike Price', yaxis_title='Days to Expiry', zaxis_title='Implied Volatility (%)', camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))), template=CHART_THEME['template'], height=450, margin=dict(l=10, r=10, t=40, b=10))
    add_watermark(fig_surface)
    with col_surf:
        st.plotly_chart(fig_surface, use_container_width=True)

# ========================= 11. ML ENGINE =========================
def render_ml_engine(ticker, is_crypto):
    st.subheader(f"🤖 11. Machine Learning Predictive Engine")
    df = fetch_data(ticker, period="2y", interval="1d", is_crypto=is_crypto)
    if df is None or len(df) < 50: return st.warning("Insufficient data for ML.")
    
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20D'] = df['Log_Returns'].rolling(20).std() * np.sqrt(252)
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    ml_data = df.dropna().copy()
    
    X = ml_data[['Log_Returns', 'Vol_20D']].iloc[:-1]
    y = ml_data['Target'].iloc[:-1]
    
    model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)
    
    prob = model.predict_proba(ml_data[['Log_Returns', 'Vol_20D']].iloc[[-1]])[0][1] * 100
    st.metric("Probability of Up-Day Tomorrow", f"{prob:.1f}%", "Bullish Bias" if prob > 50 else "Bearish Bias")

# ========================= MAIN DASHBOARD APP =========================
def main():
    with st.sidebar:
        st.title("🎛️ Terminal Settings")
        asset_class = st.radio("Asset Class", ["Indian Equities", "Crypto"])
        is_crypto = (asset_class == "Crypto")
        
        if not is_crypto:
            ASSET_DICT = INDIAN_ASSETS
            div1, div2, div1_name, div2_name, currency, trading_days = "^NSEI", "^NSEBANK", "Nifty 50", "Bank Nifty", "₹", 252
        else:
            ASSET_DICT = CRYPTO_ASSETS
            div1, div2, div1_name, div2_name, currency, trading_days = "BTCUSD", "ETHUSD", "Bitcoin", "Ethereum", "$", 365
            
        selected_name = st.selectbox("Target Asset", options=list(ASSET_DICT.keys()), index=0)
        ticker = ASSET_DICT[selected_name]
        
        st.divider()
        st.caption("AlphaQuant Master Terminal v10.0")
        st.caption("Mode: Single-Page View")

    st.title(f"AlphaQuant Terminal: {selected_name}")
    st.markdown("All analytical modules are actively rendering on this single unified view. Scroll down to analyze.")
    st.markdown("<hr>", unsafe_allow_html=True)

    with st.spinner("Initializing complete quantitative matrix..."):
        # ROW 1: Real-Time
        render_realtime_chart(selected_name, ticker, is_crypto)
        st.markdown("<hr>", unsafe_allow_html=True)

        # ROW 2: ML & Synthesis
        render_ml_engine(ticker, is_crypto)
        st.markdown("<br>", unsafe_allow_html=True)
        render_market_synthesis(selected_name, ticker, asset_class, div1, div2, div1_name, div2_name, currency, trading_days, is_crypto)
        st.markdown("<hr>", unsafe_allow_html=True)

        # ROW 3: Options
        render_options_greeks(selected_name, ticker, asset_class, is_crypto)
        st.markdown("<hr>", unsafe_allow_html=True)

        # ROW 4: Divergence & VRP (Full Width)
        render_index_divergence(div1, div2, div1_name, div2_name, currency, is_crypto)
        st.markdown("<br>", unsafe_allow_html=True)
        render_vrp(selected_name, ticker, asset_class, trading_days, is_crypto)
        st.markdown("<hr>", unsafe_allow_html=True)

        # ROW 5: IVR & Expected Move (Side-by-Side)
        col1, col2 = st.columns(2)
        with col1: render_volatility_metrics(asset_class, ticker, is_crypto)
        with col2: render_expected_move(selected_name, ticker, asset_class, currency, trading_days, is_crypto)
        st.markdown("<hr>", unsafe_allow_html=True)

        # ROW 6: Hurst & Vol Cone (Side-by-Side)
        col3, col4 = st.columns(2)
        with col3: render_hurst_regime(selected_name, ticker, is_crypto)
        with col4: render_volatility_cone(selected_name, ticker, trading_days, is_crypto)
        st.markdown("<hr>", unsafe_allow_html=True)

        # ROW 7: Liquidity & YZ (Full Width)
        render_liquidity_sweep(selected_name, ticker, is_crypto)
        st.markdown("<br>", unsafe_allow_html=True)
        render_advanced_volatility(selected_name, ticker, trading_days, is_crypto)

if __name__ == "__main__":
    main()