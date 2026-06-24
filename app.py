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

# ========================= ALADDIN TERMINAL THEME =========================
st.set_page_config(page_title="Aladdin-Quant Terminal", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
        /* Aladdin-Style Container Styling */
        .stApp { background-color: #05070a; color: #D1D5DB; font-family: 'Inter', sans-serif; }
        .module-card { 
            background-color: #0f131a; 
            border: 1px solid #1e293b; 
            padding: 15px; 
            border-radius: 4px; 
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        h1, h2, h3 { color: #F8FAFC !important; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
        .metric-label { color: #94a3b8; font-size: 11px; text-transform: uppercase; }
        .metric-value { color: #E2E8F0; font-size: 18px; font-weight: 600; font-family: 'Courier New', monospace; }
        div[data-testid="stMetricValue"] { color: #67e8f9; font-family: 'Courier New', monospace; font-weight: bold; }
        hr { margin-top: 2em; margin-bottom: 2em; border-color: #1e293b; }
        
        /* Streamlit native container border override to match Aladdin */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #1e293b !important;
            background-color: #0f131a !important;
            border-radius: 4px !important;
        }
        
        .news-headline { font-size: 13px; margin-bottom: 8px; border-left: 3px solid; padding-left: 8px;}
        .news-publisher { font-size: 10px; color: #64748b; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

CHART_THEME = {
    "template": "plotly_dark",
    "primary": "#67e8f9",     # Cyan
    "secondary": "#fbbf24",   # Amber
    "bullish": "#22c55e",     # Green
    "bearish": "#ef4444",     # Red
    "neutral": "white",
    "watermark": "#334155"    # Slate
}

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

# ========================= GLOBAL HELPERS =========================
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_scalar(series):
    if isinstance(series, pd.Series):
        if series.empty: return 0.0
        val = series.iloc[-1]
    else:
        val = series
    if hasattr(val, 'item'): return float(val.item())
    return float(val)

def add_watermark(fig):
    fig.add_annotation(text="ALADDIN QUANT TERMINAL v12.0", xref="paper", yref="paper", x=0.99, y=0.01, showarrow=False, font=dict(size=10, color=CHART_THEME["watermark"]), opacity=0.6)

def aladdin_metric(label, value, delta=None, invert_color=False):
    if delta:
        if invert_color:
            color = "#ef4444" if "+" in delta else "#22c55e" # Red for positive (e.g. DXY up is bad)
        else:
            color = "#22c55e" if "+" in delta else "#ef4444" # Green for positive
    else:
        color = "white"
        
    st.markdown(f"""
        <div class="module-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value} {f'<span style="color:{color}">({delta})</span>' if delta else ''}</div>
        </div>
    """, unsafe_allow_html=True)

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


# ========================= 0.1 GLOBAL MACRO PULSE =========================
def render_macro_pulse():
    st.subheader("GLOBAL MACRO PULSE & SYSTEMIC LIQUIDITY")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_dxy = executor.submit(fetch_data, "DX-Y.NYB", "5d", "1d", False)
        f_tnx = executor.submit(fetch_data, "^TNX", "5d", "1d", False)
        f_hyg = executor.submit(fetch_data, "HYG", "5d", "1d", False)
        
        dxy_data = f_dxy.result()
        tnx_data = f_tnx.result()
        hyg_data = f_hyg.result()
        
    def get_metrics(df):
        if df is None or len(df) < 2: return 0.0, 0.0
        cur = get_scalar(df['Close'].iloc[-1])
        prev = get_scalar(df['Close'].iloc[-2])
        pct = ((cur - prev) / prev) * 100
        return cur, pct
        
    dxy_cur, dxy_pct = get_metrics(dxy_data)
    tnx_cur, tnx_pct = get_metrics(tnx_data)
    hyg_cur, hyg_pct = get_metrics(hyg_data)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        aladdin_metric("US DOLLAR INDEX (DXY) [Risk-Off Indicator]", f"{dxy_cur:.2f}", f"{dxy_pct:+.2f}%", invert_color=True)
    with c2:
        aladdin_metric("10-YEAR TREASURY YIELD (^TNX)", f"{tnx_cur:.2f}%", f"{tnx_pct:+.2f}%", invert_color=True)
    with c3:
        aladdin_metric("HIGH YIELD CREDIT (HYG) [Risk-On Indicator]", f"${hyg_cur:.2f}", f"{hyg_pct:+.2f}%", invert_color=False)

# ========================= 0.2 NLP NEWS SENTIMENT ENGINE =========================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_news_sentiment(ticker, is_crypto):
    # Map Delta crypto tickers to Yahoo Finance tickers for news
    y_ticker = ticker
    if is_crypto:
        y_ticker = ticker.replace("USD", "-USD")
        
    try:
        tkr = yf.Ticker(y_ticker)
        news = tkr.news
    except:
        news = []
        
    if not news:
        return None, []
        
    # Institutional Lexicon Matrix
    pos_words = ['surge', 'jump', 'gain', 'rally', 'bull', 'upgrade', 'high', 'soar', 'outperform', 'beat', 'growth', 'positive', 'strong', 'buy', 'record', 'soars', 'boom', 'adopt', 'adoption', 'approve', 'etf', 'profit', 'launch']
    neg_words = ['crash', 'fall', 'drop', 'bear', 'downgrade', 'low', 'hack', 'probe', 'inflation', 'rate', 'hike', 'weak', 'sell', 'miss', 'negative', 'plunge', 'loss', 'lawsuit', 'sec', 'sue', 'fined', 'fraud', 'bankrupt', 'ban', 'illegal', 'investigate', 'plummets', 'slips', 'debt']
    
    score = 0
    analyzed_headlines = []
    
    for item in news[:6]:
        title = item.get('title', '').lower()
        if not title: continue
        
        p_count = sum([1 for w in pos_words if w in title])
        n_count = sum([1 for w in neg_words if w in title])
        
        net = p_count - n_count
        score += net
        
        if net > 0: 
            tag, color = "BULLISH", CHART_THEME['bullish']
        elif net < 0: 
            tag, color = "BEARISH", CHART_THEME['bearish']
        else: 
            tag, color = "NEUTRAL", CHART_THEME['secondary']
            
        analyzed_headlines.append({
            'title': item.get('title', ''),
            'publisher': item.get('publisher', 'WIRE'),
            'tag': tag,
            'color': color
        })
        
    # Normalize score between -100 and 100
    max_possible = len(analyzed_headlines) * 2
    normalized_score = (score / max_possible) * 100 if max_possible > 0 else 0
    normalized_score = max(-100, min(100, normalized_score))
    
    return normalized_score, analyzed_headlines

def render_nlp_sentiment(ticker, is_crypto):
    st.subheader("NLP NEWS SENTIMENT ENGINE")
    
    score, headlines = fetch_news_sentiment(ticker, is_crypto)
    
    if score is None:
        st.warning("No recent news context found.")
        return
        
    gauge_color = CHART_THEME['bullish'] if score > 15 else (CHART_THEME['bearish'] if score < -15 else CHART_THEME['neutral'])
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = score, domain = {'x': [0, 1], 'y': [0, 1]}, title = {'text': "Live Media Sentiment"},
        gauge = {
            'axis': {'range': [-100, 100]}, 'bar': {'color': gauge_color},
            'steps': [
                {'range': [-100, -20], 'color': "rgba(239, 68, 68, 0.2)"},
                {'range': [-20, 20], 'color': "rgba(255, 255, 255, 0.1)"},
                {'range': [20, 100], 'color': "rgba(34, 197, 94, 0.2)"}],
        }
    ))
    fig_gauge.update_layout(template=CHART_THEME["template"], height=220, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("<div style='margin-bottom: 10px; color: #94a3b8; font-size: 11px; text-transform: uppercase;'>REAL-TIME HEADLINES SCANNED</div>", unsafe_allow_html=True)
    
    for h in headlines:
        st.markdown(f"""
            <div class="news-headline" style="border-color: {h['color']};">
                {h['title']} <br>
                <span class="news-publisher">[{h['tag']}] • {h['publisher']}</span>
            </div>
        """, unsafe_allow_html=True)


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
    st.subheader(f"MARKET PRICE & VOLUME")
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
        
    fig.update_layout(template=CHART_THEME['template'], height=500, xaxis_rangeslider_visible=False, hovermode='x unified', margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    if timeframe in ['15m', '1h', '4h']: fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])]) 
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    last_24h_data = data.loc[data.index >= data.index.max() - pd.Timedelta(days=1)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close", f"{get_scalar(data['Close']):,.2f}")
    c2.metric("24h High", f"{last_24h_data['High'].max():,.2f}")
    c3.metric("24h Low", f"{last_24h_data['Low'].min():,.2f}")
    if not data['VWAP'].isna().all():
        c4.metric("Current VWAP", f"{get_scalar(data['VWAP']):,.2f}")

# ========================= 1. IVR & IVP =========================
def render_volatility_metrics(asset_class, ticker, is_crypto):
    st.subheader("1. IMPLIED VOLATILITY RANK")
    vix_name = "India VIX" if asset_class == "Indian Equities" else "Synthetic IV (30D HV)"
    data = get_vix_data(asset_class, ticker, period="1y", is_crypto=is_crypto)
    if data is None: return st.warning(f"{vix_name} data unavailable.")
    
    close = data['Close']
    current_iv, high_52w, low_52w = get_scalar(close), get_scalar(close.max()), get_scalar(close.min())
    denom = (high_52w - low_52w)
    ivr = ((current_iv - low_52w) / denom * 100) if denom != 0 else 0.0
    ivp = (close[close < current_iv].count() / len(close)) * 100
    regime = "HIGH VOLATILITY" if ivr > 50 else "LOW VOLATILITY"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, mode='lines', name=vix_name, line=dict(color=CHART_THEME["primary"], width=2.5)))
    fig.add_hline(y=high_52w, line_dash="dash", line_color=CHART_THEME["bearish"], annotation_text=f"52W High: {high_52w:.1f}")
    fig.add_hline(y=low_52w, line_dash="dash", line_color=CHART_THEME["bullish"], annotation_text=f"52W Low: {low_52w:.1f}")
    fig.add_hline(y=current_iv, line_color=CHART_THEME["neutral"], line_width=3, annotation_text=f"Current: {current_iv:.1f}")
    fig.update_layout(template=CHART_THEME["template"], height=300, margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("IV Rank", f"{ivr:.1f}%")
    c2.metric("IV Percentile", f"{ivp:.1f}%")
    c3.metric("Regime", regime)

# ========================= 2. Expected Move =========================
def render_expected_move(selected_name, ticker, asset_class, currency, trading_days, is_crypto):
    st.subheader("2. EXPECTED MOVE (1σ)")
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
    fig.update_layout(template=CHART_THEME["template"], height=300, margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Spot Price", f"{currency}{spot:,.2f}")
    c2.metric("Vol Benchmark", f"{current_vix:.2f}%")
    c3.metric("Implied Move", f"± {currency}{exp_move:,.1f}")

# ========================= 3. Index Divergence =========================
def render_index_divergence(div1, div2, name1, name2, currency, is_crypto):
    st.subheader(f"3. SYSTEMIC DIVERGENCE ({name1} vs {name2})")
    d1 = fetch_data(div1, period="1y", is_crypto=is_crypto)
    d2 = fetch_data(div2, period="1y", is_crypto=is_crypto)
    if d1 is None or d2 is None: return st.warning("Divergence data unavailable.")

    data = pd.merge(d1['Close'].to_frame(name1), d2['Close'].to_frame(name2), left_index=True, right_index=True).dropna()
    normalized = (data / data.iloc[0]) * 100
    rolling_corr = np.log(data / data.shift(1)).dropna()[name1].rolling(20).corr(np.log(data / data.shift(1)).dropna()[name2]).dropna()
    current_corr = float(rolling_corr.iloc[-1])
    regime = "HIGH CORRELATION" if current_corr > 0.80 else ("SEVERE DIVERGENCE" if current_corr < 0.50 else "MODERATE DIVERGENCE")

    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[name1], name=name1, line=dict(color=CHART_THEME["primary"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[name2], name=name2, line=dict(color=CHART_THEME["secondary"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, name='20D Corr', line=dict(color=CHART_THEME["neutral"])), row=2, col=1)
    fig.update_layout(template=CHART_THEME["template"], height=350, margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric(name1, f"{currency}{float(data[name1].iloc[-1]):,.2f}")
    c2.metric(name2, f"{currency}{float(data[name2].iloc[-1]):,.2f}")
    c3.metric("20D Correlation", f"{current_corr:.3f}", delta=regime, delta_color="inverse" if current_corr < 0.5 else "normal")

# ========================= 4. Volatility Cone =========================
def render_volatility_cone(selected_name, ticker, trading_days, is_crypto):
    st.subheader("4. VOLATILITY TERM STRUCTURE (CONE)")
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
    fig.update_layout(template=CHART_THEME["template"], height=300, margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

# ========================= 5. Volatility Risk Premium =========================
def render_vrp(selected_name, ticker, asset_class, trading_days, is_crypto):
    st.subheader("5. VOLATILITY RISK PREMIUM (VRP)")
    main_data = fetch_data(ticker, period="6mo", is_crypto=is_crypto)
    vix = get_vix_data(asset_class, ticker, period="6mo", is_crypto=is_crypto)
    if main_data is None or vix is None: return st.warning("Data unavailable.")

    hv = np.log(main_data['Close'] / main_data['Close'].shift(1)).rolling(20).std() * np.sqrt(trading_days) * 100
    df = pd.merge(vix['Close'].to_frame('VIX'), hv.to_frame('HV'), left_index=True, right_index=True).dropna()
    df['VRP'] = df['VIX'] - df['HV']
    current_vrp = float(df['VRP'].iloc[-1])
    regime = "POSITIVE VRP (Sellers Edge)" if current_vrp > 0 else "NEGATIVE VRP (Buyers Edge)"

    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['VIX'], name='Implied Vol', line=dict(color=CHART_THEME["primary"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['HV'], name='Realized Vol', line=dict(color=CHART_THEME["secondary"])), row=1, col=1)
    colors = np.where(df['VRP'] > 0, CHART_THEME["bullish"], CHART_THEME["bearish"])
    fig.add_trace(go.Bar(x=df.index, y=df['VRP'], name='VRP Spread', marker_color=colors), row=2, col=1)
    fig.update_layout(template=CHART_THEME["template"], height=350, margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Vol Benchmark", f"{float(df['VIX'].iloc[-1]):.2f}")
    c2.metric("HV (20D)", f"{float(df['HV'].iloc[-1]):.2f}")
    c3.metric("Current VRP", f"{current_vrp:+.2f}%", regime, delta_color="normal")

# ========================= 6. Hurst Regime =========================
def render_hurst_regime(selected_name, ticker, is_crypto):
    st.subheader("6. HURST EXPONENT (REGIME)")
    data = fetch_data(ticker, period="1y", is_crypto=is_crypto)
    if data is None: return st.warning("Data unavailable.")

    def calculate_hurst(ts):
        if len(ts) < 20: return np.nan
        lags = range(2, 20)
        reg_val = [np.std(ts.values[lag:] - ts.values[:-lag]) for lag in lags]
        return np.polyfit(np.log(lags), np.log(reg_val), 1)[0]

    log_prices = np.log(data['Close'])
    hurst_series = log_prices.rolling(window=60).apply(calculate_hurst, raw=False)
    df = pd.DataFrame({'Close': data['Close'], 'Hurst': hurst_series}).dropna()
    current_hurst = float(df['Hurst'].iloc[-1])
    regime = "MEAN REVERTING" if current_hurst < 0.45 else ("TRENDING" if current_hurst > 0.55 else "RANDOM WALK")

    fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35], shared_xaxes=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color=CHART_THEME["neutral"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Hurst'], name='Hurst', line=dict(color=CHART_THEME["primary"])), row=2, col=1)
    fig.update_layout(template=CHART_THEME["template"], height=300, margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Hurst Value", f"{current_hurst:.3f}")
    c2.metric("Regime", regime)

# ========================= 7. Liquidity Sweeps =========================
def render_liquidity_sweep(selected_name, ticker, is_crypto):
    st.subheader("7. INTRADAY LIQUIDITY SWEEPS (15M)")
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
    
    fig.update_layout(template=CHART_THEME["template"], height=300, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    last_regime = "SUPPLY SWEEP (Resistance)" if df['Supply_Sweep'].iloc[-1] else ("DEMAND SWEEP (Support)" if df['Demand_Sweep'].iloc[-1] else "PRICE DISCOVERY")
    st.metric("Latest Micro-Structure Regime", last_regime)

# ========================= 8. Advanced Volatility =========================
def render_advanced_volatility(selected_name, ticker, trading_days, is_crypto):
    st.subheader("8. YANG-ZHANG VOLATILITY")
    data = fetch_data(ticker, period="1y", is_crypto=is_crypto)
    if data is None: return st.warning("Data unavailable.")

    df = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
    o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
    N = len(o)
    
    vol_o, vol_c, vol_rs = np.log(o / c.shift(1)).std()**2, np.log(c / o).std()**2, ((np.log(h/o) * np.log(h/c)) + (np.log(l/o) * np.log(l/c))).mean()
    k = 0.34 / (1.34 + (N + 1) / (N - 1)) if N > 1 else 0
    yz_vol = np.sqrt(vol_o + k * vol_c + (1 - k) * vol_rs) * np.sqrt(trading_days) * 100
    c2c_vol = np.log(c / c.shift(1)).std() * np.sqrt(trading_days) * 100
    
    st.caption("Captures overnight gap risk & intraday trend mathematically.")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Yang-Zhang True Volatility", f"{yz_vol:.2f}%")
    c2.metric("Close-to-Close Vol", f"{c2c_vol:.2f}%")
    c3.metric("Hidden Gap Risk", f"{yz_vol - c2c_vol:+.2f}%", delta_color="inverse")

# ========================= 9. Market Synthesis =========================
def render_market_synthesis(selected_name, ticker, asset_class, div1, div2, div1_name, div2_name, currency, trading_days, is_crypto):
    st.subheader(f"9. MULTI-TIMEFRAME MARKET SYNTHESIS")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_vix = executor.submit(get_vix_data, asset_class, ticker, "6mo", is_crypto)
        f_daily = executor.submit(fetch_data, ticker, "1y", "1d", is_crypto)
        f_intra = executor.submit(fetch_data, ticker, "30d", "15m", is_crypto)
        f_div1 = executor.submit(fetch_data, div1, "1y", "1d", is_crypto)
        f_div2 = executor.submit(fetch_data, div2, "1y", "1d", is_crypto)

        vix_data, daily_data, intra_data, d1_data, d2_data = f_vix.result(), f_daily.result(), f_intra.result(), f_div1.result(), f_div2.result()

    if any(d is None for d in [vix_data, daily_data, intra_data, d1_data, d2_data]): return st.warning("Synthesis data incomplete.")

    current_vix = get_scalar(vix_data['Close'])
    ivr = ((current_vix - get_scalar(vix_data['Close'].min())) / (get_scalar(vix_data['Close'].max()) - get_scalar(vix_data['Close'].min())) * 100) if (get_scalar(vix_data['Close'].max()) - get_scalar(vix_data['Close'].min())) != 0 else 0
    vrp_val = current_vix - (np.log(daily_data['Close'] / daily_data['Close'].shift(1)).rolling(20).std() * np.sqrt(trading_days) * 100).iloc[-1]
    
    data_div = pd.merge(d1_data['Close'].to_frame('A'), d2_data['Close'].to_frame('B'), left_index=True, right_index=True).dropna()
    corr_val = float(np.log(data_div / data_div.shift(1)).dropna()['A'].rolling(20).corr(np.log(data_div / data_div.shift(1)).dropna()['B']).iloc[-1])

    log_prices = np.log(daily_data['Close'])
    def calc_h(ts):
        if len(ts) < 20: return np.nan
        lags = range(2, 20)
        return np.polyfit(np.log(lags), np.log([np.std(ts.values[lag:] - ts.values[:-lag]) for lag in lags]), 1)[0]
    hurst_series = log_prices.rolling(window=60).apply(calc_h, raw=False).dropna()
    hurst_val = float(hurst_series.iloc[-1]) if not hurst_series.empty else 0.5
    hurst_regime = "MEAN REVERTING" if hurst_val < 0.45 else ("TRENDING" if hurst_val > 0.55 else "RANDOM WALK")

    df_liq = intra_data.copy()
    df_liq['Prev_High'] = df_liq['High'].rolling(20).max().shift(1)
    df_liq['Prev_Low'] = df_liq['Low'].rolling(20).min().shift(1)
    df_liq['Supply_Sweep'] = (df_liq['High'] > df_liq['Prev_High']) & (df_liq['Close'] < df_liq['Prev_High'])
    df_liq['Demand_Sweep'] = (df_liq['Low'] < df_liq['Prev_Low']) & (df_liq['Close'] > df_liq['Prev_Low'])
    liq_regime = "SUPPLY SWEEP" if df_liq['Supply_Sweep'].iloc[-1] else ("DEMAND SWEEP" if df_liq['Demand_Sweep'].iloc[-1] else "PRICE DISCOVERY")

    bias = 0 
    if hurst_regime == "TRENDING": bias += 1
    if hurst_regime == "MEAN REVERTING": bias -= 1
    if "DEMAND" in liq_regime: bias += 1
    if "SUPPLY" in liq_regime: bias -= 1
    if corr_val < 0.50: bias -= 1
    
    macro_state = "BULLISH / TRENDING" if bias > 0 else ("BEARISH / REVERSION" if bias < 0 else "NEUTRAL / CHOPPY")
    option_strategy = "NET SHORT PREMIUM" if ivr > 50 and vrp_val > 0 else "NET LONG PREMIUM"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. VOLATILITY & PRICING**")
        st.metric("IV Rank", f"{ivr:.1f}%", "High Premium" if ivr > 50 else "Low Premium", delta_color="inverse")
        st.metric("VRP Edge", f"{vrp_val:+.2f}%", "Sellers Edge" if vrp_val > 0 else "Buyers Edge", delta_color="normal")
    with c2:
        st.markdown("**2. MARKET BEHAVIOR**")
        st.metric("Hurst Exponent", f"{hurst_val:.3f}", hurst_regime, delta_color="off")
        st.metric("Intraday Liquidity", "Latest Bias", liq_regime, delta_color="off")
    with c3:
        st.markdown("**3. SYSTEMIC RISK**")
        st.metric(f"{div1_name}/{div2_name} Corr", f"{corr_val:.2f}", "SEVERE DIVERGENCE" if corr_val < 0.5 else "HIGH CORRELATION", delta_color="inverse" if corr_val < 0.50 else "normal")

    categories = ['IV Rank', 'Trend (Hurst)', 'Correlation', 'VRP Premium']
    norm_ivr = min(ivr / 100, 1.0)
    norm_hurst = min(hurst_val, 1.0)
    norm_corr = max(0, min(corr_val, 1.0))
    norm_vrp = max(0, min((vrp_val + 5) / 10, 1.0)) 

    colA, colB = st.columns([1, 2])
    with colA:
        fig_radar = go.Figure(data=go.Scatterpolar(r=[norm_ivr, norm_hurst, norm_corr, norm_vrp], theta=categories, fill='toself', line=dict(color=CHART_THEME["primary"])))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), showlegend=False, template=CHART_THEME["template"], title="REGIME PROFILE", height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=30, b=20))
        add_watermark(fig_radar)
        st.plotly_chart(fig_radar, use_container_width=True)

    with colB:
        st.markdown("### AI DAILY MARKET ANALYSIS")
        st.info(f"""
        **Systemic Risk:** Correlation is {corr_val:.2f}. {"Assets confirm the broader trend." if corr_val > 0.5 else "Assets are acting independently, breaking systemic correlation."}\n
        **Volatility:** IVR is {ivr:.1f}%. VRP is {vrp_val:+.2f}%. {"Market is overpricing risk, giving sellers an edge." if vrp_val > 0 else "Market is underpricing risk, making options cheap for buyers."}\n
        **Behavior:** Hurst is {hurst_val:.3f} ({hurst_regime}). {"Follow momentum breakouts." if hurst_regime == "TRENDING" else "Fade extremes at support/resistance."}\n
        **Conclusion:** Quantitative bias is **{macro_state}**. Optimal options approach is **{option_strategy}**.
        """)

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
    st.subheader(f"10. ADVANCED OPTIONS PRICING & GREEKS")
    asset_data = fetch_data(ticker, period="5d", is_crypto=is_crypto)
    vix = get_vix_data(asset_class, ticker, period="5d", is_crypto=is_crypto)
    if asset_data is None: return st.warning("Spot price data unavailable.")
    
    live_spot = get_scalar(asset_data['Close'])
    live_iv = get_scalar(vix['Close']) if vix is not None else 30.0
    default_rate = 7.0 if asset_class == "Indian Equities" else 5.0

    c1, c2, c3, c4 = st.columns(4)
    with c1: spot = st.number_input("Spot Price", value=float(live_spot), step=10.0)
    with c2: strike = st.number_input("Strike Price (K)", value=float(round(live_spot/100)*100), step=100.0)
    with c3: dte = st.number_input("Days to Expiry (DTE)", value=7, min_value=0, max_value=1000)
    with c4: iv = st.number_input("Implied Volatility (%)", value=float(live_iv), step=1.0)

    greeks = bs_greeks(spot, strike, dte, default_rate, iv)
    
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
        
    with c1: render_greek_card("CALL OPTION", greeks['Call'], CHART_THEME["bullish"])
    with c2: render_greek_card("PUT OPTION", greeks['Put'], CHART_THEME["bearish"])

    # --- DYNAMIC QUANT RISK INTERPRETATION ---
    st.markdown("### AI OPTIONS ANALYTICS & RISK BREAKDOWN")
    pct_from_strike = ((spot - strike) / strike) * 100
    
    if abs(pct_from_strike) < 1.0: moneyness_status, moneyness_desc = "At-The-Money (ATM)", "The spot price is sitting directly on the strike."
    elif spot > strike: moneyness_status, moneyness_desc = "Call ITM / Put OTM", f"Spot is trading **{abs(pct_from_strike):.2f}% above** the strike."
    else: moneyness_status, moneyness_desc = "Call OTM / Put ITM", f"Spot is trading **{abs(pct_from_strike):.2f}% below** the strike."

    st.info(f"""
    🌐 **Moneyness Matrix:** {moneyness_status}. {moneyness_desc}  
    🎯 **Direction (Delta):** Call assigns **{greeks['Call']['Delta']*100:.1f}% theoretical probability** of expiring ITM.  
    ⚡ **Acceleration (Gamma):** {"CRITICAL GAMMA RISK" if (dte <= 7 and abs(pct_from_strike) < 2.0) else "STABLE GAMMA PROFILE"}.  
    ⏳ **The Clock (Theta):** Call sheds **{abs(greeks['Call']['Theta']):.2f} pts/day**.  
    🌊 **Vol Risk (Vega):** A 1% Vol Crush drops contract value by **{greeks['Call']['Vega']:.2f} pts**.
    """)

    # --- EXPOSURE CURVES ---
    st.markdown("### OPTION DELTA EXPOSURE CURVE")
    
    spot_range = np.linspace(spot * 0.85, spot * 1.15, 100)
    call_deltas = [bs_greeks(s, strike, dte, default_rate, iv)['Call']['Delta'] for s in spot_range]
    put_deltas = [bs_greeks(s, strike, dte, default_rate, iv)['Put']['Delta'] for s in spot_range]
    
    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(x=spot_range, y=call_deltas, name='Call Delta', line=dict(color=CHART_THEME['bullish'], width=3)))
    fig_delta.add_trace(go.Scatter(x=spot_range, y=put_deltas, name='Put Delta', line=dict(color=CHART_THEME['bearish'], width=3)))
    fig_delta.add_vline(x=spot, line_dash="dash", line_color=CHART_THEME['neutral'], annotation_text="Current Spot")
    fig_delta.add_vline(x=strike, line_dash="dot", line_color=CHART_THEME['secondary'], annotation_text="Strike Price")
    fig_delta.update_layout(title="Option Delta Exposure Curve", template=CHART_THEME['template'], height=350, hovermode='x unified', margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    add_watermark(fig_delta)
    st.plotly_chart(fig_delta, use_container_width=True)


# ========================= 11. ML ENGINE =========================
def render_ml_engine(ticker, is_crypto):
    st.subheader(f"AI PREDICTIVE ENGINE")
    st.caption("Training isolated Random Forest on engineered features.")
    
    df = fetch_data(ticker, period="2y", interval="1d", is_crypto=is_crypto)
    if df is None or len(df) < 50: return st.warning("Insufficient data for ML.")
    
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20D'] = df['Log_Returns'].rolling(20).std() * np.sqrt(252)
    df['Momentum_10D'] = df['Close'] - df['Close'].shift(10)
    df['SMA_20_Dist'] = (df['Close'] / df['Close'].rolling(20).mean()) - 1
    df['RSI_14'] = calculate_rsi(df['Close'], 14)
    
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    ml_data = df.dropna().copy()
    
    features = ['Log_Returns', 'Vol_20D', 'Momentum_10D', 'SMA_20_Dist', 'RSI_14']
    X = ml_data[features].iloc[:-1]
    y = ml_data['Target'].iloc[:-1]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    model.fit(X_scaled, y)
    
    live_data = ml_data[features].iloc[[-1]]
    live_scaled = scaler.transform(live_data)
    
    prob_bullish = model.predict_proba(live_scaled)[0][1] * 100
    prob_bearish = model.predict_proba(live_scaled)[0][0] * 100
    prediction = "BULLISH" if prob_bullish > 50 else "BEARISH"
    color = CHART_THEME["bullish"] if prediction == "BULLISH" else CHART_THEME["bearish"]

    c1, c2, c3 = st.columns(3)
    c1.metric("ML Model Bias", prediction)
    c2.metric("Up-Day Prob", f"{prob_bullish:.1f}%")
    c3.metric("Down-Day Prob", f"{prob_bearish:.1f}%")

    col1, col2 = st.columns([1, 1])
    with col1:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = prob_bullish, domain = {'x': [0, 1], 'y': [0, 1]}, title = {'text': "Bullish Prob (%)"},
            gauge = {
                'axis': {'range': [0, 100]}, 'bar': {'color': CHART_THEME["primary"]},
                'steps': [
                    {'range': [0, 45], 'color': "rgba(239, 68, 68, 0.3)"},
                    {'range': [45, 55], 'color': "rgba(255, 255, 255, 0.1)"},
                    {'range': [55, 100], 'color': "rgba(34, 197, 94, 0.3)"}],
            }
        ))
        fig_gauge.update_layout(template=CHART_THEME["template"], height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=True)
        fig_imp = go.Figure(go.Bar(x=feat_imp_df['Importance'], y=feat_imp_df['Feature'], orientation='h', marker_color=CHART_THEME["secondary"]))
        fig_imp.update_layout(title="Feature Importance", template=CHART_THEME["template"], height=250, xaxis_title="Weight", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=40, b=10))
        add_watermark(fig_imp)
        st.plotly_chart(fig_imp, use_container_width=True)

# ========================= ALADDIN UI ROUTER =========================
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

    # Header Section
    col_h1, col_h2 = st.columns([4, 1])
    col_h1.title(f"ALADDIN // QUANT TERMINAL // {selected_name.upper()}")
    with col_h2:
        aladdin_metric("SYSTEM STATUS", "ONLINE", "OPTIMIZED")

    with st.spinner("Initializing complete Aladdin quantitative matrix..."):
        
        # ROW 0: Macro Pulse (New)
        with st.container(border=True):
            render_macro_pulse()
        
        # Grid Layout - Row 1 (Realtime & NLP)
        tab_row1_c1, tab_row1_c2 = st.columns([2, 1])
        
        with tab_row1_c1:
            with st.container(border=True):
                render_realtime_chart(selected_name, ticker, is_crypto)
            
        with tab_row1_c2:
            with st.container(border=True):
                render_nlp_sentiment(ticker, is_crypto)

        # Grid Layout - Row 2 (ML Engine & Market Synthesis)
        tab_row2_c1, tab_row2_c2 = st.columns([1, 2])
        
        with tab_row2_c1:
            with st.container(border=True):
                render_ml_engine(ticker, is_crypto)
                
        with tab_row2_c2:
            with st.container(border=True):
                render_market_synthesis(selected_name, ticker, asset_class, div1, div2, div1_name, div2_name, currency, trading_days, is_crypto)

        # Grid Layout - Row 3 (Options Pricing)
        with st.container(border=True):
            render_options_greeks(selected_name, ticker, asset_class, is_crypto)

        # Secondary Grid - Row 4 (Divergence & VRP)
        col_row4_1, col_row4_2 = st.columns(2)
        
        with col_row4_1:
            with st.container(border=True):
                render_index_divergence(div1, div2, div1_name, div2_name, currency, is_crypto)
            
        with col_row4_2:
            with st.container(border=True):
                render_vrp(selected_name, ticker, asset_class, trading_days, is_crypto)

        # Secondary Grid - Row 5 (Volatility & Expected Move & Cone)
        col_row5_1, col_row5_2, col_row5_3 = st.columns(3)
        
        with col_row5_1:
            with st.container(border=True):
                render_volatility_metrics(asset_class, ticker, is_crypto)
            
        with col_row5_2:
            with st.container(border=True):
                render_expected_move(selected_name, ticker, asset_class, currency, trading_days, is_crypto)
            
        with col_row5_3:
            with st.container(border=True):
                render_volatility_cone(selected_name, ticker, trading_days, is_crypto)

        # Secondary Grid - Row 6 (Hurst, Liquidity, YZ Vol)
        col_row6_1, col_row6_2, col_row6_3 = st.columns([1, 2, 1])
        
        with col_row6_1:
            with st.container(border=True):
                render_hurst_regime(selected_name, ticker, is_crypto)
            
        with col_row6_2:
            with st.container(border=True):
                render_liquidity_sweep(selected_name, ticker, is_crypto)
            
        with col_row6_3:
            with st.container(border=True):
                render_advanced_volatility(selected_name, ticker, trading_days, is_crypto)

if __name__ == "__main__":
    main()