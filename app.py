# ============================== IMPORTS ==============================
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

# ML IMPORTS
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

from textblob import TextBlob
import logging
import yaml
import os
from typing import Optional, Tuple, List, Dict, Any, Union, Callable

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

warnings.filterwarnings('ignore')

# ============================== LOGGING ==============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================== CONFIGURATION ==============================
class Config:
    DEFAULTS = {
        'thresholds': {
            'ivr_high': 50.0,
            'hurst_trend': 0.55,
            'hurst_mean_revert': 0.45,
            'corr_high': 0.80,
            'corr_divergence': 0.50,
            'vrp_sellers': 0.0,
        },
        'ml': {
            'n_estimators': 150,
            'max_depth': 4,
            'walk_forward_splits': 5,
        },
        'retry': {
            'max_attempts': 3,
            'backoff_factor': 2.0,
        }
    }

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.DEFAULTS.copy()
        self._load_yaml(config_path)
        self._apply_env_overrides()

    def _load_yaml(self, path: str) -> None:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    user_config = yaml.safe_load(f)
                if user_config: self._deep_update(self.config, user_config)
            except Exception as e:
                logger.warning(f"Could not load {path}: {e}")

    def _apply_env_overrides(self) -> None:
        for section, params in self.config.items():
            if not isinstance(params, dict): continue
            for key, val in params.items():
                env_key = f"ALADDIN_{section.upper()}_{key.upper()}"
                if env_key in os.environ:
                    try:
                        default_type = type(self.DEFAULTS.get(section, {}).get(key, val))
                        self.config[section][key] = default_type(os.environ[env_key])
                    except Exception as e:
                        logger.warning(f"Could not apply env {env_key}: {e}")

    def _deep_update(self, d: dict, u: dict) -> None:
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v

    def get(self, section: str, key: str) -> Any:
        return self.config.get(section, {}).get(key)

CONFIG = Config()

# ============================== CONSTANTS ==============================
CHART_THEME = {
    "template": "plotly_dark",
    "primary": "#67e8f9",
    "secondary": "#fbbf24",
    "bullish": "#22c55e",
    "bearish": "#ef4444",
    "neutral": "white",
    "watermark": "#334155"
}

INDIAN_ASSETS = {
    "Nifty 50 (Index)": "^NSEI",
    "Bank Nifty (Index)": "^NSEBANK",
    "FinNifty (Index)": "NIFTY_FIN_SERVICE.NS",
    "Nifty IT (Index)": "^CNXIT",
    "Nifty Auto (Index)": "^CNXAUTO",
    "Nifty Metal (Index)": "^CNXMETAL",
    "Sensex (Index)": "^BSESN",
    "Reliance Ind (Stock)": "RELIANCE.NS",
    "HDFC Bank (Stock)": "HDFCBANK.NS",
    "Infosys (Stock)": "INFY.NS",
    "TCS (Stock)": "TCS.NS",
    "ICICI Bank (Stock)": "ICICIBANK.NS"
}

CRYPTO_ASSETS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD"
}

# ============================== HELPERS ==============================
def calculate_rsi(data: pd.Series, periods: int = 14) -> pd.Series:
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def safe_get_scalar(series: Union[pd.Series, float, int, None], default: float = 0.0) -> float:
    if series is None: return default
    if isinstance(series, pd.Series):
        if series.empty: return default
        val = series.iloc[-1]
    else: val = series
    try:
        if hasattr(val, 'item'): val = val.item()
        val = float(val)
    except (ValueError, TypeError): val = default
    if np.isnan(val) or np.isinf(val): val = default
    return val

def add_watermark(fig: go.Figure) -> None:
    fig.add_annotation(
        text="ALADDIN QUANT TERMINAL v18.0",
        xref="paper", yref="paper", x=0.99, y=0.01,
        showarrow=False, font=dict(size=10, color=CHART_THEME["watermark"]), opacity=0.6
    )

def aladdin_metric(label: str, value: str, delta: Optional[str] = None, invert_color: bool = False) -> None:
    if delta:
        color = ("#ef4444" if "+" in delta else "#22c55e") if invert_color else ("#22c55e" if "+" in delta else "#ef4444")
    else:
        color = "white"
    st.markdown(f"""
        <div class="module-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value} {f'<span style="color:{color}">({delta})</span>' if delta else ''}</div>
        </div>
    """, unsafe_allow_html=True)

def safe_render(func: Callable, *args, **kwargs) -> None:
    try:
        func(*args, **kwargs)
    except Exception as e:
        st.error(f"Module Offline: {func.__name__} encountered an error.")
        logger.error(f"Component Failure in {func.__name__}: {str(e)}")

# ============================== DATA INGESTION (L1 + L2 CACHING) ==============================
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_data_internal(ticker: str, period: str, interval: str, is_crypto: bool) -> Optional[pd.DataFrame]:
    if interval == '15m': 
        period = '1mo' 
    elif interval in ['1h', '4h'] and period in ['max', '5y', '10y', '2y']: 
        period = '730d'

    data = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if (data is None or data.empty) and interval == "15m":
        data = yf.download(ticker, period="5d", interval=interval, progress=False)
        
    if data is None or data.empty: 
        return None
        
    if isinstance(data.columns, pd.MultiIndex): 
        data.columns = [col[0] for col in data.columns]
        
    if data.index.tz is not None: 
        data.index = data.index.tz_localize(None)
        
    return data.dropna(subset=['Close'])

def fetch_data(ticker: str, period: str = "1y", interval: str = "1d", is_crypto: bool = False) -> Optional[pd.DataFrame]:
    if 'market_data' not in st.session_state:
        st.session_state.market_data = {}
        
    cache_key = f"{ticker}_{period}_{interval}_{is_crypto}"
    if cache_key in st.session_state.market_data:
        return st.session_state.market_data[cache_key]

    max_attempts = CONFIG.get('retry', 'max_attempts')
    backoff_factor = CONFIG.get('retry', 'backoff_factor')
    attempt, delay = 0, 1.0
    
    while attempt < max_attempts:
        try:
            df = _fetch_data_internal(ticker, period, interval, is_crypto)
            if df is not None:
                st.session_state.market_data[cache_key] = df
            return df
        except Exception as e:
            attempt += 1
            if attempt == max_attempts:
                logger.error(f"Failed to fetch {ticker}: {e}")
                return None
            time.sleep(delay)
            delay *= backoff_factor

def get_vix_data(asset_class: str, ticker: str, period: str = "1y", is_crypto: bool = False) -> Optional[pd.DataFrame]:
    if asset_class == "Indian Equities":
        return fetch_data("^INDIAVIX", period=period, is_crypto=False)
    else:
        data = fetch_data(ticker, period="2y", is_crypto=True)
        if data is None: return None
        ret = np.log(data['Close'] / data['Close'].shift(1))
        synth_vix = ret.rolling(30).std() * np.sqrt(365) * 100
        vix_df = data[['Close']].copy()
        vix_df['Close'] = synth_vix
        return vix_df.dropna().tail(365 if period == "1y" else 180)

# ============================== NLP NEWS SENTIMENT ENGINE (UPGRADED) ==============================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_news_sentiment(ticker: str, is_crypto: bool) -> Tuple[Optional[float], Optional[float], List[Dict]]:
    try:
        tkr = yf.Ticker(ticker)
        news = tkr.news
    except Exception as e:
        logger.error(f"Failed to fetch news for {ticker}: {e}")
        news = []

    if not news: return None, None, []

    total_polarity = 0.0
    total_subjectivity = 0.0
    analyzed_headlines = []
    
    fin_bull = {'surge', 'rally', 'bullish', 'breakout', 'growth', 'outperform', 'etf', 'buy', 'acquisition', 'profit', 'gain', 'soar', 'approve'}
    fin_bear = {'crash', 'bearish', 'drop', 'lawsuit', 'sec', 'regulatory', 'probe', 'deficit', 'sell', 'inflation', 'dump', 'hack', 'miss'}

    for item in news[:8]:
        title = item.get('title', '')
        publisher = item.get('publisher', 'WIRE')
        if not title: continue
        
        blob = TextBlob(title)
        polarity = blob.sentiment.polarity       
        subjectivity = blob.sentiment.subjectivity 
        
        title_lower = title.lower()
        for word in fin_bull:
            if word in title_lower: polarity += 0.15
        for word in fin_bear:
            if word in title_lower: polarity -= 0.15
            
        polarity = max(-1.0, min(1.0, polarity))
        total_polarity += polarity
        total_subjectivity += subjectivity
        
        if polarity > 0.05: 
            tag, color = "BULLISH", CHART_THEME['bullish']
        elif polarity < -0.05: 
            tag, color = "BEARISH", CHART_THEME['bearish']
        else: 
            tag, color = "NEUTRAL", CHART_THEME['secondary']
            
        analyzed_headlines.append({
            'title': title, 
            'publisher': publisher, 
            'tag': tag, 
            'color': color,
            'polarity': polarity
        })

    count = len(analyzed_headlines)
    if count == 0: return None, None, []
    
    avg_polarity = (total_polarity / count) * 100 
    avg_subjectivity = (total_subjectivity / count) * 100 
    return avg_polarity, avg_subjectivity, analyzed_headlines

def render_nlp_sentiment(ticker: str, is_crypto: bool) -> None:
    st.subheader("NLP NEWS SENTIMENT ENGINE")
    score_data = fetch_news_sentiment(ticker, is_crypto)
    if score_data[0] is None: return st.warning("No recent news context found.")
    
    score, subjectivity, headlines = score_data

    gauge_color = CHART_THEME['bullish'] if score > 10 else (CHART_THEME['bearish'] if score < -10 else CHART_THEME['secondary'])
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=score, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "TextBlob Financial Sentiment"},
        gauge={
            'axis': {'range': [-100, 100]}, 'bar': {'color': gauge_color},
            'steps': [
                {'range': [-100, -15], 'color': "rgba(239, 68, 68, 0.2)"}, 
                {'range': [-15, 15], 'color': "rgba(255, 255, 255, 0.1)"}, 
                {'range': [15, 100], 'color': "rgba(34, 197, 94, 0.2)"}
            ]
        }
    ))
    fig_gauge.update_layout(template=CHART_THEME["template"], height=200, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_gauge, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Net Bias Score", f"{score:+.1f}")
    c2.metric("Media Subjectivity", f"{subjectivity:.1f}%")

    st.markdown("<div style='margin-top: 15px; margin-bottom: 10px; color: #94a3b8; font-size: 11px; text-transform: uppercase;'>REAL-TIME HEADLINES & POLARITY SCORES</div>", unsafe_allow_html=True)
    for h in headlines:
        st.markdown(f"""
            <div class="news-headline" style="border-left: 3px solid {h['color']}; padding-left: 8px; margin-bottom: 8px; font-size: 13px;">
                {h['title']} <br>
                <span style="font-size: 10px; color: #64748b; text-transform: uppercase;">[{h['tag']} • Score: {h['polarity']:+.2f}] • {h['publisher']}</span>
            </div>
        """, unsafe_allow_html=True)

# ============================== GAP-FREE REAL-TIME CHART, VWAP & LIQUIDITY ==============================
def render_realtime_chart(selected_name: str, ticker: str, is_crypto: bool) -> None:
    st.subheader("MARKET PRICE, VOLUME & MOMENTUM")
    timeframe = st.radio("Select Timeframe", ["15m", "1h", "4h", "1d"], index=1, horizontal=True, label_visibility="collapsed")
    period, interval = ("1mo", "15m") if timeframe == "15m" else ("730d", "1h") if timeframe in ["1h", "4h"] else ("2y", "1d")

    data = fetch_data(ticker, period=period, interval=interval, is_crypto=is_crypto)
    if data is None or data.empty: return st.warning("Real-time data unavailable.")

    if timeframe == "4h": data = data.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    data = data.tail(300 if timeframe in ["15m", "1h", "4h"] else 252).copy()
    
    if 'Volume' in data.columns and data['Volume'].sum() > 0:
        data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3
        data['VP'] = data['Typical_Price'] * data['Volume']
        grouper = data.index.date if timeframe in ['15m', '1h', '4h'] else data.index.to_period('M')
        data['VWAP'] = data.groupby(grouper)['VP'].cumsum() / data.groupby(grouper)['Volume'].cumsum()
    else:
        data['VWAP'] = np.nan
        
    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    data['RSI'] = calculate_rsi(data['Close'], 14)
    
    data['Prev_High'] = data['High'].rolling(20).max().shift(1)
    data['Prev_Low'] = data['Low'].rolling(20).min().shift(1)
    data['Supply_Sweep'] = (data['High'] > data['Prev_High']) & (data['Close'] < data['Prev_High'])
    data['Demand_Sweep'] = (data['Low'] < data['Prev_Low']) & (data['Close'] > data['Prev_Low'])

    last_close = safe_get_scalar(data['Close'])
    x_format = '%Y-%m-%d' if timeframe == "1d" else '%Y-%m-%d %H:%M'
    x_axis_string = data.index.strftime(x_format)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.65, 0.15, 0.20], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=x_axis_string, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], 
        name='Price', increasing_line_color=CHART_THEME['bullish'], decreasing_line_color=CHART_THEME['bearish']
    ), row=1, col=1)
    
    if not data['EMA_9'].isna().all():
        fig.add_trace(go.Scatter(x=x_axis_string, y=data['EMA_9'], mode='lines', name='EMA 9', line=dict(color='#fbbf24', width=1.5)), row=1, col=1)
    if not data['EMA_21'].isna().all():
        fig.add_trace(go.Scatter(x=x_axis_string, y=data['EMA_21'], mode='lines', name='EMA 21', line=dict(color='#8b5cf6', width=2)), row=1, col=1)
    if not data['VWAP'].isna().all():
        fig.add_trace(go.Scatter(x=x_axis_string, y=data['VWAP'], mode='lines', name='VWAP', line=dict(color='#e2e8f0', width=1.5, dash='dot')), row=1, col=1)

    supply_data = data[data['Supply_Sweep']]
    demand_data = data[data['Demand_Sweep']]
    
    if not supply_data.empty:
        fig.add_trace(go.Scatter(
            x=supply_data.index.strftime(x_format), y=supply_data['High'] * 1.002, mode='markers', 
            marker=dict(symbol='triangle-down', size=12, color=CHART_THEME["bearish"], line=dict(width=1, color='white')), 
            name='Supply Sweep'
        ), row=1, col=1)
    if not demand_data.empty:
        fig.add_trace(go.Scatter(
            x=demand_data.index.strftime(x_format), y=demand_data['Low'] * 0.998, mode='markers', 
            marker=dict(symbol='triangle-up', size=12, color=CHART_THEME["bullish"], line=dict(width=1, color='white')), 
            name='Demand Sweep'
        ), row=1, col=1)

    fig.add_hline(y=last_close, line_dash="dot", line_color=CHART_THEME["primary"], line_width=1.5, 
                  annotation_text=f"{last_close:,.2f}", annotation_position="right", 
                  annotation_font_color=CHART_THEME["primary"], row=1, col=1)

    if 'Volume' in data.columns and not (data['Volume'] == 0).all():
        colors = [CHART_THEME['bullish'] if row['Close'] >= row['Open'] else CHART_THEME['bearish'] for _, row in data.iterrows()]
        fig.add_trace(go.Bar(x=x_axis_string, y=data['Volume'], name='Volume', marker_color=colors, opacity=0.85, marker_line_width=0), row=2, col=1)

    if not data['RSI'].isna().all():
        fig.add_trace(go.Scatter(x=x_axis_string, y=data['RSI'], mode='lines', name='RSI 14', line=dict(color=CHART_THEME['primary'], width=1.5)), row=3, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255, 255, 255, 0.05)", line_width=0, row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=CHART_THEME['bearish'], line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=CHART_THEME['bullish'], line_width=1, row=3, col=1)

    fig.update_layout(
        template=CHART_THEME['template'], height=650, xaxis_rangeslider_visible=False, hovermode='x unified', 
        bargap=0, bargroupgap=0, margin=dict(l=10, r=60, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False
    )
    
    fig.update_xaxes(type='category', showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)', showticklabels=False, row=1, col=1)
    fig.update_xaxes(type='category', showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)', showticklabels=False, row=2, col=1)
    fig.update_xaxes(type='category', categoryorder='category ascending', showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)', showticklabels=True, showspikes=True, spikemode='across', spikethickness=1, spikedash='dot', spikecolor='rgba(255,255,255,0.3)', row=3, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    last_24h = data.loc[data.index >= data.index.max() - pd.Timedelta(days=1)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close", f"{last_close:,.2f}")
    c2.metric("24h High", f"{last_24h['High'].max():,.2f}")
    c3.metric("24h Low", f"{last_24h['Low'].min():,.2f}")
    
    last_supply = safe_get_scalar(data['Supply_Sweep'])
    last_demand = safe_get_scalar(data['Demand_Sweep'])
    regime = "SUPPLY SWEEP" if last_supply else ("DEMAND SWEEP" if last_demand else "DISCOVERY")
    c4.metric("Micro-Regime", regime)

# ============================== VOLATILITY & SYSTEMIC METRICS ==============================
def render_volatility_metrics(asset_class: str, ticker: str, is_crypto: bool) -> None:
    st.subheader("1. IMPLIED VOLATILITY RANK")
    vix_name = "India VIX" if asset_class == "Indian Equities" else "Synthetic IV (30D HV)"
    data = get_vix_data(asset_class, ticker, period="1y", is_crypto=is_crypto)
    if data is None: return st.warning(f"{vix_name} data unavailable.")

    close = data['Close']
    current_iv, high_52w, low_52w = safe_get_scalar(close), safe_get_scalar(close.max()), safe_get_scalar(close.min())
    denom = high_52w - low_52w
    ivr = ((current_iv - low_52w) / denom * 100) if denom != 0 else 0.0
    ivp = (close[close < current_iv].count() / len(close)) * 100 if len(close) > 0 else 0.0
    regime = "HIGH VOLATILITY" if ivr > CONFIG.get('thresholds', 'ivr_high') else "LOW VOLATILITY"

    fig = go.Figure()
    fig.add_hrect(y0=low_52w, y1=high_52w, line_width=0, fillcolor="rgba(255, 255, 255, 0.05)", opacity=1)
    fig.add_trace(go.Scatter(x=close.index, y=close, mode='lines', name=vix_name, line=dict(color=CHART_THEME["primary"], width=2.5), fill='tozeroy', fillcolor='rgba(103, 232, 249, 0.05)'))
    fig.add_hline(y=high_52w, line_dash="dash", line_color=CHART_THEME["bearish"])
    fig.add_hline(y=low_52w, line_dash="dash", line_color=CHART_THEME["bullish"])
    fig.add_hline(y=current_iv, line_color=CHART_THEME["neutral"], line_width=3)
    
    fig.update_layout(template=CHART_THEME["template"], height=300, margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
    
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("IV Rank", f"{ivr:.1f}%")
    c2.metric("IV Percentile", f"{ivp:.1f}%")
    c3.metric("Regime", regime)

def render_expected_move(selected_name: str, ticker: str, asset_class: str, currency: str, trading_days: int, is_crypto: bool) -> None:
    st.subheader("2. EXPECTED MOVE (1σ)")
    asset_data, vix = fetch_data(ticker, period="1mo", is_crypto=is_crypto), get_vix_data(asset_class, ticker, period="5d", is_crypto=is_crypto)
    if asset_data is None or vix is None: return st.warning("Data fetch failed for Expected Move.")

    spot, current_vix = safe_get_scalar(asset_data['Close']), safe_get_scalar(vix['Close'])
    exp_move = spot * ((current_vix / 100) * np.sqrt(1 / trading_days))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=asset_data.index, y=asset_data['Close'], mode='lines', name=selected_name, line=dict(color=CHART_THEME["primary"], width=3)))
    
    tomorrow = asset_data.index[-1] + timedelta(days=1)
    fig.add_trace(go.Scatter(
        x=[asset_data.index[-1], tomorrow, tomorrow, asset_data.index[-1]], 
        y=[spot, spot + exp_move, spot - exp_move, spot], 
        fill='toself', fillcolor='rgba(251, 191, 36, 0.15)', line=dict(color='rgba(255,255,255,0)'),
        name='1σ Range', showlegend=False, hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(x=[tomorrow], y=[spot + exp_move], mode='markers+text', name='+1σ', marker=dict(color=CHART_THEME["bullish"], size=12, symbol='triangle-up'), text=[f"+{exp_move:,.0f}"], textposition="top right"))
    fig.add_trace(go.Scatter(x=[tomorrow], y=[spot - exp_move], mode='markers+text', name='-1σ', marker=dict(color=CHART_THEME["bearish"], size=12, symbol='triangle-down'), text=[f"-{exp_move:,.0f}"], textposition="bottom right"))
    
    fig.update_layout(template=CHART_THEME["template"], height=300, margin=dict(t=10, b=10, r=40), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
    
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Spot Price", f"{currency}{spot:,.2f}")
    c2.metric("Vol Benchmark", f"{current_vix:.2f}%")
    c3.metric("Implied Move", f"± {currency}{exp_move:,.1f}")

def render_index_divergence(div1: str, div2: str, name1: str, name2: str, currency: str, is_crypto: bool) -> None:
    st.subheader(f"3. SYSTEMIC DIVERGENCE ({name1} vs {name2})")
    d1, d2 = fetch_data(div1, period="1y", is_crypto=is_crypto), fetch_data(div2, period="1y", is_crypto=is_crypto)
    if d1 is None or d2 is None: return st.warning("Divergence data unavailable.")

    data = pd.merge(d1['Close'].to_frame(name1), d2['Close'].to_frame(name2), left_index=True, right_index=True, how='outer').ffill().dropna()
    if data.empty: return st.warning("Insufficient overlapping data.")
    
    normalized = (data / data.iloc[0]) * 100
    rolling_corr = np.log(data / data.shift(1)).dropna()[name1].rolling(20).corr(np.log(data / data.shift(1)).dropna()[name2]).dropna()
    current_corr = safe_get_scalar(rolling_corr)
    
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[name1], name=name1, line=dict(color=CHART_THEME["primary"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[name2], name=name2, line=dict(color=CHART_THEME["secondary"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, name='20D Corr', line=dict(color=CHART_THEME["neutral"], width=1.5), fill='tozeroy', fillcolor='rgba(255,255,255,0.1)'), row=2, col=1)
    
    fig.update_layout(template=CHART_THEME["template"], height=350, margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric(name1, f"{currency}{safe_get_scalar(data[name1]):,.2f}")
    c2.metric(name2, f"{currency}{safe_get_scalar(data[name2]):,.2f}")
    c3.metric("20D Correlation", f"{current_corr:.3f}")

def render_volatility_cone(selected_name: str, ticker: str, trading_days: int, is_crypto: bool) -> None:
    st.subheader("4. VOLATILITY TERM STRUCTURE (CONE)")
    data = fetch_data(ticker, period="max", is_crypto=is_crypto)
    if data is None: return st.warning("Data unavailable.")

    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    windows = [10, 20, 30, 60, 90, 120, 180, trading_days]
    stats = []
    for w in windows:
        if w > len(returns): continue
        vol_series = returns.rolling(w).std() * np.sqrt(trading_days) * 100
        stats.append({
            'window': w, 'max': safe_get_scalar(vol_series.max()), 'min': safe_get_scalar(vol_series.min()),
            'median': safe_get_scalar(vol_series.median()), 'current': safe_get_scalar(vol_series.iloc[-1])
        })
    df_stats = pd.DataFrame(stats)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_stats['window'], y=df_stats['max'], name='Max Vol', mode='lines+markers', line=dict(color=CHART_THEME["bearish"])))
    fig.add_trace(go.Scatter(x=df_stats['window'], y=df_stats['min'], name='Min Vol', mode='lines+markers', line=dict(color=CHART_THEME["bullish"])))
    fig.add_trace(go.Scatter(x=df_stats['window'], y=df_stats['median'], name='Median Vol', mode='lines+markers', line=dict(color=CHART_THEME["neutral"], dash='dash')))
    fig.add_trace(go.Scatter(x=df_stats['window'], y=df_stats['current'], name='Current Vol', mode='lines+markers', line=dict(color=CHART_THEME["primary"], width=4)))
    fig.update_layout(template=CHART_THEME["template"], height=300, margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

def render_vrp(selected_name: str, ticker: str, asset_class: str, trading_days: int, is_crypto: bool) -> None:
    st.subheader("5. VOLATILITY RISK PREMIUM (VRP)")
    main_data, vix = fetch_data(ticker, period="6mo", is_crypto=is_crypto), get_vix_data(asset_class, ticker, period="6mo", is_crypto=is_crypto)
    if main_data is None or vix is None: return st.warning("Data unavailable.")

    hv = np.log(main_data['Close'] / main_data['Close'].shift(1)).rolling(20).std() * np.sqrt(trading_days) * 100
    df = pd.merge(vix['Close'].to_frame('VIX'), hv.to_frame('HV'), left_index=True, right_index=True, how='outer').ffill().dropna()
    if df.empty: return st.warning("Insufficient data.")
    df['VRP'] = df['VIX'] - df['HV']
    current_vrp = safe_get_scalar(df['VRP'])

    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=df.index, y=df['VIX'], name='Implied Vol', line=dict(color=CHART_THEME["primary"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['HV'], name='Realized Vol', line=dict(color=CHART_THEME["secondary"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['HV'], showlegend=False, line=dict(width=0), hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VIX'], fill='tonexty', fillcolor='rgba(103, 232, 249, 0.15)', line=dict(width=0), name='Spread', showlegend=False, hoverinfo='skip'), row=1, col=1)

    colors = np.where(df['VRP'] > 0, CHART_THEME["bullish"], CHART_THEME["bearish"])
    fig.add_trace(go.Bar(x=df.index, y=df['VRP'], name='VRP Spread', marker_color=colors, opacity=0.8), row=2, col=1)
    
    fig.update_layout(template=CHART_THEME["template"], height=350, margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified', bargap=0)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Vol Benchmark", f"{safe_get_scalar(df['VIX']):.2f}")
    c2.metric("HV (20D)", f"{safe_get_scalar(df['HV']):.2f}")
    c3.metric("Current VRP", f"{current_vrp:+.2f}%")

def render_hurst_regime(selected_name: str, ticker: str, is_crypto: bool) -> None:
    st.subheader("6. HURST EXPONENT (REGIME)")
    data = fetch_data(ticker, period="1y", is_crypto=is_crypto)
    if data is None: return st.warning("Data unavailable.")

    def calculate_hurst(ts: pd.Series) -> float:
        if len(ts) < 20: return np.nan
        lags = range(2, 20)
        reg_val = [np.std(ts.values[lag:] - ts.values[:-lag]) for lag in lags]
        try: return np.polyfit(np.log(lags), np.log(reg_val), 1)[0]
        except: return np.nan

    log_prices = np.log(data['Close'])
    hurst_series = log_prices.rolling(window=60).apply(calculate_hurst, raw=False)
    df = pd.DataFrame({'Close': data['Close'], 'Hurst': hurst_series}).dropna()
    if df.empty: return st.warning("Insufficient data for Hurst calculation.")
    
    current_hurst = safe_get_scalar(df['Hurst'])
    trend_thresh = CONFIG.get('thresholds', 'hurst_trend')
    mean_revert_thresh = CONFIG.get('thresholds', 'hurst_mean_revert')
    if current_hurst < mean_revert_thresh: regime = "MEAN REVERTING"
    elif current_hurst > trend_thresh: regime = "TRENDING"
    else: regime = "RANDOM WALK"

    fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35], shared_xaxes=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color=CHART_THEME["neutral"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Hurst'], name='Hurst', line=dict(color=CHART_THEME["primary"])), row=2, col=1)
    fig.update_layout(template=CHART_THEME["template"], height=300, margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Hurst Value", f"{current_hurst:.3f}")
    c2.metric("Regime", regime)

def render_advanced_volatility(selected_name: str, ticker: str, trading_days: int, is_crypto: bool) -> None:
    st.subheader("7. YANG-ZHANG VOLATILITY")
    data = fetch_data(ticker, period="1y", is_crypto=is_crypto)
    if data is None: return st.warning("Data unavailable.")

    df = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
    if len(df) < 2: return st.warning("Insufficient data for Yang-Zhang.")

    o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
    N = len(o)

    vol_o = np.log(o / c.shift(1)).std() ** 2
    vol_c = np.log(c / o).std() ** 2
    vol_rs = ((np.log(h / o) * np.log(h / c)) + (np.log(l / o) * np.log(l / c))).mean()
    k = 0.34 / (1.34 + (N + 1) / (N - 1)) if N > 1 else 0
    yz_vol = np.sqrt(vol_o + k * vol_c + (1 - k) * vol_rs) * np.sqrt(trading_days) * 100
    c2c_vol = np.log(c / c.shift(1)).std() * np.sqrt(trading_days) * 100

    st.caption("Captures overnight gap risk & intraday trend mathematically.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Yang-Zhang True Volatility", f"{yz_vol:.2f}%")
    c2.metric("Close-to-Close Vol", f"{c2c_vol:.2f}%")
    c3.metric("Hidden Gap Risk", f"{yz_vol - c2c_vol:+.2f}%", delta_color="inverse")

def render_market_synthesis(selected_name: str, ticker: str, asset_class: str, div1: str, div2: str, div1_name: str, div2_name: str, currency: str, trading_days: int, is_crypto: bool) -> None:
    st.subheader("8. MULTI-TIMEFRAME MARKET SYNTHESIS")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_vix = executor.submit(get_vix_data, asset_class, ticker, "6mo", is_crypto)
        f_daily = executor.submit(fetch_data, ticker, "1y", "1d", is_crypto)
        f_div1 = executor.submit(fetch_data, div1, "1y", "1d", is_crypto)
        f_div2 = executor.submit(fetch_data, div2, "1y", "1d", is_crypto)
        vix_data, daily_data, d1_data, d2_data = f_vix.result(), f_daily.result(), f_div1.result(), f_div2.result()

    if any(d is None for d in [vix_data, daily_data, d1_data, d2_data]): return st.warning("Synthesis data incomplete.")

    current_vix = safe_get_scalar(vix_data['Close'])
    vix_min = safe_get_scalar(vix_data['Close'].min())
    vix_max = safe_get_scalar(vix_data['Close'].max())
    ivr = ((current_vix - vix_min) / (vix_max - vix_min) * 100) if (vix_max - vix_min) != 0 else 0.0
    
    hv_20 = np.log(daily_data['Close'] / daily_data['Close'].shift(1)).rolling(20).std() * np.sqrt(trading_days) * 100
    vrp_val = current_vix - safe_get_scalar(hv_20)

    data_div = pd.merge(d1_data['Close'].to_frame('A'), d2_data['Close'].to_frame('B'), left_index=True, right_index=True, how='outer').ffill().dropna()
    corr_val = safe_get_scalar(np.log(data_div / data_div.shift(1)).dropna()['A'].rolling(20).corr(np.log(data_div / data_div.shift(1)).dropna()['B']).dropna(), default=0.5)

    categories = ['IV Rank', 'Correlation', 'VRP Premium']
    norm_ivr, norm_corr, norm_vrp = min(ivr / 100, 1.0), max(0, min(corr_val, 1.0)), max(0, min((vrp_val + 5) / 10, 1.0))

    colA, colB = st.columns([1, 2])
    with colA:
        fig_radar = go.Figure(data=go.Scatterpolar(r=[norm_ivr, norm_corr, norm_vrp], theta=categories, fill='toself', line=dict(color=CHART_THEME["primary"])))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), showlegend=False, template=CHART_THEME["template"], title="REGIME PROFILE", height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_radar, use_container_width=True)

    with colB:
        st.info(f"""
        **Systemic Risk:** Correlation is {corr_val:.2f}. {"Assets confirm the broader trend." if corr_val > 0.5 else "Assets are acting independently, breaking systemic correlation."}
        **Volatility:** IVR is {ivr:.1f}%. VRP is {vrp_val:+.2f}%. {"Market is overpricing risk, giving sellers an edge." if vrp_val > 0 else "Market is underpricing risk, making options cheap for buyers."}
        """)

# ============================== ML ENGINE (HARDENED & REGULARIZED) ==============================
@st.cache_resource(ttl=3600, show_spinner=False)
def train_and_validate_ml_model(ticker: str, is_crypto: bool):
    df = _fetch_data_internal(ticker, period="2y", interval="1d", is_crypto=is_crypto)
    macro_tnx = _fetch_data_internal("^TNX", period="2y", interval="1d", is_crypto=False)
    macro_dxy = _fetch_data_internal("DX-Y.NYB", period="2y", interval="1d", is_crypto=False)
    
    if df is None or len(df) < 100: return None, None, None, None

    if macro_tnx is not None: df['Macro_TNX'] = macro_tnx['Close'].ffill()
    if macro_dxy is not None: df['Macro_DXY'] = macro_dxy['Close'].ffill()

    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20D'] = df['Log_Returns'].rolling(20).std() * np.sqrt(252)
    df['Momentum_10D'] = df['Close'] - df['Close'].shift(10)
    df['SMA_20_Dist'] = (df['Close'] / df['Close'].rolling(20).mean()) - 1
    df['RSI_14'] = calculate_rsi(df['Close'], 14)
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    features = ['Log_Returns', 'Vol_20D', 'Momentum_10D', 'SMA_20_Dist', 'RSI_14']
    if 'Macro_TNX' in df.columns: features.append('Macro_TNX')
    if 'Macro_DXY' in df.columns: features.append('Macro_DXY')

    ml_data = df.dropna().copy()
    X, y = ml_data[features], ml_data['Target']

    tscv = TimeSeriesSplit(n_splits=CONFIG.get('ml', 'walk_forward_splits'), gap=5)
    accuracies, precisions, recalls = [], [], []

    xgb_params = {
        'n_estimators': CONFIG.get('ml', 'n_estimators'), 
        'max_depth': CONFIG.get('ml', 'max_depth'), 
        'learning_rate': 0.05, 
        'objective': 'binary:logistic', 
        'random_state': 42,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_weight': 3,
        'gamma': 0.1
    }

    for train_index, test_index in tscv.split(X):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))

    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X)
    model_full = xgb.XGBClassifier(**xgb_params)
    model_full.fit(X_full_scaled, y)

    metrics = {"acc": np.mean(accuracies), "prec": np.mean(precisions), "rec": np.mean(recalls)}
    return model_full, scaler_full, features, metrics

def render_ml_engine(ticker: str, is_crypto: bool) -> None:
    st.subheader("9. AI PREDICTIVE ENGINE (XGBoost + Purged CV)")
    model_full, scaler_full, features, metrics = train_and_validate_ml_model(ticker, is_crypto)
    if model_full is None: return st.warning("Insufficient data to train ML model.")

    df = fetch_data(ticker, period="60d", interval="1d", is_crypto=is_crypto)
    macro_tnx = fetch_data("^TNX", period="60d", interval="1d", is_crypto=False)
    macro_dxy = fetch_data("DX-Y.NYB", period="60d", interval="1d", is_crypto=False)

    if macro_tnx is not None: df['Macro_TNX'] = macro_tnx['Close'].ffill()
    if macro_dxy is not None: df['Macro_DXY'] = macro_dxy['Close'].ffill()

    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20D'] = df['Log_Returns'].rolling(20).std() * np.sqrt(252)
    df['Momentum_10D'] = df['Close'] - df['Close'].shift(10)
    df['SMA_20_Dist'] = (df['Close'] / df['Close'].rolling(20).mean()) - 1
    df['RSI_14'] = calculate_rsi(df['Close'], 14)
    
    live_data = df.dropna().tail(1)[features]
    if live_data.empty: return st.warning("Not enough live feature data.")

    live_scaled = scaler_full.transform(live_data)
    prob_bullish = model_full.predict_proba(live_scaled)[0][1] * 100
    prob_bearish = model_full.predict_proba(live_scaled)[0][0] * 100
    prediction = "BULLISH" if prob_bullish > 50 else "BEARISH"

    c1, c2, c3 = st.columns(3)
    c1.metric("ML Model Bias", prediction)
    c2.metric("Up-Day Prob", f"{prob_bullish:.1f}%")
    c3.metric("Down-Day Prob", f"{prob_bearish:.1f}%")

    col1, col2 = st.columns([1, 1])
    with col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob_bullish, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Bullish Prob (%)"},
            gauge={
                'axis': {'range': [0, 100]}, 'bar': {'color': CHART_THEME["primary"]},
                'steps': [
                    {'range': [0, 45], 'color': "rgba(239, 68, 68, 0.3)"},
                    {'range': [45, 55], 'color': "rgba(255, 255, 255, 0.1)"},
                    {'range': [55, 100], 'color': "rgba(34, 197, 94, 0.3)"}
                ]
            }
        ))
        fig_gauge.update_layout(template=CHART_THEME["template"], height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        importances = model_full.feature_importances_
        feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=True)
        fig_imp = go.Figure(go.Bar(x=feat_imp_df['Importance'], y=feat_imp_df['Feature'], orientation='h', marker_color=CHART_THEME["secondary"]))
        fig_imp.update_layout(title="Feature Importance", template=CHART_THEME["template"], height=250, xaxis_title="Weight", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)

# ============================== RISK PORTFOLIO & OPTIONS GREEKS ==============================
def render_portfolio_risk(is_crypto: bool, currency: str) -> None:
    st.subheader("10. MULTI-ASSET PORTFOLIO STRESS TEST (Risk Parity & Hist VaR)")
    basket = list(CRYPTO_ASSETS.values()) if is_crypto else list(INDIAN_ASSETS.values())[:3]
    basket_names = list(CRYPTO_ASSETS.keys()) if is_crypto else list(INDIAN_ASSETS.keys())[:3]
    
    data_dict, successful_names = {}, []
    for ticker, name in zip(basket, basket_names):
        df = fetch_data(ticker, period="2y", interval="1d", is_crypto=is_crypto)
        if df is not None and not df.empty:
            data_dict[ticker] = df['Close']
            successful_names.append(name.split(" ")[0])
            
    if len(data_dict) < 2: return st.warning("Insufficient portfolio data.")

    port_df = pd.DataFrame(data_dict).ffill().dropna()
    returns = np.log(port_df / port_df.shift(1)).dropna()
    std_devs = returns.std()
    inv_vol = 1.0 / std_devs
    weights = (inv_vol / inv_vol.sum()).values
    
    cov_matrix = returns.cov()
    port_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    hist_port_returns = returns.dot(weights)
    var_95 = abs(np.percentile(hist_port_returns, 5)) * 100
    weight_str = ", ".join([f"{n}: {w*100:.0f}%" for n, w in zip(successful_names, weights)])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Parity Weights", weight_str)
    c2.metric("Portfolio Annual Volatility", f"{port_std_dev*100:.2f}%")
    c3.metric("Historical Daily VaR (95%)", f"-{var_95:.2f}%", "Capital at Risk", delta_color="inverse")

def bs_greeks_advanced(S: float, K: float, T_days: int, r_pct: float, sigma_pct: float, q_pct: float) -> Dict[str, Dict[str, float]]:
    T, r, sigma, q = T_days / 365.0, r_pct / 100.0, sigma_pct / 100.0, q_pct / 100.0
    if T <= 0: T = 1e-5
    if sigma <= 0: sigma = 1e-5
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return {
        'Call': {
            'Price': S * np.exp(-q * T) * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2),
            'Delta': np.exp(-q * T) * si.norm.cdf(d1), 'Gamma': (np.exp(-q * T) * si.norm.pdf(d1)) / (S * sigma * np.sqrt(T)),
            'Theta': ((-S * si.norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) + q * S * si.norm.cdf(d1) * np.exp(-q * T) - r * K * np.exp(-r * T) * si.norm.cdf(d2)) / 365,
            'Vega': S * np.exp(-q * T) * si.norm.pdf(d1) * np.sqrt(T) / 100, 'Rho': K * T * np.exp(-r * T) * si.norm.cdf(d2) / 100
        },
        'Put': {
            'Price': K * np.exp(-r * T) * si.norm.cdf(-d2) - S * np.exp(-q * T) * si.norm.cdf(-d1),
            'Delta': np.exp(-q * T) * (si.norm.cdf(d1) - 1), 'Gamma': (np.exp(-q * T) * si.norm.pdf(d1)) / (S * sigma * np.sqrt(T)),
            'Theta': ((-S * si.norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) - q * S * si.norm.cdf(-d1) * np.exp(-q * T) + r * K * np.exp(-r * T) * si.norm.cdf(-d2)) / 365,
            'Vega': S * np.exp(-q * T) * si.norm.pdf(d1) * np.sqrt(T) / 100, 'Rho': -K * T * np.exp(-r * T) * si.norm.cdf(-d2) / 100
        }
    }

def render_options_greeks(selected_name: str, ticker: str, asset_class: str, is_crypto: bool) -> None:
    st.subheader("11. ADVANCED OPTIONS PRICING (Merton Extension)")
    asset_data, vix = fetch_data(ticker, period="5d", is_crypto=is_crypto), get_vix_data(asset_class, ticker, period="5d", is_crypto=is_crypto)
    tnx_data = fetch_data("^TNX", period="5d", is_crypto=False)
    live_r = safe_get_scalar(tnx_data['Close']) if tnx_data is not None else (7.0 if asset_class == "Indian Equities" else 4.5)
    
    if asset_data is None: return st.warning("Spot price data unavailable.")

    live_spot = safe_get_scalar(asset_data['Close'])
    live_iv = safe_get_scalar(vix['Close']) if vix is not None else 30.0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: spot = st.number_input("Spot Price", value=float(live_spot), step=10.0)
    with c2: strike = st.number_input("Strike (K)", value=float(round(live_spot/100)*100), step=100.0)
    with c3: dte = st.number_input("DTE", value=7, min_value=0)
    with c4: iv = st.number_input("Implied Vol (%)", value=float(live_iv), step=1.0)
    with c5: div = st.number_input("Div Yield (%)", value=0.0 if is_crypto else 1.2, step=0.1)

    st.caption(f"Risk-Free Rate dynamically locked at live 10Y Yield: **{live_r:.2f}%**")
    greeks = bs_greeks_advanced(spot, strike, dte, live_r, iv, div)

    c1, c2 = st.columns(2)
    def render_greek_card(title: str, data: Dict[str, float], color: str) -> None:
        st.markdown(f"#### <span style='color:{color}'>{title}</span>", unsafe_allow_html=True)
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Theo Price", f"${data['Price']:.2f}")
        rc2.metric("Delta", f"{data['Delta']:.4f}")
        rc3.metric("Gamma", f"{data['Gamma']:.4f}")
        rc4, rc5, rc6 = st.columns(3)
        rc4.metric("Theta", f"{data['Theta']:.2f}/day")
        rc5.metric("Vega", f"{data['Vega']:.4f}")
        rc6.metric("Rho", f"{data['Rho']:.4f}")

    with c1: render_greek_card("CALL OPTION", greeks['Call'], CHART_THEME["bullish"])
    with c2: render_greek_card("PUT OPTION", greeks['Put'], CHART_THEME["bearish"])

# ============================== ALADDIN UI ROUTER ==============================
def main() -> None:
    st.set_page_config(page_title="Aladdin-Quant Terminal", layout="wide", page_icon="⚡")

    if HAS_AUTOREFRESH:
        st_autorefresh(interval=60000, key="aladdin_refresh") 

    st.markdown("""
        <style>
            .stApp { background-color: #05070a; color: #D1D5DB; font-family: 'Inter', sans-serif; }
            .module-card { 
                background-color: #0f131a; border: 1px solid #1e293b; padding: 15px; 
                border-radius: 4px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            h1, h2, h3 { color: #F8FAFC !important; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
            .metric-label { color: #94a3b8; font-size: 11px; text-transform: uppercase; }
            .metric-value { color: #E2E8F0; font-size: 18px; font-weight: 600; font-family: 'Courier New', monospace; }
            div[data-testid="stMetricValue"] { color: #67e8f9; font-family: 'Courier New', monospace; font-weight: bold; }
            hr { margin-top: 2em; margin-bottom: 2em; border-color: #1e293b; }
            [data-testid="stVerticalBlockBorderWrapper"] {
                border: 1px solid #1e293b !important; background-color: #0f131a !important; border-radius: 4px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("🎛️ Terminal Settings")
        if st.button("🔄 Force Sync Market Data", use_container_width=True):
            st.session_state.market_data = {}
            st.cache_data.clear()
            st.rerun()
            
        st.divider()
        asset_class = st.radio("Asset Class", ["Indian Equities", "Crypto"])
        is_crypto = (asset_class == "Crypto")

        if not is_crypto:
            ASSET_DICT = INDIAN_ASSETS
            div1, div2, div1_name, div2_name, currency, trading_days = "^NSEI", "^NSEBANK", "Nifty 50", "Bank Nifty", "₹", 252
        else:
            ASSET_DICT = CRYPTO_ASSETS
            div1, div2, div1_name, div2_name, currency, trading_days = "BTC-USD", "ETH-USD", "Bitcoin", "Ethereum", "$", 365

        selected_name = st.selectbox("Target Asset", options=list(ASSET_DICT.keys()), index=0)
        ticker = ASSET_DICT[selected_name]

    col_h1, col_h2 = st.columns([4, 1])
    col_h1.title(f"ALADDIN // QUANT TERMINAL // {selected_name.upper()}")
    with col_h2:
        aladdin_metric("SYSTEM STATUS", "AUTO-SYNC ACTIVE" if HAS_AUTOREFRESH else "ONLINE", "OPTIMIZED")

    with st.spinner("Initializing complete Aladdin quantitative matrix..."):
        tab_row1_c1, tab_row1_c2 = st.columns([2, 1])
        with tab_row1_c1:
            with st.container(border=True): safe_render(render_realtime_chart, selected_name, ticker, is_crypto)
        with tab_row1_c2:
            with st.container(border=True): safe_render(render_nlp_sentiment, ticker, is_crypto)

        col_row2_1, col_row2_2, col_row2_3 = st.columns(3)
        with col_row2_1:
            with st.container(border=True): safe_render(render_volatility_metrics, asset_class, ticker, is_crypto)
        with col_row2_2:
            with st.container(border=True): safe_render(render_expected_move, selected_name, ticker, asset_class, currency, trading_days, is_crypto)
        with col_row2_3:
            with st.container(border=True): safe_render(render_index_divergence, div1, div2, div1_name, div2_name, currency, is_crypto)

        col_row3_1, col_row3_2 = st.columns(2)
        with col_row3_1:
            with st.container(border=True): safe_render(render_volatility_cone, selected_name, ticker, trading_days, is_crypto)
        with col_row3_2:
            with st.container(border=True): safe_render(render_vrp, selected_name, ticker, asset_class, trading_days, is_crypto)

        col_row4_1, col_row4_2 = st.columns(2)
        with col_row4_1:
            with st.container(border=True): safe_render(render_hurst_regime, selected_name, ticker, is_crypto)
        with col_row4_2:
            with st.container(border=True): safe_render(render_advanced_volatility, selected_name, ticker, trading_days, is_crypto)

        tab_row5_c1, tab_row5_c2 = st.columns([1, 2])
        with tab_row5_c1:
            with st.container(border=True): safe_render(render_market_synthesis, selected_name, ticker, asset_class, div1, div2, div1_name, div2_name, currency, trading_days, is_crypto)
        with tab_row5_c2:
            with st.container(border=True): safe_render(render_ml_engine, ticker, is_crypto)
                
        with st.container(border=True): safe_render(render_portfolio_risk, is_crypto, currency)
        with st.container(border=True): safe_render(render_options_greeks, selected_name, ticker, asset_class, is_crypto)

if __name__ == "__main__":
    main()