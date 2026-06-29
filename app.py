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
import re
import urllib.parse
import xml.etree.ElementTree as ET
import os
import yaml
import logging
from typing import Optional, Tuple, List, Dict, Any, Union, Callable

# ML IMPORTS
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from textblob import TextBlob

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

warnings.filterwarnings('ignore')

# ============================== LOGGING ==============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    "accent": "#a78bfa",
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
def markdown_to_html(text: str) -> str:
    return re.sub(r'\*\*(.*?)\*\*', r'<strong style="color:#67e8f9;font-weight:700;">\1</strong>', text)

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
        text="ALADDIN QUANT TERMINAL v22.0",
        xref="paper", yref="paper", x=0.99, y=0.01,
        showarrow=False, font=dict(size=9, color=CHART_THEME["watermark"]), opacity=0.5
    )

def section_header(number: str, title: str, icon: str = "◈") -> None:
    st.markdown(f"""
        <div class="section-header-wrap">
            <span class="section-num">{number}</span>
            <span class="section-icon">{icon}</span>
            <span class="section-title">{title}</span>
        </div>
    """, unsafe_allow_html=True)

def safe_render(func: Callable, *args, **kwargs) -> None:
    try:
        func(*args, **kwargs)
    except Exception as e:
        st.markdown(f'<div class="module-offline">⚠ MODULE OFFLINE — {func.__name__}</div>', unsafe_allow_html=True)
        logger.error(f"Component Failure in {func.__name__}: {str(e)}")

# ============================== DATA INGESTION (L1 + L2) ==============================
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_data_internal(ticker: str, period: str, interval: str, is_crypto: bool) -> Optional[pd.DataFrame]:
    if interval == '15m': period = '1mo'
    elif interval in ['1h', '4h'] and period in ['max', '5y', '10y', '2y']: period = '730d'

    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if (data is None or data.empty) and interval == "15m":
        data = yf.download(ticker, period="5d", interval=interval, progress=False)

    if data is None or data.empty: return None

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

# ============================== EXECUTIVE MARKET SUMMARY ==============================
def render_executive_summary(
    selected_name: str, ticker: str, asset_class: str,
    div1: str, div2: str, div1_name: str, div2_name: str,
    currency: str, trading_days: int, is_crypto: bool
) -> None:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_vix = executor.submit(get_vix_data, asset_class, ticker, "6mo", is_crypto)
        f_daily = executor.submit(fetch_data, ticker, "1y", "1d", is_crypto)
        f_div1 = executor.submit(fetch_data, div1, "1y", "1d", is_crypto)
        f_div2 = executor.submit(fetch_data, div2, "1y", "1d", is_crypto)
        vix_data_summ, daily_data_summ, d1_data_summ, d2_data_summ = (
            f_vix.result(), f_daily.result(), f_div1.result(), f_div2.result()
        )

    if any(d is None for d in [vix_data_summ, daily_data_summ, d1_data_summ, d2_data_summ]):
        st.warning("Synthesis data incomplete.")
        return

    current_vix = safe_get_scalar(vix_data_summ['Close'])
    vix_min = safe_get_scalar(vix_data_summ['Close'].min())
    vix_max = safe_get_scalar(vix_data_summ['Close'].max())
    ivr = ((current_vix - vix_min) / (vix_max - vix_min) * 100) if (vix_max - vix_min) != 0 else 0.0

    hv_20 = np.log(daily_data_summ['Close'] / daily_data_summ['Close'].shift(1)).rolling(20).std() * np.sqrt(trading_days) * 100
    vrp_val = current_vix - safe_get_scalar(hv_20)

    data_div = pd.merge(
        d1_data_summ['Close'].to_frame('A'), d2_data_summ['Close'].to_frame('B'),
        left_index=True, right_index=True, how='outer'
    ).ffill().dropna()
    
    if data_div.empty:
        corr_val = 0.5
    else:
        corr_val = safe_get_scalar(
            np.log(data_div / data_div.shift(1)).dropna()['A']
            .rolling(20).corr(np.log(data_div / data_div.shift(1)).dropna()['B']).dropna(),
            default=0.5
        )

    df_metrics = daily_data_summ.copy()
    df_metrics['EMA_89'] = df_metrics['Close'].ewm(span=89, adjust=False).mean()
    df_metrics['EMA_21'] = df_metrics['Close'].ewm(span=21, adjust=False).mean()
    df_metrics['RSI'] = calculate_rsi(df_metrics['Close'], 14)

    last_close = safe_get_scalar(df_metrics['Close'])
    last_ema89 = safe_get_scalar(df_metrics['EMA_89'])
    last_ema21 = safe_get_scalar(df_metrics['EMA_21'])
    last_rsi = safe_get_scalar(df_metrics['RSI'])

    if last_close > last_ema89 and last_ema89 > last_ema21:
        trend_narrative = f"exhibiting an active **Bullish Expansion Structure**. Spot is comfortably elevated above both the momentum 89-period EMA ({currency}{last_ema89:,.2f}) and the intermediate 21-period EMA ({currency}{last_ema21:,.2f}), confirming sustainable upward velocity across timeframes."
        trend_signal, trend_color = "BULLISH EXPANSION", CHART_THEME['bullish']
    elif last_close < last_ema89 and last_ema89 < last_ema21:
        trend_narrative = f"stuck in a strong **Bearish Markdown Sequence**. Price action remains structurally pinned beneath the descending 89-period EMA ({currency}{last_ema89:,.2f}) and 21-period EMA ({currency}{last_ema21:,.2f}), alerting option buyers to step carefully."
        trend_signal, trend_color = "BEARISH MARKDOWN", CHART_THEME['bearish']
    else:
        trend_narrative = f"experiencing a **Mean Reversion / Consolidation Phase**. Spot pricing is weaving through its 89-period and 21-period EMAs, showing range containment ahead of any directional breakout."
        trend_signal, trend_color = "CONSOLIDATION", CHART_THEME['secondary']

    if last_rsi > 70:
        rsi_narrative = f"The 14-period RSI reads **{last_rsi:.1f}** — **overbought** territory, flagging momentum exhaustion risk."
        rsi_signal, rsi_color = "OVERBOUGHT", CHART_THEME['bearish']
    elif last_rsi < 30:
        rsi_narrative = f"The 14-period RSI sits at a washed-out **{last_rsi:.1f}**, signalling **seller capitulation** and potential reversal setups."
        rsi_signal, rsi_color = "OVERSOLD", CHART_THEME['bullish']
    else:
        rsi_narrative = f"The 14-period RSI is balanced at **{last_rsi:.1f}**, providing clean runway for linear expansions in either direction."
        rsi_signal, rsi_color = "NEUTRAL", CHART_THEME['primary']

    if vrp_val > 0:
        vol_narrative = f"Implied parameters are **overpricing** realized movements (VRP: **{vrp_val:+.2f}%**, IVR: **{ivr:.1f}%**). Structural edge favours **premium sellers** — **credit spreads**, **covered writes**, or **short straddles**."
        vol_signal, vol_color = "SELL PREMIUM", CHART_THEME['secondary']
    else:
        vol_narrative = f"Implied vol is **underpricing** historical risk (VRP: **{vrp_val:+.2f}%**, IVR: **{ivr:.1f}%**). Options premium is cheap — structural advantage for **directional buyers** and **long gamma** strategies."
        vol_signal, vol_color = "BUY OPTIONS", CHART_THEME['primary']

    corr_text = "**Unified macro-driven capital flows** confirm high index-wide systematic risk." if corr_val > 0.5 else "**Fragmented, independent asset movement** — diversification is currently effective."

    # Radar Chart & Summary Layout
    norm_ivr = min(ivr / 100, 1.0)
    norm_corr = max(0, min(corr_val, 1.0))
    norm_vrp = max(0, min((vrp_val + 5) / 10, 1.0))
    categories = ['IV Rank', 'Correlation', 'VRP Premium']

    col_radar, col_summary = st.columns([1, 2.5])
    
    with col_radar:
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[norm_ivr, norm_corr, norm_vrp, norm_ivr],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(103,232,249,0.08)',
            line=dict(color=CHART_THEME["primary"], width=2)
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=False, range=[0, 1]),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.06)', tickfont=dict(size=10, color='#94a3b8'))
            ),
            showlegend=False,
            template=CHART_THEME["template"],
            title=dict(text="REGIME PROFILE", font=dict(size=10, color='#64748b'), x=0.5),
            height=220,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=10)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_summary:
        st.markdown(f"""
            <div class="exec-summary-card" style="margin-top: 0px;">
                <div class="exec-summary-header">
                    <span class="exec-dot"></span>
                    EXECUTIVE MARKET SUMMARY — {selected_name.upper()}
                </div>
                <div class="exec-grid">
                    <div class="exec-block">
                        <div class="exec-block-label">PRICE MOMENTUM STRUCTURE</div>
                        <div class="exec-signal" style="color:{trend_color};">{trend_signal}</div>
                        <div class="exec-block-text">{markdown_to_html(trend_narrative)}</div>
                        <div class="exec-sub" style="color:{rsi_color};">RSI STATUS: {rsi_signal}</div>
                        <div class="exec-block-text">{markdown_to_html(rsi_narrative)}</div>
                    </div>
                    <div class="exec-block">
                        <div class="exec-block-label">VOLATILITY & OPTIONS STRATEGY</div>
                        <div class="exec-signal" style="color:{vol_color};">{vol_signal}</div>
                        <div class="exec-block-text">{markdown_to_html(vol_narrative)}</div>
                    </div>
                    <div class="exec-block">
                        <div class="exec-block-label">INTERMARKET CORRELATION</div>
                        <div class="exec-signal" style="color:{'#22c55e' if corr_val > 0.5 else '#67e8f9'};">
                            {div1_name} / {div2_name}: {corr_val:.3f}
                        </div>
                        <div class="exec-block-text">{markdown_to_html(corr_text)}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ============================== NLP NEWS SENTIMENT ENGINE ==============================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_news_sentiment(ticker: str, is_crypto: bool) -> Tuple[Optional[float], Optional[float], List[Dict]]:
    news = []
    try:
        tkr = yf.Ticker(ticker)
        news = tkr.news
    except Exception: pass

    if not news or not isinstance(news, list) or len(news) == 0:
        try:
            search_res = yf.Search(ticker, news_count=8)
            news = search_res.news
        except Exception: pass

    if not news or not isinstance(news, list) or len(news) == 0:
        try:
            clean_ticker = ticker.replace('^', '').replace('.NS', '')
            query_suffix = "crypto" if is_crypto else "stock market"
            search_query = urllib.parse.quote(f"{clean_ticker} {query_suffix} news")
            url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                news = []
                for item in root.findall('./channel/item')[:8]:
                    title_elem = item.find('title')
                    source_elem = item.find('source')
                    if title_elem is not None:
                        news.append({
                            'title': title_elem.text,
                            'publisher': source_elem.text if source_elem is not None else 'Google News'
                        })
        except Exception as e:
            logger.error(f"Google News RSS Fallback failed for {ticker}: {e}")

    if not news or not isinstance(news, list) or len(news) == 0:
        return None, None, []

    total_polarity, total_subjectivity = 0.0, 0.0
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

        if polarity > 0.05: tag, color = "BULLISH", CHART_THEME['bullish']
        elif polarity < -0.05: tag, color = "BEARISH", CHART_THEME['bearish']
        else: tag, color = "NEUTRAL", CHART_THEME['secondary']

        analyzed_headlines.append({'title': title, 'publisher': publisher, 'tag': tag, 'color': color, 'polarity': polarity})

    count = len(analyzed_headlines)
    if count == 0: return None, None, []

    avg_polarity = (total_polarity / count) * 100
    avg_subjectivity = (total_subjectivity / count) * 100
    return avg_polarity, avg_subjectivity, analyzed_headlines

def render_nlp_sentiment(ticker: str, is_crypto: bool) -> None:
    section_header("", "NLP NEWS SENTIMENT", "◈")
    score_data = fetch_news_sentiment(ticker, is_crypto)
    if score_data[0] is None:
        st.warning("No recent news context found. IP Rate Limited.")
        return

    score, subjectivity, headlines = score_data
    gauge_color = CHART_THEME['bullish'] if score > 10 else (CHART_THEME['bearish'] if score < -10 else CHART_THEME['secondary'])

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=score, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "TextBlob Financial Sentiment", 'font': {'size': 12, 'color': '#94a3b8'}},
        number={'font': {'color': gauge_color, 'size': 28}},
        gauge={
            'axis': {'range': [-100, 100], 'tickcolor': '#334155', 'tickfont': {'size': 9}},
            'bar': {'color': gauge_color, 'thickness': 0.25},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [-100, -15], 'color': "rgba(239, 68, 68, 0.15)"},
                {'range': [-15, 15], 'color': "rgba(255, 255, 255, 0.04)"},
                {'range': [15, 100], 'color': "rgba(34, 197, 94, 0.15)"}
            ],
            'threshold': {'line': {'color': gauge_color, 'width': 2}, 'thickness': 0.75, 'value': score}
        }
    ))
    fig_gauge.update_layout(
        template=CHART_THEME["template"], height=190,
        margin=dict(l=20, r=20, t=35, b=5),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Net Bias Score", f"{score:+.1f}")
    c2.metric("Media Subjectivity", f"{subjectivity:.1f}%")

    st.markdown('<div class="news-section-label">LIVE HEADLINES & POLARITY</div>', unsafe_allow_html=True)
    for h in headlines:
        st.markdown(f"""
            <div class="news-headline" style="border-left: 3px solid {h['color']};">
                <div class="news-title">{h['title']}</div>
                <div class="news-meta" style="color:{h['color']};">
                    [{h['tag']} · {h['polarity']:+.2f}] · <span style="color:#475569;">{h['publisher']}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ============================== PRE-MARKET & POST-MARKET TABS ==============================
def render_pre_market_analysis(ticker: str, is_crypto: bool, asset_class: str, currency: str) -> None:
    section_header("PM1", "AUTOMATED PRE-MARKET PLAN & MACRO SYNTHESIS", "◈")
    if is_crypto:
        st.info("Crypto markets operate 24/7. Pre-market concepts are mapped to a 00:00 UTC rollover context.")

    # Fetch macro indicators to judge overnight sentiment
    macro_tnx = fetch_data("^TNX", period="5d", is_crypto=False)
    macro_dxy = fetch_data("DX-Y.NYB", period="5d", is_crypto=False)
    asset_data = fetch_data(ticker, period="1mo", is_crypto=is_crypto)

    tnx_val = safe_get_scalar(macro_tnx['Close']) if macro_tnx is not None else 0.0
    dxy_val = safe_get_scalar(macro_dxy['Close']) if macro_dxy is not None else 0.0
    
    if macro_tnx is not None and len(macro_tnx) >= 2:
        tnx_change = (macro_tnx['Close'].iloc[-1] - macro_tnx['Close'].iloc[-2]) / macro_tnx['Close'].iloc[-2] * 100
    else: tnx_change = 0.0

    if macro_dxy is not None and len(macro_dxy) >= 2:
        dxy_change = (macro_dxy['Close'].iloc[-1] - macro_dxy['Close'].iloc[-2]) / macro_dxy['Close'].iloc[-2] * 100
    else: dxy_change = 0.0
    
    if asset_data is not None and not asset_data.empty:
        prev_close = asset_data['Close'].iloc[-2] if len(asset_data) > 1 else asset_data['Close'].iloc[-1]
        prev_high = asset_data['High'].iloc[-2] if len(asset_data) > 1 else asset_data['High'].iloc[-1]
        prev_low = asset_data['Low'].iloc[-2] if len(asset_data) > 1 else asset_data['Low'].iloc[-1]
        
        last_close = asset_data['Close'].iloc[-1]
        gap_pct = ((last_close - prev_close) / prev_close) * 100
        pivot = (prev_high + prev_low + prev_close) / 3

        # Macro Bias Logic
        if tnx_change > 0.5 and dxy_change > 0.2:
            macro_bias = "BEARISH HEADWINDS"
            macro_color = CHART_THEME['bearish']
            macro_desc = "Rising US Yields and a strong Dollar are historically toxic for risk assets. Expect pressure on long positions."
        elif tnx_change < -0.5 and dxy_change < -0.2:
            macro_bias = "BULLISH TAILWINDS"
            macro_color = CHART_THEME['bullish']
            macro_desc = "Falling US Yields and a weak Dollar provide a highly supportive environment for risk assets and equities."
        else:
            macro_bias = "NEUTRAL / MIXED"
            macro_color = CHART_THEME['secondary']
            macro_desc = "Macro cross-currents are mixed. The asset will likely trade on its own technicals and intrinsic catalysts today."

        # Gap Logic
        if gap_pct > 0.5: gap_bias = f"Gap Up (+{gap_pct:.2f}%). Watch for gap-and-go continuation or morning profit-taking."
        elif gap_pct < -0.5: gap_bias = f"Gap Down ({gap_pct:.2f}%). Watch for panic selling or a quick mean-reversion bounce."
        else: gap_bias = f"Flat Open ({gap_pct:+.2f}%). Expect initial chop as the market establishes the morning range."

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Previous Close", f"{currency}{prev_close:,.2f}")
        c2.metric("Overnight Gap (Proxy)", f"{gap_pct:+.2f}%", delta_color="normal")
        c3.metric("US 10Y Yield", f"{tnx_val:.3f}%", f"{tnx_change:+.2f}%", delta_color="inverse")
        c4.metric("DXY Index", f"{dxy_val:.2f}", f"{dxy_change:+.2f}%", delta_color="inverse")
        
        st.divider()

        # Automated Pre-Market Synthesis (Replaced HTML grids with native Streamlit Columns)
        st.markdown("#### 🤖 AI PRE-MARKET SYNTHESIS & TRADE PLAN")
        pm_col1, pm_col2 = st.columns(2)
        
        with pm_col1:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02); padding:15px; border-radius:4px; border:1px solid #111c2e; height: 100%;">
                <div style="font-size:10px; color:#64748b; letter-spacing:1px; text-transform:uppercase; margin-bottom:5px;">Global Macro Regime</div>
                <div style="font-size:16px; font-weight:700; color:{macro_color}; margin-bottom:8px;">{macro_bias}</div>
                <div style="font-size:12px; color:#94a3b8; line-height:1.5;">{macro_desc}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with pm_col2:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02); padding:15px; border-radius:4px; border:1px solid #111c2e; height: 100%;">
                <div style="font-size:10px; color:#64748b; letter-spacing:1px; text-transform:uppercase; margin-bottom:5px;">Opening Action Plan</div>
                <div style="font-size:14px; font-weight:700; color:#E2E8F0; margin-bottom:8px;">{gap_bias}</div>
                <div style="font-size:12px; color:#94a3b8; line-height:1.5;">
                    <strong>Bull Trigger:</strong> Sustained trading above Resistance at {currency}{prev_high:,.2f}<br>
                    <strong>Bear Trigger:</strong> Breakdown below Support at {currency}{prev_low:,.2f}<br>
                    <strong>Daily Pivot:</strong> {currency}{pivot:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("") # Spacer
        render_nlp_sentiment(ticker, is_crypto)
    else:
        st.warning("Pre-market asset data unavailable.")

def render_post_market_analysis(ticker: str, is_crypto: bool, asset_class: str, currency: str, trading_days: int) -> None:
    section_header("PM2", "AUTOMATED END-OF-DAY WRAP & VERDICT", "◈")
    asset_data = fetch_data(ticker, period="3mo", is_crypto=is_crypto)
    
    if asset_data is not None and not asset_data.empty:
        today = asset_data.iloc[-1]
        yest = asset_data.iloc[-2] if len(asset_data) > 1 else today
        
        # Calculate Average True Range (ATR) safely
        high_low = asset_data['High'] - asset_data['Low']
        high_close = np.abs(asset_data['High'] - asset_data['Close'].shift())
        low_close = np.abs(asset_data['Low'] - asset_data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr_20 = true_range.rolling(20).mean().iloc[-1] if len(true_range) >= 20 else true_range.mean()
        
        day_range = today['High'] - today['Low']
        if pd.isna(atr_20) or atr_20 == 0:
            atr_20 = day_range if day_range > 0 else 1e-5
            
        range_pct_atr = (day_range / atr_20) * 100
        
        # Safely handle missing/zero Volume (like Nifty 50)
        has_volume = 'Volume' in asset_data.columns and not (asset_data['Volume'] == 0).all()
        vol_20 = asset_data['Volume'].rolling(20).mean().iloc[-1] if has_volume else 0.0
        day_vol = today['Volume'] if has_volume else 0.0
        vol_pct = (day_vol / vol_20) * 100 if (has_volume and vol_20 > 0) else 0.0

        day_return = ((today['Close'] - yest['Close']) / yest['Close']) * 100

        # EOD Analysis Logic
        # 1. Candle Shape (Where did it close relative to its range?)
        if day_range == 0: candle_close_pct = 0.5
        else: candle_close_pct = (today['Close'] - today['Low']) / day_range

        if candle_close_pct > 0.7:
            candle_bias = "Strong Bullish Close"
            candle_desc = "Buyers seized absolute control into the close, shutting down sellers near the highs. High probability of continuation."
            c_color = CHART_THEME['bullish']
        elif candle_close_pct < 0.3:
            candle_bias = "Strong Bearish Close"
            candle_desc = "Aggressive selling into the bell. The asset closed near its lows, indicating trapped buyers and downside risk."
            c_color = CHART_THEME['bearish']
        else:
            candle_bias = "Indecision / Doji"
            candle_desc = "Price closed near the middle of the daily range. The market is waiting for a catalyst to break the equilibrium."
            c_color = CHART_THEME['secondary']

        # 2. Volume & Conviction Profile
        if not has_volume or vol_20 == 0 or pd.isna(vol_20):
            vol_conviction = "DATA UNAVAILABLE"
            vol_desc = "Volume data is not consistently reported for this index/asset by the current feed."
        elif vol_pct > 120:
            vol_conviction = "HIGH CONVICTION"
            if day_return > 0: vol_desc = "Institutional Accumulation. Heavy volume supporting the upward move."
            else: vol_desc = "Institutional Distribution. Heavy volume confirming the sell-off."
        elif vol_pct < 80:
            vol_conviction = "LOW CONVICTION"
            vol_desc = "Retail-driven or algorithmic drift. Lack of heavy institutional participation."
        else:
            vol_conviction = "AVERAGE VOLUME"
            vol_desc = "Standard daily rotation. No extreme flow anomalies detected."

        # 3. Volatility Profile
        if range_pct_atr > 120:
            atr_bias = "Trend Expansion"
            atr_desc = f"The day's range ({currency}{day_range:,.2f}) vastly exceeded the 20-day average. Volatility is expanding."
        elif range_pct_atr < 80:
            atr_bias = "Volatility Compression"
            atr_desc = f"A tight, compressed trading range. Expect an explosive directional breakout soon."
        else:
            atr_bias = "Normal Range"
            atr_desc = "Price action moved within expected historical bounds."

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EOD Close", f"{currency}{today['Close']:,.2f}", f"{day_return:+.2f}%")
        c2.metric("Day's Range", f"{currency}{day_range:,.2f}")
        c3.metric("Range vs ATR(20)", f"{range_pct_atr:.1f}%")
        c4.metric("Volume vs Avg(20)", f"{vol_pct:.1f}%" if has_volume and vol_pct > 0 else "N/A")
        
        st.divider()

        # Automated Post-Market Synthesis (Replaced HTML grids with native Streamlit Columns)
        st.markdown("#### 🌃 AI POST-MARKET E.O.D. VERDICT")
        eod_col1, eod_col2, eod_col3 = st.columns(3)
        
        with eod_col1:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.015); padding:15px; border-radius:4px; border:1px solid #111c2e; height: 100%;">
                <div style="font-size:10px; color:#64748b; letter-spacing:1px; text-transform:uppercase; margin-bottom:5px;">Price Action</div>
                <div style="font-size:14px; font-weight:700; color:{c_color}; margin-bottom:8px;">{candle_bias}</div>
                <div style="font-size:11px; color:#94a3b8; line-height:1.4;">{candle_desc}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with eod_col2:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.015); padding:15px; border-radius:4px; border:1px solid #111c2e; height: 100%;">
                <div style="font-size:10px; color:#64748b; letter-spacing:1px; text-transform:uppercase; margin-bottom:5px;">Volume Flow</div>
                <div style="font-size:14px; font-weight:700; color:#E2E8F0; margin-bottom:8px;">{vol_conviction}</div>
                <div style="font-size:11px; color:#94a3b8; line-height:1.4;">{vol_desc}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with eod_col3:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.015); padding:15px; border-radius:4px; border:1px solid #111c2e; height: 100%;">
                <div style="font-size:10px; color:#64748b; letter-spacing:1px; text-transform:uppercase; margin-bottom:5px;">ATR State</div>
                <div style="font-size:14px; font-weight:700; color:#E2E8F0; margin-bottom:8px;">{atr_bias}</div>
                <div style="font-size:11px; color:#94a3b8; line-height:1.4;">{atr_desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("") # Spacer
        col1, col2 = st.columns(2)
        with col1:
            render_expected_move(ticker, ticker, asset_class, currency, trading_days, is_crypto)
        with col2:
            render_hurst_regime(ticker, ticker, is_crypto)
    else:
        st.warning("Post-market data unavailable.")

# ============================== REALTIME CHART: EMA 89 & 21 ==============================
def render_realtime_chart(selected_name: str, ticker: str, is_crypto: bool) -> None:
    section_header("", "MARKET PRICE · VOLUME · MOMENTUM", "◈")
    timeframe = st.radio("Timeframe", ["15m", "1h", "4h", "1d"], index=1, horizontal=True, label_visibility="collapsed")
    period, interval = (
        ("1mo", "15m") if timeframe == "15m"
        else ("730d", "1h") if timeframe in ["1h", "4h"]
        else ("2y", "1d")
    )

    data = fetch_data(ticker, period=period, interval=interval, is_crypto=is_crypto)
    if data is None or data.empty:
        st.warning("Real-time data unavailable.")
        return

    if timeframe == "4h":
        data = data.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

    data = data.tail(300 if timeframe in ["15m", "1h", "4h"] else 252).copy()

    if 'Volume' in data.columns and data['Volume'].sum() > 0:
        data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3
        data['VP'] = data['Typical_Price'] * data['Volume']
        grouper = data.index.date if timeframe in ['15m', '1h', '4h'] else data.index.to_period('M')
        data['VWAP'] = data.groupby(grouper)['VP'].cumsum() / data.groupby(grouper)['Volume'].cumsum()
    else:
        data['VWAP'] = np.nan

    data['EMA_89'] = data['Close'].ewm(span=89, adjust=False).mean()
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    data['RSI'] = calculate_rsi(data['Close'], 14)

    data['Prev_High'] = data['High'].rolling(20).max().shift(1)
    data['Prev_Low'] = data['Low'].rolling(20).min().shift(1)
    data['Supply_Sweep'] = (data['High'] > data['Prev_High']) & (data['Close'] < data['Prev_High'])
    data['Demand_Sweep'] = (data['Low'] < data['Prev_Low']) & (data['Close'] > data['Prev_Low'])

    last_close = safe_get_scalar(data['Close'])
    x_format = '%Y-%m-%d' if timeframe == "1d" else '%Y-%m-%d %H:%M'
    x_axis_string = data.index.strftime(x_format)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.15, 0.20], vertical_spacing=0.03
    )

    fig.add_trace(go.Candlestick(
        x=x_axis_string, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
        name='Price', increasing_line_color=CHART_THEME['bullish'], decreasing_line_color=CHART_THEME['bearish'],
        increasing_fillcolor=CHART_THEME['bullish'], decreasing_fillcolor=CHART_THEME['bearish']
    ), row=1, col=1)

    if not data['EMA_89'].isna().all():
        fig.add_trace(go.Scatter(
            x=x_axis_string, y=data['EMA_89'], mode='lines', name='EMA 89',
            line=dict(color='#fbbf24', width=1.5)
        ), row=1, col=1)

    if not data['EMA_21'].isna().all():
        fig.add_trace(go.Scatter(
            x=x_axis_string, y=data['EMA_21'], mode='lines', name='EMA 21',
            line=dict(color='#a78bfa', width=2)
        ), row=1, col=1)

    if not data['VWAP'].isna().all():
        fig.add_trace(go.Scatter(
            x=x_axis_string, y=data['VWAP'], mode='lines', name='VWAP',
            line=dict(color='#e2e8f0', width=1.5, dash='dot')
        ), row=1, col=1)

    supply_data = data[data['Supply_Sweep']]
    demand_data = data[data['Demand_Sweep']]

    if not supply_data.empty:
        fig.add_trace(go.Scatter(
            x=supply_data.index.strftime(x_format), y=supply_data['High'] * 1.002, mode='markers',
            marker=dict(symbol='triangle-down', size=11, color=CHART_THEME["bearish"], line=dict(width=1, color='white')),
            name='Supply Sweep'
        ), row=1, col=1)

    if not demand_data.empty:
        fig.add_trace(go.Scatter(
            x=demand_data.index.strftime(x_format), y=demand_data['Low'] * 0.998, mode='markers',
            marker=dict(symbol='triangle-up', size=11, color=CHART_THEME["bullish"], line=dict(width=1, color='white')),
            name='Demand Sweep'
        ), row=1, col=1)

    fig.add_hline(
        y=last_close, line_dash="dot", line_color=CHART_THEME["primary"], line_width=1.5,
        annotation_text=f"  {last_close:,.2f}", annotation_position="right",
        annotation_font_color=CHART_THEME["primary"], row=1, col=1
    )

    if 'Volume' in data.columns and not (data['Volume'] == 0).all():
        colors = [CHART_THEME['bullish'] if row['Close'] >= row['Open'] else CHART_THEME['bearish'] for _, row in data.iterrows()]
        fig.add_trace(go.Bar(
            x=x_axis_string, y=data['Volume'], name='Volume',
            marker_color=colors, opacity=0.75, marker_line_width=0
        ), row=2, col=1)

    if not data['RSI'].isna().all():
        fig.add_trace(go.Scatter(
            x=x_axis_string, y=data['RSI'], mode='lines', name='RSI 14',
            line=dict(color=CHART_THEME['primary'], width=1.5)
        ), row=3, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.03)", line_width=0, row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=CHART_THEME['bearish'], line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=CHART_THEME['bullish'], line_width=1, row=3, col=1)

    grid_cfg = dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.04)')
    fig.update_layout(
        template=CHART_THEME['template'], height=660, xaxis_rangeslider_visible=False,
        hovermode='x unified', bargap=0, bargroupgap=0,
        margin=dict(l=10, r=65, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True, legend=dict(
            orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0,
            font=dict(size=10, color='#64748b'), bgcolor='rgba(0,0,0,0)', borderwidth=0
        )
    )

    fig.update_xaxes(type='category', **grid_cfg, showticklabels=False, row=1, col=1)
    fig.update_xaxes(type='category', **grid_cfg, showticklabels=False, row=2, col=1)
    fig.update_xaxes(
        type='category', categoryorder='category ascending', **grid_cfg,
        showticklabels=True, showspikes=True, spikemode='across',
        spikethickness=1, spikedash='dot', spikecolor='rgba(255,255,255,0.2)', row=3, col=1
    )
    fig.update_yaxes(**grid_cfg)
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
    section_header("1", "IMPLIED VOLATILITY RANK", "◈")
    vix_name = "India VIX" if asset_class == "Indian Equities" else "Synthetic IV (30D HV)"
    data = get_vix_data(asset_class, ticker, period="1y", is_crypto=is_crypto)
    if data is None:
        st.warning(f"{vix_name} data unavailable.")
        return

    close = data['Close']
    current_iv = safe_get_scalar(close)
    high_52w = safe_get_scalar(close.max())
    low_52w = safe_get_scalar(close.min())
    denom = high_52w - low_52w
    ivr = ((current_iv - low_52w) / denom * 100) if denom != 0 else 0.0
    ivp = (close[close < current_iv].count() / len(close)) * 100 if len(close) > 0 else 0.0
    regime = "HIGH VOL" if ivr > CONFIG.get('thresholds', 'ivr_high') else "LOW VOL"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=close.index, y=close, mode='lines', name=vix_name,
        line=dict(color=CHART_THEME["primary"], width=2),
        fill='tozeroy', fillcolor='rgba(103, 232, 249, 0.04)'
    ))
    fig.add_hline(y=high_52w, line_dash="dash", line_color=CHART_THEME["bearish"],
                  annotation_text=f"52W H: {high_52w:.1f}", annotation_font_color=CHART_THEME["bearish"])
    fig.add_hline(y=low_52w, line_dash="dash", line_color=CHART_THEME["bullish"],
                  annotation_text=f"52W L: {low_52w:.1f}", annotation_font_color=CHART_THEME["bullish"])
    fig.add_hline(y=current_iv, line_color=CHART_THEME["neutral"], line_width=2.5)
    fig.update_layout(
        template=CHART_THEME["template"], height=270,
        margin=dict(t=10, b=10, l=0, r=80),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified', showlegend=False
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.04)')
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("IV Rank", f"{ivr:.1f}%")
    c2.metric("IV Percentile", f"{ivp:.1f}%")
    c3.metric("Regime", regime)

def render_expected_move(selected_name: str, ticker: str, asset_class: str, currency: str, trading_days: int, is_crypto: bool) -> None:
    section_header("2", "EXPECTED MOVE (1σ)", "◈")
    asset_data = fetch_data(ticker, period="1mo", is_crypto=is_crypto)
    vix = get_vix_data(asset_class, ticker, period="5d", is_crypto=is_crypto)
    if asset_data is None or vix is None:
        st.warning("Data fetch failed for Expected Move.")
        return

    spot = safe_get_scalar(asset_data['Close'])
    current_vix = safe_get_scalar(vix['Close'])
    exp_move = spot * ((current_vix / 100) * np.sqrt(1 / trading_days))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=asset_data.index, y=asset_data['Close'], mode='lines', name=selected_name,
        line=dict(color=CHART_THEME["primary"], width=2.5)
    ))
    tomorrow = asset_data.index[-1] + timedelta(days=1)
    fig.add_trace(go.Scatter(
        x=[asset_data.index[-1], tomorrow, tomorrow, asset_data.index[-1]],
        y=[spot, spot + exp_move, spot - exp_move, spot],
        fill='toself', fillcolor='rgba(251, 191, 36, 0.10)',
        line=dict(color='rgba(255,255,255,0)'), name='1σ Range', showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[tomorrow], y=[spot + exp_move], mode='markers+text', name='+1σ',
        marker=dict(color=CHART_THEME["bullish"], size=10, symbol='triangle-up'),
        text=[f"+{exp_move:,.0f}"], textposition="top right",
        textfont=dict(color=CHART_THEME["bullish"], size=11)
    ))
    fig.add_trace(go.Scatter(
        x=[tomorrow], y=[spot - exp_move], mode='markers+text', name='-1σ',
        marker=dict(color=CHART_THEME["bearish"], size=10, symbol='triangle-down'),
        text=[f"-{exp_move:,.0f}"], textposition="bottom right",
        textfont=dict(color=CHART_THEME["bearish"], size=11)
    ))
    fig.update_layout(
        template=CHART_THEME["template"], height=270,
        margin=dict(t=10, b=10, l=0, r=40),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified', showlegend=False
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.04)')
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Spot Price", f"{currency}{spot:,.2f}")
    c2.metric("Vol Benchmark", f"{current_vix:.2f}%")
    c3.metric("Implied Move", f"± {currency}{exp_move:,.1f}")

def render_index_divergence(div1: str, div2: str, name1: str, name2: str, currency: str, is_crypto: bool) -> None:
    section_header("3", f"SYSTEMIC DIVERGENCE: {name1} vs {name2}", "◈")
    d1 = fetch_data(div1, period="1y", is_crypto=is_crypto)
    d2 = fetch_data(div2, period="1y", is_crypto=is_crypto)
    if d1 is None or d2 is None:
        st.warning("Divergence data unavailable.")
        return

    data = pd.merge(
        d1['Close'].to_frame(name1), d2['Close'].to_frame(name2),
        left_index=True, right_index=True, how='outer'
    ).ffill().dropna()
    if data.empty:
        st.warning("Insufficient overlapping data.")
        return

    normalized = (data / data.iloc[0]) * 100
    log_ret = np.log(data / data.shift(1)).dropna()
    rolling_corr = log_ret[name1].rolling(20).corr(log_ret[name2]).dropna()
    current_corr = safe_get_scalar(rolling_corr)

    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[name1], name=name1, line=dict(color=CHART_THEME["primary"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[name2], name=name2, line=dict(color=CHART_THEME["secondary"])), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=rolling_corr.index, y=rolling_corr, name='20D Corr',
        line=dict(color=CHART_THEME["accent"], width=1.5),
        fill='tozeroy', fillcolor='rgba(167,139,250,0.08)'
    ), row=2, col=1)
    fig.update_layout(
        template=CHART_THEME["template"], height=320,
        margin=dict(t=5, b=10, l=0, r=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified', legend=dict(font=dict(size=10, color='#64748b'), bgcolor='rgba(0,0,0,0)')
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.04)')
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric(name1, f"{currency}{safe_get_scalar(data[name1]):,.2f}")
    c2.metric(name2, f"{currency}{safe_get_scalar(data[name2]):,.2f}")
    c3.metric("20D Correlation", f"{current_corr:.3f}")

def render_volatility_cone(selected_name: str, ticker: str, trading_days: int, is_crypto: bool) -> None:
    section_header("4", "VOLATILITY TERM STRUCTURE (CONE)", "◈")
    data = fetch_data(ticker, period="max", is_crypto=is_crypto)
    if data is None:
        st.warning("Data unavailable.")
        return

    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    windows = [10, 20, 30, 60, 90, 120, 180, trading_days]
    stats = []
    for w in windows:
        if w > len(returns): continue
        vol_series = returns.rolling(w).std() * np.sqrt(trading_days) * 100
        stats.append({
            'window': w, 'max': safe_get_scalar(vol_series.max()),
            'min': safe_get_scalar(vol_series.min()),
            'median': safe_get_scalar(vol_series.median()),
            'current': safe_get_scalar(vol_series.iloc[-1])
        })
    df_stats = pd.DataFrame(stats)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_stats['window'], y=df_stats['max'], name='Max Vol', mode='lines+markers', line=dict(color=CHART_THEME["bearish"])))
    fig.add_trace(go.Scatter(x=df_stats['window'], y=df_stats['min'], name='Min Vol', mode='lines+markers', line=dict(color=CHART_THEME["bullish"])))
    fig.add_trace(go.Scatter(x=df_stats['window'], y=df_stats['median'], name='Median Vol', mode='lines+markers', line=dict(color=CHART_THEME["neutral"], dash='dash')))
    fig.add_trace(go.Scatter(x=df_stats['window'], y=df_stats['current'], name='Current Vol', mode='lines+markers', line=dict(color=CHART_THEME["primary"], width=3.5)))
    fig.update_layout(
        template=CHART_THEME["template"], height=280,
        margin=dict(t=5, b=10, l=0, r=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', font=dict(size=10, color='#64748b'), bgcolor='rgba(0,0,0,0)')
    )
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

def render_vrp(selected_name: str, ticker: str, asset_class: str, trading_days: int, is_crypto: bool) -> None:
    section_header("5", "VOLATILITY RISK PREMIUM (VRP)", "◈")
    main_data = fetch_data(ticker, period="6mo", is_crypto=is_crypto)
    vix = get_vix_data(asset_class, ticker, period="6mo", is_crypto=is_crypto)
    if main_data is None or vix is None:
        st.warning("Data unavailable.")
        return

    hv = np.log(main_data['Close'] / main_data['Close'].shift(1)).rolling(20).std() * np.sqrt(trading_days) * 100
    df = pd.merge(vix['Close'].to_frame('VIX'), hv.to_frame('HV'), left_index=True, right_index=True, how='outer').ffill().dropna()
    if df.empty:
        st.warning("Insufficient data.")
        return

    df['VRP'] = df['VIX'] - df['HV']
    current_vrp = safe_get_scalar(df['VRP'])

    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=df.index, y=df['VIX'], name='Implied Vol', line=dict(color=CHART_THEME["primary"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['HV'], name='Realized Vol', line=dict(color=CHART_THEME["secondary"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['HV'], showlegend=False, line=dict(width=0), hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VIX'], fill='tonexty', fillcolor='rgba(103, 232, 249, 0.08)', line=dict(width=0), name='Spread', showlegend=False, hoverinfo='skip'), row=1, col=1)
    colors = np.where(df['VRP'] > 0, CHART_THEME["bullish"], CHART_THEME["bearish"])
    fig.add_trace(go.Bar(x=df.index, y=df['VRP'], name='VRP Spread', marker_color=colors, opacity=0.8), row=2, col=1)
    fig.update_layout(
        template=CHART_THEME["template"], height=320,
        margin=dict(t=5, b=10, l=0, r=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified', bargap=0,
        legend=dict(orientation='h', font=dict(size=10, color='#64748b'), bgcolor='rgba(0,0,0,0)')
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.04)')
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Vol Benchmark", f"{safe_get_scalar(df['VIX']):.2f}")
    c2.metric("HV (20D)", f"{safe_get_scalar(df['HV']):.2f}")
    c3.metric("Current VRP", f"{current_vrp:+.2f}%")

def render_hurst_regime(selected_name: str, ticker: str, is_crypto: bool) -> None:
    section_header("6", "HURST EXPONENT — REGIME CLASSIFIER", "◈")
    data = fetch_data(ticker, period="1y", is_crypto=is_crypto)
    if data is None:
        st.warning("Data unavailable.")
        return

    def calculate_hurst(ts: pd.Series) -> float:
        if len(ts) < 20: return np.nan
        lags = range(2, 20)
        reg_val = [np.std(ts.values[lag:] - ts.values[:-lag]) for lag in lags]
        try: return np.polyfit(np.log(lags), np.log(reg_val), 1)[0]
        except: return np.nan

    log_prices = np.log(data['Close'])
    hurst_series = log_prices.rolling(window=60).apply(calculate_hurst, raw=False)
    df = pd.DataFrame({'Close': data['Close'], 'Hurst': hurst_series}).dropna()
    if df.empty:
        st.warning("Insufficient data for Hurst calculation.")
        return

    current_hurst = safe_get_scalar(df['Hurst'])
    trend_thresh = CONFIG.get('thresholds', 'hurst_trend')
    mean_revert_thresh = CONFIG.get('thresholds', 'hurst_mean_revert')

    if current_hurst < mean_revert_thresh:
        regime, hurst_color = "MEAN REVERTING", CHART_THEME['bullish']
    elif current_hurst > trend_thresh:
        regime, hurst_color = "TRENDING", CHART_THEME['secondary']
    else:
        regime, hurst_color = "RANDOM WALK", CHART_THEME['primary']

    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], shared_xaxes=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color=CHART_THEME["neutral"], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Hurst'], name='Hurst', line=dict(color=hurst_color, width=2)), row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color='rgba(255,255,255,0.3)', line_width=1, row=2, col=1)
    fig.update_layout(
        template=CHART_THEME["template"], height=280,
        margin=dict(t=5, b=10, l=0, r=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.04)')
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Hurst Value", f"{current_hurst:.3f}")
    c2.metric("Regime", regime)

def render_advanced_volatility(selected_name: str, ticker: str, trading_days: int, is_crypto: bool) -> None:
    section_header("7", "YANG-ZHANG TRUE VOLATILITY", "◈")
    data = fetch_data(ticker, period="1y", is_crypto=is_crypto)
    if data is None:
        st.warning("Data unavailable.")
        return

    df = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
    if len(df) < 2:
        st.warning("Insufficient data for Yang-Zhang.")
        return

    o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
    N = len(o)
    vol_o = np.log(o / c.shift(1)).std() ** 2
    vol_c = np.log(c / o).std() ** 2
    vol_rs = ((np.log(h / o) * np.log(h / c)) + (np.log(l / o) * np.log(l / c))).mean()
    k = 0.34 / (1.34 + (N + 1) / (N - 1)) if N > 1 else 0
    yz_vol = np.sqrt(vol_o + k * vol_c + (1 - k) * vol_rs) * np.sqrt(trading_days) * 100
    c2c_vol = np.log(c / c.shift(1)).std() * np.sqrt(trading_days) * 100
    gap_risk = yz_vol - c2c_vol

    st.markdown(
        '<p style="color:#64748b;font-size:12px;margin-bottom:12px;">Captures overnight gap risk & intraday trend mathematically.</p>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Yang-Zhang Vol", f"{yz_vol:.2f}%")
    c2.metric("Close-to-Close Vol", f"{c2c_vol:.2f}%")
    c3.metric("Hidden Gap Risk", f"{gap_risk:+.2f}%", delta_color="inverse")

def render_microstructure(ticker: str, is_crypto: bool) -> None:
    section_header("8", "MARKET MICROSTRUCTURE & ORDER FLOW (INTRADAY)", "◈")
    data = fetch_data(ticker, period="30d", interval="1h", is_crypto=is_crypto)
    if data is None or data.empty or 'Volume' not in data.columns:
        st.warning("Intraday volume data required for microstructure analysis is unavailable.")
        return
        
    df = data.copy()
    epsilon = 1e-8
    
    # 1. Kyle's Lambda
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1)).abs()
    df['Lambda'] = df['Log_Ret'] / (df['Volume'] + epsilon)
    df['Lambda_Smooth'] = df['Lambda'].rolling(24).median()
    
    curr_lambda = safe_get_scalar(df['Lambda_Smooth'])
    lambda_min = df['Lambda_Smooth'].min()
    lambda_max = df['Lambda_Smooth'].max()
    lambda_rank = ((curr_lambda - lambda_min) / (lambda_max - lambda_min + epsilon)) * 100
    
    # 2. VPIN Proxy
    range_hl = df['High'] - df['Low']
    buy_pct = np.where(range_hl > 0, (df['Close'] - df['Low']) / range_hl, 0.5)
    df['Buy_Vol'] = df['Volume'] * buy_pct
    df['Sell_Vol'] = df['Volume'] - df['Buy_Vol']
    df['Roll_Buy'] = df['Buy_Vol'].rolling(24).sum()
    df['Roll_Sell'] = df['Sell_Vol'].rolling(24).sum()
    df['Roll_Vol'] = df['Volume'].rolling(24).sum()
    df['VPIN'] = abs(df['Roll_Buy'] - df['Roll_Sell']) / (df['Roll_Vol'] + epsilon) * 100
    curr_vpin = safe_get_scalar(df['VPIN'])
    
    # 3. Shannon Entropy
    def calc_entropy(series):
        if len(series) < 10: return np.nan
        hist, _ = np.histogram(series, bins=10, density=False)
        p = hist / np.sum(hist)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))
        
    df['Returns'] = data['Close'].pct_change()
    df['Entropy'] = df['Returns'].rolling(48).apply(calc_entropy)
    
    curr_entropy = safe_get_scalar(df['Entropy'])
    entropy_norm = (curr_entropy / 3.3219) * 100 if curr_entropy else 0.0

    lambda_color = CHART_THEME['bearish'] if lambda_rank > 70 else (CHART_THEME['bullish'] if lambda_rank < 30 else CHART_THEME['secondary'])
    lambda_state = "BRITTLE (High Slippage)" if lambda_rank > 70 else ("LIQUID (Low Slippage)" if lambda_rank < 30 else "NORMAL")
    
    vpin_color = CHART_THEME['bearish'] if curr_vpin > 40 else CHART_THEME['bullish']
    vpin_state = "TOXIC (Smart Money Active)" if curr_vpin > 40 else "BALANCED (Retail Flow)"
    
    ent_color = CHART_THEME['primary'] if entropy_norm < 70 else CHART_THEME['bearish']
    ent_state = "PREDICTABLE (Structured)" if entropy_norm < 70 else "CHAOTIC (Random Walk)"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'''
        <div class="module-card">
            <div class="metric-label">Kyle\'s Lambda Rank</div>
            <div class="metric-value">{lambda_rank:.1f}%</div>
            <div style="color:{lambda_color};font-size:11px;font-weight:700;margin-top:4px;">{lambda_state}</div>
        </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''
        <div class="module-card">
            <div class="metric-label">VPIN (Toxicity)</div>
            <div class="metric-value">{curr_vpin:.1f}%</div>
            <div style="color:{vpin_color};font-size:11px;font-weight:700;margin-top:4px;">{vpin_state}</div>
        </div>''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''
        <div class="module-card">
            <div class="metric-label">Shannon Entropy</div>
            <div class="metric-value">{entropy_norm:.1f}%</div>
            <div style="color:{ent_color};font-size:11px;font-weight:700;margin-top:4px;">{ent_state}</div>
        </div>''', unsafe_allow_html=True)

# ============================== ML ENGINE ==============================
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
        'n_estimators': CONFIG.get('ml', 'n_estimators'), 'max_depth': CONFIG.get('ml', 'max_depth'),
        'learning_rate': 0.05, 'objective': 'binary:logistic', 'random_state': 42,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'min_child_weight': 3, 'gamma': 0.1
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
    section_header("9", "AI PREDICTIVE ENGINE (XGBoost + Purged CV)", "◈")
    
    model_full, scaler_full, features, metrics = train_and_validate_ml_model(ticker, is_crypto)
    
    if model_full is None:
        st.warning("Insufficient historical data to train ML model.")
        return

    df = fetch_data(ticker, period="60d", interval="1d", is_crypto=is_crypto)
    macro_tnx = fetch_data("^TNX", period="60d", interval="1d", is_crypto=False)
    macro_dxy = fetch_data("DX-Y.NYB", period="60d", interval="1d", is_crypto=False)

    if df is None or df.empty:
        st.warning("Live spot data unavailable for prediction.")
        return

    # Safely assign macro features if API responds, otherwise assign neutral 0.0 values to prevent KeyErrors
    if macro_tnx is not None and not macro_tnx.empty:
        df['Macro_TNX'] = macro_tnx['Close'].ffill()
    if macro_dxy is not None and not macro_dxy.empty:
        df['Macro_DXY'] = macro_dxy['Close'].ffill()

    # Calculate Technical Features
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20D'] = df['Log_Returns'].rolling(20).std() * np.sqrt(252)
    df['Momentum_10D'] = df['Close'] - df['Close'].shift(10)
    df['SMA_20_Dist'] = (df['Close'] / df['Close'].rolling(20).mean()) - 1
    df['RSI_14'] = calculate_rsi(df['Close'], 14)

    # BULLETPROOF FIX: Ensure all required features exist for prediction regardless of live API failures
    for f in features:
        if f not in df.columns:
            df[f] = 0.0 

    # Isolate last row, fill any lingering NaNs created by rolling windows that didn't fully resolve
    live_data = df.tail(1).copy()
    live_data = live_data[features].fillna(0.0)

    if live_data.empty:
        st.warning("Prediction calculation failed. Not enough live feature data.")
        return

    live_scaled = scaler_full.transform(live_data)
    prob_bullish = model_full.predict_proba(live_scaled)[0][1] * 100
    prob_bearish = model_full.predict_proba(live_scaled)[0][0] * 100
    prediction = "BULLISH" if prob_bullish > 50 else "BEARISH"
    pred_color = CHART_THEME['bullish'] if prediction == "BULLISH" else CHART_THEME['bearish']

    c1, c2, c3 = st.columns(3)
    c1.metric("ML Model Bias", prediction)
    c2.metric("Up-Day Prob", f"{prob_bullish:.1f}%")
    c3.metric("Down-Day Prob", f"{prob_bearish:.1f}%")

    col1, col2 = st.columns([1, 1])
    with col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob_bullish, domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Bullish Probability (%)", 'font': {'size': 12, 'color': '#94a3b8'}},
            number={'font': {'color': pred_color, 'size': 30}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#334155', 'tickfont': {'size': 9}},
                'bar': {'color': pred_color, 'thickness': 0.25},
                'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
                'steps': [
                    {'range': [0, 45], 'color': "rgba(239, 68, 68, 0.12)"},
                    {'range': [45, 55], 'color': "rgba(255, 255, 255, 0.04)"},
                    {'range': [55, 100], 'color': "rgba(34, 197, 94, 0.12)"}
                ],
                'threshold': {'line': {'color': pred_color, 'width': 2}, 'thickness': 0.75, 'value': prob_bullish}
            }
        ))
        fig_gauge.update_layout(
            template=CHART_THEME["template"], height=260,
            margin=dict(l=20, r=20, t=50, b=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        importances = model_full.feature_importances_
        feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=True)
        fig_imp = go.Figure(go.Bar(
            x=feat_imp_df['Importance'], y=feat_imp_df['Feature'],
            orientation='h', marker_color=CHART_THEME["secondary"],
            marker_line_width=0, opacity=0.85
        ))
        fig_imp.update_layout(
            title=dict(text="Feature Importance Weights", font=dict(size=12, color='#94a3b8')),
            template=CHART_THEME["template"], height=260,
            xaxis_title="Weight",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=40, b=10)
        )
        fig_imp.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
        fig_imp.update_yaxes(showgrid=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    if metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("WF Accuracy", f"{metrics['acc']*100:.1f}%")
        m2.metric("WF Precision", f"{metrics['prec']*100:.1f}%")
        m3.metric("WF Recall", f"{metrics['rec']*100:.1f}%")

# ============================== PORTFOLIO RISK & OPTIONS GREEKS ==============================
def render_portfolio_risk(is_crypto: bool, currency: str) -> None:
    section_header("10", "MULTI-ASSET PORTFOLIO STRESS TEST (Risk Parity & Hist VaR)", "◈")
    basket = list(CRYPTO_ASSETS.values()) if is_crypto else list(INDIAN_ASSETS.values())[:3]
    basket_names = list(CRYPTO_ASSETS.keys()) if is_crypto else list(INDIAN_ASSETS.keys())[:3]

    data_dict, successful_names = {}, []
    for ticker, name in zip(basket, basket_names):
        df = fetch_data(ticker, period="2y", interval="1d", is_crypto=is_crypto)
        if df is not None and not df.empty:
            data_dict[ticker] = df['Close']
            successful_names.append(name.split(" ")[0])

    if len(data_dict) < 2:
        st.warning("Insufficient portfolio data.")
        return

    port_df = pd.DataFrame(data_dict).ffill().dropna()
    returns = np.log(port_df / port_df.shift(1)).dropna()
    std_devs = returns.std()
    inv_vol = 1.0 / std_devs
    weights = (inv_vol / inv_vol.sum()).values
    cov_matrix = returns.cov()
    port_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    hist_port_returns = returns.dot(weights)
    var_95 = abs(np.percentile(hist_port_returns, 5)) * 100
    weight_str = " / ".join([f"{n}: {w*100:.0f}%" for n, w in zip(successful_names, weights)])

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Parity Weights", weight_str)
    c2.metric("Portfolio Annual Vol", f"{port_std_dev*100:.2f}%")
    c3.metric("Daily VaR (95%)", f"-{var_95:.2f}%", "Capital at Risk", delta_color="inverse")

def bs_greeks_advanced(S: float, K: float, T_days: int, r_pct: float, sigma_pct: float, q_pct: float) -> Dict[str, Dict[str, float]]:
    T, r, sigma, q = T_days / 365.0, r_pct / 100.0, sigma_pct / 100.0, q_pct / 100.0
    if T <= 0: T = 1e-5
    if sigma <= 0: sigma = 1e-5
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return {
        'Call': {
            'Price': S * np.exp(-q * T) * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2),
            'Delta': np.exp(-q * T) * si.norm.cdf(d1),
            'Gamma': (np.exp(-q * T) * si.norm.pdf(d1)) / (S * sigma * np.sqrt(T)),
            'Theta': ((-S * si.norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) + q * S * si.norm.cdf(d1) * np.exp(-q * T) - r * K * np.exp(-r * T) * si.norm.cdf(d2)) / 365,
            'Vega': S * np.exp(-q * T) * si.norm.pdf(d1) * np.sqrt(T) / 100,
            'Rho': K * T * np.exp(-r * T) * si.norm.cdf(d2) / 100
        },
        'Put': {
            'Price': K * np.exp(-r * T) * si.norm.cdf(-d2) - S * np.exp(-q * T) * si.norm.cdf(-d1),
            'Delta': np.exp(-q * T) * (si.norm.cdf(d1) - 1),
            'Gamma': (np.exp(-q * T) * si.norm.pdf(d1)) / (S * sigma * np.sqrt(T)),
            'Theta': ((-S * si.norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) - q * S * si.norm.cdf(-d1) * np.exp(-q * T) + r * K * np.exp(-r * T) * si.norm.cdf(-d2)) / 365,
            'Vega': S * np.exp(-q * T) * si.norm.pdf(d1) * np.sqrt(T) / 100,
            'Rho': -K * T * np.exp(-r * T) * si.norm.cdf(-d2) / 100
        }
    }

def render_options_greeks(selected_name: str, ticker: str, asset_class: str, is_crypto: bool) -> None:
    section_header("11", "OPTIONS PRICING ENGINE (Merton Extension)", "◈")
    asset_data = fetch_data(ticker, period="5d", is_crypto=is_crypto)
    vix = get_vix_data(asset_class, ticker, period="5d", is_crypto=is_crypto)
    tnx_data = fetch_data("^TNX", period="5d", is_crypto=False)
    live_r = safe_get_scalar(tnx_data['Close']) if tnx_data is not None else (7.0 if asset_class == "Indian Equities" else 4.5)

    if asset_data is None:
        st.warning("Spot price data unavailable.")
        return

    live_spot = safe_get_scalar(asset_data['Close'])
    live_iv = safe_get_scalar(vix['Close']) if vix is not None else 30.0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: spot = st.number_input("Spot Price (S)", value=float(live_spot), step=10.0)
    with c2: strike = st.number_input("Strike (K)", value=float(round(live_spot / 100) * 100), step=100.0)
    with c3: dte = st.number_input("Days to Expiry", value=7, min_value=0)
    with c4: iv = st.number_input("Implied Vol (%)", value=float(live_iv), step=1.0)
    with c5: div = st.number_input("Div Yield (%)", value=0.0 if is_crypto else 1.2, step=0.1)

    st.markdown(
        f'<p style="color:#64748b;font-size:12px;margin-bottom:12px;">Risk-Free Rate locked at live 10Y Yield: <strong style="color:#67e8f9;">{live_r:.2f}%</strong></p>',
        unsafe_allow_html=True
    )

    greeks = bs_greeks_advanced(spot, strike, dte, live_r, iv, div)

    def render_greek_card(title: str, data: Dict[str, float], color: str, bg: str) -> None:
        st.markdown(f"""
            <div class="greek-card" style="border-top:3px solid {color};background:{bg};">
                <div class="greek-title" style="color:{color};">{title}</div>
            </div>
        """, unsafe_allow_html=True)
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("Theoretical Price", f"${data['Price']:.2f}")
        r1c2.metric("Delta (Δ)", f"{data['Delta']:.4f}")
        r1c3.metric("Gamma (Γ)", f"{data['Gamma']:.6f}")
        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric("Theta (Θ)", f"{data['Theta']:.3f}/day")
        r2c2.metric("Vega (ν)", f"{data['Vega']:.4f}")
        r2c3.metric("Rho (ρ)", f"{data['Rho']:.4f}")

    col_call, col_put = st.columns(2)
    with col_call:
        render_greek_card("▲ CALL OPTION", greeks['Call'], CHART_THEME["bullish"], "rgba(34,197,94,0.04)")
    with col_put:
        render_greek_card("▼ PUT OPTION", greeks['Put'], CHART_THEME["bearish"], "rgba(239,68,68,0.04)")

# ============================== MAIN UI ROUTER ==============================
def main() -> None:
    st.set_page_config(
        page_title="Aladdin Quant Terminal",
        layout="wide",
        page_icon="⚡",
        initial_sidebar_state="expanded"
    )

    if HAS_AUTOREFRESH:
        st_autorefresh(interval=60000, key="aladdin_refresh")

    # COMPREHENSIVE CSS
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

        .stApp {
            background: #03050a;
            color: #C8D1DC;
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #070c18 0%, #040810 100%) !important;
            border-right: 1px solid #111c2e !important;
        }
        [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
        [data-testid="stSidebar"] h1 {
            font-size: 16px !important;
            letter-spacing: 1.5px !important;
            color: #67e8f9 !important;
            border-bottom: 1px solid #111c2e;
            padding-bottom: 10px;
            margin-bottom: 14px;
        }
        [data-testid="stSidebar"] label { color: #64748b !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }
        [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
            background: #07101f !important;
            border: 1px solid #1a2840 !important;
            border-radius: 4px !important;
            color: #E2E8F0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] { margin-bottom: 6px; }
        [data-testid="stSidebar"] hr { border-color: #111c2e !important; margin: 12px 0 !important; }

        .block-container {
            padding: 1.2rem 1.5rem 2rem !important;
            max-width: 100% !important;
        }

        h1 {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 19px !important;
            font-weight: 700 !important;
            color: #F1F5F9 !important;
            letter-spacing: 3px !important;
            text-transform: uppercase !important;
            border-bottom: 1px solid #111c2e;
            padding-bottom: 12px;
            margin-bottom: 4px !important;
        }

        .section-header-wrap {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 16px;
            margin-top: 4px;
            padding: 8px 14px;
            background: linear-gradient(90deg, rgba(103,232,249,0.06) 0%, rgba(103,232,249,0.01) 100%);
            border-left: 3px solid #67e8f9;
            border-radius: 0 4px 4px 0;
        }
        .section-num {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: #67e8f9;
            background: rgba(103,232,249,0.1);
            padding: 2px 7px;
            border-radius: 3px;
            font-weight: 700;
            display: none;
        }
        .section-num:not(:empty) { display: inline-block; }
        .section-icon { font-size: 14px; color: #67e8f9; }
        .section-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 700;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 2.5px;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #111c2e !important;
            background: linear-gradient(145deg, #070d1a 0%, #040a14 100%) !important;
            border-radius: 6px !important;
            padding: 6px !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.45), 0 1px 0 rgba(103,232,249,0.04) inset !important;
        }
        .module-card {
            background: linear-gradient(145deg, #0b1524, #07101e);
            border: 1px solid #111c2e;
            border-top: 2px solid rgba(103,232,249,0.3);
            padding: 14px 16px;
            border-radius: 5px;
            margin-bottom: 12px;
        }
        .metric-label {
            color: #475569;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
            margin-bottom: 4px;
            font-family: 'JetBrains Mono', monospace;
        }
        .metric-value {
            color: #E2E8F0;
            font-size: 19px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.2;
        }

        div[data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.25em !important;
            font-weight: 700 !important;
            color: #67e8f9 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 10px !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            color: #475569 !important;
            font-family: 'JetBrains Mono', monospace !important;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 11px !important;
            font-family: 'JetBrains Mono', monospace !important;
        }
        [data-testid="stMetricDeltaIcon-Up"] { color: #22c55e !important; }
        [data-testid="stMetricDeltaIcon-Down"] { color: #ef4444 !important; }

        .exec-summary-card {
            background: linear-gradient(135deg, #07111f 0%, #050d18 60%, #060f1c 100%);
            border: 1px solid #1a2840;
            border-top: 2px solid #67e8f9;
            border-radius: 6px;
            padding: 20px 22px;
            margin-bottom: 6px;
            box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 1px 0 rgba(103,232,249,0.08) inset;
        }
        .exec-summary-header {
            display: flex;
            align-items: center;
            gap: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            font-weight: 700;
            color: #64748b;
            letter-spacing: 2.5px;
            text-transform: uppercase;
            margin-bottom: 18px;
            border-bottom: 1px solid #0f1e35;
            padding-bottom: 12px;
        }
        .exec-dot {
            display: inline-block;
            width: 7px; height: 7px;
            border-radius: 50%;
            background: #22c55e;
            box-shadow: 0 0 6px #22c55e88;
            animation: pulse-dot 2s infinite;
        }
        @keyframes pulse-dot {
            0%, 100% { opacity: 1; box-shadow: 0 0 6px #22c55e88; }
            50% { opacity: 0.4; box-shadow: 0 0 2px #22c55e22; }
        }
        .exec-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 16px;
        }
        .exec-block {
            padding: 14px 16px;
            background: rgba(255,255,255,0.015);
            border: 1px solid #111c2e;
            border-radius: 5px;
        }
        .exec-block-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 9px;
            font-weight: 700;
            color: #334155;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 6px;
        }
        .exec-signal {
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .exec-sub {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            margin: 8px 0 4px 0;
        }
        .exec-block-text {
            font-size: 12px;
            color: #94a3b8;
            line-height: 1.6;
            word-wrap: break-word;
        }
        .exec-block-text strong {
            font-weight: 700;
            color: #67e8f9;
        }

        .news-section-label {
            color: #334155;
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            margin: 10px 0 8px 0;
        }
        .news-headline {
            background: rgba(10,18,32,0.8);
            border-radius: 4px;
            padding: 8px 10px 8px 14px;
            margin-bottom: 7px;
        }
        .news-title {
            font-size: 12px;
            color: #C8D1DC;
            line-height: 1.45;
            margin-bottom: 4px;
        }
        .news-meta {
            font-size: 9px;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.5px;
        }

        .greek-card {
            border-radius: 5px;
            padding: 10px 14px 8px;
            margin-bottom: 10px;
            border: 1px solid #111c2e;
        }
        .greek-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            margin-bottom: 2px;
        }

        .module-offline {
            background: rgba(239,68,68,0.06);
            border: 1px solid rgba(239,68,68,0.2);
            color: #ef4444;
            border-radius: 4px;
            padding: 10px 14px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            letter-spacing: 1px;
        }

        .stButton > button {
            background: linear-gradient(135deg, #07111f, #0a1626) !important;
            border: 1px solid #1a2840 !important;
            color: #67e8f9 !important;
            border-radius: 4px !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 11px !important;
            letter-spacing: 1.5px !important;
            text-transform: uppercase !important;
            font-weight: 600 !important;
            transition: all 0.15s ease !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #0d1c35, #0f2040) !important;
            border-color: #67e8f9 !important;
            box-shadow: 0 0 12px rgba(103,232,249,0.15) !important;
        }

        div[role="radiogroup"] label {
            color: #64748b !important;
            font-size: 11px !important;
            font-family: 'JetBrains Mono', monospace !important;
        }
        div[role="radiogroup"] label[data-baseweb="radio"] { background: transparent !important; }

        div[data-testid="stNumberInput"] input {
            background: #06101d !important;
            border: 1px solid #1a2840 !important;
            border-radius: 4px !important;
            color: #67e8f9 !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 13px !important;
        }

        div[data-testid="stAlert"] {
            background: rgba(103,232,249,0.04) !important;
            border: 1px solid rgba(103,232,249,0.12) !important;
            border-radius: 5px !important;
            color: #94a3b8 !important;
            font-size: 12px !important;
        }

        div[data-testid="stSpinner"] > div {
            border-top-color: #67e8f9 !important;
        }

        hr { border-color: #0f1e35 !important; margin: 1.2em 0 !important; }

        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #040810; }
        ::-webkit-scrollbar-thumb { background: #1a2840; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #253758; }

        h2, h3 {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            color: #94a3b8 !important;
            letter-spacing: 2px !important;
            text-transform: uppercase !important;
        }

        /* Customizing Streamlit Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            padding-top: 10px;
            padding-bottom: 10px;
            font-family: 'JetBrains Mono', monospace !important;
            color: #64748b;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(103,232,249,0.05);
            border-bottom: 2px solid #67e8f9 !important;
            color: #F1F5F9 !important;
        }
        
        .stPlotlyChart { border-radius: 4px; overflow: hidden; }
        </style>
    """, unsafe_allow_html=True)

    # SIDEBAR
    with st.sidebar:
        st.title("⚡ ALADDIN v22.0")

        sync_col1, sync_col2 = st.columns([3, 1])
        with sync_col1:
            if st.button("↻ Force Sync", use_container_width=True):
                st.session_state.market_data = {}
                st.cache_data.clear()
                st.rerun()
        with sync_col2:
            status_label = "●" if HAS_AUTOREFRESH else "○"
            st.markdown(
                f'<div style="color:{"#22c55e" if HAS_AUTOREFRESH else "#ef4444"};font-size:20px;text-align:center;padding-top:5px;">{status_label}</div>',
                unsafe_allow_html=True
            )

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

        st.divider()
        st.markdown(
            '<div style="font-family:JetBrains Mono;font-size:9px;color:#1e3a5f;text-align:center;letter-spacing:1px;">EMA 89 · EMA 21 · VWAP · RSI(14)<br>YANG-ZHANG · HURST · VRP<br>VPIN · XGBOOST ML · BLACK-SCHOLES</div>',
            unsafe_allow_html=True
        )

    # PAGE HEADER
    h_col1, h_col2 = st.columns([5, 1])
    with h_col1:
        live_tag = "⬤ LIVE" if HAS_AUTOREFRESH else "⬤ ONLINE"
        live_color = "#22c55e" if HAS_AUTOREFRESH else "#fbbf24"
        st.markdown(f"""
            <div style="display:flex;align-items:baseline;gap:14px;margin-bottom:14px;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;
                            color:#F1F5F9;letter-spacing:3px;text-transform:uppercase;">
                    ALADDIN // QUANT TERMINAL
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#67e8f9;
                            letter-spacing:1px;border:1px solid rgba(103,232,249,0.25);
                            padding:2px 10px;border-radius:3px;background:rgba(103,232,249,0.05);">
                    {selected_name.upper()}
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:{live_color};letter-spacing:1px;">
                    {live_tag}
                </div>
            </div>
        """, unsafe_allow_html=True)

    # CREATE TABS
    tab_live, tab_pre, tab_post = st.tabs(["🔴 LIVE MATRIX", "🌅 PRE-MARKET PREP", "🌃 POST-MARKET WRAP"])

    with tab_live:
        with st.spinner("Initialising Aladdin quantitative matrix…"):
            
            # ROW 0: EXECUTIVE SUMMARY
            safe_render(
                render_executive_summary,
                selected_name, ticker, asset_class,
                div1, div2, div1_name, div2_name,
                currency, trading_days, is_crypto
            )

            # ROW 1: REALTIME CHART & NLP
            tab_row1_c1, tab_row1_c2 = st.columns([2, 1])
            with tab_row1_c1:
                with st.container(border=True):
                    safe_render(render_realtime_chart, selected_name, ticker, is_crypto)
            with tab_row1_c2:
                with st.container(border=True):
                    safe_render(render_nlp_sentiment, ticker, is_crypto)

            # ROW 2: IV RANK, EXPECTED MOVE, DIVERGENCE
            col_row2_1, col_row2_2, col_row2_3 = st.columns(3)
            with col_row2_1:
                with st.container(border=True):
                    safe_render(render_volatility_metrics, asset_class, ticker, is_crypto)
            with col_row2_2:
                with st.container(border=True):
                    safe_render(render_expected_move, selected_name, ticker, asset_class, currency, trading_days, is_crypto)
            with col_row2_3:
                with st.container(border=True):
                    safe_render(render_index_divergence, div1, div2, div1_name, div2_name, currency, is_crypto)

            # ROW 3: VOL CONE & VRP
            col_row3_1, col_row3_2 = st.columns(2)
            with col_row3_1:
                with st.container(border=True):
                    safe_render(render_volatility_cone, selected_name, ticker, trading_days, is_crypto)
            with col_row3_2:
                with st.container(border=True):
                    safe_render(render_vrp, selected_name, ticker, asset_class, trading_days, is_crypto)

            # ROW 4: HURST & YANG-ZHANG
            col_row4_1, col_row4_2 = st.columns(2)
            with col_row4_1:
                with st.container(border=True):
                    safe_render(render_hurst_regime, selected_name, ticker, is_crypto)
            with col_row4_2:
                with st.container(border=True):
                    safe_render(render_advanced_volatility, selected_name, ticker, trading_days, is_crypto)

            # ROW 5: MICROSTRUCTURE
            with st.container(border=True):
                safe_render(render_microstructure, ticker, is_crypto)

            # ROW 6: ML ENGINE
            with st.container(border=True):
                safe_render(render_ml_engine, ticker, is_crypto)

            # ROW 7: PORTFOLIO RISK
            with st.container(border=True):
                safe_render(render_portfolio_risk, is_crypto, currency)

            # ROW 8: OPTIONS GREEKS
            with st.container(border=True):
                safe_render(render_options_greeks, selected_name, ticker, asset_class, is_crypto)

    with tab_pre:
        with st.container(border=True):
            safe_render(render_pre_market_analysis, ticker, is_crypto, asset_class, currency)

    with tab_post:
        with st.container(border=True):
            safe_render(render_post_market_analysis, ticker, is_crypto, asset_class, currency, trading_days)

    # FOOTER
    st.markdown("""
        <div style="text-align:center;margin-top:32px;padding:16px;
                    border-top:1px solid #0f1e35;
                    font-family:'JetBrains Mono',monospace;
                    font-size:9px;color:#1e3a5f;letter-spacing:2px;">
            ALADDIN QUANT TERMINAL v22.0 &nbsp;·&nbsp; EMA(89,21) &nbsp;·&nbsp; YANG-ZHANG &nbsp;·&nbsp;
            HURST &nbsp;·&nbsp; VRP &nbsp;·&nbsp; VPIN &nbsp;·&nbsp; XGBOOST ML &nbsp;·&nbsp; MERTON B-S<br>
            FOR EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()