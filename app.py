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

# ============================== HELPERS & DEFENSIVE RENDER ==============================
def markdown_to_html(text: str) -> str:
    return re.sub(r'\*\*(.*?)\*\*', r'<strong style="color:#67e8f9;font-weight:700;">\1</strong>', text)

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
        text="ALADDIN QUANT TERMINAL v26.1",
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
    """Wrapper to encapsulate UI components and prevent app-wide crashes."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
        st.error(f"Component Error: {func.__name__.replace('_', ' ').title()}")
        return None

def get_pivots(high, low, close):
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    return pivot, r1, s1

# ============================== DATA INGESTION (L1 + L2) ==============================
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_data_internal(ticker: str, period: str, interval: str, is_crypto: bool) -> Optional[pd.DataFrame]:
    if interval == '15m': period = '1mo'
    elif interval in ['1h', '4h'] and period in ['max', '5y', '10y', '2y']: period = '730d'

    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if (data is None or data.empty) and interval == "15m":
        data = yf.download(ticker, period="5d", interval=interval, progress=False)

    if data is None or data.empty: return None

    # CRITICAL: Normalize MultiIndex columns (yfinance API update issue fix)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # CRITICAL: Prevent processing of empty/corrupted dataframes
    if 'Close' not in data.columns:
        return None

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
        if data is None or data.empty or 'Close' not in data.columns: 
            return None
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

    if any(d is None or d.empty for d in [vix_data_summ, daily_data_summ, d1_data_summ, d2_data_summ]):
        st.warning("Synthesis data incomplete. Some metrics may be unavailable.")
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

    last_close = safe_get_scalar(df_metrics['Close'])
    last_ema89 = safe_get_scalar(df_metrics['EMA_89'])
    last_ema21 = safe_get_scalar(df_metrics['EMA_21'])

    if last_close > last_ema89 and last_ema89 > last_ema21:
        trend_narrative = f"exhibiting an active **Bullish Expansion Structure**. Spot is comfortably elevated above both the momentum 89-period EMA ({currency}{last_ema89:,.2f}) and the intermediate 21-period EMA ({currency}{last_ema21:,.2f}), confirming sustainable upward velocity across timeframes."
        trend_signal, trend_color = "BULLISH EXPANSION", CHART_THEME['bullish']
    elif last_close < last_ema89 and last_ema89 < last_ema21:
        trend_narrative = f"stuck in a strong **Bearish Markdown Sequence**. Price action remains structurally pinned beneath the descending 89-period EMA ({currency}{last_ema89:,.2f}) and 21-period EMA ({currency}{last_ema21:,.2f}), alerting option buyers to step carefully."
        trend_signal, trend_color = "BEARISH MARKDOWN", CHART_THEME['bearish']
    else:
        trend_narrative = f"experiencing a **Mean Reversion / Consolidation Phase**. Spot pricing is weaving through its 89-period and 21-period EMAs, showing range containment ahead of any directional breakout."
        trend_signal, trend_color = "CONSOLIDATION", CHART_THEME['secondary']

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

# ============================== ML ENGINE ==============================
@st.cache_resource(ttl=3600, show_spinner=False)
def train_and_validate_ml_model(ticker: str, is_crypto: bool):
    df = fetch_data(ticker, period="2y", interval="1d", is_crypto=is_crypto)

    if df is None or len(df) < 100 or 'Close' not in df.columns: 
        return None, None, None, None

    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20D'] = df['Log_Returns'].rolling(20).std() * np.sqrt(252)
    df['Momentum_10D'] = df['Close'] - df['Close'].shift(10)
    df['SMA_20_Dist'] = (df['Close'] / df['Close'].rolling(20).mean()) - 1
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    features = ['Log_Returns', 'Vol_20D', 'Momentum_10D', 'SMA_20_Dist']

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
    section_header("8", "AI PREDICTIVE ENGINE (XGBoost + Purged CV)", "◈")
    
    model_full, scaler_full, features, metrics = train_and_validate_ml_model(ticker, is_crypto)
    
    if model_full is None:
        st.warning("Insufficient historical data to train ML model.")
        return

    df = fetch_data(ticker, period="60d", interval="1d", is_crypto=is_crypto)

    if df is None or df.empty or 'Close' not in df.columns:
        st.warning("Live spot data unavailable for prediction.")
        return

    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20D'] = df['Log_Returns'].rolling(20).std() * np.sqrt(252)
    df['Momentum_10D'] = df['Close'] - df['Close'].shift(10)
    df['SMA_20_Dist'] = (df['Close'] / df['Close'].rolling(20).mean()) - 1

    for f in features:
        if f not in df.columns:
            df[f] = 0.0 

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

def render_portfolio_risk(is_crypto: bool, currency: str) -> None:
    section_header("9", "MULTI-ASSET PORTFOLIO STRESS TEST (Risk Parity & Hist VaR)", "◈")
    basket = list(CRYPTO_ASSETS.values()) if is_crypto else list(INDIAN_ASSETS.values())[:3]
    basket_names = list(CRYPTO_ASSETS.keys()) if is_crypto else list(INDIAN_ASSETS.keys())[:3]

    data_dict, successful_names = {}, []
    for ticker, name in zip(basket, basket_names):
        df = fetch_data(ticker, period="2y", interval="1d", is_crypto=is_crypto)
        if df is not None and not df.empty and 'Close' in df.columns:
            data_dict[ticker] = df['Close']
            successful_names.append(name.split(" ")[0])

    if len(data_dict) < 2:
        st.warning("Insufficient portfolio data.")
        return

    port_df = pd.DataFrame(data_dict).ffill().dropna()
    if port_df.empty:
        st.warning("Insufficient overlapping data for portfolio risk.")
        return

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

# ============================== REALTIME CHART ==============================
def render_realtime_chart(selected_name: str, ticker: str, is_crypto: bool) -> None:
    section_header("", "MARKET PRICE · VOLUME · MOMENTUM", "◈")
    timeframe = st.radio("Timeframe", ["15m", "1h", "4h", "1d"], index=1, horizontal=True, label_visibility="collapsed")
    period, interval = (
        ("1mo", "15m") if timeframe == "15m"
        else ("730d", "1h") if timeframe in ["1h", "4h"]
        else ("2y", "1d")
    )

    data = fetch_data(ticker, period=period, interval=interval, is_crypto=is_crypto)
    
    if data is None or data.empty or 'Close' not in data.columns:
        st.caption(f"Real-time data currently unavailable for {ticker}.")
        return

    if timeframe == "4h":
        if 'Volume' in data.columns:
            data = data.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        else:
            data = data.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()

    if data.empty:
        st.warning(f"Insufficient data after resampling for {ticker}.")
        return

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

    data['Prev_High'] = data['High'].rolling(20).max().shift(1)
    data['Prev_Low'] = data['Low'].rolling(20).min().shift(1)
    data['Supply_Sweep'] = (data['High'] > data['Prev_High']) & (data['Close'] < data['Prev_High'])
    data['Demand_Sweep'] = (data['Low'] < data['Prev_Low']) & (data['Close'] > data['Prev_Low'])

    last_close = safe_get_scalar(data['Close'])
    x_format = '%Y-%m-%d' if timeframe == "1d" else '%Y-%m-%d %H:%M'
    x_axis_string = data.index.strftime(x_format)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.80, 0.20], vertical_spacing=0.03
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

    grid_cfg = dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.04)')
    fig.update_layout(
        template=CHART_THEME['template'], height=560, xaxis_rangeslider_visible=False,
        hovermode='x unified', bargap=0, bargroupgap=0,
        margin=dict(l=10, r=65, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True, legend=dict(
            orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0,
            font=dict(size=10, color='#64748b'), bgcolor='rgba(0,0,0,0)', borderwidth=0
        )
    )

    fig.update_xaxes(type='category', **grid_cfg, showticklabels=False, row=1, col=1)
    fig.update_xaxes(
        type='category', categoryorder='category ascending', **grid_cfg,
        showticklabels=True, showspikes=True, spikemode='across',
        spikethickness=1, spikedash='dot', spikecolor='rgba(255,255,255,0.2)', row=2, col=1
    )
    fig.update_yaxes(**grid_cfg)

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

def render_volatility_metrics(asset_class: str, ticker: str, is_crypto: bool) -> None:
    section_header("1", "IMPLIED VOLATILITY RANK", "◈")
    vix_name = "India VIX" if asset_class == "Indian Equities" else "Synthetic IV (30D HV)"
    data = get_vix_data(asset_class, ticker, period="1y", is_crypto=is_crypto)
    if data is None or data.empty or 'Close' not in data.columns:
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
    
    if asset_data is None or asset_data.empty or vix is None or vix.empty or 'Close' not in asset_data.columns:
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
    
    if d1 is None or d1.empty or d2 is None or d2.empty or 'Close' not in d1.columns or 'Close' not in d2.columns:
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
    if data is None or data.empty or 'Close' not in data.columns:
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
    
    if df_stats.empty:
        st.warning("Insufficient historical data for volatility cone.")
        return

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
    
    if main_data is None or main_data.empty or vix is None or vix.empty or 'Close' not in main_data.columns:
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
    if data is None or data.empty or 'Close' not in data.columns:
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
    if data is None or data.empty:
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
            word-break: break-word !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 10px !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            color: #475569 !important;
            font-family: 'JetBrains Mono', monospace !important;
            white-space: normal !important;
            word-wrap: break-word !important;
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
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
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

        /* Custom Grids for Pre/Post Market */
        .pm-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .eod-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; }

        /* ================= ULTRA-RESPONSIVE / MOBILE CSS ================= */
        @media (max-width: 768px) {
            .block-container { padding: 0.8rem 0.5rem !important; }
            .top-header-container { flex-direction: column !important; gap: 6px !important; align-items: flex-start !important; }
            .top-header-title { font-size: 15px !important; letter-spacing: 1px !important; margin-bottom: 2px !important; }
            .top-header-tag { font-size: 9px !important; padding: 2px 6px !important; display: inline-block !important; margin-bottom: 4px !important;}
            .exec-grid, .pm-grid, .eod-grid { grid-template-columns: 1fr !important; gap: 10px !important; }
            .exec-block, .module-card { padding: 10px 12px !important; margin-bottom: 8px !important; }
            [data-testid="stVerticalBlockBorderWrapper"] { padding: 8px !important; border-radius: 4px !important; }
            .section-header-wrap { padding: 6px 10px !important; margin-bottom: 12px !important; }
            .section-title { font-size: 10px !important; letter-spacing: 1px !important; }
            div[data-testid="stMetricValue"] { font-size: 16px !important; }
            div[data-testid="stMetricLabel"] { font-size: 9px !important; }
            div[data-testid="stMetricDelta"] { font-size: 10px !important; }
            div[data-testid="stNumberInput"] label { font-size: 9px !important; }
            div[data-testid="stNumberInput"] input { font-size: 12px !important; height: 32px !important; min-height: 32px !important; }
            .stTabs [data-baseweb="tab-list"] { gap: 10px !important; overflow-x: auto !important; padding-bottom: 4px !important; }
            .stTabs [data-baseweb="tab"] { height: 38px !important; font-size: 10px !important; padding: 6px 10px !important; white-space: nowrap !important; }
        }
        </style>
    """, unsafe_allow_html=True)

    # SIDEBAR
    with st.sidebar:
        st.title("⚡ ALADDIN v26.1")

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
            '<div style="font-family:JetBrains Mono;font-size:9px;color:#1e3a5f;text-align:center;letter-spacing:1px;">EMA 89 · EMA 21 · VWAP<br>YANG-ZHANG · HURST · VRP<br>TEXTBLOB NLP · XGBOOST ML</div>',
            unsafe_allow_html=True
        )

    # PAGE HEADER
    h_col1, h_col2 = st.columns([5, 1])
    with h_col1:
        live_tag = "⬤ LIVE" if HAS_AUTOREFRESH else "⬤ ONLINE"
        live_color = "#22c55e" if HAS_AUTOREFRESH else "#fbbf24"
        st.markdown(f"""
            <div class="top-header-container" style="display:flex;align-items:baseline;gap:14px;margin-bottom:14px;flex-wrap:wrap;">
                <div class="top-header-title" style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;
                            color:#F1F5F9;letter-spacing:3px;text-transform:uppercase;">
                    ALADDIN // QUANT TERMINAL
                </div>
                <div class="top-header-tag" style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#67e8f9;
                            letter-spacing:1px;border:1px solid rgba(103,232,249,0.25);
                            padding:2px 10px;border-radius:3px;background:rgba(103,232,249,0.05);">
                    {selected_name.upper()}
                </div>
                <div class="top-header-tag" style="font-family:'JetBrains Mono',monospace;font-size:10px;color:{live_color};letter-spacing:1px;">
                    {live_tag}
                </div>
            </div>
        """, unsafe_allow_html=True)

    # CREATE TABS
    tab_live, tab_pre, tab_post, tab_vix = st.tabs(["🔴 LIVE MATRIX", "🌅 PRE-TRADE", "🌃 POST-TRADE", "📉 INDIA VIX"])

    with tab_live:
        with st.spinner("Initialising Aladdin quantitative matrix…"):
            
            safe_render(render_executive_summary, selected_name, ticker, asset_class, div1, div2, div1_name, div2_name, currency, trading_days, is_crypto)

            tab_row1_c1, tab_row1_c2 = st.columns([2, 1])
            with tab_row1_c1:
                with st.container(border=True):
                    safe_render(render_realtime_chart, selected_name, ticker, is_crypto)
            with tab_row1_c2:
                with st.container(border=True):
                    safe_render(render_nlp_sentiment, ticker, is_crypto)

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

            col_row3_1, col_row3_2 = st.columns(2)
            with col_row3_1:
                with st.container(border=True):
                    safe_render(render_volatility_cone, selected_name, ticker, trading_days, is_crypto)
            with col_row3_2:
                with st.container(border=True):
                    safe_render(render_vrp, selected_name, ticker, asset_class, trading_days, is_crypto)

            col_row4_1, col_row4_2 = st.columns(2)
            with col_row4_1:
                with st.container(border=True):
                    safe_render(render_hurst_regime, selected_name, ticker, is_crypto)
            with col_row4_2:
                with st.container(border=True):
                    safe_render(render_advanced_volatility, selected_name, ticker, trading_days, is_crypto)

            with st.container(border=True):
                safe_render(render_ml_engine, ticker, is_crypto)

            with st.container(border=True):
                safe_render(render_portfolio_risk, is_crypto, currency)

    with tab_pre:
        with st.container(border=True):
            # Pre-Trade Analysis Step-by-Step
            st.markdown("## 🌅 PRE-TRADE ANALYSIS (8-Step Institutional Setup)")
            
            asset_data = fetch_data(ticker, period="1y", interval="1d", is_crypto=is_crypto)
            vix_data = get_vix_data(asset_class, ticker, period="1y", is_crypto=is_crypto)
            
            if asset_data is not None and len(asset_data) > 200 and 'Close' in asset_data.columns:
                # Steps 1 & 2: Environment & Technical
                section_header("1 & 2", "ENVIRONMENT & TECHNICAL STRUCTURE", "◈")
                
                df_tech = asset_data.copy()
                df_tech['EMA_20'] = df_tech['Close'].ewm(span=20, adjust=False).mean()
                df_tech['EMA_50'] = df_tech['Close'].ewm(span=50, adjust=False).mean()
                df_tech['EMA_200'] = df_tech['Close'].ewm(span=200, adjust=False).mean()
                
                last = df_tech.iloc[-1]
                prev = df_tech.iloc[-2]
                c_close, ema20, ema50, ema200 = last['Close'], last['EMA_20'], last['EMA_50'], last['EMA_200']
                
                if c_close > ema20 and ema20 > ema50 and ema50 > ema200:
                    trend_bias = "UPTREND"
                    trend_desc = "Higher Highs + Higher Lows. Market directional hai."
                    trend_col = CHART_THEME['bullish']
                elif c_close < ema20 and ema20 < ema50 and ema50 < ema200:
                    trend_bias = "DOWNTREND"
                    trend_desc = "Lower Highs + Lower Lows. Market directional hai."
                    trend_col = CHART_THEME['bearish']
                else:
                    trend_bias = "SIDEWAYS / CHOPPY"
                    trend_desc = "MAs are flat or crossing. Neutral strategy better."
                    trend_col = CHART_THEME['secondary']
                
                c1, c2 = st.columns(2)
                c1.markdown(f"<div class='module-card'><div class='metric-label'>Trend vs Range</div><div class='metric-value' style='color:{trend_col}'>{trend_bias}</div><div style='font-size:11px;color:#94a3b8;margin-top:4px;'>{trend_desc}</div></div>", unsafe_allow_html=True)
                
                pivot, r1, s1 = get_pivots(prev['High'], prev['Low'], prev['Close'])
                c2.markdown(f"<div class='module-card'><div class='metric-label'>Key Levels (Pivot/R1/S1)</div><div class='metric-value'>{currency}{pivot:,.2f}</div><div style='font-size:11px;color:#94a3b8;margin-top:4px;'>R1: {currency}{r1:,.2f} | S1: {currency}{s1:,.2f}</div></div>", unsafe_allow_html=True)

                # Step 3: Volatility Analysis
                section_header("3", "VOLATILITY ANALYSIS (OPTIONS PRICING)", "◈")
                
                current_iv = 15.0
                ivr = 0.0
                if vix_data is not None and not vix_data.empty and 'Close' in vix_data.columns:
                    current_iv = safe_get_scalar(vix_data['Close'])
                    v_max = safe_get_scalar(vix_data['Close'].max())
                    v_min = safe_get_scalar(vix_data['Close'].min())
                    if v_max - v_min > 0: ivr = ((current_iv - v_min) / (v_max - v_min)) * 100
                
                if ivr > 50:
                    iv_bias = "HIGH IV (Expensive)"
                    iv_action = "Selling is favorable (Credit Spreads)"
                else:
                    iv_bias = "LOW IV (Cheap)"
                    iv_action = "Buying is favorable (Debit Spreads)"
                
                v1, v2 = st.columns(2)
                v1.markdown(f"<div class='module-card'><div class='metric-label'>Implied Volatility (IV) Rank</div><div class='metric-value'>{ivr:.1f}%</div><div style='font-size:11px;color:#94a3b8;margin-top:4px;'>Current IV: {current_iv:.2f}</div></div>", unsafe_allow_html=True)
                v2.markdown(f"<div class='module-card'><div class='metric-label'>Pricing Status</div><div class='metric-value'>{iv_bias}</div><div style='font-size:11px;color:#94a3b8;margin-top:4px;'>{iv_action}</div></div>", unsafe_allow_html=True)

                # Step 7: Strategy Selection
                section_header("7", "AI STRATEGY COMBINER", "◈")
                
                strat = ""
                if "UPTREND" in trend_bias or "DOWNTREND" in trend_bias:
                    if ivr < 50: strat = "Directional + Low IV → **Option Buying / Debit Spreads**"
                    else: strat = "Directional + High IV → **Credit Spreads**"
                else:
                    if ivr > 50: strat = "Range + High IV → **Iron Condor / Short Strangle**"
                    else: strat = "Explosive Expected → **Long Straddle / Strangle**"
                    
                st.markdown(f"<div style='background:rgba(103,232,249,0.05); padding:15px; border-left:3px solid {CHART_THEME['primary']}; border-radius:4px;'><strong>Strategy Fit:</strong> {strat}</div>", unsafe_allow_html=True)
                st.divider()

                # Steps 4, 5, 6, 8 (Manual Checklist)
                section_header("4-5-6-8", "RISK, EVENT & SENTIMENT CHECKLIST", "◈")
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**Step 5: Event & Catalyst Check**")
                    st.text_input("Shock aa sakta hai kya? (Earnings/Data/Expiry)", placeholder="e.g., Fed/RBI Policy today")
                    st.markdown("**Step 6: Sentiment / Positioning**")
                    st.text_input("Open Interest (OI) / Put-Call Ratio (PCR)", placeholder="e.g., Call writing high at 22,000")
                with colB:
                    st.markdown("**Step 4 & 8: Quantitative & Risk Decision**")
                    st.checkbox("Are Greeks acceptable? (Delta, Gamma, Vega, Theta)")
                    st.checkbox("Is Risk-Reward sensible? (Good Expectancy)")
                    st.text_input("Position Size & Max Acceptable Loss", placeholder="e.g., 2 Lots, Risk ₹5000")
                    st.text_input("Exit Logic (Agar trade galat gaya toh?)", placeholder="e.g., Exit if close < 20 EMA")
                
                st.markdown(f"<div style='color:{CHART_THEME['bearish']}; font-size:12px; font-weight:bold; margin-top:10px;'>🚫 Golden Rule: If any answer unclear → NO TRADE.</div>", unsafe_allow_html=True)
            else:
                st.warning("Insufficient data for Pre-Market Analysis.")

    with tab_post:
        with st.container(border=True):
            st.markdown("## 🌃 POST-TRADE ANALYSIS (10-Step Workflow)")
            
            asset_data = fetch_data(ticker, period="1mo", interval="1d", is_crypto=is_crypto)
            if asset_data is not None and not asset_data.empty and len(asset_data) > 1 and 'Close' in asset_data.columns:
                today = asset_data.iloc[-1]
                yest = asset_data.iloc[-2]
                
                # Step 1: Price Behavior
                section_header("1-5", "PRICE BEHAVIOR & REGIME SHIFT", "◈")
                c1, c2, c3 = st.columns(3)
                c1.metric("EOD Close", f"{currency}{today['Close']:,.2f}", f"{((today['Close']-yest['Close'])/yest['Close'])*100:+.2f}%")
                c2.metric("Day's High", f"{currency}{today['High']:,.2f}")
                c3.metric("Day's Low", f"{currency}{today['Low']:,.2f}")
                
                st.markdown("""
                <div class="pm-grid">
                    <div class="module-card">
                        <strong style="color:#67e8f9;font-size:12px;">Step 1: Price vs Thesis</strong><br>
                        <span style="font-size:11px;color:#94a3b8;">Is price staying inside zone? Is rejection visible? Don’t ask "profit ho raha?", ask "thesis valid hai?"</span>
                    </div>
                    <div class="module-card">
                        <strong style="color:#fbbf24;font-size:12px;">Step 5: Market Environment Shift</strong><br>
                        <span style="font-size:11px;color:#94a3b8;">Range → suddenly trending? Calm → news shock? If regime shifts, strategy may become invalid.</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.divider()
                
                # Step 2 & 3: Greeks & Vol
                section_header("2-3", "GREEKS & VOLATILITY MONITORING", "◈")
                st.markdown("""
                <p style='color:#94a3b8; font-size:12px;'>Options P&L ≠ Price only. Monitor your Greeks!</p>
                """, unsafe_allow_html=True)
                g1, g2, g3, g4 = st.columns(4)
                g1.checkbox("Delta (Direction moving?)")
                g2.checkbox("Gamma (Speed risk high?)")
                g3.checkbox("Vega (IV crush starting?)")
                g4.checkbox("Theta (Time decay hitting?)")
                
                st.divider()

                # Step 6, 7, 8, 9: Review
                section_header("6-9", "DEFENSE & EXIT QUALITY", "◈")
                r1, r2 = st.columns(2)
                with r1:
                    st.selectbox("**Step 6: Adjustment Decision**", ["No Adjustment", "Delta Hedge", "Roll Position", "Reduce Size", "Close Trade"])
                    st.selectbox("**Step 7: Exit Quality**", ["Open Position", "Rule-based Exit", "Panic Exit", "Greed Exit"])
                with r2:
                    st.selectbox("**Step 8: P&L Source (Decomposition)**", ["N/A", "Delta Move (Price)", "Vega Crush/Spike (IV)", "Theta Decay (Time)"])
                    st.markdown("**Step 9: Mistake vs Outcome**")
                    st.radio("Evaluation:", ["Good Decision + Profit", "Good Decision + Loss", "Bad Decision + Profit", "Bad Decision + Loss"], index=0, horizontal=True)
                
                st.divider()
                
                # Step 10: Journal
                section_header("10", "TRADE JOURNAL ENTRY (NON-NEGOTIABLE)", "◈")
                if 'trade_journal' not in st.session_state:
                    st.session_state.trade_journal = pd.DataFrame(columns=[
                        'Setup', 'Thesis', 'Greeks Cond', 'Exit Reason', 'P&L', 'Lessons'
                    ])
                st.session_state.trade_journal = st.data_editor(st.session_state.trade_journal, num_rows="dynamic", use_container_width=True)
            else:
                st.warning("Insufficient data for Post-Market Analysis.")

    with tab_vix:
        with st.container(border=True):
            st.markdown("## 📉 INDIA VIX & SMART MONEY POSITIONING")
            st.markdown("<p style='color:#94a3b8; font-size:13px;'>VIX Regimes, Option Chain PCR, and Volume Profile Logic.</p>", unsafe_allow_html=True)
            
            vix_ticker = "^INDIAVIX" if not is_crypto else ticker
            vix_data = fetch_data(vix_ticker, period="1y", interval="1d", is_crypto=is_crypto)

            if vix_data is not None and not vix_data.empty and 'Close' in vix_data.columns:
                if is_crypto:
                    ret = np.log(vix_data['Close'] / vix_data['Close'].shift(1))
                    synth_vix = ret.rolling(30).std() * np.sqrt(365) * 100
                    vix_data['Close'] = synth_vix
                    vix_data = vix_data.dropna()
                    vix_ticker_name = "Crypto Synth VIX (30D)"
                else:
                    vix_ticker_name = "India VIX"

                current_vix = safe_get_scalar(vix_data['Close'])
                prev_vix = safe_get_scalar(vix_data['Close'].iloc[-2]) if len(vix_data) > 1 else current_vix
                
                # STEP 1: READ VIX
                section_header("1", f"READ {vix_ticker_name.upper()} (MARKET CONDITION)", "◈")
                
                if current_vix < 14:
                    v_regime = "10 - 14 (Dead / Range)"
                    v_bias = "Option Buying (cheap)"
                    v_col = CHART_THEME['primary']
                elif current_vix < 18:
                    v_regime = "15 - 18 (Normal)"
                    v_bias = "Directional"
                    v_col = CHART_THEME['bullish']
                elif current_vix < 25:
                    v_regime = "18 - 25 (Volatile)"
                    v_bias = "Option Selling"
                    v_col = CHART_THEME['secondary']
                else:
                    v_regime = "25+ (Panic)"
                    v_bias = "Mean Reversion / Quick trades"
                    v_col = CHART_THEME['bearish']

                v1, v2, v3 = st.columns(3)
                v1.metric(f"{vix_ticker_name} Level", f"{current_vix:.2f}", f"{current_vix - prev_vix:+.2f}", delta_color="inverse")
                v2.metric("VIX Condition", v_regime)
                v3.metric("Strategy Bias", v_bias)

                fig = go.Figure(go.Scatter(x=vix_data.index[-90:], y=vix_data['Close'].tail(90), mode='lines', line=dict(color=v_col, width=2)))
                fig.update_layout(template=CHART_THEME['template'], height=150, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False), yaxis=dict(visible=False))
                st.plotly_chart(fig, use_container_width=True)
                st.divider()

                # STEP 2: OPTION CHAIN PCR & SMART MONEY
                section_header("2", "OPTION CHAIN (SMART MONEY POSITIONING)", "◈")
                colA, colB = st.columns(2)
                with colA:
                    pcr_val = st.number_input("Enter Live PCR (Put Call Ratio)", value=1.0, step=0.1)
                    pcr_sig = "Bullish" if pcr_val > 1.2 else "Bearish" if pcr_val < 0.8 else "Neutral"
                    st.info(f"**PCR Interpretation:** {pcr_sig}")
                with colB:
                    st.markdown("**Hidden Signal (OI Shift + VIX):**")
                    oi_sel = st.selectbox("Select observed OI build-up:", ["Put OI increasing", "Call OI increasing", "Neutral"])
                    if oi_sel == "Put OI increasing" and current_vix < prev_vix:
                        sig = "Strong Bullish (Support forming, fear dropping)"
                        sig_col = CHART_THEME['bullish']
                    elif oi_sel == "Call OI increasing" and current_vix > prev_vix:
                        sig = "Bearish Pressure (Resistance forming, fear rising)"
                        sig_col = CHART_THEME['bearish']
                    else:
                        sig = "Mixed Signal"
                        sig_col = CHART_THEME['secondary']
                    st.markdown(f"<div style='color:{sig_col}; font-weight:bold;'>{sig}</div>", unsafe_allow_html=True)
                st.divider()

                # STEP 3: COMPLETE STRATEGY SETUPS
                section_header("3", "COMPLETE STRATEGY SETUPS", "◈")
                st.markdown("<p style='color:#94a3b8; font-size:12px;'>AI evaluating conditions for the 4 Master Setups...</p>", unsafe_allow_html=True)

                s1_act = "✅ ACTIVE" if current_vix <= 13 else "❌ Inactive"
                s2_act = "✅ ACTIVE" if current_vix >= 22 else "❌ Inactive"
                s3_act = "✅ ACTIVE" if (current_vix < prev_vix and oi_sel == "Put OI increasing") else "❌ Inactive"
                s4_act = "✅ ACTIVE" if (current_vix > prev_vix and oi_sel == "Call OI increasing") else "❌ Inactive"

                st.markdown(f"""
                <div class="pm-grid">
                    <div class="module-card">
                        <strong style="color:{CHART_THEME['primary']};">SETUP 1: LOW VIX BREAKOUT</strong><br>
                        <span style="font-size:11px; color:#C8D1DC;">Status: <strong>{s1_act}</strong><br>Cond: VIX=11-13. Market sleeping $\\rightarrow$ breakout gives explosive move.<br>Trade: Buy ATM options on VIX spike.</span>
                    </div>
                    <div class="module-card">
                        <strong style="color:{CHART_THEME['bearish']};">SETUP 2: HIGH VIX MEAN REVERSION</strong><br>
                        <span style="font-size:11px; color:#C8D1DC;">Status: <strong>{s2_act}</strong><br>Cond: VIX=22+. Panic already priced $\\rightarrow$ market cools down.<br>Trade: Sell options (Iron Condor).</span>
                    </div>
                    <div class="module-card">
                        <strong style="color:{CHART_THEME['bullish']};">SETUP 3: TREND CONTINUATION (BEST)</strong><br>
                        <span style="font-size:11px; color:#C8D1DC;">Status: <strong>{s3_act}</strong><br>Cond: VIX falling, Put writing up. Smart money accumulating.<br>Trade: Buy on dip (CE or futures).</span>
                    </div>
                    <div class="module-card">
                        <strong style="color:{CHART_THEME['secondary']};">SETUP 4: REVERSAL WARNING (ADVANCED)</strong><br>
                        <span style="font-size:11px; color:#C8D1DC;">Status: <strong>{s4_act}</strong><br>Cond: Market rising BUT VIX also rising. Hidden fear building.<br>Trade: Prepare for short.</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("VIX Data unavailable for Analysis.")

    # FOOTER
    st.markdown("""
        <div style="text-align:center;margin-top:32px;padding:16px;
                    border-top:1px solid #0f1e35;
                    font-family:'JetBrains Mono',monospace;
                    font-size:9px;color:#1e3a5f;letter-spacing:2px;">
            ALADDIN QUANT TERMINAL v26.1 &nbsp;·&nbsp; EMA(20,50,200) &nbsp;·&nbsp; VIX REGIMES &nbsp;·&nbsp;
            PRE/POST PLAYBOOK &nbsp;·&nbsp; VRP &nbsp;·&nbsp; TEXTBLOB NLP &nbsp;·&nbsp; XGBOOST ML<br>
            FOR EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
