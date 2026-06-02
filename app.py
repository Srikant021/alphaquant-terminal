# AlphaQuant Terminal — Complete with Free AI Agent, All Technicals, Daily Snapshots, Habit Tracker
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import scipy.stats as si
from arch import arch_model
import requests, time, logging, yaml, os, calendar

try:
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

# ───── PAGE CONFIG & CSS ─────
st.set_page_config(page_title="AlphaQuant Terminal", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, .stApp { font-family: 'Inter', sans-serif; background: #0E1117; color: #E0E0E0; }
    .header-bar { background: #1A1D24; padding: 10px 20px; border-bottom: 1px solid #2A2E39; display: flex; justify-content: space-between; align-items: center; }
    [data-testid="stSidebar"] { background: #13161C; border-right: 1px solid #2A2E39; }
    .chart-container { background: #13161C; border: 1px solid #2A2E39; border-radius: 8px; padding: 10px; margin-bottom: 15px; }
    .metric-row { display: flex; gap: 10px; margin: 15px 0; flex-wrap: wrap; }
    .metric-card { background: #1A1D24; border: 1px solid #2A2E39; border-radius: 8px; padding: 12px 18px; flex: 1; min-width: 120px; }
    .metric-card .label { font-size: 0.75rem; color: #A0A7B8; margin-bottom: 4px; }
    .metric-card .value { font-size: 1.3rem; font-weight: 700; color: #FFFFFF; }
    .metric-card .sub { font-size: 0.8rem; color: #7A8296; }
    .stButton>button { background: #2A3A5C; color: white; border: none; border-radius: 4px; font-weight: 500; }
    .stButton>button:hover { background: #3A4D7A; }
    .stChatMessage { background: #1A1D24; border-radius: 8px; padding: 10px; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

# ───── SESSION STATE ─────
for key, default in [
    ('live_mode', False), ('refresh_interval', 120), ('selected_market', 'Crypto'),
    ('paper_balance', 100000), ('paper_positions', []), ('auto_exit_enabled', True),
    ('ml_model_trained', False), ('snapshots', []), ('chart_tf', '1D'),
    ('active_tab', '📊 Dashboard'), ('habit_data', pd.DataFrame()),
    ('ai_messages', [])
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ───── CONFIG ─────
CONFIG_PATH = "crypto_config.yaml"
def load_config():
    default = {
        'cryptos': {'Bitcoin':'BTC-USD','Ethereum':'ETH-USD','Dogecoin':'DOGE-USD','XRP':'XRP-USD'},
        'indian_market': {'Nifty 50':'^NSEI','Sensex':'^BSESN','Bank Nifty':'^NSEBANK',
                         'Gold (MCX)':'GOLDM.NS','Silver (MCX)':'SILVERM.NS',
                         'Crude Oil (MCX)':'CRUDEOIL.NS','Natural Gas (MCX)':'NATURALGAS.NS'},
        'cache_ttl': {'long_hist':3600,'garch':1800,'live_price':300,'intraday':300}
    }
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH,'r') as f:
            user = yaml.safe_load(f)
            for k,v in user.items():
                if isinstance(v, dict) and k in default: default[k].update(v)
                else: default[k]=v
    return default
CONFIG = load_config()
CACHE_TTL = CONFIG['cache_ttl']

plt.style.use('dark_background')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ───── SNAPSHOT FILE ─────
SNAPSHOT_FILE = "daily_snapshots.csv"
def load_snapshots():
    if os.path.exists(SNAPSHOT_FILE):
        try:
            df = pd.read_csv(SNAPSHOT_FILE)
            st.session_state['snapshots'] = df.to_dict('records')
        except: 
            st.session_state['snapshots'] = []
    else: 
        st.session_state['snapshots'] = []
        
def save_snapshot(snap_dict):
    df_new = pd.DataFrame([snap_dict])
    if os.path.exists(SNAPSHOT_FILE):
        df_new.to_csv(SNAPSHOT_FILE, mode='a', header=False, index=False)
    else:
        df_new.to_csv(SNAPSHOT_FILE, index=False)
        
if 'snapshots_loaded' not in st.session_state:
    load_snapshots()
    st.session_state['snapshots_loaded'] = True

# ───── HABIT TRACKER SETUP ─────
HABIT_FILE = "habit_tracker.csv"
TASKS = [
    "Pre Market Testing Range & Trend by 9:00",
    "Global Market Check (IV/Oil/USDINR/US Market)",
    "Attended PT Session",
    "Paper/Real Trade Done",
    "Mindfulness (Reading/Meditation)",
    "Trading Journal Maintained",
    "Goal Journaling"
]

def load_habit_data():
    if os.path.exists(HABIT_FILE):
        try:
            df = pd.read_csv(HABIT_FILE, parse_dates=['Date'])
            for task in TASKS:
                if task not in df.columns:
                    df[task] = False
            if 'Score' not in df.columns:
                df['Score'] = df[TASKS].sum(axis=1) / len(TASKS) * 100
            return df
        except: 
            return pd.DataFrame(columns=['Date'] + TASKS + ['Score'])
    else: 
        return pd.DataFrame(columns=['Date'] + TASKS + ['Score'])

def save_habit_data(df):
    df.to_csv(HABIT_FILE, index=False)

def initialize_monthly_habit():
    today = datetime.now().date()
    year, month = today.year, today.month
    df = st.session_state['habit_data']
    if df.empty:
        all_dates = pd.date_range(start=datetime(year, month, 1), end=today, freq='D')
        new_rows = [{'Date': d.date(), **{t:False for t in TASKS}} for d in all_dates]
        df = pd.DataFrame(new_rows)
    else:
        existing = set(pd.to_datetime(df['Date']).dt.date)
        d = datetime(year, month, 1).date()
        while d <= today:
            if d not in existing:
                df = pd.concat([df, pd.DataFrame([{'Date': d, **{t:False for t in TASKS}}])], ignore_index=True)
            d += timedelta(days=1)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df = df[df['Date'] <= today]
    df[TASKS] = df[TASKS].fillna(False).astype(bool)
    df['Score'] = df[TASKS].sum(axis=1) / len(TASKS) * 100
    st.session_state['habit_data'] = df
    save_habit_data(df)

if 'habit_data' not in st.session_state or st.session_state['habit_data'].empty:
    st.session_state['habit_data'] = load_habit_data()
initialize_monthly_habit()

# ───── TECHNICAL INDICATORS ─────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bbands(series, period=20, std=2):
    sma = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper = sma + (rolling_std * std)
    lower = sma - (rolling_std * std)
    return upper, sma, lower

def calculate_hurst_exponent(ts):
    if len(ts) < 20: return np.nan
    lags = range(2, min(20, len(ts)//5))
    if len(lags)<3: return np.nan
    try:
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]*2.0
    except: 
        return np.nan

def calculate_parkinson_volatility(high_px, low_px, periods_per_year=252):
    if len(high_px)!=len(low_px) or len(high_px)<2: return 0.0
    log_hl = np.log(high_px/low_px)**2
    N = len(log_hl)
    return np.sqrt((log_hl.sum()/(4*N*np.log(2)))*periods_per_year)*100

def calculate_iv_rank_percentile(close_px, window=20):
    if len(close_px)<window: return 50.0, 50.0
    log_ret = np.log(close_px/close_px.shift(1)).dropna()
    rolling_vol = log_ret.rolling(window).std()*np.sqrt(252)*100
    cur_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else 50
    if rolling_vol.empty or np.isnan(cur_vol): return 50.0, 50.0
    vol_min, vol_max = rolling_vol.min(), rolling_vol.max()
    ivr = ((cur_vol-vol_min)/(vol_max-vol_min))*100 if vol_max!=vol_min else 50.0
    ivp = (rolling_vol<cur_vol).sum()/len(rolling_vol)*100
    return ivr, ivp

def compute_trend_strength(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    up = high - high.shift()
    down = low.shift() - low
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx.iloc[-1] if not adx.empty else 25.0

def compute_volume_profile(df):
    avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
    last_vol = df['Volume'].iloc[-1]
    return last_vol, avg_vol

def compute_full_analysis(hist_data, asset_spot, garch_vol, trading_days, park_vol, ivr_val):
    if hist_data.empty or not all(c in hist_data.columns for c in ['Close','High','Low','Volume']):
        return None
    tech_df = hist_data[['Close','High','Low','Volume']].copy()
    tech_df['RSI'] = compute_rsi(tech_df['Close'])
    tech_df['MACD'], tech_df['Signal'], _ = compute_macd(tech_df['Close'])
    tech_df['SMA20'] = tech_df['Close'].rolling(20).mean()
    tech_df['SMA50'] = tech_df['Close'].rolling(50).mean()
    adx = compute_trend_strength(tech_df['High'], tech_df['Low'], tech_df['Close'])
    last_vol, avg_vol = compute_volume_profile(tech_df)
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

    last_rsi = tech_df['RSI'].iloc[-1]
    last_macd = tech_df['MACD'].iloc[-1]
    last_signal = tech_df['Signal'].iloc[-1]
    last_sma20 = tech_df['SMA20'].iloc[-1]
    last_sma50 = tech_df['SMA50'].iloc[-1]

    log_p = np.log(tech_df['Close'])
    hurst = calculate_hurst_exponent(log_p.values[-200:])
    if np.isnan(hurst): 
        regime = "Unknown"
    elif hurst > 0.55: 
        regime = "Trending"
    elif hurst < 0.45: 
        regime = "Mean‑Reverting"
    else: 
        regime = "Random Walk"

    sma_alignment = (asset_spot > last_sma20) and (last_sma20 > last_sma50)
    macd_bullish = last_macd > last_signal
    trend_score = 0
    if sma_alignment: trend_score += 2
    if macd_bullish: trend_score += 1
    if adx > 25: trend_score += 1
    if trend_score >= 3: 
        bias = "Bullish"
    elif trend_score == 0: 
        bias = "Bearish"
    else: 
        bias = "Neutral"

    if garch_vol > 70: 
        vol_regime = "High Volatility"
    elif garch_vol < 30: 
        vol_regime = "Low Volatility"
    else: 
        vol_regime = "Moderate"

    daily_move = asset_spot * (garch_vol/100) * np.sqrt(1/trading_days)

    return {
        'adx': adx, 'last_rsi': last_rsi, 'last_macd': last_macd,
        'last_signal': last_signal, 'last_sma20': last_sma20, 'last_sma50': last_sma50,
        'vol_ratio': vol_ratio, 'regime': regime, 'bias': bias, 'vol_regime': vol_regime,
        'hurst': hurst, 'macd_bullish': macd_bullish, 'daily_move': daily_move,
        'park_vol': park_vol, 'garch_vol': garch_vol, 'ivr_val': ivr_val
    }

# ───── ML ─────
def train_ml_model(close_px):
    if not ML_AVAILABLE or len(close_px) < 150: return False
    try:
        df = pd.DataFrame(close_px, columns=['close'])
        df['ret'] = df['close'].pct_change()
        df['vol'] = df['ret'].rolling(10).std()
        df['rsi'] = 100 - 100 / (1 + df['ret'].rolling(14).mean() / df['ret'].rolling(14).std())
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df.dropna(inplace=True)
        if len(df) < 100: return False
        X = df[['ret', 'vol', 'rsi', 'macd']].values[-500:]
        y = df['target'].values[-500:]
        model = XGBClassifier(n_estimators=100, max_depth=3)
        model.fit(X, y)
        st.session_state['ml_model'] = model
        st.session_state['ml_model_trained'] = True
        return True
    except Exception as e:
        logger.warning(f"ML training failed: {e}")
        return False

# ───── DATA UTILITIES ─────
def yf_download_retry(*args, max_retries=2, **kwargs):
    for attempt in range(max_retries):
        try:
            data = yf.download(*args, progress=False, **kwargs)
            if data is not None and not data.empty:
                return data
        except: 
            pass
        time.sleep(2**attempt)
    return pd.DataFrame()

def flatten_df(df_raw):
    if isinstance(df_raw.columns, pd.MultiIndex):
        df = df_raw.copy()
        df.columns = df_raw.columns.get_level_values(0)
        return df
    return df_raw

@st.cache_data(ttl=CACHE_TTL['long_hist'], show_spinner=False)
def fetch_long_hist(ticker):
    raw = yf_download_retry(ticker, period="2y")
    return flatten_df(raw) if not raw.empty else pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL['live_price'], show_spinner=False)
def live_price(ticker):
    raw = yf_download_retry(ticker, period="2d")
    if raw.empty: return None
    df = flatten_df(raw)
    if len(df)<2: return None
    last = float(df['Close'].iloc[-1])
    prev = float(df['Close'].iloc[-2])
    chg = last-prev
    pct = ((last-prev)/prev)*100 if prev else 0
    return {'spot':last,'prev_close':prev,'change':chg,'pct':pct,'ts':datetime.now().strftime('%H:%M:%S')}

@st.cache_data(ttl=CACHE_TTL['garch'])
def garch_both(ticker):
    df = yf_download_retry(ticker, period="1y", interval="1d")
    if df.empty: return 80, 80
    close = df['Close'].squeeze()
    ret = 100*close.pct_change().dropna()
    if len(ret)<100: return 80, 80
    try:
        m1 = arch_model(ret, vol='GARCH', p=1, q=1, rescale=True).fit(disp='off')
        vol1 = np.sqrt(m1.forecast(horizon=1).variance.iloc[-1].values[0])*np.sqrt(252)
    except: 
        vol1 = 80
    try:
        m2 = arch_model(ret, vol='GARCH', p=1, o=1, q=1, rescale=True).fit(disp='off')
        vol2 = np.sqrt(m2.forecast(horizon=1).variance.iloc[-1].values[0])*np.sqrt(252)
    except: 
        vol2 = 80
    return vol1, vol2

@st.cache_data(ttl=300)
def get_india_vix(period="5d"):
    v = yf_download_retry("^INDIAVIX", period=period)['Close'].squeeze()
    return v if not v.empty else None

# ───── CHART FUNCTIONS ─────
def chart_correlation():
    corr_tickers = ['BTC-USD','ETH-USD'] if selected_market=="Crypto" else ['^NSEI','^NSEBANK']
    corr_data = yf_download_retry(corr_tickers, period="1y")['Close']
    if corr_data.empty: return None
    names = ("Bitcoin","Ethereum") if selected_market=="Crypto" else ("Nifty","Bank Nifty")
    df = corr_data.dropna()
    df.columns = names
    norm = df / df.iloc[0] * 100
    log_ret = np.log(df / df.shift(1)).dropna()
    roll_corr = log_ret[names[0]].rolling(20).corr(log_ret[names[1]])
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,6), gridspec_kw={'height_ratios':[2,1]})
    ax1.plot(norm.index, norm[names[0]], label=names[0])
    ax1.plot(norm.index, norm[names[1]], label=names[1])
    ax1.legend()
    ax1.set_title("Correlation")
    ax2.plot(roll_corr.index, roll_corr, color='white')
    ax2.axhline(0.8, color='green', ls='--')
    ax2.axhline(0.5, color='red', ls='--')
    ax2.set_ylim(-0.2, 1.1)
    plt.tight_layout()
    return fig

def chart_expected_move():
    iv_use = garch_vol
    if selected_market == "Indian Market":
        vix = get_india_vix("5d")
        if vix is not None: 
            iv_use = float(vix.iloc[-1])
    recent_close = hist_data['Close'].squeeze().tail(20)
    daily_vol = iv_use/100/np.sqrt(trading_days)
    dm = asset_spot * daily_vol
    wm = dm * np.sqrt(7)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(recent_close.index, recent_close.values, color='white')
    last_idx = recent_close.index[-1]
    next_d = last_idx + pd.Timedelta(days=1)
    next_w = last_idx + pd.Timedelta(days=7)
    ax.hlines(asset_spot+dm, last_idx, next_d, colors='cyan', linestyles='--')
    ax.hlines(asset_spot-dm, last_idx, next_d, colors='cyan', linestyles='--')
    ax.hlines(asset_spot+wm, last_idx, next_w, colors='orange', linestyles='--')
    ax.hlines(asset_spot-wm, last_idx, next_w, colors='orange', linestyles='--')
    ax.set_title("Expected Moves")
    plt.tight_layout()
    return fig

# ============================================================================
# SIDEBAR & NAVIGATION
# ============================================================================
with st.sidebar:
    st.markdown("## ⚡ AlphaQuant Terminal")
    market_type = st.radio("Market", ["Crypto", "Indian Market"], horizontal=True,
                           index=0 if st.session_state['selected_market'] == 'Crypto' else 1, 
                           key="market_radio")
    if market_type != st.session_state['selected_market']:
        st.session_state['selected_market'] = market_type
        st.cache_data.clear()
        st.rerun()
    selected_market = st.session_state['selected_market']

    if selected_market == "Crypto":
        ticker_dict = CONFIG['cryptos']
        trading_days = 365
        currency = "$"
    else:
        ticker_dict = CONFIG['indian_market']
        trading_days = 252
        currency = "₹"

    asset_choice = st.selectbox("Asset", list(ticker_dict.keys()), key="asset_select")
    ticker = ticker_dict[asset_choice]

    st.markdown("---")
    tab_options = ["📊 Dashboard", "📈 Technical", "🤖 AI Agent", "📓 Habit Tracker"]
    active_tab = st.radio("Navigate", tab_options, index=tab_options.index(st.session_state['active_tab']))
    if active_tab != st.session_state['active_tab']:
        st.session_state['active_tab'] = active_tab
        st.rerun()

    st.markdown("---")
    live_mode = st.checkbox("🟢 Live Mode", value=st.session_state['live_mode'])
    if live_mode != st.session_state['live_mode']:
        st.session_state['live_mode'] = live_mode

    if ML_AVAILABLE:
        with st.expander("🧠 ML Model"):
            if st.button("Train ML Model"):
                hist = fetch_long_hist(ticker)
                if not hist.empty:
                    close = hist['Close'].squeeze()
                    if train_ml_model(close):
                        st.success("Model trained!")
                    else:
                        st.warning("Insufficient data.")
    st.markdown("---")
    if st.button("🔄 Refresh All Data"):
        st.cache_data.clear()
        st.rerun()
    st.caption("AlphaQuant Terminal Pro · AI-Powered Trading Assistant")

# ============================================================================
# DATA LOADING
# ============================================================================
hist_data = fetch_long_hist(ticker)
lp = live_price(ticker)
if lp is None and not hist_data.empty:
    close = hist_data['Close'].squeeze()
    if len(close) >= 2:
        spot = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        lp = {'spot':spot, 'prev_close':prev, 'change':spot-prev, 'pct':((spot-prev)/prev)*100, 'ts':'hist'}
    else:
        lp = {'spot':0, 'prev_close':0, 'change':0, 'pct':0, 'ts':'unavailable'}
asset_spot = lp['spot']
garch_vol, gjrgarch_vol = garch_both(ticker)

park_vol = ivr_val = ivp_val = None
if not hist_data.empty and all(c in hist_data.columns for c in ['High','Low','Close']):
    high = hist_data['High'].squeeze().tail(60)
    low = hist_data['Low'].squeeze().tail(60)
    close_px = hist_data['Close'].squeeze().tail(60)
    park_vol = calculate_parkinson_volatility(high, low, periods_per_year=trading_days)
    ivr_val, ivp_val = calculate_iv_rank_percentile(close_px)

selected_analysis = compute_full_analysis(hist_data, asset_spot, garch_vol, trading_days, park_vol, ivr_val)

# ============================================================================
# AUTO-SAVE SNAPSHOTS
# ============================================================================
today_str = datetime.now().strftime('%Y-%m-%d')
snaps = st.session_state['snapshots']
if not any(s.get('date')==today_str and s.get('market')==selected_market and s.get('asset')==asset_choice for s in snaps):
    save_snapshot({'date':today_str, 'asset':asset_choice, 'market':selected_market,
                   'spot':asset_spot, 'garch_vol':garch_vol, 'park_vol':park_vol if park_vol else 0.0,
                   'ivr':ivr_val if ivr_val else 0.0, 'ivp':ivp_val if ivp_val else 0.0,
                   'daily_move': selected_analysis['daily_move'] if selected_analysis else 0})
    st.session_state['snapshots'] = load_snapshots()

# ============================================================================
# AI AGENT FUNCTION
# ============================================================================
def ai_agent_response(query, market_data, analysis):
    """Simple rule-based AI agent for trading assistance"""
    query_lower = query.lower()
    
    # Market condition responses
    if any(word in query_lower for word in ['trend', 'direction', 'bias']):
        if analysis:
            return f"""📈 **Market Trend Analysis**
- Current Bias: **{analysis['bias']}**
- ADX (Trend Strength): {analysis['adx']:.1f} {'(Strong Trend)' if analysis['adx'] > 25 else '(Weak Trend)'}
- Market Regime: {analysis['regime']}
- SMA Alignment: {'Bullish' if asset_spot > analysis['last_sma20'] > analysis['last_sma50'] else 'Bearish'}
- MACD: {'Bullish Cross' if analysis['macd_bullish'] else 'Bearish Cross'}

**Suggested Action:** {'Consider bullish strategies' if analysis['bias'] == 'Bullish' else 'Consider bearish/hedging strategies' if analysis['bias'] == 'Bearish' else 'Range-bound - consider neutral strategies'}"""
    
    elif any(word in query_lower for word in ['volatility', 'vol', 'risk']):
        return f"""⚡ **Volatility Analysis**
- GARCH Forecast: {garch_vol:.1f}% 
- Parkinson Vol: {park_vol:.1f}% (if available)
- IV Rank: {ivr_val:.0f}% {'(High - Favor Selling Premium)' if ivr_val and ivr_val > 65 else '(Low - Favor Buying Premium)' if ivr_val and ivr_val < 30 else '(Moderate)'}
- IV Percentile: {ivp_val:.0f}%
- Volatility Regime: {analysis['vol_regime'] if analysis else 'Moderate'}

**Option Strategy Suggestion:** 
{'🔥 **SELL** options (credit spreads, Iron Condors) - Premiums are rich!' if ivr_val and ivr_val > 65 else '📊 **BUY** options or use debit spreads - Volatility is cheap!' if ivr_val and ivr_val < 30 else '⚖️ Neutral strategies like Iron Condors or Strangles'}"""
    
    elif any(word in query_lower for word in ['rsi', 'overbought', 'oversold']):
        if analysis:
            rsi = analysis['last_rsi']
            if rsi > 70:
                return f"""🟡 **RSI Alert: OVERBOUGHT** (RSI = {rsi:.1f})
- Market may be due for a pullback
- Consider: Bearish spreads, put options, or profit booking
- Wait for RSI to cool below 70 before re-entry"""
            elif rsi < 30:
                return f"""🟢 **RSI Alert: OVERSOLD** (RSI = {rsi:.1f})
- Market may be due for a bounce
- Consider: Bullish spreads, call options, or accumulation
- Wait for RSI to rise above 30 for confirmation"""
            else:
                return f"""📊 **RSI Status: NEUTRAL** (RSI = {rsi:.1f})
- No extreme conditions
- Continue with your current strategy
- Watch for break of 70 (overbought) or 30 (oversold)"""
    
    elif any(word in query_lower for word in ['support', 'resistance', 'levels']):
        recent_high = hist_data['High'].squeeze().tail(20).max()
        recent_low = hist_data['Low'].squeeze().tail(20).min()
        pivot = (recent_high + recent_low + asset_spot) / 3
        r1 = 2 * pivot - recent_low
        r2 = pivot + (recent_high - recent_low)
        s1 = 2 * pivot - recent_high
        s2 = pivot - (recent_high - recent_low)
        
        return f"""📐 **Key Levels for {asset_choice}**
Current Spot: {currency}{asset_spot:,.2f}

**Resistance Levels:**
- R1: {currency}{r1:,.2f}
- R2: {currency}{r2:,.2f}

**Support Levels:**
- S1: {currency}{s1:,.2f}
- S2: {currency}{s2:,.2f}

**Recent Range:** {currency}{recent_low:,.2f} - {currency}{recent_high:,.2f}"""
    
    elif any(word in query_lower for word in ['strategy', 'trade', 'position']):
        return f"""💡 **Option Strategy Recommendation**

Based on current market conditions:
- Trend: {analysis['bias'] if analysis else 'Neutral'}
- Volatility: {garch_vol:.1f}%
- IV Rank: {ivr_val:.0f}% (if available)

**Recommended Strategies:**
1. **{'Credit Spreads' if ivr_val and ivr_val > 50 else 'Debit Spreads'}** - Aligns with current volatility environment
2. **{'Iron Condors' if analysis and analysis['vol_regime'] == 'Moderate' else 'Directional Strategies'}**
3. **Always use stop-losses** - Define risk before entry
4. **Position sizing** - Max 2-3% risk per trade

Remember: Options selling requires patience and risk management!"""
    
    elif any(word in query_lower for word in ['morning', 'pre-market', 'setup']):
        return f"""🌅 **Pre-Market Checklist ({datetime.now().strftime('%Y-%m-%d')})**

✅ **Pre-Market Testing (by 9:00 AM)**
- Check overnight global markets
- Identify key support/resistance levels
- Note any major news/events

✅ **Global Market Check**
- US Market closing: {'Green' if btc_lp and btc_lp.get('change', 0) > 0 else 'Red'} 
- Dollar Index trend
- Crude Oil movement
- INR movement vs USD

✅ **Setup for Today**
- Expected range: {currency}{asset_spot - selected_analysis['daily_move'] if selected_analysis else 0:,.2f} - {currency}{asset_spot + selected_analysis['daily_move'] if selected_analysis else 0:,.2f}
- Volatility regime: {analysis['vol_regime'] if analysis else 'Moderate'}
- Key levels identified above

**Action Items:**
1. Complete pre-market analysis
2. Mark key levels on charts
3. Prepare trade plan for both directions"""
    
    else:
        return f"""🤖 **AI Trading Assistant**

I can help you with:
- 📈 **Market Analysis**: Ask about "trend", "volatility", or "RSI"
- 📊 **Technical Levels**: Ask for "support/resistance" or "key levels"
- 💡 **Strategy Suggestions**: Ask for "strategy" or "trade ideas"
- 🌅 **Pre-Market Setup**: Ask about "morning routine" or "pre-market"

**Current Market