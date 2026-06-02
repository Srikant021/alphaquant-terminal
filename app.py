# AlphaQuant Terminal — All Technical Analyses in One Tab + Daily Snapshots for Both Markets
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
</style>
""", unsafe_allow_html=True)

for key, default in [
    ('live_mode', False), ('refresh_interval', 120), ('selected_market', 'Crypto'),
    ('paper_balance', 100000), ('paper_positions', []), ('auto_exit_enabled', True),
    ('ml_model_trained', False), ('snapshots', []), ('chart_tf', '1D'),
    ('active_tab', '📊 Dashboard'), ('habit_data', pd.DataFrame())
]:
    if key not in st.session_state:
        st.session_state[key] = default

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
logging.basicConfig(level=logging.INFO); logger=logging.getLogger(__name__)

# ───── SNAPSHOT FILE ─────
SNAPSHOT_FILE = "daily_snapshots.csv"
def load_snapshots():
    if os.path.exists(SNAPSHOT_FILE):
        try:
            df = pd.read_csv(SNAPSHOT_FILE)
            st.session_state['snapshots'] = df.to_dict('records')
        except: st.session_state['snapshots'] = []
    else: st.session_state['snapshots'] = []
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
        except: return pd.DataFrame(columns=['Date'] + TASKS + ['Score'])
    else: return pd.DataFrame(columns=['Date'] + TASKS + ['Score'])

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
    except: return np.nan

def calculate_parkinson_volatility(high_px,low_px,periods_per_year=252):
    if len(high_px)!=len(low_px) or len(high_px)<2: return 0.0
    log_hl=np.log(high_px/low_px)**2; N=len(log_hl)
    return np.sqrt((log_hl.sum()/(4*N*np.log(2)))*periods_per_year)*100

def calculate_iv_rank_percentile(close_px,window=20):
    if len(close_px)<window: return 50.0,50.0
    log_ret=np.log(close_px/close_px.shift(1)).dropna()
    rolling_vol=log_ret.rolling(window).std()*np.sqrt(252)*100
    cur_vol=rolling_vol.iloc[-1]
    if rolling_vol.empty or np.isnan(cur_vol): return 50.0,50.0
    vol_min,vol_max=rolling_vol.min(),rolling_vol.max()
    ivr=((cur_vol-vol_min)/(vol_max-vol_min))*100 if vol_max!=vol_min else 50.0
    ivp=(rolling_vol<cur_vol).sum()/len(rolling_vol)*100
    return ivr,ivp

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
    if np.isnan(hurst): regime = "Unknown"
    elif hurst > 0.55: regime = "Trending"
    elif hurst < 0.45: regime = "Mean‑Reverting"
    else: regime = "Random Walk"

    sma_alignment = (asset_spot > last_sma20) and (last_sma20 > last_sma50)
    macd_bullish = last_macd > last_signal
    trend_score = 0
    if sma_alignment: trend_score += 2
    if macd_bullish: trend_score += 1
    if adx > 25: trend_score += 1
    if trend_score >= 3: bias = "Bullish"
    elif trend_score == 0: bias = "Bearish"
    else: bias = "Neutral"

    if garch_vol > 70: vol_regime = "High Volatility"
    elif garch_vol < 30: vol_regime = "Low Volatility"
    else: vol_regime = "Moderate"

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
        except: pass
        time.sleep(2**attempt)
    return pd.DataFrame()

def flatten_df(df_raw):
    if isinstance(df_raw.columns, pd.MultiIndex):
        df=df_raw.copy(); df.columns=df_raw.columns.get_level_values(0); return df
    return df_raw

@st.cache_data(ttl=CACHE_TTL['long_hist'], show_spinner=False)
def fetch_long_hist(ticker):
    raw=yf_download_retry(ticker, period="2y")
    return flatten_df(raw) if not raw.empty else pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL['live_price'], show_spinner=False)
def live_price(ticker):
    raw=yf_download_retry(ticker, period="2d")
    if raw.empty: return None
    df=flatten_df(raw)
    if len(df)<2: return None
    last=float(df['Close'].iloc[-1]); prev=float(df['Close'].iloc[-2])
    chg=last-prev; pct=((last-prev)/prev)*100 if prev else 0
    return {'spot':last,'prev_close':prev,'change':chg,'pct':pct,'ts':datetime.now().strftime('%H:%M:%S')}

@st.cache_data(ttl=CACHE_TTL['garch'])
def garch_both(ticker):
    df=yf_download_retry(ticker, period="1y", interval="1d")
    if df.empty: return 80, 80
    close=df['Close'].squeeze()
    ret=100*close.pct_change().dropna()
    if len(ret)<100: return 80, 80
    try:
        m1 = arch_model(ret, vol='GARCH', p=1, q=1, rescale=True).fit(disp='off')
        vol1 = np.sqrt(m1.forecast(horizon=1).variance.iloc[-1].values[0])*np.sqrt(252)
    except: vol1 = 80
    try:
        m2 = arch_model(ret, vol='GARCH', p=1, o=1, q=1, rescale=True).fit(disp='off')
        vol2 = np.sqrt(m2.forecast(horizon=1).variance.iloc[-1].values[0])*np.sqrt(252)
    except: vol2 = 80
    return vol1, vol2

@st.cache_data(ttl=300)
def get_india_vix(period="5d"):
    v = yf_download_retry("^INDIAVIX", period=period)['Close'].squeeze()
    return v if not v.empty else None

# ───── SIDEBAR & NAVIGATION ─────
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
        ticker_dict = CONFIG['cryptos']; trading_days = 365; currency = "$"
    else:
        ticker_dict = CONFIG['indian_market']; trading_days = 252; currency = "₹"

    asset_choice = st.selectbox("Asset", list(ticker_dict.keys()), key="asset_select")
    ticker = ticker_dict[asset_choice]

    st.markdown("---")
    tab_options = ["📊 Dashboard", "📈 Technical", "📓 Habit Tracker"]
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
    st.caption("AlphaQuant Terminal Pro · For learning purposes only.")

# ───── DATA LOADING ─────
hist_data = fetch_long_hist(ticker)
lp = live_price(ticker)
if lp is None and not hist_data.empty:
    close = hist_data['Close'].squeeze()
    if len(close) >= 2:
        spot = float(close.iloc[-1]); prev = float(close.iloc[-2])
        lp = {'spot':spot,'prev_close':prev,'change':spot-prev,'pct':((spot-prev)/prev)*100,'ts':'hist'}
    else:
        lp = {'spot':0,'prev_close':0,'change':0,'pct':0,'ts':'unavailable'}
asset_spot = lp['spot']
garch_vol, gjrgarch_vol = garch_both(ticker)

park_vol = ivr_val = ivp_val = None
if not hist_data.empty and all(c in hist_data.columns for c in ['High','Low','Close']):
    high = hist_data['High'].squeeze().tail(60); low = hist_data['Low'].squeeze().tail(60); close_px = hist_data['Close'].squeeze().tail(60)
    park_vol = calculate_parkinson_volatility(high,low,periods_per_year=trading_days)
    ivr_val, ivp_val = calculate_iv_rank_percentile(close_px)

selected_analysis = compute_full_analysis(hist_data, asset_spot, garch_vol, trading_days, park_vol, ivr_val)

# ───── BTC & NIFTY DATA FOR OVERVIEWS ─────
btc_ticker = 'BTC-USD'
btc_hist = fetch_long_hist(btc_ticker)
btc_lp = live_price(btc_ticker)
if btc_lp is None and not btc_hist.empty:
    close_btc = btc_hist['Close'].squeeze()
    if len(close_btc) >= 2:
        btc_spot = float(close_btc.iloc[-1]); btc_prev = float(close_btc.iloc[-2])
        btc_lp = {'spot':btc_spot,'change':btc_spot-btc_prev,'pct':((btc_spot-btc_prev)/btc_prev)*100}
    else: btc_lp = {'spot':0,'change':0,'pct':0}
btc_garch, _ = garch_both(btc_ticker)
btc_park = None; btc_ivr = None
if not btc_hist.empty and all(c in btc_hist.columns for c in ['High','Low','Close']):
    high_btc = btc_hist['High'].squeeze().tail(60); low_btc = btc_hist['Low'].squeeze().tail(60)
    close_btc_px = btc_hist['Close'].squeeze().tail(60)
    btc_park = calculate_parkinson_volatility(high_btc, low_btc, periods_per_year=365)
    btc_ivr, _ = calculate_iv_rank_percentile(close_btc_px)
btc_analysis = compute_full_analysis(btc_hist, btc_lp['spot'], btc_garch, 365, btc_park, btc_ivr)

nifty_ticker = '^NSEI'
nifty_hist = fetch_long_hist(nifty_ticker)
nifty_lp = live_price(nifty_ticker)
if nifty_lp is None and not nifty_hist.empty:
    close_nifty = nifty_hist['Close'].squeeze()
    if len(close_nifty) >= 2:
        nifty_spot = float(close_nifty.iloc[-1]); nifty_prev = float(close_nifty.iloc[-2])
        nifty_lp = {'spot':nifty_spot,'change':nifty_spot-nifty_prev,'pct':((nifty_spot-nifty_prev)/nifty_prev)*100}
    else: nifty_lp = {'spot':0,'change':0,'pct':0}
nifty_garch, _ = garch_both(nifty_ticker)
nifty_park = None; nifty_ivr = None
if not nifty_hist.empty and all(c in nifty_hist.columns for c in ['High','Low','Close']):
    high_nifty = nifty_hist['High'].squeeze().tail(60); low_nifty = nifty_hist['Low'].squeeze().tail(60)
    close_nifty_px = nifty_hist['Close'].squeeze().tail(60)
    nifty_park = calculate_parkinson_volatility(high_nifty, low_nifty, periods_per_year=252)
    nifty_ivr, _ = calculate_iv_rank_percentile(close_nifty_px)
nifty_analysis = compute_full_analysis(nifty_hist, nifty_lp['spot'], nifty_garch, 252, nifty_park, nifty_ivr)

# Auto‑save snapshots
today_str = datetime.now().strftime('%Y-%m-%d')
snaps = st.session_state['snapshots']
if not any(s.get('date')==today_str and s.get('market')==selected_market and s.get('asset')==asset_choice for s in snaps):
    save_snapshot({'date':today_str, 'asset':asset_choice, 'market':selected_market,
                   'spot':asset_spot, 'garch_vol':garch_vol, 'park_vol':park_vol if park_vol else 0.0,
                   'ivr':ivr_val if ivr_val else 0.0, 'ivp':ivp_val if ivp_val else 0.0,
                   'daily_move': selected_analysis['daily_move'] if selected_analysis else 0})
    st.session_state['snapshots'] = load_snapshots()
if not any(s.get('date')==today_str and s.get('asset')=='Bitcoin' for s in snaps):
    save_snapshot({'date':today_str, 'asset':'Bitcoin', 'market':'Crypto',
                   'spot':btc_lp['spot'], 'garch_vol':btc_garch, 'park_vol':btc_park if btc_park else 0.0,
                   'ivr':btc_ivr if btc_ivr else 0.0, 'daily_move': btc_analysis['daily_move'] if btc_analysis else 0})
    st.session_state['snapshots'] = load_snapshots()
if not any(s.get('date')==today_str and s.get('asset')=='Nifty 50' for s in snaps):
    save_snapshot({'date':today_str, 'asset':'Nifty 50', 'market':'Indian Market',
                   'spot':nifty_lp['spot'], 'garch_vol':nifty_garch, 'park_vol':nifty_park if nifty_park else 0.0,
                   'ivr':nifty_ivr if nifty_ivr else 0.0, 'daily_move': nifty_analysis['daily_move'] if nifty_analysis else 0})
    st.session_state['snapshots'] = load_snapshots()

# ───── COMMON CHART FUNCTIONS (defined globally to avoid NameError) ─────
def chart_correlation():
    corr_tickers = ['BTC-USD','ETH-USD'] if selected_market=="Crypto" else ['^NSEI','^NSEBANK']
    corr_data = yf_download_retry(corr_tickers, period="1y")['Close']
    if corr_data.empty: return None
    names = ("Bitcoin","Ethereum") if selected_market=="Crypto" else ("Nifty","Bank Nifty")
    df = corr_data.dropna(); df.columns = names
    norm = df / df.iloc[0] * 100
    log_ret = np.log(df / df.shift(1)).dropna()
    roll_corr = log_ret[names[0]].rolling(20).corr(log_ret[names[1]])
    cur = roll_corr.iloc[-1]
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,6), gridspec_kw={'height_ratios':[2,1]})
    ax1.plot(norm.index, norm[names[0]], label=names[0])
    ax1.plot(norm.index, norm[names[1]], label=names[1])
    ax1.legend(); ax1.set_title("Correlation")
    ax2.plot(roll_corr.index, roll_corr, color='white')
    ax2.axhline(0.8,color='green',ls='--'); ax2.axhline(0.5,color='red',ls='--'); ax2.set_ylim(-0.2,1.1)
    plt.tight_layout(); return fig

def chart_expected_move():
    iv_use = garch_vol
    if selected_market == "Indian Market":
        vix = get_india_vix("5d")
        if vix is not None: iv_use = float(vix.iloc[-1])
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
    plt.tight_layout(); return fig

def chart_hurst():
    close = hist_data['Close'].squeeze()
    log_p = np.log(close)
    hurst_series = log_p.rolling(60).apply(lambda x: calculate_hurst_exponent(x) if len(x)>=20 else np.nan, raw=False)
    df = pd.DataFrame({'Close':close,'Hurst':hurst_series}).dropna()
    if df.empty: return None
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,7), gridspec_kw={'height_ratios':[2,1]})
    ax1.plot(df.index, df['Close'], color='white')
    ax2.plot(df.index, df['Hurst'], color='cyan')
    ax2.axhline(0.55,color='green',ls='--'); ax2.axhline(0.45,color='red',ls='--')
    ax2.set_ylim(0.3,0.7)
    return fig

def chart_ivr():
    close = hist_data['Close'].squeeze()
    rolling_vol = np.log(close/close.shift(1)).dropna().rolling(20).std()*np.sqrt(252)*100
    vol_series = rolling_vol.dropna()
    cur = vol_series.iloc[-1]
    vmin, vmax = vol_series.min(), vol_series.max()
    ivr = ((cur-vmin)/(vmax-vmin))*100 if vmax!=vmin else 50
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(vol_series.index, vol_series, color='cyan')
    ax.axhline(vmax, color='red', ls='--'); ax.axhline(vmin, color='green', ls='--')
    ax.set_title(f"IV Rank: {ivr:.0f}% | IV Percentile: {ivp_val:.0f}%")
    return fig

def chart_liquidity():
    intra = yf_download_retry(ticker, period="5d", interval="30m")
    if intra is None: return None
    intra = flatten_df(intra).tail(60)
    df = intra.copy()
    df['Prev_High'] = df['High'].rolling(20).max().shift(1)
    df['Prev_Low'] = df['Low'].rolling(20).min().shift(1)
    df['Supply'] = (df['High']>df['Prev_High']) & (df['Close']<df['Prev_High'])
    df['Demand'] = (df['Low']<df['Prev_Low']) & (df['Close']>df['Prev_Low'])
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.index, df['Close'], color='white')
    ax.scatter(df.index[df['Supply']], df['High'][df['Supply']]+10, color='red', marker='v')
    ax.scatter(df.index[df['Demand']], df['Low'][df['Demand']]-10, color='green', marker='^')
    return fig

def chart_oi():
    step = 500 if asset_spot>10000 else 50
    base = round(asset_spot/step)*step
    strikes = np.arange(base-8*step, base+9*step, step)
    np.random.seed(int(asset_spot)%1234)
    calls = np.random.randint(10,80,len(strikes))*50000
    puts = np.random.randint(10,80,len(strikes))*50000
    pain = {k: np.sum(np.maximum(0,k-strikes)*calls + np.maximum(0,strikes-k)*puts) for k in strikes}
    max_pain = min(pain, key=pain.get)
    fig, ax = plt.subplots(figsize=(12,7))
    ax.barh(strikes, calls/1e5, color='red', alpha=0.8, label='Call OI')
    ax.barh(strikes, -puts/1e5, color='green', alpha=0.8, label='Put OI')
    ax.axhline(asset_spot, color='cyan', linewidth=2, label=f'Spot {asset_spot:,.0f}')
    ax.axhline(max_pain, color='white', linestyle='--', label=f'Max Pain {max_pain}')
    ax.legend(); ax.invert_yaxis(); return fig

def chart_park():
    high = hist_data['High'].squeeze().tail(60); low = hist_data['Low'].squeeze().tail(60)
    park_val = calculate_parkinson_volatility(high,low,trading_days)
    fig, ax = plt.subplots()
    ax.bar(['Parkinson Vol'], [park_val], color='orange')
    ax.set_ylabel('%'); ax.set_title(f"Parkinson Vol = {park_val:.1f}%")
    return fig

def chart_cone():
    close = hist_data['Close'].squeeze()
    log_ret = np.log(close/close.shift(1)).dropna()
    windows = [10,20,30,60,90,120,180,252]
    max_v,min_v,med_v,cur_v = [],[],[],[]
    for w in windows:
        rv = log_ret.rolling(w).std()*np.sqrt(trading_days)*100
        if not rv.dropna().empty:
            max_v.append(rv.max()); min_v.append(rv.min()); med_v.append(rv.median()); cur_v.append(rv.iloc[-1])
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(windows, max_v, 'o-', color='red', label='Max')
    ax.plot(windows, min_v, 'o-', color='green', label='Min')
    ax.plot(windows, med_v, 's--', color='white', label='Median')
    ax.plot(windows, cur_v, 'X-', color='yellow', markersize=10, label='Current')
    ax.fill_between(windows, min_v, max_v, alpha=0.2)
    ax.legend(); ax.set_xlabel("Window (days)"); ax.set_ylabel("Vol (%)")
    return fig

def chart_vrp():
    close = hist_data['Close'].squeeze()
    log_ret = np.log(close/close.shift(1)).dropna()
    hv = log_ret.rolling(20).std()*np.sqrt(trading_days)*100
    iv = garch_vol
    if selected_market=="Indian Market":
        v = get_india_vix("6mo")
        if v is not None: iv_series = v; iv = v.iloc[-1]
    else: iv_series = pd.Series([iv]*len(hv), index=hv.index)
    common = hv.index.intersection(iv_series.index)
    hv_c = hv[common]; iv_c = iv_series[common]
    vrp = iv_c - hv_c
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,6), gridspec_kw={'height_ratios':[2,1]})
    ax1.plot(common, iv_c, color='cyan', label='Implied')
    ax1.plot(common, hv_c, color='orange', label='Realised')
    ax1.fill_between(common, hv_c, iv_c, where=(iv_c>hv_c), color='green', alpha=0.3)
    ax1.fill_between(common, hv_c, iv_c, where=(iv_c<=hv_c), color='red', alpha=0.3)
    ax1.legend(); ax1.set_title("Volatility Risk Premium")
    colors = ['green' if v>0 else 'red' for v in vrp]
    ax2.bar(common, vrp, color=colors); ax2.axhline(0, color='white')
    return fig

# ───── ROUTING TO TABS ─────
if st.session_state['active_tab'] == "📊 Dashboard":
    st.title("📊 Market Intelligence Dashboard")

    # Multi‑timeframe chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    tf = st.radio("Chart Timeframe", ["1D", "5m", "15m", "1h"], horizontal=True, key="chart_tf")
    with st.spinner(f"Loading {tf} chart..."):
        if tf == "1D":
            df_chart = yf_download_retry(ticker, period="6mo", interval="1d")
            if not df_chart.empty:
                df_chart = flatten_df(df_chart).tail(60)
                bb_upper, _, bb_lower = compute_bbands(df_chart['Close'])
                fig_main = make_subplots(specs=[[{"secondary_y": False}]])
                fig_main.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
                                                  low=df_chart['Low'], close=df_chart['Close'], name='Price'))
                fig_main.add_trace(go.Scatter(x=df_chart.index, y=bb_upper, line=dict(color='gray',width=1,dash='dot'), name='BB Upper'))
                fig_main.add_trace(go.Scatter(x=df_chart.index, y=bb_lower, line=dict(color='gray',width=1,dash='dot'), name='BB Lower'))
                fig_main.update_layout(template='plotly_dark', height=500,
                                       title=f"{asset_choice} — Daily (last 60 days)",
                                       xaxis_rangeslider_visible=False, hovermode='x unified')
                st.plotly_chart(fig_main, width='stretch')
            else: st.warning("Daily data unavailable.")
        else:
            df_chart = yf_download_retry(ticker, period="5d" if tf in ["5m","15m"] else "1mo", interval=tf)
            if not df_chart.empty:
                df_chart = flatten_df(df_chart).tail(100)
                bb_upper, _, bb_lower = compute_bbands(df_chart['Close'])
                fig_main = make_subplots(specs=[[{"secondary_y": False}]])
                fig_main.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
                                                  low=df_chart['Low'], close=df_chart['Close'], name='Price'))
                fig_main.add_trace(go.Scatter(x=df_chart.index, y=bb_upper, line=dict(color='gray',width=1,dash='dot'), name='BB Upper'))
                fig_main.add_trace(go.Scatter(x=df_chart.index, y=bb_lower, line=dict(color='gray',width=1,dash='dot'), name='BB Lower'))
                daily_move_ = selected_analysis['daily_move'] if selected_analysis else 0
                fig_main.add_hline(y=asset_spot + daily_move_, line_dash="dash", line_color="cyan", annotation_text="Daily Upper")
                fig_main.add_hline(y=asset_spot - daily_move_, line_dash="dash", line_color="cyan", annotation_text="Daily Lower")
                fig_main.update_layout(template='plotly_dark', height=500,
                                       title=f"{asset_choice} — {tf} Chart",
                                       xaxis_rangeslider_visible=False, hovermode='x unified')
                st.plotly_chart(fig_main, width='stretch')
            else: st.warning(f"Could not load {tf} data.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Metrics row
    st.markdown('<div class="metric-row">', unsafe_allow_html=True)
    cols = st.columns(5)
    metrics = [
        ("Spot", f"{currency}{asset_spot:,.2f}", f"{lp['change']:+.2f} ({lp['pct']:+.2f}%)"),
        ("GARCH Vol", f"{garch_vol:.1f}%", "1-day forecast"),
        ("Parkinson", f"{park_vol:.1f}%" if park_vol else "N/A", "High-Low"),
        ("IV Rank", f"{ivr_val:.0f}%" if ivr_val else "N/A", "Sell >65, Buy <30"),
        ("IV Percentile", f"{ivp_val:.0f}%" if ivp_val else "N/A", "vs 1yr")
    ]
    for i, (label, value, sub) in enumerate(metrics):
        with cols[i]:
            st.markdown(f'''<div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
                <div class="sub">{sub}</div>
            </div>''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Same‑market weekly comparison
    with st.expander("📊 Same‑Market Weekly Comparison", expanded=True):
        st.markdown(f"**{selected_market} Assets — Last 7 Days Performance**")
        weekly_data = {}
        for name, tkr in ticker_dict.items():
            df_wk = yf_download_retry(tkr, period="1wk", interval="1d")
            if not df_wk.empty:
                df_wk = flatten_df(df_wk)
                if len(df_wk) >= 2:
                    start_price = df_wk['Close'].iloc[0]
                    end_price = df_wk['Close'].iloc[-1]
                    pct_change = ((end_price - start_price) / start_price) * 100
                    weekly_data[name] = {
                        'current': end_price,
                        'change_pct': pct_change,
                        'start': start_price,
                        'end': end_price
                    }
        if weekly_data:
            comp_df = pd.DataFrame(weekly_data).T.reset_index().rename(columns={'index':'Asset'})
            comp_df['Change %'] = comp_df['change_pct'].apply(lambda x: f"{x:+.2f}%")
            st.dataframe(comp_df[['Asset', 'current', 'Change %']].rename(columns={'current': 'Price'}),
                         use_container_width=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['#00C48C' if v >=0 else '#FF4D4D' for v in comp_df['change_pct']]
            ax.bar(comp_df['Asset'], comp_df['change_pct'], color=colors)
            ax.axhline(0, color='white', linewidth=0.5)
            ax.set_ylabel("Weekly Change %")
            ax.set_title("Same‑Market Weekly Returns")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("No weekly data available for this market.")

    # ───── SEPARATE MARKET OVERVIEW, PREDICTION, AND SNAPSHOT ANALYSIS ─────
    col_left, col_right = st.columns(2)
    with col_left:
        with st.expander("₿ Crypto Market Overview & Prediction", expanded=True):
            if btc_analysis:
                st.subheader("📊 Overview")
                st.markdown(f"**Regime:** {btc_analysis['regime']} (Hurst {btc_analysis['hurst']:.3f})")
                st.markdown(f"**Trend Bias:** {btc_analysis['bias']} (ADX {btc_analysis['adx']:.1f})")
                st.markdown(f"**Volatility:** {btc_analysis['vol_regime']} (GARCH {btc_garch:.1f}%)")
                st.markdown(f"**RSI:** {btc_analysis['last_rsi']:.1f}")
                st.markdown(f"**Expected Move:** ±${btc_analysis['daily_move']:,.2f}")
                st.subheader("🔮 Prediction")
                st.markdown(f"**Today:** {'Bullish' if btc_analysis['bias']=='Bullish' else 'Bearish' if btc_analysis['bias']=='Bearish' else 'Neutral'}")
                st.markdown(f"**Tomorrow:** {'Trend continuation' if btc_analysis['regime']=='Trending' else 'Reversion' if btc_analysis['regime']=='Mean‑Reverting' else 'Rangebound'}")
            else:
                st.warning("Data unavailable")
        # Crypto snapshot analysis
        with st.expander("📸 Crypto Daily Snapshots", expanded=False):
            crypto_snaps = [s for s in st.session_state['snapshots'] if s.get('market') == 'Crypto']
            if crypto_snaps:
                df_c = pd.DataFrame(crypto_snaps)
                df_c['date'] = pd.to_datetime(df_c['date'])
                df_c = df_c.sort_values('date')
                st.dataframe(df_c.tail(5)[['date','asset','spot','garch_vol','ivr']], use_container_width=True)
                if len(df_c) >= 2:
                    latest = df_c.iloc[-1]
                    prev = df_c.iloc[-2]
                    st.markdown(f"**Day‑over‑Day Change:** Spot {latest['spot']-prev['spot']:+.2f} ({(latest['spot']-prev['spot'])/prev['spot']*100:+.2f}%)")
                fig, ax1 = plt.subplots(figsize=(10,4))
                ax1.plot(df_c['date'], df_c['spot'], marker='o', color='white', label='Spot')
                ax2 = ax1.twinx()
                ax2.plot(df_c['date'], df_c['garch_vol'], marker='s', color='cyan', label='GARCH Vol')
                ax1.legend(); plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.info("No Crypto snapshots yet.")

    with col_right:
        with st.expander("🇮🇳 Indian Market Overview & Prediction", expanded=True):
            if nifty_analysis:
                st.subheader("📊 Overview")
                st.markdown(f"**Regime:** {nifty_analysis['regime']} (Hurst {nifty_analysis['hurst']:.3f})")
                st.markdown(f"**Trend Bias:** {nifty_analysis['bias']} (ADX {nifty_analysis['adx']:.1f})")
                st.markdown(f"**Volatility:** {nifty_analysis['vol_regime']} (GARCH {nifty_garch:.1f}%)")
                st.markdown(f"**RSI:** {nifty_analysis['last_rsi']:.1f}")
                st.markdown(f"**Expected Move:** ±₹{nifty_analysis['daily_move']:,.2f}")
                st.subheader("🔮 Prediction")
                st.markdown(f"**Today:** {'Bullish' if nifty_analysis['bias']=='Bullish' else 'Bearish' if nifty_analysis['bias']=='Bearish' else 'Neutral'}")
                st.markdown(f"**Tomorrow:** {'Trend continuation' if nifty_analysis['regime']=='Trending' else 'Reversion' if nifty_analysis['regime']=='Mean‑Reverting' else 'Rangebound'}")
            else:
                st.warning("Data unavailable")
        # Indian snapshot analysis
        with st.expander("📸 Indian Market Daily Snapshots", expanded=False):
            indian_snaps = [s for s in st.session_state['snapshots'] if s.get('market') == 'Indian Market']
            if indian_snaps:
                df_i = pd.DataFrame(indian_snaps)
                df_i['date'] = pd.to_datetime(df_i['date'])
                df_i = df_i.sort_values('date')
                st.dataframe(df_i.tail(5)[['date','asset','spot','garch_vol','ivr']], use_container_width=True)
                if len(df_i) >= 2:
                    latest = df_i.iloc[-1]
                    prev = df_i.iloc[-2]
                    st.markdown(f"**Day‑over‑Day Change:** Spot {latest['spot']-prev['spot']:+.2f} ({(latest['spot']-prev['spot'])/prev['spot']*100:+.2f}%)")
                fig, ax1 = plt.subplots(figsize=(10,4))
                ax1.plot(df_i['date'], df_i['spot'], marker='o', color='white', label='Spot')
                ax2 = ax1.twinx()
                ax2.plot(df_i['date'], df_i['garch_vol'], marker='s', color='cyan', label='GARCH Vol')
                ax1.legend(); plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.info("No Indian market snapshots yet.")

elif st.session_state['active_tab'] == "📈 Technical":
    st.title("📈 Full Technical Analysis Suite")
    st.caption("All modules displayed together with a comprehensive summary.")

    # Combined summary
    if selected_analysis:
        summary_parts = []
        summary_parts.append(f"**Market Regime:** {selected_analysis['regime']} (Hurst={selected_analysis['hurst']:.3f}), **Trend Bias:** {selected_analysis['bias']} (ADX={selected_analysis['adx']:.1f}).")
        summary_parts.append(f"**Volatility:** {selected_analysis['vol_regime']} (GARCH={garch_vol:.1f}%, Parkinson={park_vol:.1f}%). Expected daily move: ±{currency}{selected_analysis['daily_move']:,.2f}.")
        rsi = selected_analysis['last_rsi']
        rsi_status = "Overbought" if rsi>70 else ("Oversold" if rsi<30 else "Neutral")
        summary_parts.append(f"**RSI(14):** {rsi:.1f} ({rsi_status}).")
        macd_status = "Bullish cross" if selected_analysis['macd_bullish'] else "Bearish cross"
        summary_parts.append(f"**MACD:** {macd_status}.")
        sma_line = f"SMA20={selected_analysis['last_sma20']:,.2f}, SMA50={selected_analysis['last_sma50']:,.2f} → {'Bullish alignment' if selected_analysis['last_sma20'] > selected_analysis['last_sma50'] else 'Bearish alignment'}."
        summary_parts.append(sma_line)
        summary_parts.append(f"**Volume:** {selected_analysis['vol_ratio']:.1f}x average.")
        ivr_text = f"**IV Rank:** {ivr_val:.0f}% → {'High – favor selling premium' if ivr_val>65 else ('Low – favor buying premium' if ivr_val<30 else 'Moderate')}."
        summary_parts.append(ivr_text)

        # Correlation
        corr_tickers = ['BTC-USD','ETH-USD'] if selected_market=="Crypto" else ['^NSEI','^NSEBANK']
        corr_data = yf_download_retry(corr_tickers, period="1y")['Close']
        if not corr_data.empty:
            names = ("Bitcoin","Ethereum") if selected_market=="Crypto" else ("Nifty","Bank Nifty")
            df_corr = corr_data.dropna(); df_corr.columns = names
            log_ret = np.log(df_corr / df_corr.shift(1)).dropna()
            roll_corr = log_ret[names[0]].rolling(20).corr(log_ret[names[1]])
            current_corr = roll_corr.iloc[-1]
            corr_desc = "High" if current_corr>0.8 else ("Low" if current_corr<0.5 else "Moderate")
            summary_parts.append(f"**Correlation (20‑day):** {current_corr:.2f} ({corr_desc}).")
        else:
            summary_parts.append("Correlation data unavailable.")

        # Liquidity sweep
        intra = yf_download_retry(ticker, period="5d", interval="30m")
        if intra is not None:
            intra = flatten_df(intra).tail(60)
            supply = ((intra['High'] > intra['High'].rolling(20).max().shift(1)) & (intra['Close'] < intra['High'].rolling(20).max().shift(1))).any()
            demand = ((intra['Low'] < intra['Low'].rolling(20).min().shift(1)) & (intra['Close'] > intra['Low'].rolling(20).min().shift(1))).any()
            if supply:
                summary_parts.append("**Liquidity Sweep:** Supply sweep detected (bearish).")
            elif demand:
                summary_parts.append("**Liquidity Sweep:** Demand sweep detected (bullish).")
            else:
                summary_parts.append("**Liquidity Sweep:** No clear sweep.")
        else:
            summary_parts.append("**Liquidity Sweep:** Intraday data unavailable.")

        # Max Pain
        step = 500 if asset_spot>10000 else 50
        base = round(asset_spot/step)*step
        strikes = np.arange(base-8*step, base+9*step, step)
        np.random.seed(int(asset_spot)%1234)
        calls = np.random.randint(10,80,len(strikes))*50000
        puts = np.random.randint(10,80,len(strikes))*50000
        pain = {k: np.sum(np.maximum(0,k-strikes)*calls + np.maximum(0,strikes-k)*puts) for k in strikes}
        max_pain = min(pain, key=pain.get)
        summary_parts.append(f"**Max Pain (simulated):** {currency}{max_pain:,.0f}.")
        summary_parts.append("**Volatility Cone:** current vol position shown in chart below.")
        summary_parts.append("**VRP:** implied vs realised spread shown in chart below.")

        st.markdown("### 📋 Combined Technical Summary")
        for line in summary_parts:
            st.markdown(f"- {line}")

    # All charts in two‑column grid
    analysis_modules = [
        ("Correlation Analysis", chart_correlation),
        ("Expected Daily & Weekly Move", chart_expected_move),
        ("Hurst Exponent (Market Regime)", chart_hurst),
        ("IV Rank & IV Percentile", chart_ivr),
        ("Liquidity Detector (Sweep)", chart_liquidity),
        ("Open Interest Profile (Max Pain)", chart_oi),
        ("Parkinson Volatility Estimator", chart_park),
        ("Volatility Cone", chart_cone),
        ("Volatility Risk Premium (VRP)", chart_vrp)
    ]
    for i in range(0, len(analysis_modules), 2):
        cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx < len(analysis_modules):
                name, func = analysis_modules[idx]
                with cols[j]:
                    with st.expander(f"📈 {name}", expanded=False):
                        try:
                            fig = func()
                            if fig:
                                st.pyplot(fig)
                            else:
                                st.warning("Data unavailable")
                        except Exception as e:
                            st.error(f"Error: {e}")

elif st.session_state['active_tab'] == "📓 Habit Tracker":
    st.title("📓 Options Seller Habit Tracker")
    st.caption("Track your daily trading disciplines. Data auto‑saves to CSV.")

    df = st.session_state['habit_data']
    today = datetime.now().date()
    current_year, current_month = today.year, today.month
    st.subheader(f"Daily Checklist — {calendar.month_name[current_month]} {current_year}")

    today_date = today
    today_row = df[df['Date'] == today_date]
    if today_row.empty:
        st.warning("No entry for today. Refreshing...")
        initialize_monthly_habit()
        st.rerun()
    else:
        today_idx = today_row.index[0]

    st.markdown("### Today's Tasks")
    col1, col2 = st.columns([3, 1])
    with col1:
        updated_tasks = {}
        for i, task in enumerate(TASKS):
            current_val = bool(df.loc[today_idx, task])
            checked = st.checkbox(task, value=current_val, key=f"habit_{i}")
            updated_tasks[task] = checked
        changes_made = any(updated_tasks[task] != bool(df.loc[today_idx, task]) for task in TASKS)
        if changes_made:
            for task, val in updated_tasks.items():
                df.at[today_idx, task] = val
            df['Score'] = df[TASKS].sum(axis=1) / len(TASKS) * 100
            st.session_state['habit_data'] = df
            save_habit_data(df)
            st.rerun()

    today_score = df.loc[today_idx, 'Score']
    st.metric("Today's Score", f"{today_score:.0f}%")

    st.subheader("Weekly Performance")
    df['Week'] = pd.to_datetime(df['Date']).dt.isocalendar().week
    weekly_avg = df.groupby('Week')['Score'].mean().reset_index()
    weekly_avg.columns = ['Week', 'Average Score']
    def rating(score):
        if score >= 80: return "EXCELLENT"
        elif score >= 60: return "GOOD"
        elif score >= 40: return "FAIR"
        else: return "NEEDS WORK"
    weekly_avg['Rating'] = weekly_avg['Average Score'].apply(rating)
    st.dataframe(weekly_avg, use_container_width=True)

    st.subheader("Monthly Analytics")
    monthly_avg = df['Score'].mean()
    best_day_score = df['Score'].max()
    consistency = (df['Score'] >= 80).mean() * 100
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Monthly Average", f"{monthly_avg:.1f}%")
    col_m2.metric("Best Day", f"{best_day_score:.0f}%")
    col_m3.metric("Consistency (≥80%)", f"{consistency:.1f}%")

    st.subheader("Score Over Time")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df['Date'], df['Score'], marker='o', linestyle='-', color='cyan')
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
    ax.set_ylim(0,105)
    ax.set_ylabel("Score %")
    ax.set_title("Daily Habit Score")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    with st.expander("View Full Month Data"):
        display_df = df[['Date'] + TASKS + ['Score']].copy()
        display_df['Date'] = display_df['Date'].apply(lambda d: d.strftime('%d-%b'))
        st.dataframe(display_df, use_container_width=True)

# ───── AUTO REFRESH ─────
if st.session_state['live_mode']:
    time.sleep(st.session_state['refresh_interval'])
    st.rerun()