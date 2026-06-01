# AlphaQuant Terminal – Auto‑Snapshots, Cross‑Market Comparison, 1D Chart
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
import requests, time, logging, yaml, os

# Optional ML
try:
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

# ───── PAGE CONFIG & FYERS‑INSPIRED CSS ─────
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
    div[data-testid="stTabs"] { background: #13161C; border-radius: 8px; padding: 4px; }
    div[data-testid="stTabs"] button { background: #1A1D24; color: #A0A7B8; border: none; border-radius: 6px; padding: 6px 12px; }
    div[data-testid="stTabs"] button[aria-selected="true"] { background: #2A3A5C; color: white; }
</style>
""", unsafe_allow_html=True)

# Session state
for key, default in [
    ('live_mode', False), ('refresh_interval', 120), ('selected_market', 'Crypto'),
    ('paper_balance', 100000), ('paper_positions', []), ('auto_exit_enabled', True),
    ('ml_model_trained', False), ('snapshots', []), ('chart_tf', '1D')
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Load config
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

# ───── SNAPSHOT FILE MANAGEMENT ─────
SNAPSHOT_FILE = "daily_snapshots.csv"

def load_snapshots():
    if os.path.exists(SNAPSHOT_FILE):
        try:
            df = pd.read_csv(SNAPSHOT_FILE)
            st.session_state['snapshots'] = df.to_dict('records')
        except Exception as e:
            logger.error(f"Snapshot load error: {e}")
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

@st.cache_data(ttl=300)
def fetch_deribit_option_chain(coin="BTC"):
    base="https://www.deribit.com/api/v2/public/"
    try:
        r=requests.get(base+"get_instruments",params={"currency":coin,"kind":"option","expired":"false"},timeout=10)
        if r.status_code!=200: return None
        instruments=r.json()['result']
        option_data=[]
        for inst in instruments[:50]:
            name=inst['instrument_name']
            r2=requests.get(base+"get_order_book",params={"instrument_name":name,"depth":1},timeout=5)
            if r2.status_code==200:
                book=r2.json()['result']
                option_data.append({'instrument':name,'strike':inst['strike'],
                    'option_type':'call' if 'C' in name.split('-')[-1] else 'put',
                    'expiry':inst['expiration_timestamp'],'mark_iv':book.get('mark_iv',0),
                    'underlying_price':book.get('underlying_price',0),
                    'open_interest':book.get('open_interest',0),
                    'delta':book.get('greeks',{}).get('delta',0),
                    'gamma':book.get('greeks',{}).get('gamma',0),
                    'theta':book.get('greeks',{}).get('theta',0),
                    'vega':book.get('greeks',{}).get('vega',0)})
        return pd.DataFrame(option_data)
    except: return None

# ───── HEADER BAR ─────
st.markdown('<div class="header-bar">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns([2,2,1,1])
with col1:
    market_type = st.radio("Market", ["Crypto","Indian"], horizontal=True, label_visibility="collapsed")
    selected_market = "Crypto" if market_type == "Crypto" else "Indian Market"
with col2:
    if selected_market == "Crypto":
        ticker_dict = CONFIG['cryptos']; trading_days=365; currency="$"
    else:
        ticker_dict = CONFIG['indian_market']; trading_days=252; currency="₹"
    asset_choice = st.selectbox("Asset", list(ticker_dict.keys()), label_visibility="collapsed")
    ticker = ticker_dict[asset_choice]
with col3:
    st.session_state['live_mode'] = st.checkbox("Live", value=st.session_state['live_mode'])
with col4:
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ───── MAIN DATA LOADING (selected asset) ─────
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

iv_est = garch_vol / 100
daily_move = asset_spot * iv_est * np.sqrt(1/trading_days)
weekly_move = daily_move * np.sqrt(7)

# ───── AUTO‑SAVE SNAPSHOT FOR TODAY (if not exists) ─────
today_str = datetime.now().strftime('%Y-%m-%d')
existing_snaps = st.session_state['snapshots']
already_saved = any(snap.get('date') == today_str and snap.get('market') == selected_market and snap.get('asset') == asset_choice for snap in existing_snaps)
if not already_saved:
    snap_auto = {
        'date': today_str,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'asset': asset_choice,
        'market': selected_market,
        'spot': asset_spot,
        'garch_vol': garch_vol,
        'park_vol': park_vol if park_vol else 0.0,
        'ivr': ivr_val if ivr_val else 0.0,
        'ivp': ivp_val if ivp_val else 0.0,
        'daily_move': daily_move,
    }
    st.session_state['snapshots'].append(snap_auto)
    save_snapshot(snap_auto)

# ───── CROSS‑MARKET DATA (for comparison) ─────
# Fetch representative asset from the other market
if selected_market == "Crypto":
    other_ticker = "^NSEI"
    other_name = "Nifty 50"
    other_currency = "₹"
    other_trading_days = 252
else:
    other_ticker = "BTC-USD"
    other_name = "Bitcoin"
    other_currency = "$"
    other_trading_days = 365

other_lp = live_price(other_ticker)
if other_lp is None:
    other_hist = fetch_long_hist(other_ticker)
    if not other_hist.empty:
        close_o = other_hist['Close'].squeeze()
        if len(close_o) >= 2:
            spot_o = float(close_o.iloc[-1]); prev_o = float(close_o.iloc[-2])
            other_lp = {'spot':spot_o,'prev_close':prev_o,'change':spot_o-prev_o,
                        'pct':((spot_o-prev_o)/prev_o)*100 if prev_o else 0,'ts':'hist'}
    if other_lp is None:
        other_lp = {'spot':0,'prev_close':0,'change':0,'pct':0,'ts':'unavail'}

other_spot = other_lp['spot']
other_garch, _ = garch_both(other_ticker)

# Correlation between the two markets (using last 6 months daily data)
corr_series = None
try:
    df_cross = yf_download_retry([ticker, other_ticker], period="6mo")['Close']
    if not df_cross.empty:
        df_cross = flatten_df(df_cross)
        # compute daily returns correlation
        rets = df_cross.pct_change().dropna()
        if rets.shape[1] == 2:
            corr_val = rets.iloc[:,0].corr(rets.iloc[:,1])
            corr_series = corr_val
except:
    corr_series = None

# Auto‑save snapshot for the other market (if not already saved)
other_already_saved = any(snap.get('date') == today_str and snap.get('market') == ("Indian Market" if selected_market=="Crypto" else "Crypto") and snap.get('asset') == other_name for snap in existing_snaps)
if not other_already_saved:
    # compute parkinson/ivr for other asset
    other_hist = fetch_long_hist(other_ticker)
    other_park = None; other_ivr = None; other_ivp = None
    if not other_hist.empty and all(c in other_hist.columns for c in ['High','Low','Close']):
        h_o = other_hist['High'].squeeze().tail(60); l_o = other_hist['Low'].squeeze().tail(60); c_o = other_hist['Close'].squeeze().tail(60)
        other_park = calculate_parkinson_volatility(h_o,l_o,periods_per_year=other_trading_days)
        other_ivr, other_ivp = calculate_iv_rank_percentile(c_o)
    snap_other = {
        'date': today_str,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'asset': other_name,
        'market': "Indian Market" if selected_market=="Crypto" else "Crypto",
        'spot': other_spot,
        'garch_vol': other_garch,
        'park_vol': other_park if other_park else 0.0,
        'ivr': other_ivr if other_ivr else 0.0,
        'ivp': other_ivp if other_ivp else 0.0,
        'daily_move': other_spot * (other_garch/100) * np.sqrt(1/other_trading_days),
    }
    st.session_state['snapshots'].append(snap_other)
    save_snapshot(snap_other)

# ───── SIDEBAR (watchlist, ML, snapshot status) ─────
with st.sidebar:
    st.markdown("### Watchlist")
    for name, tkr in ticker_dict.items():
        try:
            live = live_price(tkr)
            if live:
                spot, chg = live['spot'], live['change']
                col1, col2 = st.columns([3,1])
                col1.markdown(f"**{name}**  {currency}{spot:,.2f}")
                delta_class = "positive" if chg >=0 else "negative"
                col2.markdown(f"<span style='color:{'#00C48C' if chg>=0 else '#FF4D4D'}'>{chg:+.2f}</span>", unsafe_allow_html=True)
            else:
                st.text(f"{name} --")
        except:
            st.text(f"{name} --")
    st.markdown("---")
    if ML_AVAILABLE:
        if st.button("Train ML Model"):
            if not hist_data.empty:
                close = hist_data['Close'].squeeze()
                train_ml_model(close)
                st.success("Model trained!")
    st.markdown("---")
    st.caption(f"📸 Auto‑saved snapshot for {today_str} ({selected_market} & other market)")

# ───── CHART SECTION (multi‑timeframe) ─────
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
tf = st.radio("Chart Timeframe", ["1D", "5m", "15m", "1h"], horizontal=True, key="chart_tf")
with st.spinner(f"Loading {tf} chart..."):
    if tf == "1D":
        df_chart = yf_download_retry(ticker, period="6mo", interval="1d")
        if not df_chart.empty:
            df_chart = flatten_df(df_chart).tail(60)
            bb_upper, bb_mid, bb_lower = compute_bbands(df_chart['Close'])
            fig_main = make_subplots(specs=[[{"secondary_y": False}]])
            fig_main.add_trace(go.Candlestick(
                x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
                low=df_chart['Low'], close=df_chart['Close'], name='Price'
            ))
            fig_main.add_trace(go.Scatter(
                x=df_chart.index, y=bb_upper, line=dict(color='gray',width=1,dash='dot'), name='BB Upper'
            ))
            fig_main.add_trace(go.Scatter(
                x=df_chart.index, y=bb_lower, line=dict(color='gray',width=1,dash='dot'), name='BB Lower'
            ))
            fig_main.update_layout(
                template='plotly_dark', height=500,
                title=f"{asset_choice} — Daily (last 60 days)",
                xaxis_rangeslider_visible=False, hovermode='x unified'
            )
            st.plotly_chart(fig_main, width='stretch')
        else:
            st.warning("Daily data unavailable.")
    else:
        df_chart = yf_download_retry(ticker, period="5d" if tf in ["5m","15m"] else "1mo", interval=tf)
        if not df_chart.empty:
            df_chart = flatten_df(df_chart).tail(100)
            bb_upper, bb_mid, bb_lower = compute_bbands(df_chart['Close'])
            last_date = df_chart.index[-1]
            fig_main = make_subplots(specs=[[{"secondary_y": False}]])
            fig_main.add_trace(go.Candlestick(
                x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
                low=df_chart['Low'], close=df_chart['Close'], name='Price'
            ))
            fig_main.add_trace(go.Scatter(
                x=df_chart.index, y=bb_upper, line=dict(color='gray',width=1,dash='dot'), name='BB Upper'
            ))
            fig_main.add_trace(go.Scatter(
                x=df_chart.index, y=bb_lower, line=dict(color='gray',width=1,dash='dot'), name='BB Lower'
            ))
            fig_main.add_hline(y=asset_spot + daily_move, line_dash="dash", line_color="cyan", annotation_text="Daily Upper")
            fig_main.add_hline(y=asset_spot - daily_move, line_dash="dash", line_color="cyan", annotation_text="Daily Lower")
            fig_main.update_layout(
                template='plotly_dark', height=500,
                title=f"{asset_choice} — {tf} Chart",
                xaxis_rangeslider_visible=False, hovermode='x unified'
            )
            st.plotly_chart(fig_main, width='stretch')
        else:
            st.warning(f"Could not load {tf} data.")
st.markdown('</div>', unsafe_allow_html=True)

# ───── METRICS ROW ─────
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

# ───── ANALYSIS MODULES (all nine, fully implemented) ─────
analysis_modules = [
    "Correlation Analysis",
    "Expected Daily & Weekly Move",
    "Hurst Exponent (Market Regime)",
    "IV Rank & IV Percentile",
    "Liquidity Detector (Sweep)",
    "Open Interest Profile (Max Pain)",
    "Parkinson Volatility Estimator",
    "Volatility Cone",
    "Volatility Risk Premium (VRP)"
]
selected_module = st.selectbox("📊 Detailed Analytics", analysis_modules)

def show_module(title, explanation, plot_func, *args, **kwargs):
    st.markdown(f"## {title}")
    st.markdown(explanation)
    with st.spinner("Generating..."):
        fig = plot_func(*args, **kwargs) if callable(plot_func) else None
        if fig:
            st.pyplot(fig)
        else:
            st.warning("Chart could not be generated.")

# Correlation
if selected_module == "Correlation Analysis":
    explanation = """
    **Correlation Analysis** measures how closely two assets move together.  
    - A **rolling 20‑day correlation** between Bitcoin and Ethereum (for Crypto) or Nifty and Bank Nifty (for Indian markets) is computed.  
    - **Above 0.8** → high lockstep movement; trend‑following strategies tend to work.  
    - **Below 0.5** → decoupling; consider neutral or pair‑trading strategies.  
    """
    corr_tickers = ['BTC-USD','ETH-USD'] if selected_market=="Crypto" else ['^NSEI','^NSEBANK']
    def plot_corr():
        corr_data = yf_download_retry(corr_tickers, period="1y")['Close']
        if corr_data.empty: return None
        names = ("Bitcoin","Ethereum") if selected_market=="Crypto" else ("Nifty","Bank Nifty")
        df = corr_data.dropna(); df.columns = names
        norm = df / df.iloc[0] * 100
        log_ret = np.log(df / df.shift(1)).dropna()
        roll_corr = log_ret[names[0]].rolling(20).corr(log_ret[names[1]])
        cur = roll_corr.iloc[-1]
        fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,7), gridspec_kw={'height_ratios':[2,1]})
        ax1.plot(norm.index, norm[names[0]], label=names[0])
        ax1.plot(norm.index, norm[names[1]], label=names[1])
        ax1.legend(); ax1.set_title("Correlation")
        ax2.plot(roll_corr.index, roll_corr, color='white')
        ax2.axhline(0.8,color='green',ls='--'); ax2.axhline(0.5,color='red',ls='--'); ax2.set_ylim(-0.2,1.1)
        plt.tight_layout(); return fig
    show_module("Correlation Analysis", explanation, plot_corr)

# Expected Move
elif selected_module == "Expected Daily & Weekly Move":
    explanation = """
    **Expected Moves** project the ±1σ range for the next day and next week.  
    - **Daily move** = Spot × IV × √(1/trading_days).  
    - **Weekly move** scales the daily move by √7.  
    """
    iv_use = garch_vol
    if selected_market == "Indian Market":
        vix = get_india_vix("5d")
        if vix is not None: iv_use = float(vix.iloc[-1])
    recent_close = hist_data['Close'].squeeze().tail(20)
    def plot_expected():
        daily_vol = iv_use/100/np.sqrt(trading_days)
        daily_move = asset_spot * daily_vol
        weekly_move = daily_move * np.sqrt(7)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(recent_close.index, recent_close.values, color='white')
        last_idx = recent_close.index[-1]
        next_d = last_idx + pd.Timedelta(days=1)
        next_w = last_idx + pd.Timedelta(days=7)
        ax.hlines(asset_spot+daily_move, last_idx, next_d, colors='cyan', linestyles='--')
        ax.hlines(asset_spot-daily_move, last_idx, next_d, colors='cyan', linestyles='--')
        ax.hlines(asset_spot+weekly_move, last_idx, next_w, colors='orange', linestyles='--')
        ax.hlines(asset_spot-weekly_move, last_idx, next_w, colors='orange', linestyles='--')
        ax.set_title("Expected Moves")
        plt.tight_layout(); return fig
    show_module("Expected Daily & Weekly Move", explanation, plot_expected)

# Hurst
elif selected_module == "Hurst Exponent (Market Regime)":
    explanation = """
    **Hurst Exponent (H)** quantifies long‑term memory.  
    - **H > 0.55** → trending.  
    - **H < 0.45** → mean‑reverting.  
    - **H ≈ 0.50** → random walk.  
    """
    def plot_hurst():
        close = hist_data['Close'].squeeze()
        log_p = np.log(close)
        hurst_series = log_p.rolling(60).apply(lambda x: calculate_hurst_exponent(x) if len(x)>=20 else np.nan, raw=False)
        df = pd.DataFrame({'Close':close,'Hurst':hurst_series}).dropna()
        if df.empty: return None
        fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,8), gridspec_kw={'height_ratios':[2,1]})
        ax1.plot(df.index, df['Close'], color='white')
        ax2.plot(df.index, df['Hurst'], color='cyan')
        ax2.axhline(0.55,color='green',ls='--'); ax2.axhline(0.45,color='red',ls='--')
        ax2.set_ylim(0.3,0.7)
        return fig
    show_module("Hurst Exponent", explanation, plot_hurst)

# IVR/IVP
elif selected_module == "IV Rank & IV Percentile":
    explanation = """
    **IV Rank (IVR)** and **IV Percentile (IVP)** tell you if volatility is high or low.  
    - **IVR > 65** → sell premium.  
    - **IVR < 30** → buy premium.  
    """
    def plot_ivr():
        close = hist_data['Close'].squeeze()
        rolling_vol = np.log(close/close.shift(1)).dropna().rolling(20).std()*np.sqrt(252)*100
        vol_series = rolling_vol.dropna()
        cur = vol_series.iloc[-1]
        vmin, vmax = vol_series.min(), vol_series.max()
        ivr = ((cur-vmin)/(vmax-vmin))*100 if vmax!=vmin else 50
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(vol_series.index, vol_series, color='cyan')
        ax.axhline(vmax, color='red', ls='--'); ax.axhline(vmin, color='green', ls='--')
        ax.set_title(f"IV Rank: {ivr:.0f}% | IV Percentile: {ivp_val:.0f}%")
        return fig
    show_module("IV Rank & IV Percentile", explanation, plot_ivr)

# Liquidity
elif selected_module == "Liquidity Detector (Sweep)":
    explanation = """
    **Liquidity sweeps** detect institutional absorption.  
    - **Supply sweep** = bearish.  
    - **Demand sweep** = bullish.  
    """
    intra = yf_download_retry(ticker, period="5d", interval="30m")
    if intra is not None:
        intra = flatten_df(intra).tail(60)
    def plot_liquidity():
        if intra is None: return None
        df = intra.copy()
        df['Prev_High'] = df['High'].rolling(20).max().shift(1)
        df['Prev_Low'] = df['Low'].rolling(20).min().shift(1)
        df['Supply'] = (df['High']>df['Prev_High']) & (df['Close']<df['Prev_High'])
        df['Demand'] = (df['Low']<df['Prev_Low']) & (df['Close']>df['Prev_Low'])
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df.index, df['Close'], color='white')
        ax.scatter(df.index[df['Supply']], df['High'][df['Supply']]+10, color='red', marker='v')
        ax.scatter(df.index[df['Demand']], df['Low'][df['Demand']]-10, color='green', marker='^')
        return fig
    show_module("Liquidity Detector", explanation, plot_liquidity)

# OI Profile
elif selected_module == "Open Interest Profile (Max Pain)":
    explanation = """
    **Open Interest** and **Max Pain** (simulated).  
    - Price often gravitates toward Max Pain.
    """
    step = 500 if asset_spot>10000 else 50
    base = round(asset_spot/step)*step
    strikes = np.arange(base-8*step, base+9*step, step)
    np.random.seed(int(asset_spot)%1234)
    calls = np.random.randint(10,80,len(strikes))*50000
    puts = np.random.randint(10,80,len(strikes))*50000
    pain = {k: np.sum(np.maximum(0,k-strikes)*calls + np.maximum(0,strikes-k)*puts) for k in strikes}
    max_pain = min(pain, key=pain.get)
    def plot_oi():
        fig, ax = plt.subplots(figsize=(14,8))
        ax.barh(strikes, calls/1e5, color='red', alpha=0.8, label='Call OI')
        ax.barh(strikes, -puts/1e5, color='green', alpha=0.8, label='Put OI')
        ax.axhline(asset_spot, color='cyan', linewidth=2, label=f'Spot {asset_spot:,.0f}')
        ax.axhline(max_pain, color='white', linestyle='--', label=f'Max Pain {max_pain}')
        ax.legend(); ax.invert_yaxis(); return fig
    show_module("Open Interest Profile", explanation, plot_oi)

# Parkinson
elif selected_module == "Parkinson Volatility Estimator":
    explanation = """
    **Parkinson volatility** uses high‑low range.  
    - High values → intraday turbulence.  
    """
    def plot_park():
        high = hist_data['High'].squeeze().tail(60); low = hist_data['Low'].squeeze().tail(60)
        park_val = calculate_parkinson_volatility(high,low,trading_days)
        fig, ax = plt.subplots()
        ax.bar(['Parkinson Vol'], [park_val], color='orange')
        ax.set_ylabel('%'); ax.set_title(f"Parkinson Vol = {park_val:.1f}%")
        return fig
    show_module("Parkinson Volatility Estimator", explanation, plot_park)

# Volatility Cone
elif selected_module == "Volatility Cone":
    explanation = """
    **Volatility Cone** shows the historical distribution of vol across lookback windows.  
    - Yellow crosses = current vol.  
    """
    def plot_cone():
        close = hist_data['Close'].squeeze()
        log_ret = np.log(close/close.shift(1)).dropna()
        windows = [10,20,30,60,90,120,180,252]
        max_v,min_v,med_v,cur_v = [],[],[],[]
        for w in windows:
            rv = log_ret.rolling(w).std()*np.sqrt(trading_days)*100
            if not rv.dropna().empty:
                max_v.append(rv.max()); min_v.append(rv.min()); med_v.append(rv.median()); cur_v.append(rv.iloc[-1])
        fig, ax = plt.subplots(figsize=(12,7))
        ax.plot(windows, max_v, 'o-', color='red', label='Max')
        ax.plot(windows, min_v, 'o-', color='green', label='Min')
        ax.plot(windows, med_v, 's--', color='white', label='Median')
        ax.plot(windows, cur_v, 'X-', color='yellow', markersize=10, label='Current')
        ax.fill_between(windows, min_v, max_v, alpha=0.2)
        ax.legend(); ax.set_xlabel("Window (days)"); ax.set_ylabel("Vol (%)")
        return fig
    show_module("Volatility Cone", explanation, plot_cone)

# VRP
elif selected_module == "Volatility Risk Premium (VRP)":
    explanation = """
    **VRP** = implied vol – realised vol.  
    - Positive → sell premium.  
    - Negative → buy premium.  
    """
    def plot_vrp():
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
        fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,7), gridspec_kw={'height_ratios':[2,1]})
        ax1.plot(common, iv_c, color='cyan', label='Implied')
        ax1.plot(common, hv_c, color='orange', label='Realised')
        ax1.fill_between(common, hv_c, iv_c, where=(iv_c>hv_c), color='green', alpha=0.3)
        ax1.fill_between(common, hv_c, iv_c, where=(iv_c<=hv_c), color='red', alpha=0.3)
        ax1.legend(); ax1.set_title("Volatility Risk Premium")
        colors = ['green' if v>0 else 'red' for v in vrp]
        ax2.bar(common, vrp, color=colors); ax2.axhline(0, color='white')
        return fig
    show_module("Volatility Risk Premium (VRP)", explanation, plot_vrp)

# ───── CROSS‑MARKET COMPARISON ─────
with st.expander("🌐 Cross‑Market Comparison", expanded=True):
    st.markdown(f"### {asset_choice} ({selected_market}) vs {other_name} ({'Indian Market' if selected_market=='Crypto' else 'Crypto'})")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.metric(f"{asset_choice} Spot", f"{currency}{asset_spot:,.2f}", f"{lp['change']:+.2f} ({lp['pct']:+.2f}%)")
    with col_c2:
        st.metric(f"{other_name} Spot", f"{other_currency}{other_spot:,.2f}", f"{other_lp['change']:+.2f} ({other_lp['pct']:+.2f}%)")
    col_c3, col_c4 = st.columns(2)
    with col_c3:
        st.metric("GARCH Vol", f"{garch_vol:.1f}%")
    with col_c4:
        st.metric("Other GARCH Vol", f"{other_garch:.1f}%")
    if corr_series is not None:
        st.metric("6‑Month Correlation", f"{corr_series:.2f}")
    # Interpretation
    spot_diff_pct = ((asset_spot - other_spot) / other_spot) * 100 if other_spot else 0
    st.markdown(f"**Relative Performance:** {asset_choice} is {'outperforming' if spot_diff_pct > 0 else 'underperforming'} {other_name} by {abs(spot_diff_pct):.2f}%.")
    vol_spread = garch_vol - other_garch
    st.markdown(f"**Volatility Spread:** {asset_choice} volatility is {abs(vol_spread):.1f}% {'higher' if vol_spread>0 else 'lower'} than {other_name}.")
    if corr_series is not None:
        if corr_series > 0.7:
            st.info("High positive correlation – both markets are moving in tandem.")
        elif corr_series < -0.3:
            st.warning("Negative correlation – markets are moving opposite directions; diversification opportunities.")
        else:
            st.info("Low correlation – markets are largely independent.")

# ───── ENHANCED DAILY MARKET OVERVIEW (from snapshots) ─────
with st.expander("📅 Daily Market Overview (Snapshot Analysis)", expanded=True):
    snaps = st.session_state['snapshots']
    if not snaps:
        st.info("No snapshots yet.")
    else:
        df_snaps = pd.DataFrame(snaps)
        df_snaps['date'] = pd.to_datetime(df_snaps['date'])
        df_snaps = df_snaps.sort_values('date')
        st.subheader("Recent Snapshots")
        st.dataframe(df_snaps.tail(5), use_container_width=True)

        # Technical analysis of selected asset
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

        st.subheader("📊 Comprehensive Market Analysis")
        summary_lines = [
            f"**Market Regime:** {regime} (Hurst = {hurst:.3f})",
            f"**Trend Bias:** {bias} (ADX={adx:.1f}, SMA20={last_sma20:,.2f}, SMA50={last_sma50:,.2f})",
            f"**Volatility Regime:** {vol_regime} (GARCH={garch_vol:.1f}%, Parkinson={park_vol:.1f}%)",
            f"**RSI(14):** {last_rsi:.1f} – {'Overbought' if last_rsi>70 else 'Oversold' if last_rsi<30 else 'Neutral'}",
            f"**MACD:** {'Bullish cross' if macd_bullish else 'Bearish cross'}",
            f"**Volume:** {vol_ratio:.1f}x average (Last: {last_vol:,.0f}, Avg: {avg_vol:,.0f})",
            f"**Expected Daily Move:** ±{currency}{daily_move:,.2f}",
            f"**IV Rank:** {ivr_val:.0f}% – {'Sell premium' if ivr_val>65 else 'Buy premium' if ivr_val<30 else 'Neutral'}",
        ]
        for line in summary_lines:
            st.markdown(line)

        if len(df_snaps) >= 2:
            latest = df_snaps.iloc[-1]
            prev = df_snaps.iloc[-2]
            st.subheader("Day‑over‑Day Changes")
            changes = {
                'Spot': latest['spot'] - prev['spot'],
                'Spot %': ((latest['spot'] - prev['spot']) / prev['spot']) * 100,
                'GARCH Vol': latest['garch_vol'] - prev['garch_vol'],
                'IV Rank': latest['ivr'] - prev['ivr'],
                'Daily Move': latest['daily_move'] - prev['daily_move'],
            }
            changes_df = pd.DataFrame.from_dict(changes, orient='index', columns=['Change'])
            st.dataframe(changes_df.style.format("{:,.2f}"), use_container_width=True)

        st.subheader("Snapshot History")
        fig, ax1 = plt.subplots(figsize=(12,6))
        ax1.plot(df_snaps['date'], df_snaps['spot'], marker='o', color='white', label='Spot')
        ax2 = ax1.twinx()
        ax2.plot(df_snaps['date'], df_snaps['garch_vol'], marker='s', color='cyan', label='GARCH Vol')
        ax2.plot(df_snaps['date'], df_snaps['park_vol'], marker='^', color='orange', label='Parkinson Vol')
        fig.legend(loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

# ───── AUTO REFRESH ─────
if st.session_state['live_mode']:
    time.sleep(st.session_state['refresh_interval'])
    st.rerun()