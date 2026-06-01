# AlphaQuant Terminal — Full Code + Daily Snapshot & After‑Close Analysis
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
</style>
""", unsafe_allow_html=True)

# Session state defaults
for key, default in [
    ('live_mode', False), ('refresh_interval', 120), ('selected_market', 'Crypto'),
    ('paper_balance', 100000), ('paper_positions', []), ('auto_exit_enabled', True),
    ('ml_model_trained', False), ('snapshots', [])
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
    """Load snapshots from CSV into session state if file exists."""
    if os.path.exists(SNAPSHOT_FILE):
        try:
            df = pd.read_csv(SNAPSHOT_FILE)
            # Convert date column to string to avoid parsing issues
            st.session_state['snapshots'] = df.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to load snapshots: {e}")
            st.session_state['snapshots'] = []
    else:
        st.session_state['snapshots'] = []

def save_snapshot(snap_dict):
    """Append a snapshot dictionary to the CSV file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(SNAPSHOT_FILE) if os.path.dirname(SNAPSHOT_FILE) else '.', exist_ok=True)
    df_new = pd.DataFrame([snap_dict])
    if os.path.exists(SNAPSHOT_FILE):
        df_new.to_csv(SNAPSHOT_FILE, mode='a', header=False, index=False)
    else:
        df_new.to_csv(SNAPSHOT_FILE, index=False)

# Load snapshots on first run
if 'snapshots_loaded' not in st.session_state:
    load_snapshots()
    st.session_state['snapshots_loaded'] = True

# ───── CUSTOM TECHNICAL INDICATORS (NO EXTERNAL TA LIBRARY) ─────
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

def calc_greeks(S,K,T,r,sigma,otype="call"):
    try:
        T=max(T,1e-5)
        d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2=d1-sigma*np.sqrt(T)
        if otype=="call":
            pr=S*si.norm.cdf(d1)-K*np.exp(-r*T)*si.norm.cdf(d2)
            delta=si.norm.cdf(d1); theta=(-(S*si.norm.pdf(d1)*sigma)/(2*np.sqrt(T))-r*K*np.exp(-r*T)*si.norm.cdf(d2))/365
        else:
            pr=K*np.exp(-r*T)*si.norm.cdf(-d2)-S*si.norm.cdf(-d1)
            delta=si.norm.cdf(d1)-1; theta=(-(S*si.norm.pdf(d1)*sigma)/(2*np.sqrt(T))+r*K*np.exp(-r*T)*si.norm.cdf(-d2))/365
        gamma=si.norm.pdf(d1)/(S*sigma*np.sqrt(T)); vega=(S*np.sqrt(T)*si.norm.pdf(d1))/100
        return {'price':max(0.5,pr),'delta':delta,'gamma':gamma,'theta':theta,'vega':vega}
    except: return {'price':25,'delta':0.5,'gamma':0.001,'theta':-1.5,'vega':10}

# ───── DATA UTILITIES (rate‑limit friendly) ─────
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

# ───── ML MODEL ─────
def train_ml_model(close_px):
    if not ML_AVAILABLE: return None
    try:
        df=pd.DataFrame(close_px, columns=['close'])
        df['ret']=df['close'].pct_change(); df['vol']=df['ret'].rolling(10).std()
        df['rsi']=100-100/(1+df['ret'].rolling(14).mean()/df['ret'].rolling(14).std())
        exp12=df['close'].ewm(span=12,adjust=False).mean(); exp26=df['close'].ewm(span=26,adjust=False).mean()
        df['macd']=exp12-exp26; df['target']=(df['close'].shift(-1)>df['close']).astype(int)
        df.dropna(inplace=True)
        if len(df)<100: return None
        X=df[['ret','vol','rsi','macd']].values[-500:]; y=df['target'].values[-500:]
        model=XGBClassifier(n_estimators=100, max_depth=3); model.fit(X,y)
        st.session_state['ml_model']=model; st.session_state['ml_model_trained']=True
        return model
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

# ───── SIDEBAR: WATCHLIST, ML, SNAPSHOT ─────
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
            hist = fetch_long_hist(ticker)
            if not hist.empty:
                close = hist['Close'].squeeze()
                train_ml_model(close)
                st.success("Model trained!")
    st.markdown("---")
    # Daily Snapshot button
    if st.button("📸 Save Today's Snapshot"):
        now = datetime.now()
        snap = {
            'date': now.strftime('%Y-%m-%d'),
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'asset': asset_choice,
            'market': selected_market,
            'spot': asset_spot,
            'garch_vol': garch_vol,
            'park_vol': park_vol if park_vol else 0.0,
            'ivr': ivr_val if ivr_val else 0.0,
            'ivp': ivp_val if ivp_val else 0.0,
            'daily_move': asset_spot * (garch_vol/100) * np.sqrt(1/trading_days),
        }
        # Append to session state and save to CSV
        st.session_state['snapshots'].append(snap)
        save_snapshot(snap)
        st.success("Snapshot saved!")

# ───── MAIN DATA LOADING (only selected asset) ─────
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
    high = hist_data['High'].squeeze().tail(60); low = hist_data['Low'].squeeze().tail(60); close = hist_data['Close'].squeeze().tail(60)
    park_vol = calculate_parkinson_volatility(high,low,periods_per_year=trading_days)
    ivr_val, ivp_val = calculate_iv_rank_percentile(close)

# ───── MAIN CHART ─────
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
with st.spinner("Rendering advanced chart..."):
    df_chart = yf_download_retry(ticker, period="6mo", interval="1d")
    if not df_chart.empty:
        df_chart = flatten_df(df_chart)
        bb_upper, bb_mid, bb_lower = compute_bbands(df_chart['Close'])
        iv_est = garch_vol / 100
        daily_move = asset_spot * iv_est * np.sqrt(1/trading_days)
        weekly_move = daily_move * np.sqrt(7)
        last_date = df_chart.index[-1]
        next_day = last_date + pd.Timedelta(days=1)
        next_week = last_date + pd.Timedelta(days=7)

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
        # Daily expected move (cyan)
        for bound, label in [(asset_spot+daily_move, 'Daily Upper'), (asset_spot-daily_move, 'Daily Lower')]:
            fig_main.add_trace(go.Scatter(
                x=[last_date, next_day, next_week],
                y=[bound]*3,
                mode='lines', line=dict(color='cyan', width=2, dash='dash'), name=label
            ))
        # Weekly expected move (orange)
        for bound, label in [(asset_spot+weekly_move, 'Weekly Upper'), (asset_spot-weekly_move, 'Weekly Lower')]:
            fig_main.add_trace(go.Scatter(
                x=[last_date, next_day, next_week],
                y=[bound]*3,
                mode='lines', line=dict(color='orange', width=2, dash='dot'), name=label
            ))
        # Shading between daily move
        fig_main.add_trace(go.Scatter(
            x=[last_date, next_day], y=[asset_spot-daily_move, asset_spot-daily_move],
            fill=None, mode='lines', line=dict(width=0), showlegend=False
        ))
        fig_main.add_trace(go.Scatter(
            x=[last_date, next_day], y=[asset_spot+daily_move, asset_spot+daily_move],
            fill='tonexty', fillcolor='rgba(0,255,255,0.05)', mode='lines', line=dict(width=0), showlegend=False
        ))
        fig_main.update_layout(
            template='plotly_dark', height=600,
            title=f"{asset_choice} — Daily Chart with Expected Moves",
            xaxis_rangeslider_visible=False, hovermode='x unified'
        )
        st.plotly_chart(fig_main, width='stretch')
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

# ───── ANALYSIS MODULES (with deep explanations) ─────
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

# ───── CORRELATION ANALYSIS ─────
if selected_module == "Correlation Analysis":
    explanation = """
    **Correlation Analysis** measures how closely two assets move together.  
    - A **rolling 20‑day correlation** between Bitcoin and Ethereum (for Crypto) or Nifty and Bank Nifty (for Indian markets) is computed.  
    - **Above 0.8** → high lockstep movement; trend‑following strategies tend to work.  
    - **Below 0.5** → decoupling; consider neutral or pair‑trading strategies.  
    - Watch for sudden drops in correlation – they often precede volatility regime shifts.
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

# ───── EXPECTED MOVE ─────
elif selected_module == "Expected Daily & Weekly Move":
    explanation = """
    **Expected Moves** project the ±1σ range for the next day and next week, derived from the current implied volatility (GARCH forecast or India VIX).  
    - **Daily move** = Spot × IV × √(1/365) for Crypto (or 1/252 for Indian stocks).  
    - **Weekly move** scales the daily move by √7.  
    - These bands are **not** predictions – they represent the range within which the price is expected to stay roughly 68% of the time, assuming a normal distribution.  
    - Use the lower/upper bounds as potential strike areas for options, or as zones for profit‑taking and stop‑loss placement.
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

# ───── HURST ─────
elif selected_module == "Hurst Exponent (Market Regime)":
    explanation = """
    **Hurst Exponent (H)** quantifies the long‑term memory of a price series.  
    - **H > 0.55** → trending market; ride momentum, use trailing stops.  
    - **H < 0.45** → mean‑reverting; fade breakouts, take profits at the mean.  
    - **H ≈ 0.50** → random walk; avoid aggressive directional bets, reduce position size.  
    The chart below shows the 60‑day rolling Hurst overlaid on price.
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

# ───── IV RANK & PERCENTILE ─────
elif selected_module == "IV Rank & IV Percentile":
    explanation = """
    **IV Rank (IVR)** and **IV Percentile (IVP)** tell you whether current volatility is high or low relative to the past.  
    - **IVR** = (current vol – 1‑year low) / (1‑year high – 1‑year low) × 100.  
    - **IVP** = percentage of days where vol was lower than today.  
    - **IVR > 65** → options are expensive; favour selling premium (Iron Condors, Credit Spreads).  
    - **IVR < 30** → options are cheap; favour buying premium (Debit Spreads, Long Straddles).  
    The chart displays the rolling 20‑day historical volatility (or India VIX).
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

# ───── LIQUIDITY SWEEP ─────
elif selected_module == "Liquidity Detector (Sweep)":
    explanation = """
    **Liquidity sweeps** occur when price breaches a previous high/low but then reverses, indicating that larger players absorbed the breakout.  
    - A **supply sweep** (price breaks above recent high, then closes below) is bearish.  
    - A **demand sweep** (price breaks below recent low, then closes above) is bullish.  
    The chart marks these events with markers.
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

# ───── OI PROFILE (MAX PAIN) ─────
elif selected_module == "Open Interest Profile (Max Pain)":
    explanation = """
    **Open Interest (OI)** shows where option positions are clustered.  
    - **Max Pain** is the strike at which option buyers (mainly retail) lose the most money and option sellers profit the most.  
    - Price often gravitates toward Max Pain near expiration.  
    - If spot is below Max Pain → mild bullish bias; above → mild bearish.  
    **Note:** For live markets, OI data is simulated due to API limitations. In a production terminal, this would be real‑time.
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

# ───── PARKINSON ─────
elif selected_module == "Parkinson Volatility Estimator":
    explanation = """
    **Parkinson volatility** uses the daily high‑low range to estimate intraday volatility.  
    - It is a more efficient estimator than close‑to‑close when markets have large intraday swings.  
    - Formula: σ = √( (1/(4·n·ln2)) · Σ(ln(High/Low)² ) ) · √(trading_days)  
    - A rising Parkinson indicates increased intraday turbulence – widen stops, reduce size.
    """
    def plot_park():
        high = hist_data['High'].squeeze().tail(60); low = hist_data['Low'].squeeze().tail(60)
        park_val = calculate_parkinson_volatility(high,low,trading_days)
        fig, ax = plt.subplots()
        ax.bar(['Parkinson Vol'], [park_val], color='orange')
        ax.set_ylabel('%'); ax.set_title(f"Parkinson Vol = {park_val:.1f}%")
        return fig
    show_module("Parkinson Volatility Estimator", explanation, plot_park)

# ───── VOLATILITY CONE ─────
elif selected_module == "Volatility Cone":
    explanation = """
    **Volatility Cone** shows the distribution of historical volatility across different lookback windows.  
    - The yellow crosses show the **current** volatility for each window.  
    - If current vol is near the **max** of the cone → volatility is high, selling premium is favourable.  
    - If near the **min** → volatility is low, buying premium is favourable.  
    - Also helps compare short‑term vs long‑term vol.
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

# ───── VOLATILITY RISK PREMIUM ─────
elif selected_module == "Volatility Risk Premium (VRP)":
    explanation = """
    **Volatility Risk Premium** is the difference between implied volatility (what the market expects) and realised volatility (what actually happened).  
    - **Positive VRP** → implied > realised; on average, option sellers collect a premium.  
    - **Negative VRP** → realised > implied; buyers were right; consider buying options.  
    The chart plots the 20‑day rolling realised vol against the current IV.
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

# ───── DAILY SNAPSHOT ANALYSIS (AFTER MARKET CLOSE OVERVIEW) ─────
with st.expander("📅 Daily Market Overview (Snapshot Analysis)", expanded=True):
    snaps = st.session_state['snapshots']
    if not snaps:
        st.info("No snapshots saved yet. Use the sidebar button to capture today's data.")
    else:
        # Convert to DataFrame for analysis
        df_snaps = pd.DataFrame(snaps)
        # Ensure date is datetime and sorted
        df_snaps['date'] = pd.to_datetime(df_snaps['date'])
        df_snaps = df_snaps.sort_values('date')
        
        st.subheader("Recent Snapshots")
        st.dataframe(df_snaps.tail(5), use_container_width=True)
        
        # Compute day-over-day changes
        if len(df_snaps) >= 2:
            latest = df_snaps.iloc[-1]
            prev = df_snaps.iloc[-2]
            
            st.subheader("Day‑over‑Day Changes")
            changes = {
                'Spot': (latest['spot'] - prev['spot']),
                'Spot %': ((latest['spot'] - prev['spot']) / prev['spot']) * 100,
                'GARCH Vol': (latest['garch_vol'] - prev['garch_vol']),
                'Parkinson Vol': (latest['park_vol'] - prev['park_vol']),
                'IV Rank': (latest['ivr'] - prev['ivr']),
                'IV Percentile': (latest['ivp'] - prev['ivp']),
                'Daily Move': (latest['daily_move'] - prev['daily_move']),
            }
            changes_df = pd.DataFrame.from_dict(changes, orient='index', columns=['Change'])
            st.dataframe(changes_df.style.format("{:,.2f}"), use_container_width=True)
            
            # Generate Market Overview Summary
            st.subheader("Market Overview")
            # Determine regime based on current metrics
            spot_pct_change = changes['Spot %']
            if spot_pct_change > 0.5:
                price_statement = f"Price rose {spot_pct_change:.2f}%, indicating bullish momentum."
            elif spot_pct_change < -0.5:
                price_statement = f"Price fell {abs(spot_pct_change):.2f}%, reflecting bearish pressure."
            else:
                price_statement = "Price was relatively unchanged, showing consolidation."
                
            # Volatility interpretation
            vol_statement = ""
            if changes['GARCH Vol'] > 3:
                vol_statement = "Volatility expanded significantly, suggesting increased uncertainty."
            elif changes['GARCH Vol'] < -3:
                vol_statement = "Volatility contracted, a potential precursor to a breakout."
            else:
                vol_statement = "Volatility remained stable."
                
            # IVR interpretation
            ivr_statement = ""
            if latest['ivr'] > 65:
                ivr_statement = "IV Rank is high – options are expensive; consider selling premium."
            elif latest['ivr'] < 30:
                ivr_statement = "IV Rank is low – options are cheap; consider buying premium."
            else:
                ivr_statement = "IV Rank is moderate – neutral option strategies may be appropriate."
                
            # Expected move context
            move_statement = f"Expected daily move is {currency}{latest['daily_move']:,.2f}, providing a 1σ range for tomorrow's trading."
            
            full_summary = f"{price_statement} {vol_statement} {ivr_statement} {move_statement}"
            st.markdown(f"**Summary:** {full_summary}")
            
            # Trend chart of key metrics over time
            st.subheader("Metrics Trend")
            fig, ax1 = plt.subplots(figsize=(12,6))
            ax1.plot(df_snaps['date'], df_snaps['spot'], marker='o', color='white', label='Spot')
            ax2 = ax1.twinx()
            ax2.plot(df_snaps['date'], df_snaps['garch_vol'], marker='s', color='cyan', label='GARCH Vol')
            ax2.plot(df_snaps['date'], df_snaps['park_vol'], marker='^', color='orange', label='Parkinson Vol')
            fig.legend(loc='upper left')
            ax1.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
        else:
            st.info("At least two snapshots are needed for comparison.")

# ───── AUTO REFRESH ─────
if st.session_state['live_mode']:
    time.sleep(st.session_state['refresh_interval'])
    st.rerun()