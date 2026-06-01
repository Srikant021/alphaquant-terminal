# AlphaQuant Terminal — Fyers‑Inspired GUI, No External TA Library
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
import requests, time, logging, yaml, os, re

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
    /* Fyers‑like theme: dark background, blue accents, clean typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, .stApp { font-family: 'Inter', sans-serif; background: #0E1117; color: #E0E0E0; }
    /* Top header bar */
    .header-bar {
        background: #1A1D24; padding: 10px 20px; border-bottom: 1px solid #2A2E39;
        display: flex; justify-content: space-between; align-items: center;
    }
    .market-index { font-size: 0.9rem; color: #A0A7B8; margin-right: 20px; }
    .market-index span { font-weight: 600; color: #FFFFFF; }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #13161C; border-right: 1px solid #2A2E39;
    }
    .watchlist-item {
        padding: 8px 12px; border-radius: 6px; cursor: pointer; margin: 2px 0;
        transition: background 0.2s;
    }
    .watchlist-item:hover { background: #1E2230; }
    .watchlist-item.selected { background: #2A3A5C; }
    .watchlist-item .symbol { font-weight: 600; color: #FFFFFF; }
    .watchlist-item .change { font-size: 0.8rem; }
    .positive { color: #00C48C; }
    .negative { color: #FF4D4D; }
    /* Main chart area */
    .chart-container {
        background: #13161C; border: 1px solid #2A2E39; border-radius: 8px;
        padding: 10px; margin-bottom: 15px;
    }
    /* Metric cards row */
    .metric-row {
        display: flex; gap: 10px; margin: 15px 0; flex-wrap: wrap;
    }
    .metric-card {
        background: #1A1D24; border: 1px solid #2A2E39; border-radius: 8px;
        padding: 12px 18px; flex: 1; min-width: 120px;
    }
    .metric-card .label { font-size: 0.75rem; color: #A0A7B8; margin-bottom: 4px; }
    .metric-card .value { font-size: 1.3rem; font-weight: 700; color: #FFFFFF; }
    .metric-card .sub { font-size: 0.8rem; color: #7A8296; }
    /* Buttons */
    .stButton>button {
        background: #2A3A5C; color: white; border: none; border-radius: 4px;
        font-weight: 500; transition: background 0.2s;
    }
    .stButton>button:hover { background: #3A4D7A; }
    /* Expandable sections */
    .streamlit-expanderHeader {
        background: #1A1D24; border-radius: 6px; border: 1px solid #2A2E39;
    }
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #13161C; }
    ::-webkit-scrollbar-thumb { background: #2A2E39; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# Session state defaults
for key, default in [
    ('live_mode', False), ('refresh_interval', 120), ('selected_market', 'Crypto'),
    ('selected_analysis', 'Correlation'), ('snapshots', []), ('show_order_flow', False),
    ('trade_journal', []), ('paper_balance', 100000), ('paper_positions', []),
    ('paper_trade_history', []), ('auto_exit_enabled', True), ('ml_model_trained', False)
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
        'cache_ttl': {'long_hist':3600,'garch':1800,'live_price':300,'intraday':300},
        'theme':'dark'
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

# ───── CUSTOM TECHNICAL INDICATORS (no pandas_ta) ─────
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

# ───── DATA UTILITIES (unchanged) ─────
def yf_download_retry(*args, max_retries=3, **kwargs):
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

@st.cache_data(ttl=300)
def fetch_binance_funding_rate(symbol="BTCUSDT"):
    try:
        r=requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",params={"symbol":symbol},timeout=5)
        if r.status_code==200:
            data=r.json(); return float(data['lastFundingRate'])*100
    except: pass
    return None

@st.cache_data(ttl=CACHE_TTL['long_hist'], show_spinner=False)
def fetch_long_hist(ticker_dict):
    bundle={}
    for name,ticker in ticker_dict.items():
        raw=yf_download_retry(ticker, period="2y")
        if not raw.empty: bundle[name]=flatten_df(raw)
    return bundle

@st.cache_data(ttl=120, show_spinner=False)
def fetch_top_prices():
    try:
        data = yf.download(['BTC-USD','ETH-USD'], period="2d", progress=False)
        if not data.empty:
            close = flatten_df(data)['Close']
            btc = float(close['BTC-USD'].iloc[-1]); btc_p = float(close['BTC-USD'].iloc[-2])
            eth = float(close['ETH-USD'].iloc[-1]); eth_p = float(close['ETH-USD'].iloc[-2])
            hist_btc = yf.download('BTC-USD', period="1mo", progress=False)
            hist_eth = yf.download('ETH-USD', period="1mo", progress=False)
            vol_btc = np.log(hist_btc['Close']/hist_btc['Close'].shift(1)).std()*np.sqrt(365)*100 if not hist_btc.empty else 65
            vol_eth = np.log(hist_eth['Close']/hist_eth['Close'].shift(1)).std()*np.sqrt(365)*100 if not hist_eth.empty else 65
            mkt_vol = (vol_btc+vol_eth)/2 if not np.isnan(vol_btc) and not np.isnan(vol_eth) else 65
            return {'btc':btc,'btc_change':btc-btc_p,'btc_pct':((btc-btc_p)/btc_p)*100,
                    'eth':eth,'eth_change':eth-eth_p,'eth_pct':((eth-eth_p)/eth_p)*100 if eth_p else 0,
                    'market_vol':mkt_vol,'timestamp':datetime.now().strftime('%H:%M:%S')}
    except: pass
    return None

@st.cache_data(ttl=120, show_spinner=False)
def fetch_indian_market_summary():
    try:
        nifty = yf.download('^NSEI', period="2d", progress=False)
        sensex = yf.download('^BSESN', period="2d", progress=False)
        if nifty.empty or sensex.empty: return None
        nifty_close = nifty['Close'].squeeze(); sensex_close = sensex['Close'].squeeze()
        nifty_val = float(nifty_close.iloc[-1]); nifty_prev = float(nifty_close.iloc[-2])
        sensex_val = float(sensex_close.iloc[-1]); sensex_prev = float(sensex_close.iloc[-2])
        return {'nifty':nifty_val,'nifty_change':((nifty_val-nifty_prev)/nifty_prev)*100,
                'sensex':sensex_val,'sensex_change':((sensex_val-sensex_prev)/sensex_prev)*100,
                'timestamp':datetime.now().strftime('%H:%M:%S')}
    except: return None

@st.cache_data(ttl=CACHE_TTL['intraday'], show_spinner=False)
def fetch_intraday(ticker,interval="15m"):
    raw=yf_download_retry(ticker, period="5d", interval=interval)
    if raw.empty: return None
    df=flatten_df(raw)
    if not all(c in df.columns for c in ['Open','High','Low','Close']): return None
    return df

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

@st.cache_data(ttl=CACHE_TTL['live_price'], show_spinner=False)
def live_price(ticker):
    raw=yf_download_retry(ticker, period="2d")
    if raw.empty: return None
    df=flatten_df(raw)
    if len(df)<2: return None
    last=float(df['Close'].iloc[-1]); prev=float(df['Close'].iloc[-2])
    chg=last-prev; pct=(chg/prev)*100 if prev else 0
    return {'spot':last,'prev_close':prev,'change':chg,'pct':pct,'ts':datetime.now().strftime('%H:%M:%S')}

@st.cache_data(ttl=300)
def get_india_vix(period="5d"):
    v = yf_download_retry("^INDIAVIX", period=period)['Close'].squeeze()
    if v.empty: return None
    return v

# ───── LAYOUT: HEADER BAR ─────
st.markdown('<div class="header-bar">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns([2,2,1,1])
with col1:
    market_type = st.radio("Market", ["Crypto","Indian"], horizontal=True, label_visibility="collapsed")
    if market_type == "Crypto": selected_market = "Crypto"
    else: selected_market = "Indian Market"
with col2:
    if selected_market == "Crypto":
        ticker_dict = CONFIG['cryptos']; trading_days=365; currency="$"
    else:
        ticker_dict = CONFIG['indian_market']; trading_days=252; currency="₹"
    asset_choice = st.selectbox("Asset", list(ticker_dict.keys()), label_visibility="collapsed")
    ticker = ticker_dict[asset_choice]
with col3:
    live_mode = st.checkbox("Live", value=st.session_state['live_mode'])
    if live_mode != st.session_state['live_mode']: st.session_state['live_mode'] = live_mode
with col4:
    if st.button("Refresh", key="refresh_header"):
        st.cache_data.clear()
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ───── SIDEBAR: WATCHLIST ─────
with st.sidebar:
    st.markdown("### Watchlist")
    for name, tkr in ticker_dict.items():
        live = live_price(tkr)
        if live:
            spot = live['spot']; chg = live['change']
            col1, col2 = st.columns([3,1])
            col1.markdown(f"**{name}**  {currency}{spot:,.2f}")
            delta_class = "positive" if chg >=0 else "negative"
            col2.markdown(f"<span class='{delta_class}'>{chg:+.2f}</span>", unsafe_allow_html=True)
        else:
            st.text(f"{name} --")
    st.markdown("---")
    if ML_AVAILABLE:
        if st.button("Train ML Model"):
            hist = st.session_state.get('long_hist_data',{}).get(asset_choice)
            if hist is not None:
                close = hist['Close'].squeeze(); train_ml_model(close); st.success("Model trained!")
    st.session_state['auto_exit_enabled'] = st.checkbox("Auto Exit (1.5x ATR)", value=st.session_state['auto_exit_enabled'])

# ───── DATA LOADING ─────
if 'long_hist_data' not in st.session_state or asset_choice not in st.session_state.get('long_hist_data', {}):
    st.session_state['long_hist_data'] = fetch_long_hist(ticker_dict)
lp = live_price(ticker)
if lp is None:
    hist = st.session_state['long_hist_data'].get(asset_choice)
    if hist is not None and len(hist) >= 2:
        close = hist['Close'].squeeze()
        spot = float(close.iloc[-1]); prev = float(close.iloc[-2])
        lp = {'spot':spot,'prev_close':prev,'change':spot-prev,'pct':((spot-prev)/prev)*100,'ts':'hist'}
    else: lp = {'spot':0,'prev_close':0,'change':0,'pct':0,'ts':'unavailable'}
asset_spot = lp['spot']
garch_vol, gjrgarch_vol = garch_both(ticker)
park_vol = None; ivr_val = ivp_val = None
if asset_choice in st.session_state['long_hist_data']:
    df = st.session_state['long_hist_data'][asset_choice]
    if not df.empty and all(c in df.columns for c in ['High','Low','Close']):
        high = df['High'].squeeze().tail(60); low = df['Low'].squeeze().tail(60); close = df['Close'].squeeze().tail(60)
        park_vol = calculate_parkinson_volatility(high,low,periods_per_year=trading_days)
        ivr_val, ivp_val = calculate_iv_rank_percentile(close)

# ───── MAIN CHART PANEL ─────
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
with st.spinner("Loading chart..."):
    df_ta = yf_download_retry(ticker, period="6mo", interval="1d")
    if not df_ta.empty:
        df_ta = flatten_df(df_ta)
        df_ta['RSI'] = compute_rsi(df_ta['Close'])
        macd, signal, hist = compute_macd(df_ta['Close'])
        df_ta['MACD'] = macd; df_ta['Signal'] = signal; df_ta['Histogram'] = hist
        bb_upper, bb_mid, bb_lower = compute_bbands(df_ta['Close'])
        fig_ta = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6,0.2,0.2],
                               vertical_spacing=0.02, subplot_titles=("Price","RSI","MACD"))
        fig_ta.add_trace(go.Candlestick(x=df_ta.index, open=df_ta['Open'], high=df_ta['High'],
                                        low=df_ta['Low'], close=df_ta['Close'], name='Price'), row=1, col=1)
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=bb_upper, line=dict(color='gray',width=1), name='Upper'), row=1, col=1)
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=bb_lower, line=dict(color='gray',width=1), name='Lower'), row=1, col=1)
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig_ta.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig_ta.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MACD'], line=dict(color='cyan'), name='MACD'), row=3, col=1)
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta['Signal'], line=dict(color='orange'), name='Signal'), row=3, col=1)
        fig_ta.add_trace(go.Bar(x=df_ta.index, y=df_ta['Histogram'], marker_color='gray', name='Histogram'), row=3, col=1)
        fig_ta.update_layout(template='plotly_dark', height=600, showlegend=False, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_ta, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ───── METRICS ROW (like Fyers’ top stats) ─────
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

# ───── QUICK ORDER PANEL ─────
with st.expander("⚡ Quick Trade (Paper)", expanded=False):
    c1,c2,c3 = st.columns(3)
    dir = c1.selectbox("Direction", ["Long","Short"], key="dir")
    qty = c2.number_input("Qty", min_value=0.01, value=0.01, step=0.01)
    price = c3.number_input("Price", value=asset_spot)
    if st.button("Place Order"):
        cost = qty * price
        if cost > st.session_state['paper_balance']:
            st.error("Insufficient balance")
        else:
            st.session_state['paper_balance'] -= cost
            st.session_state['paper_positions'].append({
                'Asset':asset_choice,'Direction':dir,'Qty':qty,'Entry':price,
                'Type':'Spot','Timestamp':datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            st.success(f"{dir} {qty} {asset_choice} @ {currency}{price:,.2f}")
            st.rerun()

# ───── DETAILED ANALYTICS (collapsible) ─────
analysis_tabs = ["Correlation","Expected Move","Hurst","IVR/IVP","Liquidity","OI Profile","Parkinson","Vol Cone","VRP"]
selected_tab = st.selectbox("Analysis Module", analysis_tabs, index=0)
if selected_tab == "Correlation":
    # correlation plot (simplified)
    corr_data = yf_download_retry(['BTC-USD','ETH-USD'], period="1y")['Close']
    fig, ax = plt.subplots()
    ax.plot(corr_data.index, corr_data)
    st.pyplot(fig)
# ... (add all other analysis modules; same as original but using the custom functions)

# ───── AUTO REFRESH ─────
if live_mode:
    time.sleep(st.session_state['refresh_interval'])
    st.rerun()