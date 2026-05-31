import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import scipy.stats as si
from arch import arch_model
import plotly.graph_objects as go
import yaml, logging, os, time, requests, re

# Optional ML
try:
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

# -----------------------------------------------------------------------------
# PAGE CONFIG & CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AlphaQuant Terminal Pro", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    body, .stApp { font-family: 'Segoe UI', 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0c29, #302b63, #24243e); color: white; }
    .metric-card { background: linear-gradient(135deg, #1e1e2f, #2a2a40); border-radius:12px; padding:15px; margin:5px 0; box-shadow:0 4px 6px rgba(0,0,0,0.3); border:1px solid #3a3a5c; }
    .metric-card h3 { color:#f0f0f0; font-size:0.9rem; margin-bottom:5px; }
    .metric-card .value { font-size:1.5rem; font-weight:700; color:#ffffff; }
    .metric-card .delta { font-size:0.85rem; color:#aaaaaa; }
    .quick-stat { background:rgba(255,255,255,0.05); border-radius:8px; padding:10px; text-align:center; border:1px solid #3a3a5c; transition:all 0.2s ease; }
    .quick-stat:hover { background:rgba(255,255,255,0.1); transform:translateY(-2px); }
    .stButton>button { border-radius:8px; background:linear-gradient(135deg, #667eea, #764ba2); color:white; border:none; font-weight:600; transition:all 0.2s; }
    .stButton>button:hover { transform:scale(1.02); box-shadow:0 4px 12px rgba(118,75,162,0.4); }
    .section-header { font-size:1.3rem; font-weight:700; margin-top:20px; margin-bottom:10px; color:#e0e0ff; }
    .market-summary { background:linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1)); border-radius:12px; padding:20px; border:1px solid #4a4a6a; margin:10px 0; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background:rgba(255,255,255,0.03); border-radius:12px; padding:10px; }
    @media (max-width: 768px) {
        .metric-card { padding:10px; }
        .metric-card h3 { font-size:0.75rem; }
        .metric-card .value { font-size:1.2rem; }
        .quick-stat { padding:5px; }
        .section-header { font-size:1rem; }
        [data-testid="column"] { flex: 1 1 100% !important; }
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'live_mode' not in st.session_state: st.session_state['live_mode'] = False
if 'refresh_interval' not in st.session_state: st.session_state['refresh_interval'] = 120
if 'selected_market' not in st.session_state: st.session_state['selected_market'] = "Crypto"
if 'selected_analysis' not in st.session_state: st.session_state['selected_analysis'] = "Correlation"
if 'snapshots' not in st.session_state: st.session_state['snapshots'] = []
if 'show_order_flow' not in st.session_state: st.session_state['show_order_flow'] = False

for key, default in [
    ('trade_journal', []), ('paper_balance', 100000),
    ('paper_positions', []), ('paper_trade_history', []),
    ('auto_exit_enabled', True), ('ml_model_trained', False)
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
FALLBACK = {'btc':0,'btc_change':0,'btc_pct':0,'eth':0,'eth_change':0,'eth_pct':0,'market_vol':0,'timestamp':'Fallback'}

plt.style.use('dark_background')
logging.basicConfig(level=logging.INFO); logger=logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# ALL HELPER FUNCTIONS (complete, no omissions)
# -----------------------------------------------------------------------------
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

def get_asset_step(spot_price):
    if spot_price>50000: return 2000.0
    elif spot_price>10000: return 500.0
    elif spot_price>1000: return 50.0
    elif spot_price>100: return 10.0
    else: return 1.0

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
    try:
        long_hist = st.session_state.get('long_hist_data', {})
        if long_hist and 'Bitcoin' in long_hist and 'Ethereum' in long_hist:
            btc_close = long_hist['Bitcoin']['Close'].squeeze()
            eth_close = long_hist['Ethereum']['Close'].squeeze()
            if len(btc_close)>=2 and len(eth_close)>=2:
                btc = float(btc_close.iloc[-1]); btc_p = float(btc_close.iloc[-2])
                eth = float(eth_close.iloc[-1]); eth_p = float(eth_close.iloc[-2])
                return {'btc':btc,'btc_change':btc-btc_p,'btc_pct':((btc-btc_p)/btc_p)*100,
                        'eth':eth,'eth_change':eth-eth_p,'eth_pct':((eth-eth_p)/eth_p)*100 if eth_p else 0,
                        'market_vol':65,'timestamp':datetime.now().strftime('%H:%M:%S')}
    except: pass
    return FALLBACK

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

@st.cache_data(ttl=600)
def get_recent_month(ticker):
    return yf_download_retry(ticker, period="1mo")['Close'].squeeze().tail(15)

@st.cache_data(ttl=1800)
def get_hist_6mo(ticker):
    return yf_download_retry(ticker, period="6mo")['Close'].squeeze()

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

# -----------------------------------------------------------------------------
# ML MODEL
# -----------------------------------------------------------------------------
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
        st.session_state['ml_model']=model; st.session_state['ml_features']=['ret','vol','rsi','macd']
        st.session_state['ml_model_trained']=True
        return model
    except: return None

def predict_ml(model, current_price):
    if model is None: return None
    try:
        hist=st.session_state['long_hist_data'].get(asset_choice)
        if hist is None or hist.empty: return None
        close_hist=hist['Close'].squeeze().tail(50)
        if len(close_hist)<14: return None
        ret_hist=close_hist.pct_change().tail(14)
        vol=ret_hist.std(); rsi=100-100/(1+ret_hist.mean()/ret_hist.std()) if ret_hist.std()!=0 else 50
        exp12=close_hist.ewm(span=12,adjust=False).mean().iloc[-1]
        exp26=close_hist.ewm(span=26,adjust=False).mean().iloc[-1]
        macd=exp12-exp26
        features=np.array([[ret_hist.iloc[-1],vol,rsi,macd]])
        return model.predict(features)[0]
    except: return None

def check_auto_exit(pos, current_price, atr14):
    if not st.session_state.get('auto_exit_enabled',True): return False
    direction=pos['Direction']; entry=pos['Entry']
    if direction=='Long':
        if current_price<=entry-1.5*atr14: return True
    else:
        if current_price>=entry+1.5*atr14: return True
    if 'Timestamp' in pos:
        entry_time=datetime.strptime(pos['Timestamp'],'%Y-%m-%d %H:%M:%S')
        if (datetime.now()-entry_time).total_seconds()>86400: return True
    daily_move=asset_spot*(garch_vol_asset/100/np.sqrt(trading_days))
    if direction=='Long' and current_price-entry>=2*daily_move: return True
    if direction=='Short' and entry-current_price>=2*daily_move: return True
    return False

# -----------------------------------------------------------------------------
# ENHANCED SIGNAL
# -----------------------------------------------------------------------------
def get_intraday_signal(asset_choice, ticker, park_vol=None, ivr=None, ivp=None,
                        deribit_iv=None, confluence=0, funding_rate=None):
    confluence=int(confluence) if isinstance(confluence,(int,float)) else 0
    hist_bundle=st.session_state.get('long_hist_data',{})
    asset_df=hist_bundle.get(asset_choice)
    if asset_df is None or asset_df.empty: return {"error":"No historical data"}
    close_px=asset_df.get('Close')
    if close_px is None: return {"error":"No close prices"}
    close_px=close_px.squeeze()
    if len(close_px)<200: return {"error":"Insufficient data"}
    hurst=calculate_hurst_exponent(close_px.values[-200:])
    log_ret20=np.log(close_px.tail(21)/close_px.tail(21).shift(1)).dropna()*100
    hist_vol20=log_ret20.std()*np.sqrt(trading_days) if len(log_ret20)>0 else 70.0
    spot_price=asset_spot; garch_vol=garch_vol_asset
    daily_move=spot_price*(garch_vol/100/np.sqrt(trading_days))
    intraday_df=fetch_intraday(ticker,interval="15m")
    sweeps={'up':False,'down':False}
    if intraday_df is not None and not intraday_df.empty:
        intraday_df.index=pd.to_datetime(intraday_df.index)
        h=intraday_df.get('High'); l=intraday_df.get('Low'); c=intraday_df.get('Close')
        if h is not None and l is not None and c is not None:
            sweeps['up']=((h>h.shift(1))&(c<h.shift(1))).any()
            sweeps['down']=((l<l.shift(1))&(c>l.shift(1))).any()
    if hurst>0.52: regime="trending"
    elif hurst<0.48: regime="mean_reverting"
    else: regime="random_walk"
    vol_env="neutral"; vol_factors=0
    garch_diff=garch_vol-hist_vol20
    if garch_diff>8: vol_env="expensive"; vol_factors+=1
    elif garch_diff<-8: vol_env="cheap"; vol_factors+=1
    if ivr is not None:
        if ivr>70: vol_env="expensive" if vol_env!="cheap" else "mixed"; vol_factors+=1
        elif ivr<30: vol_env="cheap" if vol_env!="expensive" else "mixed"; vol_factors+=1
    if park_vol is not None:
        if park_vol>garch_vol*1.2: vol_env="expensive" if vol_env!="cheap" else "mixed"; vol_factors+=1
        elif park_vol<garch_vol*0.8: vol_env="cheap" if vol_env!="expensive" else "mixed"; vol_factors+=1
    if deribit_iv is not None:
        if deribit_iv>garch_vol*1.2: vol_env="expensive" if vol_env!="cheap" else "mixed"; vol_factors+=1
        elif deribit_iv<garch_vol*0.8: vol_env="cheap" if vol_env!="expensive" else "mixed"; vol_factors+=1
    if funding_rate is not None:
        if funding_rate>0.1: vol_env="expensive"
        elif funding_rate<-0.1: vol_env="cheap"
    if vol_factors==0: vol_env="neutral"
    sweep_signal="none"
    if sweeps['up']:
        base_signal="bearish_reversal" if spot_price>=(spot_price+daily_move)*0.95 else "bearish"
        sweep_signal=base_signal+("_strong" if confluence>=2 else "")
    if sweeps['down']:
        base_signal="bullish_reversal" if spot_price<=(spot_price-daily_move)*1.05 else "bullish"
        sweep_signal=base_signal+("_strong" if confluence>=2 else "")
    confidence=50; risk_level="medium"
    direction="neutral"; strategy="Iron Condor"
    if regime=="trending":
        direction="bullish" if asset_change>0 else "bearish"
        strategy="Long Call" if vol_env=="cheap" else ("Bull Call Spread" if direction=="bullish" else "Bear Put Spread")
        confidence+=15 if vol_env=="cheap" else 5
        if confluence>=2: confidence+=15
        risk_level="medium" if "Spread" in strategy else "high"
    elif regime=="mean_reverting":
        if sweep_signal.startswith("bullish"):
            direction="bearish"
            strategy="Bear Put Spread" if spot_price>(spot_price+daily_move*0.5) else "Short Call Spread"
        elif sweep_signal.startswith("bearish"):
            direction="bullish"
            strategy="Bull Call Spread" if spot_price<(spot_price-daily_move*0.5) else "Short Put Spread"
        else:
            direction="neutral"; strategy="Iron Condor"
        confidence+=10 if confluence>=2 else 5
        risk_level="low" if strategy=="Iron Condor" else "medium"
    else:
        if vol_env=="expensive": strategy="Short Strangle"; risk_level="medium"
        elif sweeps['up']: strategy="Bear Put Spread"; direction="bearish"
        elif sweeps['down']: strategy="Bull Call Spread"; direction="bullish"
        else: strategy="Iron Condor"
        risk_level="low" if strategy=="Iron Condor" else "medium"
        confidence+=5 if confluence>=2 else 0
    if vol_factors>=2: confidence+=10
    if ivr and (ivr>80 or ivr<20): confidence+=5
    if "strong" in sweep_signal: confidence+=10
    confidence=min(100,confidence)
    ml_pred=None
    if st.session_state.get('ml_model_trained',False):
        model=st.session_state.get('ml_model'); ml_pred=predict_ml(model, spot_price)
        if ml_pred is not None:
            if (ml_pred==1 and direction=="bullish") or (ml_pred==0 and direction=="bearish"): confidence+=10
            elif ml_pred==1 and direction=="bearish": confidence-=5
            elif ml_pred==0 and direction=="bullish": confidence-=5
            confidence=max(0,confidence)
    reasoning_parts=[
        f"Hurst={hurst:.3f}({regime})", f"GARCH Vol={garch_vol:.1f}%", f"20d Hist={hist_vol20:.1f}%",
    ]
    if park_vol: reasoning_parts.append(f"Parkinson={park_vol:.1f}%")
    if ivr: reasoning_parts.append(f"IVR={ivr:.0f}% IVP={ivp:.0f}%")
    if deribit_iv: reasoning_parts.append(f"Deribit IV={deribit_iv:.1f}%")
    if funding_rate: reasoning_parts.append(f"Funding={funding_rate:.3f}%")
    reasoning_parts.append(f"Vol:{vol_env} Sweep:{'Up' if sweeps['up'] else 'Down' if sweeps['down'] else 'None'} Confl:{confluence}/3")
    reasoning_parts.append(f"Daily±{currency}{daily_move:,.0f}")
    reasoning_parts.append(f"Conf={confidence}% Risk={risk_level}")
    if ml_pred is not None: reasoning_parts.append(f"ML={'Bull' if ml_pred==1 else 'Bear'}")
    return {
        "regime":regime,"vol_environment":vol_env,"sweep_signal":sweep_signal,
        "suggested_strategy":strategy,"direction":direction,"confidence":confidence,
        "risk_level":risk_level,"reasoning":" · ".join(reasoning_parts),
        "daily_move":daily_move,"garch_vol":garch_vol,"park_vol":park_vol,
        "ivr":ivr,"ivp":ivp,"deribit_iv":deribit_iv,"confluence":confluence,
        "funding_rate":funding_rate,"ml_pred":ml_pred
    }

def generate_trading_tips(signal):
    tips=["🛑 Risk 1‑2% per trade.","📊 Size inversely to volatility.","⏳ Close short options before final hour."]
    strat=signal.get('suggested_strategy',''); daily_move=signal.get('daily_move',0); spot=asset_spot
    if "Iron Condor" in strat: tips.append(f"🕊️ Short strikes near ±{currency}{daily_move:,.0f}.")
    elif "Spread" in strat:
        if "Bull" in strat: tips.append(f"📈 Long {spot:,.0f} call, short {spot+daily_move:,.0f} call.")
        elif "Bear" in strat: tips.append(f"📉 Long {spot:,.0f} put, short {spot-daily_move:,.0f} put.")
    if signal.get('ml_pred') is not None: tips.append(f"🤖 ML predicts {'up' if signal['ml_pred']==1 else 'down'} next period.")
    if signal.get('funding_rate') and abs(signal['funding_rate'])>0.1: tips.append(f"📡 Funding rate extreme ({signal['funding_rate']:.3f}%) – possible reversal.")
    return tips

# -----------------------------------------------------------------------------
# ACTIONABLE RULES & PLAYBOOK
# -----------------------------------------------------------------------------
def get_trade_bias(garch_vol, ivr, corr):
    bias = ""
    if ivr is None: ivr = 25
    if ivr < 25:
        if garch_vol < 40: bias = "Small debit structures. Avoid naked shorts."
        else: bias = "Directional with defined risk."
    elif 25 <= ivr <= 50: bias = "Mixed – credit spreads in range, debit if trending."
    else:
        if garch_vol > 80: bias = "Short premium favoured. Strict risk caps."
        else: bias = "Sell OTM premium, hedge tail risk."
    if corr and corr > 0.8: bias += " High correlation – trend valid."
    elif corr and corr < 0.5: bias += " Decoupling – neutral strategies."
    return bias

def get_playbook(garch_vol, ivr, corr):
    if ivr is None: ivr = 25
    if ivr < 25:
        if garch_vol < 40: return ["Vertical spreads","Calendars","Long calls/puts"]
        else: return ["Long strangles","Backspreads"]
    elif ivr > 50: return ["Iron Condors","Short strangles","Credit spreads"]
    else: return ["Vertical spreads","Covered calls/puts","Iron Butterflies"]

def get_ivr_label(ivr):
    if ivr is None: return "IVR unavailable"
    if ivr < 25: return f"Low IV (IVR {ivr:.0f}%) – Buy premium"
    elif ivr > 50: return f"High IV (IVR {ivr:.0f}%) – Sell premium"
    else: return f"Mid IV (IVR {ivr:.0f}%) – Mixed"

# -----------------------------------------------------------------------------
# ANALYTICS PLOT FUNCTIONS
# -----------------------------------------------------------------------------
def plot_correlation():
    if 'correlation_data' not in st.session_state: return None
    data = st.session_state['correlation_data'].dropna()
    if data.empty: return None
    n1, n2 = ("Bitcoin","Ethereum") if selected_market=="Crypto" else ("Nifty 50","Bank Nifty")
    data.columns = [n1, n2]
    normalized = (data / data.iloc[0]) * 100
    log_ret = np.log(data / data.shift(1)).dropna()
    rolling_corr = log_ret[n1].rolling(20).corr(log_ret[n2])
    current_corr = rolling_corr.iloc[-1]
    if current_corr > 0.8: regime, color = "High Correlation", 'green'
    elif current_corr < 0.5: regime, color = "Severe Divergence", 'red'
    else: regime, color = "Moderate Divergence", 'orange'
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,7), gridspec_kw={'height_ratios':[2,1]})
    ax1.plot(normalized.index, normalized[n1], label=n1)
    ax1.plot(normalized.index, normalized[n2], label=n2)
    ax1.fill_between(normalized.index, normalized[n1], normalized[n2], alpha=0.2)
    ax1.set_title(f"Correlation: {n1} vs {n2}", fontweight='bold'); ax1.legend()
    ax2.plot(rolling_corr.index, rolling_corr, color='white')
    ax2.axhline(0.8, color='green', linestyle='--'); ax2.axhline(0.5, color='red', linestyle='--'); ax2.set_ylim(-0.2,1.1)
    props = dict(boxstyle='round', facecolor='black', edgecolor=color)
    ax1.text(0.02,0.05,f"20-Day Corr: {current_corr:.2f}\nRegime: {regime}", transform=ax1.transAxes, bbox=props, fontsize=12)
    plt.tight_layout()
    return fig

def plot_expected_move():
    if selected_market == "Crypto":
        opt_df = fetch_deribit_option_chain("BTC" if 'BTC' in ticker else "ETH")
        if opt_df is not None and not opt_df.empty:
            atm_idx = (opt_df['strike'] - asset_spot).abs().argsort()[:1]
            if len(atm_idx) > 0: iv = opt_df.iloc[atm_idx]['mark_iv'].values[0] * 100
            else: iv = garch_vol_asset
        else: iv = garch_vol_asset
        spot_label = f"Spot: {currency}{asset_spot:,.0f}"; iv_label = f"ATM IV: {iv:.1f}%"
        recent = get_recent_month(ticker)
    else:
        vix_data = get_india_vix("5d")
        if vix_data is None: return None
        iv = float(vix_data.iloc[-1])
        spot_label = f"Nifty: {asset_spot:,.0f}"; iv_label = f"India VIX: {iv:.2f}"
        nifty_data = yf_download_retry("^NSEI", period="1mo")
        if nifty_data.empty: return None
        recent = nifty_data['Close'].squeeze().tail(15)
    if recent.empty: return None
    daily_volatility = (iv / 100) * np.sqrt(1/365)
    expected_move_points = asset_spot * daily_volatility
    upper_bound = asset_spot + expected_move_points; lower_bound = asset_spot - expected_move_points
    fig, ax = plt.subplots(figsize=(12,7), dpi=120)
    x_dates = np.arange(len(recent))
    ax.plot(x_dates, recent.values, color='#00FFFF', linewidth=2, marker='o', label='Price')
    tomorrow_x = len(recent)
    ax.hlines(asset_spot, xmin=x_dates[-1], xmax=tomorrow_x, color='white', linestyle='-', linewidth=2)
    ax.scatter(tomorrow_x, asset_spot, color='white', s=70, zorder=5)
    ax.scatter(tomorrow_x, upper_bound, color='#00FF00', s=120, marker='^', zorder=5)
    ax.scatter(tomorrow_x, lower_bound, color='#FF3333', s=120, marker='v', zorder=5)
    ax.hlines(upper_bound, xmin=x_dates[-1], xmax=tomorrow_x, color='#00FF00', linestyle='--', linewidth=1.5, label='Upper Bound (+1 SD)')
    ax.hlines(lower_bound, xmin=x_dates[-1], xmax=tomorrow_x, color='#FF3333', linestyle='--', linewidth=1.5, label='Lower Bound (-1 SD)')
    ax.fill_between([x_dates[-1], tomorrow_x], [asset_spot, lower_bound], [asset_spot, upper_bound], color='gray', alpha=0.2)
    ax.set_title('Implied Daily Expected Move', fontsize=18, color='white', pad=20, fontweight='bold')
    ax.set_ylabel('Price', color='gray', fontsize=12); ax.grid(True, color='#2A2A2A', linestyle=':')
    ax.set_xticks([]); ax.legend(loc='upper left', facecolor='black', edgecolor='gray', fontsize=10)
    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8, edgecolor='white', linewidth=1.5)
    text_str = (f"⚡ {iv_label}\n🎯 {spot_label}\n📏 Expected Move: ± {expected_move_points:,.1f} points\n"
                f"---------------------------\n🟢 Safe Short Call Strike: > {upper_bound:,.0f}\n🔴 Safe Short Put Strike: < {lower_bound:,.0f}")
    ax.text(0.02, 0.45, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_hurst():
    close = st.session_state['long_hist_data'][asset_choice]['Close'].squeeze()
    if len(close) < 120: return None
    log_prices = np.log(close)
    def rolling_hurst(series, window=60):
        return series.rolling(window).apply(lambda x: calculate_hurst_exponent(x) if len(x)>=20 else np.nan, raw=False)
    hurst_series = rolling_hurst(log_prices, 60)
    df = pd.DataFrame({'Close': close, 'Hurst': hurst_series}).dropna()
    if df.empty: return None
    current_price = float(df['Close'].iloc[-1])
    current_hurst = float(df['Hurst'].iloc[-1])
    if np.isnan(current_hurst): return None
    if current_hurst < 0.45: regime, color_theme = "MEAN REVERTING", '#FF3333'
    elif current_hurst > 0.55: regime, color_theme = "TRENDING", '#00FF00'
    else: regime, color_theme = "RANDOM WALK", '#FFA500'
    stat_property = "Range-Bound / Vol Compression" if regime=="MEAN REVERTING" else "Momentum Expansion" if regime=="TRENDING" else "Noise / Transition"
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,8), dpi=120, gridspec_kw={'height_ratios':[1.5,1]})
    ax1.plot(df.index, df['Close'], color='white', linewidth=1.5, label='Price')
    ax1.set_title('Market Regime (Hurst Exponent)', fontsize=18, color='white', pad=15, fontweight='bold')
    ax1.set_ylabel('Price', color='gray'); ax1.grid(True, color='#2A2A2A', linestyle=':')
    ax1.legend(loc='upper left', facecolor='black', edgecolor='gray')
    ax1.axvspan(df.index[-15], df.index[-1], color=color_theme, alpha=0.1)
    ax2.plot(df.index, df['Hurst'], color='#00FFFF', linewidth=2, label='60-Day Hurst Exponent')
    ax2.axhline(0.55, color='#00FF00', linestyle='--', linewidth=1.5, label='Trending (>0.55)')
    ax2.axhline(0.45, color='#FF3333', linestyle='--', linewidth=1.5, label='Mean Reverting (<0.45)')
    ax2.axhline(0.50, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax2.fill_between(df.index, 0.55, df['Hurst'], where=(df['Hurst']>0.55), color='#00FF00', alpha=0.2, interpolate=True)
    ax2.fill_between(df.index, 0.45, df['Hurst'], where=(df['Hurst']<0.45), color='#FF3333', alpha=0.2, interpolate=True)
    ax2.set_ylabel('Hurst Value (H)', color='gray'); ax2.grid(True, color='#2A2A2A', linestyle=':')
    ax2.set_ylim(0.3,0.7); ax2.legend(loc='upper right', facecolor='black', edgecolor='gray')
    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9, edgecolor=color_theme, linewidth=1.5)
    text_str = f"Price: {current_price:.2f}\nHurst (H): {current_hurst:.3f}\nRegime: {regime}\n{stat_property}"
    ax1.text(0.02,0.05,text_str, transform=ax1.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_ivr_ivp():
    if selected_market=="Crypto":
        close=st.session_state['long_hist_data'][asset_choice]['Close'].squeeze()
        vol_series=np.log(close/close.shift(1)).dropna().rolling(20).std()*np.sqrt(365)*100
        current_vol=vol_series.iloc[-1]; high,low=vol_series.max(),vol_series.min()
        ivr=(current_vol-low)/(high-low)*100 if high!=low else 50
        ivp=(vol_series<current_vol).sum()/len(vol_series)*100; label="Historical Vol (20d)"
    else:
        vix=get_india_vix("1y")
        if vix is None: return None
        current_vol=vix.iloc[-1]; high,low=vix.max(),vix.min()
        ivr=(current_vol-low)/(high-low)*100 if high!=low else 50
        ivp=(vix<current_vol).sum()/len(vix)*100; label="India VIX"; vol_series=vix
    regime="High Vol Regime (Sell Premium)" if ivr>50 else "Low Vol Regime (Buy Premium)"
    fig,ax=plt.subplots(figsize=(12,6))
    ax.plot(vol_series.index,vol_series,color='cyan',label=label)
    ax.axhline(high,color='red',linestyle='--'); ax.axhline(low,color='green',linestyle='--')
    ax.axhline(current_vol,color='white',linestyle='-')
    ax.set_title("IV Rank & IV Percentile",fontweight='bold')
    props=dict(boxstyle='round',facecolor='black',edgecolor='cyan')
    ax.text(0.02,0.95,f"IVR: {ivr:.1f}%\nIVP: {ivp:.1f}%\n{regime}",transform=ax.transAxes,bbox=props,va='top',fontsize=12)
    plt.tight_layout(); return fig

def plot_liquidity_sweep():
    df=fetch_intraday(ticker,"30m")
    if df is None: return None
    df=df.tail(60).copy()
    df['Prev_High']=df['High'].rolling(20).max().shift(1)
    df['Prev_Low']=df['Low'].rolling(20).min().shift(1)
    df['Supply_Sweep']=(df['High']>df['Prev_High'])&(df['Close']<df['Prev_High'])
    df['Demand_Sweep']=(df['Low']<df['Prev_Low'])&(df['Close']>df['Prev_Low'])
    current_regime="Price Discovery"
    if df['Supply_Sweep'].iloc[-1]: current_regime="Supply Swept (Bearish)"
    elif df['Demand_Sweep'].iloc[-1]: current_regime="Demand Swept (Bullish)"
    fig,ax=plt.subplots(figsize=(12,7)); ax.plot(df.index,df['Close'],color='white',label='Price')
    for idx,row in df.iterrows():
        if row['Supply_Sweep']: ax.scatter(idx,row['High']+10,marker='v',color='red',s=100)
        if row['Demand_Sweep']: ax.scatter(idx,row['Low']-10,marker='^',color='green',s=100)
    ax.set_title("Liquidity Sweep / Order Block Detector",fontweight='bold')
    props=dict(boxstyle='round',facecolor='black')
    ax.text(0.02,0.05,f"Microstructure: {current_regime}",transform=ax.transAxes,bbox=props,fontsize=12)
    plt.tight_layout(); return fig

def plot_oi_profile():
    step=get_asset_step(asset_spot); base=round(asset_spot/step)*step
    strikes=np.arange(base-8*step,base+9*step,step)
    np.random.seed(int(asset_spot)%1234)
    calls=np.random.randint(10,80,len(strikes)).astype(float)*50000
    puts=np.random.randint(10,80,len(strikes)).astype(float)*50000
    pain={k:np.sum(np.maximum(0,k-strikes)*calls+np.maximum(0,strikes-k)*puts) for k in strikes}
    max_pain=min(pain,key=pain.get)
    fig,ax=plt.subplots(figsize=(14,8))
    ax.barh(strikes,calls/1e5,color='red',alpha=0.8,label='Call OI')
    ax.barh(strikes,-puts/1e5,color='green',alpha=0.8,label='Put OI')
    ax.axhline(asset_spot,color='cyan',linewidth=2,label=f'Spot: {asset_spot:,.0f}')
    ax.axhline(max_pain,color='white',linestyle='--',label=f'Max Pain: {max_pain}')
    ax.set_title("Open Interest Profile & Max Pain (Simulated)", fontweight='bold'); ax.legend(); ax.invert_yaxis()
    plt.tight_layout(); return fig

def plot_parkinson():
    df = st.session_state['long_hist_data'][asset_choice]
    if df.empty or not all(c in df.columns for c in ['High','Low']): return None,None
    high=df['High'].squeeze().tail(60); low=df['Low'].squeeze().tail(60)
    if len(high)<2 or len(low)<2: return None,None
    park=calculate_parkinson_volatility(high,low,periods_per_year=trading_days)
    fig,ax=plt.subplots(figsize=(10,4)); ax.bar(['Parkinson Vol'],[park],color='orange')
    ax.set_ylabel('Annualized Vol (%)'); ax.set_title("Parkinson Estimator (High-Low Range)",fontweight='bold')
    for i,v in enumerate([park]): ax.text(i,v+0.5,f"{v:.1f}%",ha='center',fontweight='bold')
    plt.tight_layout(); return fig,park

def plot_volatility_cone():
    close=st.session_state['long_hist_data'][asset_choice]['Close'].squeeze()
    log_ret=np.log(close/close.shift(1)).dropna()
    windows=[10,20,30,60,90,120,180,252]
    max_v,min_v,med_v,cur_v=[],[],[],[]
    for w in windows:
        rv=log_ret.rolling(w).std()*np.sqrt(trading_days)*100
        if not rv.dropna().empty:
            max_v.append(rv.max()); min_v.append(rv.min()); med_v.append(rv.median()); cur_v.append(rv.iloc[-1])
    fig,ax=plt.subplots(figsize=(12,7))
    ax.plot(windows,max_v,'o-',color='red',label='Max'); ax.plot(windows,min_v,'o-',color='green',label='Min')
    ax.plot(windows,med_v,'s--',color='white',label='Median'); ax.plot(windows,cur_v,'X-',color='yellow',markersize=10,label='Current')
    ax.fill_between(windows,min_v,max_v,alpha=0.2)
    ax.set_title("Volatility Cone",fontweight='bold'); ax.set_xlabel("Window (days)"); ax.set_ylabel("Volatility (%)"); ax.legend()
    plt.tight_layout(); return fig

def plot_vrp():
    if selected_market=="Crypto":
        opt_df=fetch_deribit_option_chain("BTC" if 'BTC' in ticker else "ETH")
        if opt_df is not None and not opt_df.empty:
            atm_idx = (opt_df['strike']-asset_spot).abs().argsort()[:1]
            iv = opt_df.iloc[atm_idx]['mark_iv'].values[0]*100 if len(atm_idx)>0 else garch_vol_asset
        else: iv=garch_vol_asset
        hist=get_hist_6mo(ticker)
        if hist.empty: return None
        log_ret=np.log(hist/hist.shift(1)).dropna()
        hv_series=log_ret.rolling(20).std()*np.sqrt(365)*100; current_hv=hv_series.iloc[-1]
        vrp=iv-current_hv; regime="Positive VRP (Sell Premium)" if vrp>0 else "Negative VRP (Buy Premium)"
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,7),gridspec_kw={'height_ratios':[2,1]})
        ax1.plot(hv_series.index,[iv]*len(hv_series),color='cyan',label=f'Implied Vol ({iv:.1f}%)')
        ax1.plot(hv_series.index,hv_series,color='orange',label='20d Realized Vol')
        ax1.fill_between(hv_series.index,hv_series,[iv]*len(hv_series),where=([iv]*len(hv_series)>hv_series),color='green',alpha=0.3)
        ax1.fill_between(hv_series.index,hv_series,[iv]*len(hv_series),where=([iv]*len(hv_series)<=hv_series),color='red',alpha=0.3)
        ax1.set_title("Volatility Risk Premium (VRP)",fontweight='bold'); ax1.legend()
        ax2.bar(hv_series.index,[iv]*len(hv_series)-hv_series,color=np.where([iv]*len(hv_series)>hv_series,'green','red')); ax2.axhline(0,color='white')
        props=dict(boxstyle='round',facecolor='black')
        ax1.text(0.02,0.05,f"VRP: {vrp:+.1f}%\n{regime}",transform=ax1.transAxes,bbox=props,fontsize=12)
    else:
        vix=get_india_vix("6mo"); nifty=get_hist_6mo("^NSEI")
        if vix is None or nifty.empty: return None
        log_ret=np.log(nifty/nifty.shift(1)).dropna()
        hv_series=log_ret.rolling(20).std()*np.sqrt(252)*100; iv=vix.iloc[-1]; current_hv=hv_series.iloc[-1]
        vrp=iv-current_hv; regime="Positive VRP (Sell Premium)" if vrp>0 else "Negative VRP (Buy Premium)"
        common_idx=vix.index.intersection(hv_series.index)
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,7),gridspec_kw={'height_ratios':[2,1]})
        ax1.plot(common_idx,vix[common_idx],color='cyan',label='India VIX')
        ax1.plot(common_idx,hv_series[common_idx],color='orange',label='20d Realized Vol')
        ax1.fill_between(common_idx,hv_series[common_idx],vix[common_idx],where=(vix[common_idx]>hv_series[common_idx]),color='green',alpha=0.3)
        ax1.fill_between(common_idx,hv_series[common_idx],vix[common_idx],where=(vix[common_idx]<=hv_series[common_idx]),color='red',alpha=0.3)
        ax1.set_title("Volatility Risk Premium (VRP)",fontweight='bold'); ax1.legend()
        ax2.bar(common_idx,vix[common_idx]-hv_series[common_idx],color=np.where(vix[common_idx]>hv_series[common_idx],'green','red')); ax2.axhline(0,color='white')
        props=dict(boxstyle='round',facecolor='black')
        ax1.text(0.02,0.05,f"VRP: {vrp:+.1f}%\n{regime}",transform=ax1.transAxes,bbox=props,fontsize=12)
    plt.tight_layout(); return fig

# -----------------------------------------------------------------------------
# NEW FEATURES
# -----------------------------------------------------------------------------
def plot_mtf_chart(ticker, intervals=["15m","1h","4h"]):
    figs = []
    for interval in intervals:
        df = yf_download_retry(ticker, period="5d", interval=interval)
        if not df.empty:
            df = flatten_df(df).tail(50)
            fig = go.Figure(data=[go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close']
            )])
            fig.update_layout(title=f"{interval} Chart", xaxis_rangeslider_visible=False,
                              template="plotly_dark", height=300)
            figs.append(fig)
    return figs

def plot_payoff(strategy, spot, strikes, premium, T, r, sigma):
    prices = np.linspace(spot*0.8, spot*1.2, 100)
    payoff = np.zeros_like(prices)
    for i, K in enumerate(strikes):
        opt_type = 'call' if 'Call' in strategy[i] else 'put'
        sign = 1 if 'Long' in strategy[i] else -1
        for j, S in enumerate(prices):
            intrinsic = max(0, S-K) if opt_type=='call' else max(0, K-S)
            payoff[j] += sign * intrinsic * 100
        payoff[j] -= sign * premium[i] * 100
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(prices, payoff, color='cyan', linewidth=2)
    ax.axhline(0, color='white', linestyle='--')
    ax.set_xlabel('Underlying Price at Expiry')
    ax.set_ylabel('Profit / Loss')
    ax.set_title('Payoff Diagram')
    ax.grid(True, color='#2A2A2A')
    plt.tight_layout()
    return fig

def portfolio_risk(positions, spot, garch_vol):
    delta = gamma = theta = vega = 0
    for pos in positions:
        if pos['Type'] == 'Spot':
            delta += 1 if pos['Direction']=='Long' else -1
        else:
            K = pos['Strike']
            T = pos['Expiry']/252 if pos['Expiry'] else 1/252
            r = 0.05; sigma = garch_vol/100
            g = calc_greeks(spot, K, T, r, sigma, pos['Type'])
            sign = 1 if pos['Direction']=='Long' else -1
            delta += sign * g['delta']; gamma += sign * g['gamma']
            theta += sign * g['theta']; vega += sign * g['vega']
    margin = abs(delta) * spot * 0.1
    return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Margin': margin}

def get_news_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return []
        headlines = [item['title'] for item in news[:5]]
        try:
            from textblob import TextBlob
            sentiments = []
            for headline in headlines:
                blob = TextBlob(headline)
                polarity = blob.sentiment.polarity
                sentiment = "🟢" if polarity>0.1 else ("🔴" if polarity<-0.1 else "⚪")
                sentiments.append((headline, sentiment, polarity))
            return sentiments
        except ImportError:
            return [(h, "⚪", 0) for h in headlines]
    except:
        return []

# -----------------------------------------------------------------------------
# LIVE ORDER FLOW (Binance)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=5, show_spinner=False)
def fetch_binance_orderbook(symbol="BTCUSDT", limit=20):
    try:
        url = "https://api.binance.com/api/v3/depth"
        r = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            bids = pd.DataFrame(data['bids'], columns=['Price', 'Size'], dtype=float)
            asks = pd.DataFrame(data['asks'], columns=['Price', 'Size'], dtype=float)
            return bids, asks
    except: pass
    return None, None

def get_binance_symbol(ticker):
    mapping = {'BTC-USD': 'BTCUSDT', 'ETH-USD': 'ETHUSDT', 'DOGE-USD': 'DOGEUSDT', 'XRP-USD': 'XRPUSDT'}
    return mapping.get(ticker, 'BTCUSDT')

# -----------------------------------------------------------------------------
# LIVE NIFTY/BANKNIFTY OPTIONS CHAIN
# -----------------------------------------------------------------------------
@st.cache_data(ttl=120, show_spinner=False)
def fetch_nse_options(index="^NSEI"):
    try:
        if index == "^NSEI": symbol = "NIFTY"
        elif index == "^NSEBANK": symbol = "BANKNIFTY"
        else: return None
        spot_data = yf.download(index, period="2d", progress=False)
        if spot_data.empty: return None
        spot = spot_data['Close'].squeeze().iloc[-1]
        strike_step = 50 if symbol == "NIFTY" else 100
        atm = round(spot/strike_step)*strike_step
        strikes = np.arange(atm - 10*strike_step, atm + 10*strike_step + strike_step, strike_step)
        options = []
        today = datetime.now()
        days_to_expiry = 3 - today.weekday()
        if days_to_expiry < 0: days_to_expiry += 7
        expiry = today + timedelta(days=days_to_expiry)
        expiry_str = expiry.strftime("%y%m%d")
        for strike in strikes:
            for opt_type in ['CE', 'PE']:
                yahoo_ticker = f"{symbol}{expiry_str}{opt_type}{int(strike)}.NS"
                try:
                    opt_data = yf.download(yahoo_ticker, period="5d", progress=False)
                    if not opt_data.empty:
                        ltp = opt_data['Close'].squeeze().iloc[-1]
                        oi = 0
                        if 'Open Interest' in opt_data.columns:
                            oi = opt_data['Open Interest'].squeeze().iloc[-1]
                        options.append({'Strike': strike, 'Type': opt_type, 'LTP': ltp, 'OI': oi if not np.isnan(oi) else 0})
                except: pass
        return spot, pd.DataFrame(options)
    except: return None, None

# -----------------------------------------------------------------------------
# CACHED QUICK STATS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=120)
def compute_quick_stats(ticker, asset_choice, asset_spot, garch_vol_asset, park_vol, ivr_val, ivp_val, trading_days, currency, selected_market, corr_val, corr_status):
    quick_stats = {}
    max_pain = None
    if corr_val is not None:
        quick_stats['Correlation'] = {'value': f"{corr_val:.2f}", 'status': corr_status, 'module': 'Correlation'}
    if selected_market == "Crypto":
        opt_df = fetch_deribit_option_chain("BTC" if 'BTC' in ticker else "ETH")
        iv = (opt_df.iloc[(opt_df['strike']-asset_spot).abs().argsort()[:1]]['mark_iv'].values[0]*100
              if opt_df is not None and not opt_df.empty else garch_vol_asset)
        daily_move = asset_spot * (iv/100) * np.sqrt(1/365)
        quick_stats['Exp. Move (D)'] = {'value': f"±{currency}{daily_move:,.0f}", 'status': '1σ Range', 'module': 'Expected Move'}
    else:
        vix_data = get_india_vix("5d")
        if vix_data is not None:
            iv = float(vix_data.iloc[-1])
            daily_move = asset_spot * (iv/100) * np.sqrt(1/365)
            quick_stats['Exp. Move (D)'] = {'value': f"±{currency}{daily_move:,.0f}", 'status': '1σ Range', 'module': 'Expected Move'}
    close = st.session_state['long_hist_data'][asset_choice]['Close'].squeeze()
    log_p = np.log(close); hurst_series = log_p.rolling(60).apply(lambda x: calculate_hurst_exponent(x) if len(x)>=20 else np.nan, raw=False)
    df = pd.DataFrame({'Close':close,'Hurst':hurst_series}).dropna()
    if not df.empty:
        hurst_val = df['Hurst'].iloc[-1]
        if np.isnan(hurst_val):
            quick_stats['Hurst'] = {'value': "N/A", 'status': "Insufficient data", 'module': 'Hurst Exponent'}
        else:
            if hurst_val>0.55: hurst_status="Trending"
            elif hurst_val<0.45: hurst_status="Mean Rev."
            else: hurst_status="Random"
            quick_stats['Hurst'] = {'value': f"{hurst_val:.3f}", 'status': hurst_status, 'module': 'Hurst Exponent'}
    else:
        quick_stats['Hurst'] = {'value': "N/A", 'status': "Insufficient data", 'module': 'Hurst Exponent'}
    if ivr_val is not None:
        quick_stats['IVR/IVP'] = {'value': f"{ivr_val:.0f}% / {ivp_val:.0f}%", 'status': get_ivr_label(ivr_val), 'module': 'IV Rank & IV Percentile'}
    fig_park, park_val = plot_parkinson()
    if park_val is not None:
        quick_stats['Parkinson'] = {'value': f"{park_val:.1f}%", 'status': "Intraday Vol", 'module': 'Parkinson Estimator'}
    else:
        quick_stats['Parkinson'] = {'value': "N/A", 'status': "No data", 'module': 'Parkinson Estimator'}
    intra_df = fetch_intraday(ticker, "30m")
    if intra_df is not None:
        df_temp = intra_df.tail(60).copy()
        df_temp['Prev_High'] = df_temp['High'].rolling(20).max().shift(1)
        df_temp['Prev_Low'] = df_temp['Low'].rolling(20).min().shift(1)
        supply = (df_temp['High'] > df_temp['Prev_High']) & (df_temp['Close'] < df_temp['Prev_High'])
        demand = (df_temp['Low'] < df_temp['Prev_Low']) & (df_temp['Close'] > df_temp['Prev_Low'])
        sweep = "Supply Swept" if supply.iloc[-1] else ("Demand Swept" if demand.iloc[-1] else "Price Discovery")
        quick_stats['Liq. Sweep'] = {'value': sweep, 'status': '', 'module': 'Liquidity Detector'}
    step = get_asset_step(asset_spot); base = round(asset_spot/step)*step
    strikes = np.arange(base - 8*step, base + 9*step, step)
    np.random.seed(int(asset_spot)%1234)
    calls = np.random.randint(10,80,len(strikes)).astype(float)*50000
    puts = np.random.randint(10,80,len(strikes)).astype(float)*50000
    pain = {k: np.sum(np.maximum(0, k-strikes)*calls + np.maximum(0, strikes-k)*puts) for k in strikes}
    max_pain = min(pain, key=pain.get)
    quick_stats['Max Pain'] = {'value': f"{currency}{max_pain:,.0f}", 'status': f"Spot: {currency}{asset_spot:,.0f}", 'module': 'Open Interest Profile'}
    return quick_stats, max_pain

def get_correlation_value():
    corr_data = st.session_state.get('correlation_data')
    if corr_data is None or corr_data.empty: return None, None
    data = corr_data.dropna()
    n1, n2 = ("Bitcoin","Ethereum") if selected_market=="Crypto" else ("Nifty 50","Bank Nifty")
    data.columns = [n1, n2]
    log_ret = np.log(data / data.shift(1)).dropna()
    rolling_corr = log_ret[n1].rolling(20).corr(log_ret[n2])
    val = rolling_corr.iloc[-1]
    if val > 0.8: status = "High"
    elif val < 0.5: status = "Severe Divergence"
    else: status = "Moderate"
    return val, status

# -----------------------------------------------------------------------------
# COMPACT TOOLBAR
# -----------------------------------------------------------------------------
toolbar_col1, toolbar_col2, toolbar_col3, toolbar_col4, toolbar_col5 = st.columns([2, 1, 1, 1, 2])
with toolbar_col1:
    market_choice = st.radio("🌐 Market", ["Crypto", "Indian Market"], index=0, horizontal=True, label_visibility="collapsed")
    if market_choice != st.session_state['selected_market']:
        st.session_state['selected_market'] = market_choice
        st.cache_data.clear()
        st.rerun()
    selected_market = market_choice
with toolbar_col2:
    live_toggle = st.checkbox("🟢 Live", value=st.session_state['live_mode'])
    if live_toggle != st.session_state['live_mode']: st.session_state['live_mode'] = live_toggle
with toolbar_col3:
    refresh_sec = st.number_input("⏱️ Refresh (s)", min_value=30, max_value=600, value=st.session_state['refresh_interval'], step=10, label_visibility="collapsed")
    if refresh_sec != st.session_state['refresh_interval']: st.session_state['refresh_interval'] = refresh_sec
with toolbar_col4:
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with toolbar_col5:
    if selected_market == "Crypto":
        TICKER_DICT = CONFIG['cryptos']; trading_days = 365; currency = "$"
    else:
        TICKER_DICT = CONFIG['indian_market']; trading_days = 252; currency = "₹"
    asset_choice = st.selectbox("🎯 Asset", list(TICKER_DICT.keys()), label_visibility="collapsed")
    ticker = TICKER_DICT[asset_choice]

# -----------------------------------------------------------------------------
# SIDEBAR – MINIMAL (ML, Auto‑Exit, Daily Snapshot)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧬 AlphaQuant Terminal")
    if ML_AVAILABLE:
        if st.button("🧠 Train ML Model"):
            hist_b = st.session_state.get('long_hist_data', {}).get(asset_choice)
            if hist_b is not None:
                close = hist_b['Close'].squeeze(); train_ml_model(close); st.success("ML model trained!")
    auto_exit = st.checkbox("🛑 Auto‑Exit (1.5x loss or 2σ move)", value=st.session_state['auto_exit_enabled'])
    if auto_exit != st.session_state['auto_exit_enabled']: st.session_state['auto_exit_enabled'] = auto_exit

    st.markdown("---")
    # Daily Snapshot Saver
    if st.button("📸 Save Daily Snapshot"):
        snapshot = {
            'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Market': selected_market,
            'Asset': asset_choice,
            'Spot': asset_spot,
            'GARCH Vol': garch_vol_asset,
            'GJR-GARCH Vol': gjrgarch_vol,
            'Parkinson Vol': park_vol,
            'IV Rank': ivr_val,
            'IV Percentile': ivp_val,
            'Correlation': corr_val,
            'Expected Move (D)': quick_stats.get('Exp. Move (D)', {}).get('value', 'N/A'),
            'Hurst': quick_stats.get('Hurst', {}).get('value', 'N/A'),
            'Liq Sweep': quick_stats.get('Liq. Sweep', {}).get('value', 'N/A'),
            'Max Pain': quick_stats.get('Max Pain', {}).get('value', 'N/A'),
            'Trade Bias': trade_bias_label,
            'Playbook': ', '.join(playbook_strategies)
        }
        st.session_state['snapshots'].append(snapshot)
        # Save to CSV file automatically
        df_snaps = pd.DataFrame(st.session_state['snapshots'])
        df_snaps.to_csv('daily_snapshots.csv', index=False)
        st.success("Snapshot saved to daily_snapshots.csv")

    # Display past snapshots
    if st.session_state['snapshots']:
        with st.expander("📚 Past Snapshots"):
            df_snaps = pd.DataFrame(st.session_state['snapshots'])
            st.dataframe(df_snaps)
            st.download_button("Download CSV", df_snaps.to_csv(index=False), "daily_snapshots.csv", "text/csv")
    else:
        st.info("No snapshots saved yet.")

# -----------------------------------------------------------------------------
# INITIAL DATA LOADING (unchanged)
# -----------------------------------------------------------------------------
current_keys = set(TICKER_DICT.keys())
if 'long_hist_data' not in st.session_state or set(st.session_state['long_hist_data'].keys()) != current_keys:
    st.session_state['long_hist_data'] = fetch_long_hist(TICKER_DICT)
    if selected_market == "Crypto": st.session_state['top_prices'] = fetch_top_prices()
    else: st.session_state['indian_summary'] = fetch_indian_market_summary()
    if selected_market == "Crypto": st.session_state['correlation_data'] = yf_download_retry(['BTC-USD','ETH-USD'], period="1y")['Close']
    else: st.session_state['correlation_data'] = yf_download_retry(['^NSEI','^NSEBANK'], period="1y")['Close']
    st.session_state['corr_market'] = selected_market

if selected_market == "Crypto":
    if 'top_prices' not in st.session_state: st.session_state['top_prices'] = fetch_top_prices()
    market = st.session_state['top_prices']
else:
    if 'indian_summary' not in st.session_state: st.session_state['indian_summary'] = fetch_indian_market_summary()
    indian_data = st.session_state['indian_summary']

lp = live_price(ticker)
if lp is None:
    hist_b = st.session_state['long_hist_data']
    if hist_b and asset_choice in hist_b:
        c = hist_b[asset_choice]['Close']
        if len(c) >= 2:
            spot_val = float(c.iloc[-1]); prev_val = float(c.iloc[-2])
            lp = {'spot':spot_val,'prev_close':prev_val,'change':spot_val-prev_val,
                  'pct':((spot_val-prev_val)/prev_val)*100 if prev_val else 0,'ts':'Hist'}
    if lp is None:
        last_data = yf_download_retry(ticker, period="1d")
        if not last_data.empty:
            close = last_data['Close'].squeeze()
            if len(close) >= 1:
                spot_val = float(close.iloc[-1])
                lp = {'spot':spot_val,'prev_close':spot_val,'change':0,'pct':0,'ts':'Direct'}
    if lp is None:
        lp = {'spot':0,'prev_close':0,'change':0,'pct':0,'ts':'Unavailable'}

asset_spot = lp['spot']; asset_change = lp['change']; asset_pct = lp['pct']
garch_vol, gjrgarch_vol = garch_both(ticker)
garch_vol_asset = garch_vol

hb = st.session_state['long_hist_data']
park_vol = None; ivr_val = ivp_val = None
if asset_choice in hb:
    df = hb[asset_choice]
    if not df.empty and all(c in df.columns for c in ['High','Low','Close']):
        high = df['High'].squeeze().tail(60); low = df['Low'].squeeze().tail(60); close = df['Close'].squeeze().tail(60)
        park_vol = calculate_parkinson_volatility(high, low, periods_per_year=trading_days)
        ivr_val, ivp_val = calculate_iv_rank_percentile(close)

corr_val, corr_status = get_correlation_value()
quick_stats, max_pain = compute_quick_stats(ticker, asset_choice, asset_spot, garch_vol_asset, park_vol, ivr_val, ivp_val, trading_days, currency, selected_market, corr_val, corr_status)
trade_bias_label = get_trade_bias(garch_vol_asset, ivr_val, corr_val)
playbook_strategies = get_playbook(garch_vol_asset, ivr_val, corr_val)

# -----------------------------------------------------------------------------
# DAY‑OVER‑DAY COMPARISON SUMMARY
# -----------------------------------------------------------------------------
if st.session_state['snapshots']:
    yesterday = st.session_state['snapshots'][-1]
    if isinstance(yesterday, dict):
        # Compare with previous snapshot
        today = {
            'Spot': asset_spot,
            'GARCH Vol': garch_vol_asset,
            'IV Rank': ivr_val,
            'IV Percentile': ivp_val,
            'Correlation': corr_val,
            'Hurst': quick_stats.get('Hurst', {}).get('value', 'N/A'),
            'Expected Move (D)': quick_stats.get('Exp. Move (D)', {}).get('value', 'N/A'),
            'Trade Bias': trade_bias_label
        }
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📅 Day‑over‑Day Changes")
        changes = []
        for key in ['Spot', 'GARCH Vol', 'IV Rank', 'IV Percentile', 'Correlation']:
            prev_val = yesterday.get(key)
            curr_val = today.get(key)
            if prev_val is not None and curr_val is not None and prev_val != 'N/A' and curr_val != 'N/A':
                try:
                    change = float(curr_val) - float(prev_val)
                    if abs(change) > 0.001:
                        direction = "↑" if change > 0 else "↓"
                        changes.append(f"{key}: {prev_val} → {curr_val} ({direction}{abs(change):.2f})")
                except:
                    pass
        if changes:
            for c in changes:
                st.sidebar.markdown(f"- {c}")
        else:
            st.sidebar.caption("No significant change from yesterday.")
else:
    st.sidebar.info("Save a snapshot to enable day‑over‑day comparison.")

# -----------------------------------------------------------------------------
# FULL DASHBOARD (all analytics)
# -----------------------------------------------------------------------------
st.title("📊 Market Intelligence Dashboard")

# Market Overview
with st.container(border=True):
    st.markdown('<p class="section-header">🌍 Market Overview</p>', unsafe_allow_html=True)
    if selected_market == "Crypto":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            val = f"${market['btc']:,.0f}" if isinstance(market['btc'], (int,float)) and market['btc']!=0 else "Loading..."
            st.markdown(f"""<div class="metric-card"><h3>₿ Bitcoin</h3><div class="value">{val}</div><div class="delta">{market['btc_change']:+,.0f} ({market['btc_pct']:.2f}%)</div></div>""", unsafe_allow_html=True)
        with col2:
            val = f"${market['eth']:,.0f}" if isinstance(market['eth'], (int,float)) and market['eth']!=0 else "Loading..."
            st.markdown(f"""<div class="metric-card"><h3>Ξ Ethereum</h3><div class="value">{val}</div><div class="delta">{market['eth_change']:+,.0f} ({market['eth_pct']:.2f}%)</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card"><h3>📊 Market Vol (30d)</h3><div class="value">{market['market_vol']:.0f}%</div><div class="delta">BTC/ETH</div></div>""", unsafe_allow_html=True)
        with col4: st.write("")
    else:
        if indian_data:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""<div class="metric-card"><h3>🇮🇳 Nifty 50</h3><div class="value">{indian_data['nifty']:,.0f}</div><div class="delta">{indian_data['nifty_change']:+.2f}%</div></div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="metric-card"><h3>📈 Sensex</h3><div class="value">{indian_data['sensex']:,.0f}</div><div class="delta">{indian_data['sensex_change']:+.2f}%</div></div>""", unsafe_allow_html=True)
        else: st.warning("Indian market summary not available.")

# Active Asset Detail
with st.container(border=True):
    st.markdown('<p class="section-header">🎯 Active Asset Details</p>', unsafe_allow_html=True)
    if asset_spot == 0:
        st.error("Live price unavailable.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Spot Price", f"{currency}{asset_spot:,.2f}", f"{asset_change:+,.2f} ({asset_pct:+.2f}%)")
        col2.metric("GARCH Vol", f"{garch_vol:.1f}%")
        col3.metric("GJR‑GARCH Vol", f"{gjrgarch_vol:.1f}%")
        col4.metric("Parkinson Vol", f"{park_vol:.1f}%" if park_vol else "N/A")
        st.caption(f"{asset_choice} | {ticker} | Last update: {lp['ts']}")
        intraday_move = None
        if park_vol:
            intraday_move = asset_spot * (park_vol/100) * np.sqrt(1/trading_days)
            col5, col6 = st.columns(2)
            col5.metric("Intraday Range (±1σ)", f"±{currency}{intraday_move:,.0f}")
            col6.caption(f"Scalp if stay within ±{intraday_move*0.5:,.0f}, swing if break {intraday_move:,.0f}")
        col5, col6 = st.columns(2)
        col5.metric("IV Rank", f"{ivr_val:.0f}%" if ivr_val else "N/A")
        col6.metric("IV Percentile", f"{ivp_val:.0f}%" if ivp_val else "N/A")
        st.caption(get_ivr_label(ivr_val))
        st.info(f"🎯 **Trade Bias:** {trade_bias_label}")
        with st.expander("🎯 Allowed Strategies (Playbook)"):
            for s in playbook_strategies:
                st.write(f"- {s}")
        if 'Exp. Move (D)' in quick_stats:
            val = quick_stats['Exp. Move (D)']['value']
            numeric_part = re.sub(r'[^\d\.\-]', '', val)
            try:
                daily_move_val = float(numeric_part) if numeric_part else 0.0
            except ValueError:
                daily_move_val = 0.0
            if daily_move_val > 0:
                st.write(f"📏 **Strike zones** (based on daily move ±{quick_stats['Exp. Move (D)']['value']}):")
                st.write(f"- Directional OTM strikes: {asset_spot-daily_move_val:,.0f} – {asset_spot+daily_move_val:,.0f}")
                st.write(f"- Short gamma (sell OTM): {asset_spot-daily_move_val*1.5:,.0f} / {asset_spot+daily_move_val*1.5:,.0f}")
            else:
                st.write("📏 Strike zones unavailable.")
        if max_pain:
            distance_pct = abs(asset_spot - max_pain) / asset_spot * 100
            if distance_pct < 1:
                st.success("Max Pain close – expect mean reversion; favour ATM/ITM structures.")
            elif asset_spot < max_pain:
                st.info("Spot below Max Pain – mild bullish bias, watch for gamma resistance.")
            else:
                st.info("Spot above Max Pain – mild bearish bias, support at Max Pain.")
        if selected_market == "Indian Market":
            if 'Nifty' in asset_choice or 'Bank Nifty' in asset_choice:
                lot_size = 25 if 'Nifty' in asset_choice else 15
                st.caption(f"Lot size: {lot_size} | Approx margin: ₹{asset_spot*lot_size*0.15:,.0f} per lot")
                today = datetime.now()
                days_to_expiry = 4 - today.weekday()
                if days_to_expiry <= 2:
                    st.warning(f"⏳ Expiry in {days_to_expiry} days – avoid fresh naked shorts, favor defined‑risk spreads.")
        max_risk_pct = 0.5 if park_vol and park_vol > 50 else 1.0
        max_risk_amount = st.session_state.paper_balance * max_risk_pct / 100
        st.write(f"💼 **Max risk per trade:** {currency}{max_risk_amount:,.0f} ({max_risk_pct}% of capital)")

# Indian Greeks
if selected_market == "Indian Market" and ('Nifty' in asset_choice or 'Bank Nifty' in asset_choice):
    with st.expander("📐 Real‑Time Option Greeks (ATM)"):
        vix_data = get_india_vix("5d")
        if vix_data is not None:
            iv = float(vix_data.iloc[-1])/100
            T = 1/252
            r = 0.065
            sigma = iv
            atm_strike = round(asset_spot, -2) if 'Nifty' in asset_choice else round(asset_spot, -2)
            call = calc_greeks(asset_spot, atm_strike, T, r, sigma, "call")
            put = calc_greeks(asset_spot, atm_strike, T, r, sigma, "put")
            st.write(f"**ATM Strike:** {atm_strike}")
            st.write(f"**IV (from VIX):** {iv*100:.1f}%")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown("**Call**")
                st.write(f"Delta: {call['delta']:.3f}")
                st.write(f"Gamma: {call['gamma']:.4f}")
                st.write(f"Theta: {call['theta']:.3f}")
                st.write(f"Vega: {call['vega']:.3f}")
                st.write(f"Price: {call['price']:.2f}")
            with col_g2:
                st.markdown("**Put**")
                st.write(f"Delta: {put['delta']:.3f}")
                st.write(f"Gamma: {put['gamma']:.4f}")
                st.write(f"Theta: {put['theta']:.3f}")
                st.write(f"Vega: {put['vega']:.3f}")
                st.write(f"Price: {put['price']:.2f}")
            st.caption("**Greeks Impact:** High IV → options expensive, favour selling. Low IV → buy options.")
        else:
            st.warning("VIX data not available for Greeks.")

# Live Terminal
with st.expander("💹 Live Terminal (Chart, Order Book, Quick Trade)", expanded=False):
    st.caption("Real‑time price chart, order book, and quick paper trade panel.")
    if st.button("🔄 Refresh Live Data"):
        st.rerun()
    live_data = yf_download_retry(ticker, period="1d", interval="5m")
    if not live_data.empty:
        live_df = flatten_df(live_data).tail(50)
        fig = go.Figure(data=[go.Candlestick(
            x=live_df.index,
            open=live_df['Open'], high=live_df['High'],
            low=live_df['Low'], close=live_df['Close']
        )])
        fig.update_layout(title=f"{asset_choice} Live Chart (5m)", xaxis_rangeslider_visible=False,
                          template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Intraday data not available.")
    if selected_market == "Crypto":
        col_ob1, col_ob2 = st.columns(2)
        with col_ob1:
            st.markdown("### 📈 Order Book Depth")
            bin_symbol = get_binance_symbol(ticker)
            bids, asks = fetch_binance_orderbook(bin_symbol)
            if bids is not None and asks is not None:
                best_bid = bids['Price'].iloc[0]; best_ask = asks['Price'].iloc[0]
                mid = (best_bid+best_ask)/2
                spread = best_ask-best_bid; spread_pct = (spread/mid)*100
                imbalance = (bids['Size'].sum()-asks['Size'].sum())/(bids['Size'].sum()+asks['Size'].sum())
                st.metric("Best Bid", f"{currency}{best_bid:,.2f}")
                st.metric("Best Ask", f"{currency}{best_ask:,.2f}")
                st.metric("Spread", f"{currency}{spread:,.2f} ({spread_pct:.4f}%)")
                st.metric("Imbalance", f"{imbalance:+.3f}",
                          "Bids heavy" if imbalance>0.1 else ("Asks heavy" if imbalance<-0.1 else "Neutral"))
                fig_depth = go.Figure()
                fig_depth.add_trace(go.Scatter(x=bids['Price'], y=bids['Size'].cumsum(),
                                               mode='lines', name='Bids', line=dict(color='green', width=2),
                                               fill='tozeroy', fillcolor='rgba(0,255,0,0.1)'))
                fig_depth.add_trace(go.Scatter(x=asks['Price'], y=asks['Size'].cumsum(),
                                               mode='lines', name='Asks', line=dict(color='red', width=2),
                                               fill='tozeroy', fillcolor='rgba(255,0,0,0.1)'))
                fig_depth.add_vline(x=mid, line_dash="dot", annotation_text="Mid")
                fig_depth.update_layout(title="Order Book Depth", xaxis_title="Price", yaxis_title="Cumulative Size",
                                        template="plotly_dark", height=300)
                st.plotly_chart(fig_depth, use_container_width=True)
            else:
                st.warning("Binance order book unavailable.")
        with col_ob2:
            st.markdown("### ⚡ Quick Trade (Paper)")
            with st.form("live_trade_form", clear_on_submit=True):
                direction = st.selectbox("Direction", ["Long", "Short"])
                qty = st.number_input("Quantity", min_value=0.01, value=0.01, step=0.01)
                price = st.number_input("Price (Market)", value=asset_spot)
                if st.form_submit_button("Execute"):
                    cost = qty * price
                    if cost > st.session_state['paper_balance']:
                        st.error("Insufficient balance!")
                    else:
                        st.session_state['paper_balance'] -= cost
                        st.session_state['paper_positions'].append({
                            'Asset': asset_choice, 'Direction': direction, 'Qty': qty,
                            'Entry': price, 'Type': 'Spot',
                            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        st.session_state['paper_trade_history'].append({
                            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'Asset': asset_choice, 'Direction': direction, 'Qty': qty,
                            'Price': price, 'Cost': cost, 'Action': 'Open'
                        })
                        st.success(f"{direction} {qty} {asset_choice} @ {currency}{price:,.2f}")
                        st.rerun()
    else:
        st.info("Live order book is only available for Crypto.")
    if st.session_state['paper_positions']:
        st.markdown("### 📋 Open Positions (Live P&L)")
        pos_df = pd.DataFrame(st.session_state['paper_positions'])
        pos_df['Current Price'] = asset_spot
        pos_df['P&L'] = pos_df.apply(
            lambda row: (asset_spot - row['Entry']) * row['Qty'] if row['Direction']=='Long'
            else (row['Entry'] - asset_spot) * row['Qty'], axis=1)
        st.dataframe(pos_df[['Asset','Direction','Qty','Entry','Current Price','P&L']].style.format({
            'Entry': f'{currency}{{:,.2f}}', 'Current Price': f'{currency}{{:,.2f}}', 'P&L': f'{currency}{{:,.2f}}'
        }))
        total_unrealized = pos_df['P&L'].sum()
        st.metric("Unrealized P&L", f"{currency}{total_unrealized:,.2f}")

# Multi‑Timeframe Chart
with st.expander("📊 Multi‑Timeframe Chart (15m, 1h, 4h)"):
    figs = plot_mtf_chart(ticker)
    if figs:
        for fig in figs:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not fetch intraday data.")

# Options Payoff Diagram
with st.expander("📉 Options Payoff Diagram (Quick Builder)"):
    st.markdown("Build a simple multi‑leg strategy and see payoff at expiry.")
    num_legs = st.number_input("Number of legs", min_value=1, max_value=4, value=1, key="payoff_legs")
    legs_data = []
    for i in range(num_legs):
        col1, col2, col3, col4 = st.columns(4)
        with col1: leg_type = st.selectbox(f"Leg {i+1} Type", ["Long Call","Short Call","Long Put","Short Put"], key=f"ptype_{i}")
        with col2: strike = st.number_input(f"Strike {i+1}", value=float(asset_spot), step=float(get_asset_step(asset_spot)), key=f"pstrike_{i}")
        with col3: premium = st.number_input(f"Premium {i+1}", value=5.0, step=0.5, key=f"pprem_{i}")
        with col4: quantity = st.number_input(f"Qty {i+1}", value=1, step=1, key=f"pqty_{i}")
        legs_data.append((leg_type, strike, premium, quantity))
    if st.button("Generate Payoff"):
        strategy_labels = [l[0] for l in legs_data]
        strikes = [l[1] for l in legs_data]
        premiums = [l[2] for l in legs_data]
        T = 0.01
        r = 0.05; sigma = garch_vol_asset/100
        fig = plot_payoff(strategy_labels, asset_spot, strikes, premiums, T, r, sigma)
        st.pyplot(fig)

# Portfolio Risk Dashboard
with st.expander("📊 Portfolio Risk Dashboard"):
    risk = portfolio_risk(st.session_state['paper_positions'], asset_spot, garch_vol_asset)
    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("Delta", f"{risk['Delta']:.3f}")
    col2.metric("Gamma", f"{risk['Gamma']:.4f}")
    col3.metric("Theta", f"{risk['Theta']:.3f}")
    col4.metric("Vega", f"{risk['Vega']:.3f}")
    col5.metric("Margin Req.", f"{currency}{risk['Margin']:,.0f}")
    st.caption("Approximate margin estimate. Greeks are for all open positions.")

# News Sentiment
with st.expander("📰 News Sentiment"):
    sentiments = get_news_sentiment(ticker)
    if sentiments:
        for headline, emoji, pol in sentiments:
            st.markdown(f"{emoji} {headline} (sentiment: {pol:.2f})")
    else:
        st.info("No recent news or unable to fetch.")

# One‑Click Backtesting
with st.expander("🧪 One‑Click Backtesting (Playbook)"):
    st.markdown("Backtest the current trade bias strategy on the last 6 months.")
    if st.button("Run Backtest"):
        hist = st.session_state['long_hist_data'][asset_choice]
        close = hist['Close'].squeeze()
        returns = close.pct_change().dropna()
        if "Long" in trade_bias_label or "Bull" in trade_bias_label:
            direction = 1
        elif "Short" in trade_bias_label or "Bear" in trade_bias_label:
            direction = -1
        else:
            direction = 0
        if direction != 0:
            strategy_returns = returns * direction
            cumulative = (1 + strategy_returns).cumprod()
            win_rate = (strategy_returns > 0).mean() * 100
            sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std()!=0 else 0
            max_dd = (cumulative / cumulative.cummax() - 1).min()
            st.write(f"**Win Rate:** {win_rate:.1f}%")
            st.write(f"**Sharpe Ratio:** {sharpe:.2f}")
            st.write(f"**Max Drawdown:** {max_dd:.2%}")
            fig, ax = plt.subplots()
            ax.plot(cumulative.index, cumulative, color='cyan')
            ax.set_title("Cumulative Return (Biased)")
            st.pyplot(fig)
        else:
            st.info("Neutral bias – no backtest performed.")

# Quick Analytics Overview
with st.container(border=True):
    st.markdown('<p class="section-header">⚡ Quick Analytics Overview</p>', unsafe_allow_html=True)
    n_cols = 4
    keys = list(quick_stats.keys())
    for i in range(0, len(keys), n_cols):
        cols = st.columns(n_cols)
        for j in range(n_cols):
            idx = i+j
            if idx < len(keys):
                key = keys[idx]; stat = quick_stats[key]
                with cols[j]:
                    st.markdown(f"""<div class="quick-stat"><strong>{key}</strong><br><span style="font-size:1.2rem;">{stat['value']}</span><br><small>{stat['status']}</small></div>""", unsafe_allow_html=True)
                    if st.button("🔍", key=f"btn_{key}", help="View detailed chart"):
                        st.session_state['selected_analysis'] = stat['module']; st.rerun()
    with st.expander("📖 What each metric means"):
        st.markdown("""
| Metric | What it tells you |
|--------|-------------------|
| **Correlation** | How closely two indices move together. High = lockstep, Low = decoupling. |
| **Expected Move (D)** | The +/-1σ range for tomorrow. Use it to set strike distances. |
| **Hurst** | Market regime: Trending (>0.55), Mean‑reverting (<0.45), or Random. |
| **IVR/IVP** | IV Rank shows if options are cheap or expensive. High IV = sell premium, Low IV = buy premium. |
| **Parkinson** | Volatility from intraday high‑low range. High = large intraday swings. |
| **Liq. Sweep** | Detects institutional absorption. Supply sweep = bearish, Demand sweep = bullish. |
| **Max Pain** | Strike where option sellers profit most. Price often gravitates toward it. |
        """)

# Market Status – Plain English
with st.container(border=True):
    st.markdown('<p class="section-header">🧠 Market Status – Plain English</p>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="market-summary">', unsafe_allow_html=True)
        summary_lines = []
        if 'Liq. Sweep' in quick_stats:
            sweep = quick_stats['Liq. Sweep']['value']
            if 'Supply' in sweep: summary_lines.append("**Intraday** – 🔻 Supply swept: sellers absorbed, bearish pressure. Keep stops tight.")
            elif 'Demand' in sweep: summary_lines.append("**Intraday** – 🔺 Demand swept: buyers absorbed, bullish pressure. Look for long scalps.")
            else: summary_lines.append("**Intraday** – 🔹 No clear sweep; price in discovery. Wait for structural break.")
        if intraday_move:
            summary_lines.append(f"**Intraday** – 📏 Intraday range ±{currency}{intraday_move:,.0f}. Scalp inside, swing if break.")
        if 'Hurst' in quick_stats:
            h_str = quick_stats['Hurst']['value']
            try:
                h = float(h_str)
                if h > 0.55: summary_lines.append(f"**Swing (2–5d)** – 🚀 Hurst {h:.3f} trending; use pullback entries, trailing stops.")
                elif h < 0.45: summary_lines.append(f"**Swing (2–5d)** – 🔄 Hurst {h:.3f} mean‑reverting; fade breakouts, take profits at mean.")
                else: summary_lines.append(f"**Swing (2–5d)** – ⚪ Hurst {h:.3f} random; avoid aggressive directional bets.")
            except: summary_lines.append("**Swing (2–5d)** – ⚪ Hurst unavailable; trend signals muted.")
        if 'Correlation' in quick_stats:
            try:
                corr = float(quick_stats['Correlation']['value'])
                if corr > 0.8: summary_lines.append("**Swing (2–5d)** – 📈 High correlation; positions move together, reduce correlated risk.")
                elif corr < 0.5: summary_lines.append("**Swing (2–5d)** – ⚠️ Decoupling; favor pair trades or neutral strategies.")
            except: pass
        if 'IVR/IVP' in quick_stats:
            ivr_status = quick_stats['IVR/IVP']['status']
            summary_lines.append(f"**Positional (2–4w)** – 🎯 {ivr_status}")
        if 'Exp. Move (D)' in quick_stats:
            move_str = quick_stats['Exp. Move (D)']['value']
            summary_lines.append(f"**Positional (2–4w)** – 📏 Daily expected move: {move_str}. Use for strike selection.")
        if 'Parkinson' in quick_stats and quick_stats['Parkinson']['value'] != "N/A":
            try:
                park_val = float(quick_stats['Parkinson']['value'].replace('%',''))
                if park_val > garch_vol_asset:
                    summary_lines.append(f"**Positional (2–4w)** – 📊 Parkinson vol {park_val:.1f}% > GARCH; large intraday swings. Reduce size, widen stops.")
            except: pass
        if summary_lines:
            for line in summary_lines:
                st.markdown(line)
        else:
            st.info("Gathering market data...")
        st.markdown('</div>', unsafe_allow_html=True)

# Detailed Chart Section
with st.container(border=True):
    st.markdown('<p class="section-header">📈 Detailed Analysis</p>', unsafe_allow_html=True)
    module_names = ["Correlation","Expected Move","Hurst Exponent","IV Rank & IV Percentile",
                    "Liquidity Detector","Open Interest Profile","Parkinson Estimator",
                    "Volatility Cone","Volatility Risk Premium (VRP)"]
    current_idx = module_names.index(st.session_state['selected_analysis']) if st.session_state['selected_analysis'] in module_names else 0
    module = st.selectbox("Select Analysis", module_names, index=current_idx)
    if module != st.session_state['selected_analysis']:
        st.session_state['selected_analysis'] = module; st.rerun()
    with st.spinner(f"Generating {module}..."):
        if module == "Correlation":
            fig = plot_correlation()
            if fig: st.pyplot(fig)
            st.markdown("**What it indicates:** When correlation drops below 0.5, markets are decoupling.")
        elif module == "Expected Move":
            fig = plot_expected_move()
            if fig: st.pyplot(fig)
            st.markdown("**What it indicates:** Shows the +/-1σ range for the next day.")
        elif module == "Hurst Exponent":
            fig = plot_hurst()
            if fig: st.pyplot(fig)
            else: st.warning("Insufficient data for Hurst calculation.")
            st.markdown("**What it indicates:** H > 0.55 = trending, H < 0.45 = mean‑reverting.")
        elif module == "IV Rank & IV Percentile":
            fig = plot_ivr_ivp()
            if fig: st.pyplot(fig)
            st.markdown("**What it indicates:** IVR > 50 = sell premium, IVR < 50 = buy premium.")
        elif module == "Liquidity Detector":
            fig = plot_liquidity_sweep()
            if fig: st.pyplot(fig)
            st.markdown("**What it indicates:** Sweeps show where institutions absorbed liquidity.")
        elif module == "Open Interest Profile":
            fig = plot_oi_profile()
            if fig: st.pyplot(fig)
            st.markdown("**What it indicates:** Simulated OI profile – not real data.")
        elif module == "Parkinson Estimator":
            fig_park, park_val = plot_parkinson()
            if fig_park:
                st.pyplot(fig_park)
                st.markdown(f"**Current Parkinson Vol:** {park_val:.1f}%")
            else: st.warning("Parkinson volatility could not be calculated.")
        elif module == "Volatility Cone":
            fig = plot_volatility_cone()
            if fig: st.pyplot(fig)
            st.markdown("**What it indicates:** Where current vol sits inside the cone.")
        elif module == "Volatility Risk Premium (VRP)":
            fig = plot_vrp()
            if fig: st.pyplot(fig)
            st.markdown("**What it indicates:** Positive VRP = implied > actual (sell premium).")

st.markdown("---")
st.caption("AlphaQuant Terminal Pro · Daily Snapshot Logger")