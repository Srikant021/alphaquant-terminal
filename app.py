import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import scipy.stats as si
from arch import arch_model
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import yaml, logging, os, time, requests

# Optional ML / Telegram
try:
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except:
    TELEGRAM_AVAILABLE = False

# -----------------------------------------------------------------------------
# PAGE CONFIG & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AlphaQuant Terminal Pro", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    body, .stApp { font-family: 'Segoe UI', 'Inter', sans-serif; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label { color: #e0e0e0 !important; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f, #2a2a40);
        border-radius: 12px; padding: 15px; margin: 5px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #3a3a5c;
    }
    .metric-card h3 { color: #f0f0f0; font-size: 0.9rem; margin-bottom: 5px; }
    .metric-card .value { font-size: 1.5rem; font-weight: 700; color: #ffffff; }
    .metric-card .delta { font-size: 0.85rem; color: #aaaaaa; }
    .quick-stat {
        background: rgba(255,255,255,0.05); border-radius: 8px; padding: 10px;
        text-align: center; border: 1px solid #3a3a5c; transition: all 0.2s ease;
    }
    .quick-stat:hover { background: rgba(255,255,255,0.1); transform: translateY(-2px); }
    .stButton>button {
        border-radius: 8px; background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; font-weight: 600; transition: all 0.2s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 4px 12px rgba(118,75,162,0.4); }
    .section-header { font-size: 1.3rem; font-weight: 700; margin-top: 20px; margin-bottom: 10px; color: #e0e0ff; }
    .market-summary {
        background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1));
        border-radius: 12px; padding: 20px; border: 1px solid #4a4a6a; margin: 10px 0;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.03); border-radius: 12px; padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'live_mode' not in st.session_state: st.session_state['live_mode'] = False
if 'refresh_interval' not in st.session_state: st.session_state['refresh_interval'] = 120
if 'selected_market' not in st.session_state: st.session_state['selected_market'] = "Crypto"
if 'active_tab' not in st.session_state: st.session_state['active_tab'] = "📊 Dashboard & Analytics"
if 'selected_analysis' not in st.session_state: st.session_state['selected_analysis'] = "Correlation"
if 'snapshots' not in st.session_state: st.session_state['snapshots'] = []

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
        'cryptos': {
            'Bitcoin': 'BTC-USD', 'Ethereum': 'ETH-USD',
            'Dogecoin': 'DOGE-USD', 'XRP': 'XRP-USD'
        },
        'indian_market': {
            'Nifty 50': '^NSEI', 'Sensex': '^BSESN', 'Bank Nifty': '^NSEBANK',
            'Gold (MCX)': 'GOLDM.NS', 'Silver (MCX)': 'SILVERM.NS',
            'Crude Oil (MCX)': 'CRUDEOIL.NS', 'Natural Gas (MCX)': 'NATURALGAS.NS'
        },
        'cache_ttl': {'long_hist':3600,'garch':1800,'live_price':300,'intraday':300},
        'alerts': {'btc_level':None,'eth_level':None,'telegram_token':None,'telegram_chat_id':None},
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
FALLBACK = {'btc':'N/A','btc_change':0,'btc_pct':0,'eth':'N/A','eth_change':0,'eth_pct':0,'market_vol':0,'timestamp':'Fallback'}

plt.style.use('dark_background')
logging.basicConfig(level=logging.INFO); logger=logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# HELPERS
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
    lags=range(2, min(20, len(ts)//5))
    if len(lags)<3: return 0.50
    try:
        tau=[np.sqrt(np.std(np.subtract(ts[lag:],ts[:-lag]))) for lag in lags]
        return np.polyfit(np.log(lags),np.log(tau),1)[0]*2.0
    except: return 0.50

def get_asset_step(spot_price):
    if spot_price>50000: return 2000
    elif spot_price>10000: return 500
    elif spot_price>1000: return 50
    elif spot_price>100: return 10
    else: return 1

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
        data=yf_download_retry(['BTC-USD','ETH-USD'], period="2d")
        if data.empty: return FALLBACK
        close=flatten_df(data)['Close']
        btc=float(close['BTC-USD'].iloc[-1]); btc_p=float(close['BTC-USD'].iloc[-2])
        eth=float(close['ETH-USD'].iloc[-1]); eth_p=float(close['ETH-USD'].iloc[-2])
        hist_btc=yf_download_retry('BTC-USD',period="1mo")
        hist_eth=yf_download_retry('ETH-USD',period="1mo")
        vol_btc=np.log(hist_btc['Close']/hist_btc['Close'].shift(1)).std()*np.sqrt(365)*100 if not hist_btc.empty else 0
        vol_eth=np.log(hist_eth['Close']/hist_eth['Close'].shift(1)).std()*np.sqrt(365)*100 if not hist_eth.empty else 0
        mkt_vol=(vol_btc+vol_eth)/2 if vol_btc and vol_eth and not np.isnan(vol_btc) and not np.isnan(vol_eth) else 65
        return {'btc':btc,'btc_change':btc-btc_p,'btc_pct':((btc-btc_p)/btc_p)*100,
                'eth':eth,'eth_change':eth-eth_p,'eth_pct':((eth-eth_p)/eth_p)*100 if eth_p else 0,
                'market_vol':mkt_vol,'timestamp':datetime.now().strftime('%H:%M:%S')}
    except: return FALLBACK

@st.cache_data(ttl=120, show_spinner=False)
def fetch_indian_market_summary():
    try:
        nifty = yf_download_retry('^NSEI', period="2d")
        sensex = yf_download_retry('^BSESN', period="2d")
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

def send_telegram_alert(message):
    if not TELEGRAM_AVAILABLE: return
    token=CONFIG['alerts'].get('telegram_token'); chat_id=CONFIG['alerts'].get('telegram_chat_id')
    if token and chat_id:
        try: bot=Bot(token=token); bot.send_message(chat_id=chat_id, text=message)
        except: pass

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
    reasoning_parts.append(f"Daily+-{currency}{daily_move:,.0f}")
    reasoning_parts.append(f"Conf={confidence}% Risk={risk_level}")
    if ml_pred is not None: reasoning_parts.append(f"ML={'Bull' if ml_pred==1 else 'Bear'}")
    return {
        "regime":regime,"vol_environment":vol_env,"sweep_signal":sweep_signal,
        "suggested_strategy":strategy,"direction":direction,"confidence":confidence,
        "risk_level":risk_level,"reasoning":" . ".join(reasoning_parts),
        "daily_move":daily_move,"garch_vol":garch_vol,"park_vol":park_vol,
        "ivr":ivr,"ivp":ivp,"deribit_iv":deribit_iv,"confluence":confluence,
        "funding_rate":funding_rate,"ml_pred":ml_pred
    }

def generate_trading_tips(signal):
    tips=["🛑 Risk 1-2% per trade.","📊 Size inversely to volatility.","⏳ Close short options before final hour."]
    strat=signal.get('suggested_strategy',''); daily_move=signal.get('daily_move',0); spot=asset_spot
    if "Iron Condor" in strat: tips.append(f"🕊️ Short strikes near +-{currency}{daily_move:,.0f}.")
    elif "Spread" in strat:
        if "Bull" in strat: tips.append(f"📈 Long {spot:,.0f} call, short {spot+daily_move:,.0f} call.")
        elif "Bear" in strat: tips.append(f"📉 Long {spot:,.0f} put, short {spot-daily_move:,.0f} put.")
    if signal.get('ml_pred') is not None: tips.append(f"🤖 ML predicts {'up' if signal['ml_pred']==1 else 'down'} next period.")
    if signal.get('funding_rate') and abs(signal['funding_rate'])>0.1: tips.append(f"📡 Funding rate extreme ({signal['funding_rate']:.3f}%) - possible reversal.")
    return tips

# -----------------------------------------------------------------------------
# ACTIONABLE RULES & PLAYBOOK
# -----------------------------------------------------------------------------
def get_trade_bias(garch_vol, ivr, corr):
    bias = ""
    if ivr is None: ivr = 25
    if ivr < 25:
        if garch_vol < 40:
            bias = "Small debit structures (vertical spreads, calendars). Avoid naked shorts."
        else:
            bias = "Directional with defined risk. Favor long calls/puts over short premium."
    elif 25 <= ivr <= 50:
        bias = "Mixed - credit spreads in range, debit if trending. Keep size small."
    else:
        if garch_vol > 80:
            bias = "Short premium favoured (iron condors, strangles). Strict risk caps."
        else:
            bias = "Sell OTM premium, but hedge tail risk."
    if corr and corr > 0.8:
        bias += " High correlation - trend following valid, but watch for systematic risk."
    elif corr and corr < 0.5:
        bias += " Decoupling - use pairs or neutral strategies."
    return bias

def get_playbook(garch_vol, ivr, corr):
    if ivr is None: ivr = 25
    if ivr < 25:
        if garch_vol < 40:
            return ["Bull/Bear vertical spreads", "Calendars", "Long calls/puts (1 month)"]
        else:
            return ["Long strangles (defined risk)", "Backspreads", "Avoid naked shorts"]
    elif ivr > 50:
        return ["Iron Condors (45 DTE)", "Short strangles (strict risk management)", "Credit spreads"]
    else:
        return ["Vertical spreads", "Covered calls/puts", "Iron Butterflies"]

def get_ivr_label(ivr):
    if ivr is None: return "IVR unavailable"
    if ivr < 25: return f"Low IV (IVR {ivr:.0f}%) - Buy premium, defined-risk spreads"
    elif ivr > 50: return f"High IV (IVR {ivr:.0f}%) - Sell premium, small size"
    else: return f"Mid IV (IVR {ivr:.0f}%) - Mixed, favour defined risk"

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
            if len(atm_idx) > 0:
                iv = opt_df.iloc[atm_idx]['mark_iv'].values[0] * 100
            else:
                iv = garch_vol_asset
        else:
            iv = garch_vol_asset
        days_in_year, spot_label, vix_label = 365, f"Spot: {currency}{asset_spot:,.0f}", f"ATM IV: {iv:.1f}%"
    else:
        vix_data = get_india_vix("5d")
        if vix_data is None:
            st.error("Could not fetch India VIX")
            return None
        iv = float(vix_data.iloc[-1])
        days_in_year, spot_label, vix_label = 365, f"Nifty: {asset_spot:,.0f}", f"India VIX: {iv:.2f}"

    t_daily, t_weekly, t_monthly = 1/days_in_year, 7/days_in_year, 30/days_in_year
    move_daily = asset_spot * (iv/100) * np.sqrt(t_daily)
    move_weekly = asset_spot * (iv/100) * np.sqrt(t_weekly)
    move_monthly = asset_spot * (iv/100) * np.sqrt(t_monthly)

    recent = get_recent_month(ticker)
    if recent.empty: return None
    fig, ax = plt.subplots(figsize=(12,6))
    x = np.arange(len(recent)); ax.plot(x, recent.values, 'o-', color='cyan', label='Price')
    tmr = len(recent); ax.scatter(tmr, asset_spot, color='white', s=80, zorder=5)
    for color, move, label in [('yellow', move_daily,'Daily'),('orange',move_weekly,'Weekly'),('red',move_monthly,'Monthly')]:
        upper = asset_spot+move; lower = asset_spot-move
        ax.vlines(tmr, lower, upper, color=color, linewidth=2, label=f'{label} Range')
        ax.scatter(tmr, upper, color=color, marker='^', s=100)
        ax.scatter(tmr, lower, color=color, marker='v', s=100)
    ax.set_title("Expected Move (Daily, Weekly, Monthly)", fontweight='bold')
    ax.set_xticks([]); ax.legend(loc='upper left', facecolor='black', edgecolor='gray')
    text_str = (f"{vix_label}\n{spot_label}\n"
                f"Daily: +-{move_daily:,.0f}  [{asset_spot-move_daily:,.0f}-{asset_spot+move_daily:,.0f}]\n"
                f"Weekly: +-{move_weekly:,.0f}  [{asset_spot-move_weekly:,.0f}-{asset_spot+move_weekly:,.0f}]\n"
                f"Monthly: +-{move_monthly:,.0f}  [{asset_spot-move_monthly:,.0f}-{asset_spot+move_monthly:,.0f}]")
    props = dict(boxstyle='round', facecolor='black', edgecolor='white')
    ax.text(0.02,0.98, text_str, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_hurst():
    close = st.session_state['long_hist_data'][asset_choice]['Close'].squeeze()
    log_p = np.log(close); hurst_series = log_p.rolling(60).apply(calculate_hurst_exponent)
    df = pd.DataFrame({'Close':close,'Hurst':hurst_series}).dropna()
    if df.empty: return None
    current_hurst = df['Hurst'].iloc[-1]
    if current_hurst>0.55: regime,color="Trending",'green'
    elif current_hurst<0.45: regime,color="Mean Reverting",'red'
    else: regime,color="Random Walk",'orange'
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,7),gridspec_kw={'height_ratios':[2,1]})
    ax1.plot(df.index,df['Close'],color='white'); ax1.set_title("Market Regime (Hurst Exponent)",fontweight='bold')
    ax1.axvspan(df.index[-15],df.index[-1],color=color,alpha=0.1)
    ax2.plot(df.index,df['Hurst'],color='cyan')
    ax2.axhline(0.55,color='green',linestyle='--'); ax2.axhline(0.45,color='red',linestyle='--'); ax2.set_ylim(0.3,0.7)
    props=dict(boxstyle='round',facecolor='black',edgecolor=color)
    ax1.text(0.02,0.05,f"Hurst: {current_hurst:.3f}\nRegime: {regime}",transform=ax1.transAxes,bbox=props,fontsize=12)
    plt.tight_layout(); return fig

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
    plt.tight_layout()
    return fig

def plot_parkinson():
    df = st.session_state['long_hist_data'][asset_choice]
    if df.empty or not all(c in df.columns for c in ['High','Low']):
        return None, None
    high = df['High'].squeeze().tail(60)
    low = df['Low'].squeeze().tail(60)
    if len(high) < 2 or len(low) < 2:
        return None, None
    park = calculate_parkinson_volatility(high, low, periods_per_year=trading_days)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(['Parkinson Vol'], [park], color='orange')
    ax.set_ylabel('Annualized Vol (%)')
    ax.set_title("Parkinson Estimator (High-Low Range)", fontweight='bold')
    for i, v in enumerate([park]):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    return fig, park

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
        quick_stats['Exp. Move (D)'] = {'value': f"+-{currency}{daily_move:,.0f}", 'status': '1sigma Range', 'module': 'Expected Move'}
    else:
        vix_data = get_india_vix("5d")
        if vix_data is not None:
            iv = float(vix_data.iloc[-1])
            daily_move = asset_spot * (iv/100) * np.sqrt(1/365)
            quick_stats['Exp. Move (D)'] = {'value': f"+-{currency}{daily_move:,.0f}", 'status': '1sigma Range', 'module': 'Expected Move'}
    close = st.session_state['long_hist_data'][asset_choice]['Close'].squeeze()
    log_p = np.log(close); hurst_series = log_p.rolling(60).apply(calculate_hurst_exponent)
    df = pd.DataFrame({'Close':close,'Hurst':hurst_series}).dropna()
    if not df.empty:
        hurst_val = df['Hurst'].iloc[-1]
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
# COMPACT TOOLBAR (top of main area)
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

# Sidebar (ML, navigation only)
with st.sidebar:
    st.markdown("## 🧬 AlphaQuant Terminal")
    if ML_AVAILABLE:
        if st.button("🧠 Train ML Model"):
            hist_b = st.session_state.get('long_hist_data', {}).get(asset_choice)
            if hist_b is not None:
                close = hist_b['Close'].squeeze(); train_ml_model(close); st.success("ML model trained!")
    auto_exit = st.checkbox("🛑 Auto-Exit (1.5x loss or 2sigma move)", value=st.session_state['auto_exit_enabled'])
    if auto_exit != st.session_state['auto_exit_enabled']: st.session_state['auto_exit_enabled'] = auto_exit
    st.markdown("---")
    tab = st.radio("📑 Navigate", [
        "📊 Dashboard & Analytics",
        "📄 Paper Trading",
        "🧙 Strategy Wizard",
        "📓 Journal"
    ])
    st.session_state.active_tab = tab

# -----------------------------------------------------------------------------
# INITIAL DATA LOADING
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
# RENDER TABS
# -----------------------------------------------------------------------------
active_tab = st.session_state.get('active_tab', '📊 Dashboard & Analytics')

if active_tab == "📊 Dashboard & Analytics":
    st.title("📊 Market Intelligence Dashboard")
    # Market Overview card
    with st.container(border=True):
        st.markdown('<p class="section-header">🌍 Market Overview</p>', unsafe_allow_html=True)
        if selected_market == "Crypto":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                val = f"${market['btc']:,.0f}" if isinstance(market['btc'], (int,float)) else market['btc']
                st.markdown(f"""<div class="metric-card"><h3>Bitcoin</h3><div class="value">{val}</div><div class="delta">{market['btc_change']:+,.0f} ({market['btc_pct']:.2f}%)</div></div>""", unsafe_allow_html=True)
            with col2:
                val = f"${market['eth']:,.0f}" if isinstance(market['eth'], (int,float)) else market['eth']
                st.markdown(f"""<div class="metric-card"><h3>Ethereum</h3><div class="value">{val}</div><div class="delta">{market['eth_change']:+,.0f} ({market['eth_pct']:.2f}%)</div></div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class="metric-card"><h3>Market Vol (30d)</h3><div class="value">{market['market_vol']:.0f}%</div><div class="delta">BTC/ETH</div></div>""", unsafe_allow_html=True)
            with col4: st.write("")
        else:
            if indian_data:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""<div class="metric-card"><h3>Nifty 50</h3><div class="value">{indian_data['nifty']:,.0f}</div><div class="delta">{indian_data['nifty_change']:+.2f}%</div></div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""<div class="metric-card"><h3>Sensex</h3><div class="value">{indian_data['sensex']:,.0f}</div><div class="delta">{indian_data['sensex_change']:+.2f}%</div></div>""", unsafe_allow_html=True)
            else: st.warning("Indian market summary not available.")

    # Active Asset Detail card
    with st.container(border=True):
        st.markdown('<p class="section-header">Active Asset Details</p>', unsafe_allow_html=True)
        if asset_spot == 0:
            st.error("Live price unavailable.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Spot Price", f"{currency}{asset_spot:,.2f}", f"{asset_change:+,.2f} ({asset_pct:+.2f}%)")
            col2.metric("GARCH Vol", f"{garch_vol:.1f}%")
            col3.metric("GJR-GARCH Vol", f"{gjrgarch_vol:.1f}%")
            col4.metric("Parkinson Vol", f"{park_vol:.1f}%" if park_vol else "N/A")
            st.caption(f"{asset_choice} | {ticker} | Last update: {lp['ts']}")
            # Intraday range from Parkinson
            intraday_move = None
            if park_vol:
                intraday_move = asset_spot * (park_vol/100) * np.sqrt(1/trading_days)
                col5, col6 = st.columns(2)
                col5.metric("Intraday Range (+-1sigma)", f"+-{currency}{intraday_move:,.0f}")
                col6.caption(f"Scalp if stay within +-{intraday_move*0.5:,.0f}, swing if break {intraday_move:,.0f}")
            # IVR/IVP
            col5, col6 = st.columns(2)
            col5.metric("IV Rank", f"{ivr_val:.0f}%" if ivr_val else "N/A")
            col6.metric("IV Percentile", f"{ivp_val:.0f}%" if ivp_val else "N/A")
            st.caption(get_ivr_label(ivr_val))
            # Trade Bias
            st.info(f"Trade Bias: {trade_bias_label}")
            # Playbook
            with st.expander("Allowed Strategies (Playbook)"):
                for s in playbook_strategies:
                    st.write(f"- {s}")
            # Strike zones (from expected move)
            if 'Exp. Move (D)' in quick_stats:
                val = quick_stats['Exp. Move (D)']['value']
                daily_move_val = float(val.replace(currency,'').replace('+-','').replace(',',''))
                st.write(f"Strike zones (based on daily move +-{quick_stats['Exp. Move (D)']['value']}):")
                st.write(f"- Directional OTM strikes: {asset_spot-daily_move_val:,.0f} - {asset_spot+daily_move_val:,.0f}")
                st.write(f"- Short gamma (sell OTM): {asset_spot-daily_move_val*1.5:,.0f} / {asset_spot+daily_move_val*1.5:,.0f}")
            # Max Pain alignment
            if max_pain:
                distance_pct = abs(asset_spot - max_pain) / asset_spot * 100
                if distance_pct < 1:
                    st.success("Max Pain close - expect mean reversion; favour ATM/ITM structures.")
                elif asset_spot < max_pain:
                    st.info("Spot below Max Pain - mild bullish bias, watch for gamma resistance.")
                else:
                    st.info("Spot above Max Pain - mild bearish bias, support at Max Pain.")
            # Indian market specific
            if selected_market == "Indian Market":
                if 'Nifty' in asset_choice or 'Bank Nifty' in asset_choice:
                    lot_size = 25 if 'Nifty' in asset_choice else 15
                    st.caption(f"Lot size: {lot_size} | Approx margin: Rs {asset_spot*lot_size*0.15:,.0f} per lot")
                    today = datetime.now()
                    days_to_expiry = 4 - today.weekday()
                    if days_to_expiry <= 2:
                        st.warning(f"Expiry in {days_to_expiry} days - avoid fresh naked shorts, favor defined-risk spreads.")
            # Dynamic position sizing
            max_risk_pct = 0.5 if park_vol and park_vol > 50 else 1.0
            max_risk_amount = st.session_state.paper_balance * max_risk_pct / 100
            st.write(f"Max risk per trade: {currency}{max_risk_amount:,.0f} ({max_risk_pct}% of capital)")

    # Quick Analytics card
    with st.container(border=True):
        st.markdown('<p class="section-header">Quick Analytics Overview</p>', unsafe_allow_html=True)
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

    # Market Status - Plain English (time-horizon tagged)
    with st.container(border=True):
        st.markdown('<p class="section-header">Market Status - Plain English</p>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="market-summary">', unsafe_allow_html=True)
            summary_lines = []
            if 'Liq. Sweep' in quick_stats:
                sweep = quick_stats['Liq. Sweep']['value']
                if 'Supply' in sweep: summary_lines.append("**Intraday** - Supply swept: sellers absorbed, bearish pressure. Keep stops tight.")
                elif 'Demand' in sweep: summary_lines.append("**Intraday** - Demand swept: buyers absorbed, bullish pressure. Look for long scalps.")
                else: summary_lines.append("**Intraday** - No clear sweep; price in discovery. Wait for structural break.")
            if intraday_move:
                summary_lines.append(f"**Intraday** - Intraday range +-{currency}{intraday_move:,.0f}. Scalp inside, swing if break.")
            if 'Hurst' in quick_stats:
                h_str = quick_stats['Hurst']['value']
                try:
                    h = float(h_str)
                    if h > 0.55: summary_lines.append(f"**Swing (2-5d)** - Hurst {h:.3f} trending; use pullback entries, trailing stops.")
                    elif h < 0.45: summary_lines.append(f"**Swing (2-5d)** - Hurst {h:.3f} mean-reverting; fade breakouts, take profits at mean.")
                    else: summary_lines.append(f"**Swing (2-5d)** - Hurst {h:.3f} random; avoid aggressive directional bets.")
                except: summary_lines.append("**Swing (2-5d)** - Hurst unavailable; trend signals muted.")
            if 'Correlation' in quick_stats:
                try:
                    corr = float(quick_stats['Correlation']['value'])
                    if corr > 0.8: summary_lines.append("**Swing (2-5d)** - High correlation; positions move together, reduce correlated risk.")
                    elif corr < 0.5: summary_lines.append("**Swing (2-5d)** - Decoupling; favor pair trades or neutral strategies.")
                except: pass
            if 'IVR/IVP' in quick_stats:
                ivr_status = quick_stats['IVR/IVP']['status']
                summary_lines.append(f"**Positional (2-4w)** - {ivr_status}")
            if 'Exp. Move (D)' in quick_stats:
                move_str = quick_stats['Exp. Move (D)']['value']
                summary_lines.append(f"**Positional (2-4w)** - Daily expected move: {move_str}. Use for strike selection.")
            if 'Parkinson' in quick_stats and quick_stats['Parkinson']['value'] != "N/A":
                try:
                    park_val = float(quick_stats['Parkinson']['value'].replace('%',''))
                    if park_val > garch_vol_asset:
                        summary_lines.append(f"**Positional (2-4w)** - Parkinson vol {park_val:.1f}% > GARCH; large intraday swings. Reduce size, widen stops.")
                except: pass
            if summary_lines:
                for line in summary_lines:
                    st.markdown(line)
            else:
                st.info("Gathering market data...")
            st.markdown('</div>', unsafe_allow_html=True)

    # Detailed Chart Section
    with st.container(border=True):
        st.markdown('<p class="section-header">Detailed Analysis</p>', unsafe_allow_html=True)
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
                st.markdown("**What it indicates:** Shows the +/-1sigma range for the next day, week, and month.")
            elif module == "Hurst Exponent":
                fig = plot_hurst()
                if fig: st.pyplot(fig)
                else: st.warning("Insufficient data for Hurst calculation.")
                st.markdown("**What it indicates:** H > 0.55 = trending, H < 0.45 = mean-reverting.")
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
                st.markdown("**What it indicates:** Simulated OI profile - not real data. For demonstration only.")
            elif module == "Parkinson Estimator":
                fig_park, park_val = plot_parkinson()
                if fig_park:
                    st.pyplot(fig_park)
                    st.markdown(f"**Current Parkinson Vol:** {park_val:.1f}% - High values relative to GARCH indicate large intraday swings; adjust stops accordingly.")
                else: st.warning("Parkinson volatility could not be calculated.")
            elif module == "Volatility Cone":
                fig = plot_volatility_cone()
                if fig: st.pyplot(fig)
                st.markdown("**What it indicates:** Where current vol sits inside the cone helps assess if options are cheap or expensive.")
            elif module == "Volatility Risk Premium (VRP)":
                fig = plot_vrp()
                if fig: st.pyplot(fig)
                st.markdown("**What it indicates:** Positive VRP = implied > actual (sell premium). Negative VRP = actual > implied (buy premium).")

elif active_tab == "📄 Paper Trading":
    st.title("📄 Paper Trading")
    st.markdown(f"Simulate trades with a **{currency}100,000** virtual account.")
    col_bal, col_pnl = st.columns(2)
    with col_bal: st.metric("Cash Balance", f"{currency}{st.session_state['paper_balance']:,.2f}")
    unrealized_pnl = 0
    for pos in st.session_state['paper_positions']:
        if pos['Type'] == 'Spot':
            unrealized_pnl += (asset_spot - pos['Entry']) * pos['Qty'] if pos['Direction'] == 'Long' else (pos['Entry'] - asset_spot) * pos['Qty']
    total_equity = st.session_state['paper_balance'] + unrealized_pnl
    col_pnl.metric("Total Equity", f"{currency}{total_equity:,.2f}", delta=f"Unrealized: {currency}{unrealized_pnl:,.2f}")
    with st.expander("Quick Trade (Manual)"):
        with st.form("paper_trade_form"):
            c1, c2 = st.columns(2)
            asset = c1.selectbox("Asset", list(TICKER_DICT.keys()), key="paper_asset")
            direction = c2.selectbox("Direction", ["Long", "Short"])
            qty = st.number_input("Quantity", min_value=0.01, value=0.01, step=0.01)
            price = st.number_input("Price", value=asset_spot)
            if st.form_submit_button("Execute Trade"):
                cost = qty * price
                if cost > st.session_state['paper_balance']:
                    st.error("Insufficient balance!")
                else:
                    st.session_state['paper_balance'] -= cost
                    st.session_state['paper_positions'].append({
                        'Asset': asset, 'Direction': direction, 'Qty': qty,
                        'Entry': price, 'Type': 'Spot',
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    st.session_state['paper_trade_history'].append({
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Asset': asset, 'Direction': direction, 'Qty': qty,
                        'Price': price, 'Cost': cost, 'Action': 'Open'
                    })
                    st.success(f"Bought {qty} {asset} @ {currency}{price:,.2f}")
    if st.session_state['paper_positions']:
        st.subheader("Open Positions")
        pos_df = pd.DataFrame(st.session_state['paper_positions']); pos_df.index = range(1, len(pos_df)+1)
        st.dataframe(pos_df)
        close_idx = st.selectbox("Select position to close", pos_df.index)
        close_price = st.number_input("Close Price", value=asset_spot)
        if st.button("Close Position"):
            pos = pos_df.loc[close_idx]
            pnl = (close_price - pos['Entry']) * pos['Qty'] if pos['Direction']=='Long' else (pos['Entry'] - close_price) * pos['Qty']
            st.session_state['paper_balance'] += (pos['Entry'] * pos['Qty'] + pnl)
            st.session_state['paper_trade_history'].append({
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Asset': pos['Asset'], 'Direction': pos['Direction'], 'Qty': pos['Qty'],
                'Price': close_price, 'PnL': pnl, 'Action': 'Close'
            })
            st.session_state['paper_positions'].pop(close_idx-1)
            st.success(f"Closed position with P&L: {currency}{pnl:,.2f}"); st.rerun()
    if st.session_state['paper_trade_history']:
        st.subheader("Trade History")
        hist_df = pd.DataFrame(st.session_state['paper_trade_history']); st.dataframe(hist_df)
        if 'PnL' in hist_df.columns:
            total_realized = hist_df['PnL'].sum(); win_trades = hist_df[hist_df['PnL'] > 0]
            st.metric("Total Realized P&L", f"{currency}{total_realized:,.2f}")
            if len(hist_df[~hist_df['PnL'].isna()]) > 0:
                st.metric("Win Rate", f"{len(win_trades) / len(hist_df[~hist_df['PnL'].isna()]) * 100:.1f}%")
    else: st.info("No trades executed yet.")
    if st.button("Reset Paper Account"):
        st.session_state['paper_balance'] = 100000; st.session_state['paper_positions'] = []; st.session_state['paper_trade_history'] = []; st.rerun()

elif active_tab == "🧙 Strategy Wizard":
    st.title("🧙 Strategy Wizard")
    signal_w = get_intraday_signal(asset_choice, ticker)
    if signal_w is not None and 'error' not in signal_w:
        st.write(f"**Market Regime:** {signal_w['regime']}")
        st.write(f"**Vol Environment:** {signal_w['vol_environment']}")
        st.write(f"**Suggested Strategy:** {signal_w['suggested_strategy']}")
        st.write(f"**Confidence:** {signal_w['confidence']}%")
        dte_w = st.slider("Select DTE", 0, 7, 4)
        risk_perc = st.slider("Risk % per trade", 0.5, 5.0, 1.0, 0.5)
        if st.button("Execute via Paper Trading"):
            qty = (st.session_state['paper_balance'] * risk_perc / 100) / asset_spot
            direction = "Long" if "Bull" in signal_w['suggested_strategy'] or "Long" in signal_w['direction'] else "Short"
            cost = qty * asset_spot
            if cost <= st.session_state['paper_balance']:
                st.session_state['paper_balance'] -= cost
                st.session_state['paper_positions'].append({
                    'Asset': asset_choice, 'Direction': direction, 'Qty': qty,
                    'Entry': asset_spot, 'Type': 'Spot',
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                st.success(f"Opened {direction} {qty:.4f} {asset_choice} @ {currency}{asset_spot:,.2f}")
                send_telegram_alert(f"Wizard opened {direction} {qty:.4f} {asset_choice} @ {currency}{asset_spot:,.2f}")
            else:
                st.error("Insufficient balance.")
    else:
        st.warning("Signal unavailable for strategy wizard.")

elif active_tab == "📓 Journal":
    st.title("📓 Trading Journal")
    if st.button("📸 Log Current Snapshot"):
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'asset': asset_choice,
            'spot': asset_spot,
            'garch_vol': garch_vol_asset,
            'gjrgarch_vol': gjrgarch_vol,
            'park_vol': park_vol,
            'ivr': ivr_val,
            'ivp': ivp_val,
            'corr': corr_val,
            'trade_bias': trade_bias_label,
            'playbook': playbook_strategies,
        }
        st.session_state.setdefault('snapshots', []).append(snapshot)
        st.success("Snapshot saved!")
    with st.expander("+ New Trade Entry"):
        with st.form("trade_form"):
            c1,c2,c3 = st.columns(3)
            asset = c1.selectbox("Asset", list(TICKER_DICT.keys()), key="journal_asset")
            dir = c2.selectbox("Direction", ["Long","Short"])
            entry = c3.number_input("Entry Price", min_value=0.0, step=0.01, format="%.2f")
            exit_p = st.number_input("Exit Price", min_value=0.0, step=0.01, format="%.2f")
            qty = st.number_input("Quantity", min_value=0.0, step=0.01, format="%.4f")
            date = st.date_input("Date", datetime.today()); notes = st.text_area("Notes")
            if st.form_submit_button("Log Trade"):
                if entry<=0 or exit_p<=0 or qty<=0: st.error("Prices and quantity must be positive.")
                else:
                    pnl = (exit_p-entry)*qty if dir=="Long" else (entry-exit_p)*qty
                    regime_tag = f"IVR={ivr_val:.0f}, GARCH={garch_vol_asset:.0f}, Corr={corr_val:.2f}"
                    st.session_state['trade_journal'].append({
                        "Date":date.strftime("%Y-%m-%d"),"Asset":asset,"Direction":dir,
                        "Entry":entry,"Exit":exit_p,"Quantity":qty,"P&L":round(pnl,2),"Notes":notes,
                        "Regime": regime_tag
                    })
                    st.success("Trade logged!")
    snapshots = st.session_state.get('snapshots', [])
    if snapshots:
        st.subheader("Saved Snapshots")
        df_snaps = pd.DataFrame(snapshots)
        st.dataframe(df_snaps)
    if st.session_state['trade_journal']:
        jdf = pd.DataFrame(st.session_state['trade_journal']); jdf.index = range(1,len(jdf)+1)
        st.subheader("Trade Log")
        st.dataframe(jdf.style.format({
            "Entry":f"{currency}{{:,.2f}}","Exit":f"{currency}{{:,.2f}}","Quantity":"{:.4f}","P&L":f"{currency}{{:,.2f}}"
        }))
        total_pnl = jdf['P&L'].sum(); wins = jdf[jdf['P&L']>0]; losses = jdf[jdf['P&L']<0]
        win_rate = len(wins)/len(jdf)*100 if len(jdf) else 0
        avg_w = wins['P&L'].mean() if not wins.empty else 0; avg_l = losses['P&L'].mean() if not losses.empty else 0
        pf = abs(wins['P&L'].sum()/losses['P&L'].sum()) if not losses.empty and losses['P&L'].sum()!=0 else float('inf')
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Total P&L", f"{currency}{total_pnl:,.2f}"); m2.metric("Win Rate", f"{win_rate:.1f}%")
        m3.metric("Avg Win", f"{currency}{avg_w:,.2f}"); m4.metric("Avg Loss", f"{currency}{avg_l:,.2f}")
        st.metric("Profit Factor", f"{pf:.2f}" if pf!=float('inf') else "inf")
        if 'Regime' in jdf.columns:
            regimes = jdf['Regime'].unique()
            selected_regime = st.selectbox("Filter by Regime", ["All"] + list(regimes))
            if selected_regime != "All":
                fdf = jdf[jdf['Regime'] == selected_regime]
                st.dataframe(fdf); st.write(f"Filtered P&L: {currency}{fdf['P&L'].sum():,.2f}")
        csv = jdf.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "trades.csv", "text/csv")
        if st.button("Clear Journal"):
            if st.warning("Delete all trades?"): st.session_state['trade_journal'] = []; st.rerun()
    else: st.info("No trades recorded.")

st.markdown("---")
st.caption("AlphaQuant Terminal Pro - Actionable & Trader-Centric")