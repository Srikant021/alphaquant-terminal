import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import scipy.stats as si
from arch import arch_model
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml, logging, os, time, requests

# Optional Telegram
try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except:
    TELEGRAM_AVAILABLE = False

# Optional ML
try:
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

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
if 'show_order_flow' not in st.session_state: st.session_state['show_order_flow'] = False

for key, default in [
    ('trade_journal', []), ('paper_balance', 100000),
    ('paper_positions', []), ('paper_trade_history', []),
    ('auto_exit_enabled', True), ('ml_model_trained', False),
    ('telegram_token', ''), ('telegram_chat_id', '')
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
FALLBACK = {'btc':'N/A','btc_change':0,'btc_pct':0,'eth':'N/A','eth_change':0,'eth_pct':0,'market_vol':0,'timestamp':'Fallback'}

plt.style.use('dark_background')
logging.basicConfig(level=logging.INFO); logger=logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# HELPERS (all previously defined functions remain unchanged)
# -----------------------------------------------------------------------------
# (Include all existing helper functions: yf_download_retry, flatten_df,
#  calculate_hurst_exponent, get_asset_step, calculate_parkinson_volatility,
#  calculate_iv_rank_percentile, fetch_deribit_option_chain, fetch_binance_funding_rate,
#  fetch_long_hist, fetch_top_prices, fetch_indian_market_summary, fetch_intraday,
#  garch_both, live_price, get_india_vix, get_recent_month, get_hist_6mo, calc_greeks,
#  train_ml_model, predict_ml, send_telegram_alert, check_auto_exit,
#  get_intraday_signal, generate_trading_tips,
#  get_trade_bias, get_playbook, get_ivr_label,
#  plot_correlation, plot_expected_move, plot_hurst, plot_ivr_ivp, plot_liquidity_sweep,
#  plot_oi_profile, plot_parkinson, plot_volatility_cone, plot_vrp)
# They are identical to the previous version – copy them verbatim.
# For brevity, I've omitted their bodies here, but you must keep them in your file.

# -----------------------------------------------------------------------------
# NEW: LIVE ORDER FLOW (Binance)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=10, show_spinner=False)
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
# NEW: LIVE NIFTY/BANKNIFTY OPTIONS CHAIN
# -----------------------------------------------------------------------------
@st.cache_data(ttl=120, show_spinner=False)
def fetch_nse_options(index="^NSEI"):
    try:
        if index == "^NSEI":
            symbol = "NIFTY"
        elif index == "^NSEBANK":
            symbol = "BANKNIFTY"
        else:
            return None

        # Get current spot
        spot_data = yf.download(index, period="2d", progress=False)
        if spot_data.empty: return None
        spot = spot_data['Close'].squeeze().iloc[-1]

        # Get nearest expiry options (simplified: fetch all options for the current month)
        # Yahoo Finance NSE options ticker format: <SYMBOL><YY><MM><DD><C/P><STRIKE>
        # We'll fetch a range of strikes around ATM
        strike_step = 50 if symbol == "NIFTY" else 100
        strikes = np.arange(round(spot/strike_step)*strike_step - 10*strike_step,
                           round(spot/strike_step)*strike_step + 10*strike_step,
                           strike_step)
        options = []
        for strike in strikes:
            # Try current month expiry – for simplicity, we'll fetch the current week's expiry (Thursday)
            # This is approximate; a full implementation would query NSE API for exact expiry
            today = datetime.now()
            days_to_expiry = 3 - today.weekday()  # nearest Thursday
            if days_to_expiry < 0: days_to_expiry += 7
            expiry = today + timedelta(days=days_to_expiry)
            expiry_str = expiry.strftime("%y%m%d")
            for opt_type in ['CE', 'PE']:
                ticker = f"{symbol}{expiry_str}{opt_type}{strike}"
                try:
                    opt_data = yf.download(ticker+".NS", period="5d", progress=False)
                    if not opt_data.empty:
                        ltp = opt_data['Close'].squeeze().iloc[-1]
                        oi = opt_data.get('Open Interest', pd.Series([np.nan])).squeeze().iloc[-1] if 'Open Interest' in opt_data.columns else np.nan
                        options.append({'Strike': strike, 'Type': opt_type, 'LTP': ltp, 'OI': oi if not np.isnan(oi) else 0})
                except: pass
        return spot, pd.DataFrame(options)
    except: return None

# -----------------------------------------------------------------------------
# CACHED QUICK STATS (unchanged, already complete)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=120)
def compute_quick_stats(ticker, asset_choice, asset_spot, garch_vol_asset, park_vol, ivr_val, ivp_val, trading_days, currency, selected_market, corr_val, corr_status):
    # (identical to previous version)
    pass

def get_correlation_value():
    # (identical to previous version)
    pass

# -----------------------------------------------------------------------------
# NEW: CORRELATION MATRIX
# -----------------------------------------------------------------------------
def plot_correlation_matrix(ticker_dict, selected_market):
    if not ticker_dict: return None
    symbols = list(ticker_dict.values())
    names = list(ticker_dict.keys())
    data = yf.download(symbols, period="3mo", progress=False)['Close']
    if data.empty: return None
    data.columns = names
    log_ret = np.log(data/data.shift(1)).dropna()
    corr_matrix = log_ret.corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5,
                ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title(f'{selected_market} Correlation Matrix (3mo)', fontweight='bold')
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# TELEGRAM ALERT FUNCTION
# -----------------------------------------------------------------------------
def send_telegram_message(message):
    token = st.session_state.get('telegram_token', '')
    chat_id = st.session_state.get('telegram_chat_id', '')
    if TELEGRAM_AVAILABLE and token and chat_id:
        try:
            bot = Bot(token=token)
            bot.send_message(chat_id=chat_id, text=message)
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

# -----------------------------------------------------------------------------
# COMPACT TOOLBAR (unchanged)
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
# SIDEBAR (Telegram settings, navigation)
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

    with st.expander("📡 Telegram Alerts"):
        token = st.text_input("Bot Token", value=st.session_state['telegram_token'], type="password")
        chat_id = st.text_input("Chat ID", value=st.session_state['telegram_chat_id'])
        if st.button("Save Telegram Settings"):
            st.session_state['telegram_token'] = token
            st.session_state['telegram_chat_id'] = chat_id
            st.success("Settings saved! Alerts will be sent to your Telegram.")

    st.markdown("---")
    tab = st.radio("📑 Navigate", [
        "📊 Dashboard & Analytics",
        "📄 Paper Trading",
        "🧙 Strategy Wizard",
        "📓 Journal"
    ])
    st.session_state.active_tab = tab

# -----------------------------------------------------------------------------
# INITIAL DATA LOADING (unchanged)
# -----------------------------------------------------------------------------
# (same as previous version, omitted for brevity – copy the full loading block from earlier)

# After loading, all metrics are computed: asset_spot, garch_vol_asset, park_vol, ivr_val, ivp_val, corr_val, quick_stats, max_pain, etc.

# -----------------------------------------------------------------------------
# RENDER TABS
# -----------------------------------------------------------------------------
active_tab = st.session_state.get('active_tab', '📊 Dashboard & Analytics')

if active_tab == "📊 Dashboard & Analytics":
    st.title("📊 Market Intelligence Dashboard")

    # Market Overview card (unchanged)
    # Active Asset Detail card (unchanged)

    # NEW: Order Flow Toggle (Crypto only)
    if selected_market == "Crypto":
        with st.expander("📈 Live Binance Order Flow (on‑demand)"):
            show_of = st.checkbox("Show Order Book", value=st.session_state['show_order_flow'])
            if show_of:
                st.session_state['show_order_flow'] = True
                bin_symbol = get_binance_symbol(ticker)
                bids, asks = fetch_binance_orderbook(bin_symbol)
                if bids is not None and asks is not None:
                    best_bid = bids['Price'].iloc[0]; best_ask = asks['Price'].iloc[0]
                    mid = (best_bid+best_ask)/2
                    spread = best_ask-best_bid; spread_pct = (spread/mid)*100
                    imbalance = (bids['Size'].sum()-asks['Size'].sum())/(bids['Size'].sum()+asks['Size'].sum())
                    col_of1, col_of2, col_of3, col_of4 = st.columns(4)
                    col_of1.metric("Best Bid", f"{currency}{best_bid:,.2f}")
                    col_of2.metric("Best Ask", f"{currency}{best_ask:,.2f}")
                    col_of3.metric("Spread", f"{currency}{spread:,.2f}", f"{spread_pct:.4f}%")
                    col_of4.metric("Imbalance", f"{imbalance:+.3f}",
                                   "Bids heavy" if imbalance>0.1 else ("Asks heavy" if imbalance<-0.1 else "Neutral"))
                    fig_depth = go.Figure()
                    fig_depth.add_trace(go.Scatter(x=bids['Price'], y=bids['Size'].cumsum(),
                                                   mode='lines', name='Bids', line=dict(color='green', width=2),
                                                   fill='tozeroy', fillcolor='rgba(0,255,0,0.1)'))
                    fig_depth.add_trace(go.Scatter(x=asks['Price'], y=asks['Size'].cumsum(),
                                                   mode='lines', name='Asks', line=dict(color='red', width=2),
                                                   fill='tozeroy', fillcolor='rgba(255,0,0,0.1)'))
                    fig_depth.add_vline(x=mid, line_dash="dot", annotation_text="Mid")
                    fig_depth.update_layout(title="Order Book Depth", xaxis_title="Price", yaxis_title="Cumulative Size")
                    st.plotly_chart(fig_depth, use_container_width=True)
                else:
                    st.warning("Could not fetch Binance order book. Retry or use manual refresh.")

    # NEW: Correlation Matrix
    with st.expander("📊 Correlation Matrix (all assets)"):
        corr_fig = plot_correlation_matrix(TICKER_DICT, selected_market)
        if corr_fig: st.pyplot(corr_fig)

    # NEW: Live Options Chain (India only)
    if selected_market == "Indian Market" and ('Nifty' in asset_choice or 'Bank Nifty' in asset_choice):
        with st.expander("📋 Live Options Chain (NSE)"):
            spot_nse, opt_chain = fetch_nse_options(ticker)
            if opt_chain is not None and not opt_chain.empty:
                # Calculate max pain
                strikes = opt_chain['Strike'].unique()
                calls = opt_chain[opt_chain['Type']=='CE'].set_index('Strike')['OI'].reindex(strikes, fill_value=0)
                puts = opt_chain[opt_chain['Type']=='PE'].set_index('Strike')['OI'].reindex(strikes, fill_value=0)
                pain = {k: np.sum(np.maximum(0, k-strikes)*calls + np.maximum(0, strikes-k)*puts) for k in strikes}
                max_pain = min(pain, key=pain.get)
                st.metric("Spot", f"{currency}{spot_nse:,.0f}")
                st.metric("Max Pain", f"{currency}{max_pain:,.0f}")
                # OI chart
                fig_oi, ax_oi = plt.subplots(figsize=(14,8))
                ax_oi.barh(strikes, calls/1e5, color='red', alpha=0.8, label='Call OI')
                ax_oi.barh(strikes, -puts/1e5, color='green', alpha=0.8, label='Put OI')
                ax_oi.axhline(spot_nse, color='cyan', linewidth=2, label=f'Spot: {spot_nse:,.0f}')
                ax_oi.axhline(max_pain, color='white', linestyle='--', label=f'Max Pain: {max_pain}')
                ax_oi.set_title("Live NSE Options OI Profile", fontweight='bold')
                ax_oi.legend(); ax_oi.invert_yaxis()
                st.pyplot(fig_oi)
            else:
                st.warning("Could not fetch live options chain. Try again later.")

    # Quick Analytics, Market Summary, Detailed Chart – remain unchanged
    # ...

elif active_tab == "📄 Paper Trading":
    # Unchanged

elif active_tab == "🧙 Strategy Wizard":
    # Unchanged, but add Telegram alert after execution
    # if button clicked:
    #    send_telegram_message(f"Wizard opened {direction} {qty:.4f} {asset_choice} @ {currency}{asset_spot:,.2f}")

elif active_tab == "📓 Journal":
    st.title("📓 Trading Journal & Analytics")
    # Snapshot logging (unchanged)
    # Trade entry (unchanged)

    # NEW: Journal Analytics
    if st.session_state['trade_journal']:
        jdf = pd.DataFrame(st.session_state['trade_journal'])
        if not jdf.empty:
            st.subheader("📈 Performance Analytics")
            # Cumulative P&L
            jdf['Date'] = pd.to_datetime(jdf['Date'])
            jdf = jdf.sort_values('Date')
            jdf['Cumulative P&L'] = jdf['P&L'].cumsum()
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(jdf['Date'], jdf['Cumulative P&L'], marker='o', color='cyan')
            ax.set_title("Cumulative P&L", fontweight='bold')
            ax.grid(True, color='#2A2A2A')
            st.pyplot(fig)

            # Win rate by regime
            if 'Regime' in jdf.columns and not jdf['Regime'].isnull().all():
                regime_stats = jdf.groupby('Regime').agg(
                    Win_Rate = ('P&L', lambda x: (x>0).mean()*100),
                    Total_PnL = ('P&L', 'sum'),
                    Count = ('P&L', 'count')
                ).round(2)
                st.subheader("📊 Strategy Performance by Regime")
                st.dataframe(regime_stats.style.format({'Win_Rate':'{:.1f}%', 'Total_PnL':f'{currency}{{:,.2f}}'}))

            # Kelly Calculator
            total_trades = len(jdf)
            if total_trades > 0:
                wins = jdf[jdf['P&L'] > 0]
                losses = jdf[jdf['P&L'] < 0]
                win_rate = len(wins)/total_trades if total_trades else 0
                avg_win = wins['P&L'].mean() if not wins.empty else 0
                avg_loss = abs(losses['P&L'].mean()) if not losses.empty else 1
                if avg_loss > 0:
                    b = avg_win / avg_loss
                    kelly = win_rate - (1-win_rate)/b
                    kelly = max(0, min(kelly, 0.25))  # cap at 25%
                else:
                    kelly = 0
                st.metric("Optimal Kelly Fraction", f"{kelly:.2%}")
                st.caption(f"Based on {total_trades} trades – suggests risking {kelly*100:.1f}% of capital per trade.")
    else:
        st.info("No trades recorded yet.")

st.markdown("---")
st.caption("AlphaQuant Terminal Pro · Advanced Trading Cockpit")