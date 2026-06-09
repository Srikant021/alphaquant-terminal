# crypto.py
# AlphaQuant Terminal — Pure Technical Suite & Derivatives Engine

import logging
import time
from datetime import datetime, timezone, timedelta
import os
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from numba import jit
from fyers_apiv3 import fyersModel

# ─────────────────────────────────────────────
# 0. SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────
if 'fyers_authenticated' not in st.session_state:
    st.session_state.fyers_authenticated = False
if 'access_token' not in st.session_state:
    st.session_state.access_token = ""
if 'selected_tf' not in st.session_state:
    st.session_state.selected_tf = "1D"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("alphaquant")

st.set_page_config(page_title="AlphaQuant Pro", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background: #080d12; }
    .metric-box { background: linear-gradient(135deg, #0d1520 0%, #111c2b 100%); padding: 16px 18px; border-radius: 8px; border-left: 3px solid #0af; margin: 5px 0; border-top: 1px solid rgba(0,170,255,0.08); }
    .section-header { font-family: 'Space Mono', monospace; font-size: 13px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #0af; margin: 24px 0 12px 0; padding-bottom: 6px; border-bottom: 1px solid rgba(0,170,255,0.2); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECRETS AUTO-WRITER
# ─────────────────────────────────────────────
def save_token_to_secrets(client_id, secret_key, access_token):
    os.makedirs(".streamlit", exist_ok=True)
    with open(".streamlit/secrets.toml", "w") as f:
        f.write("[fyers]\n")
        f.write(f'client_id = "{client_id}"\n')
        f.write(f'secret_key = "{secret_key}"\n')
        f.write(f'access_token = "{access_token}"\n')

# ─────────────────────────────────────────────
# CORE DATA FETCHING (YFinance)
# ─────────────────────────────────────────────
def _flatten_multiindex(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

@st.cache_data(ttl=300)
def fetch_data(ticker, period="1y", interval="1d"):
    try:
        raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if raw is None or raw.empty: return None
        return _flatten_multiindex(raw).dropna()
    except Exception:
        return None

@st.cache_data(ttl=60)
def get_live_price(ticker):
    try:
        data = fetch_data(ticker, period="5d", interval="1d")
        if data is None or len(data) < 2: return None
        last = float(data['Close'].iloc[-1])
        prev = float(data['Close'].iloc[-2])
        return {
            'price': last, 
            'change': last - prev, 
            'pct': ((last - prev) / prev) * 100 if prev else 0.0,
            'high': float(data['High'].iloc[-1]), 
            'low': float(data['Low'].iloc[-1]), 
            'volume': float(data['Volume'].iloc[-1])
        }
    except Exception:
        return None

# ─────────────────────────────────────────────
# FYERS API ENGINE (INDESTRUCTIBLE VERSION)
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_real_fyers_oi(client_id, access_token, index_name="NIFTY", spot_price=None):
    try:
        fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")
        url = "https://public.fyers.in/sym_details/NSE_FO.csv"
        
        # Load without names to prevent Pandas from shifting columns if Fyers changes the file size
        df = pd.read_csv(url, header=None, low_memory=False)
        
        # Hard-map the exact column indexes we need (Col 1: Symbol, 8: Expiry, 12: Strike, 13: OptType)
        df = df.rename(columns={1: 'Symbol', 8: 'Expiry', 12: 'Strike', 13: 'OptType'})
        df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
        
        # Filter strictly for CE/PE and the requested index
        df_opt = df[df['Symbol'].str.contains(index_name, na=False) & df['Symbol'].str.endswith(('CE', 'PE'))].copy()
        if index_name == "NIFTY":
            df_opt = df_opt[~df_opt['Symbol'].str.contains("BANKNIFTY|FINNIFTY|MIDCPNIFTY", na=False)]

        if df_opt.empty: 
            return None, f"No symbols matched {index_name} Options. Check CSV format."

        # Strict numerical casting
        df_opt['Expiry'] = pd.to_numeric(df_opt['Expiry'], errors='coerce')
        df_opt['Strike'] = pd.to_numeric(df_opt['Strike'], errors='coerce')
        df_opt = df_opt.dropna(subset=['Expiry', 'Strike'])

        # Buffer to prevent local-clock timezone mismatches (last 5 days)
        current_time = int(time.time()) - (86400 * 5)
        future_expiries = sorted([e for e in df_opt['Expiry'].unique() if e > current_time])
        
        if not future_expiries: 
            return None, "No future expiries found after filtering."

        # Hunt through up to the next 5 expiries for active Open Interest
        for current_expiry in future_expiries[:5]: 
            df_exp = df_opt[df_opt['Expiry'] == current_expiry].copy()
            
            if spot_price and spot_price > 0:
                df_exp = df_exp[(df_exp['Strike'] >= spot_price * 0.85) & (df_exp['Strike'] <= spot_price * 1.15)]

            symbols = df_exp['Symbol'].tolist()
            if not symbols: continue

            strikes_data = {}
            active_oi_found = False
            
            for i in range(0, len(symbols), 50):
                batch = symbols[i:i+50]
                response = fyers.quotes({"symbols": ",".join(batch)})
                
                if response and response.get("s") == "ok":
                    for item in response.get("d", []):
                        if item.get("s") == "ok":
                            sym = item.get("n", "")
                            oi = item.get("v", {}).get("open_interest", 0) 
                            if oi > 0: active_oi_found = True
                            
                            match = df_exp[df_exp['Symbol'] == sym]
                            if not match.empty:
                                strike = float(match.iloc[0]['Strike'])
                                opt_type = 'C' if 'CE' in match.iloc[0]['OptType'] else 'P'
                                if strike not in strikes_data: strikes_data[strike] = {'C': 0, 'P': 0}
                                strikes_data[strike][opt_type] += oi
            
            # If the chain is liquid, return it immediately
            if active_oi_found:
                try:
                    exp_date = datetime.fromtimestamp(int(current_expiry), tz=timezone.utc).strftime('%d %b %Y')
                except Exception:
                    exp_date = str(current_expiry)
                return pd.DataFrame.from_dict(strikes_data, orient='index').fillna(0).sort_index(), f"Live Expiry: {exp_date}"

        return None, "All checked expiries returned 0 Open Interest."
    except Exception as e:
        return None, f"API Error: {str(e)}"

# ─────────────────────────────────────────────
# MATPLOTLIB FUNCTIONS (VOLATILITY & STRUCTURE)
# ─────────────────────────────────────────────
def plot_fyers_oi_profile(oi_df, spot_price, current_expiry_str):
    calls = oi_df['C']
    puts = oi_df['P']
    strikes = oi_df.index
    
    pain_values = {}
    for test_strike in strikes:
        call_loss = np.maximum(0, test_strike - strikes) * calls.values
        put_loss = np.maximum(0, strikes - test_strike) * puts.values
        pain_values[test_strike] = np.sum(call_loss) + np.sum(put_loss)
    max_pain_strike = min(pain_values, key=pain_values.get)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8), dpi=120)

    ax.barh(strikes, calls.values / 100000, height=25, color='#FF3333', alpha=0.8, label='Call OI (Resistance)')
    ax.barh(strikes, -puts.values / 100000, height=25, color='#00FF00', alpha=0.8, label='Put OI (Support)')

    ax.axhline(spot_price, color='#00FFFF', linestyle='-', linewidth=2, label=f'Spot: {spot_price:.2f}')
    ax.axhline(max_pain_strike, color='white', linestyle='--', linewidth=2.5, label=f'Max Pain: {max_pain_strike}')

    ax.set_title(f'NIFTY 50 Institutional OI ({current_expiry_str})', fontsize=18, color='white', pad=20, fontweight='bold')
    ax.set_xlabel('Open Interest (in Lakhs)', color='gray', fontsize=12)
    ax.set_ylabel('Strike Price', color='gray', fontsize=12)

    ticks = ax.get_xticks()
    ax.set_xticklabels([str(abs(int(tick))) for tick in ticks])
    ax.set_ylim(spot_price - 600, spot_price + 600)

    ax.grid(True, color='#2A2A2A', linestyle=':')
    ax.legend(loc='upper right', facecolor='black', edgecolor='gray', fontsize=11)

    props = {'boxstyle': 'round,pad=0.6', 'facecolor': 'black', 'alpha': 0.9, 'edgecolor': 'gray', 'linewidth': 1.5}
    
    text_str = (
        f"Live Spot: {spot_price:.2f}\n"
        f"Max Pain Strike: {max_pain_strike}\n"
        f"---------------------------\n"
        f"Highest Put Wall: {puts.idxmax()}\n"
        f"Highest Call Wall: {calls.idxmax()}"
    )

    ax.text(0.02, 0.05, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_index_divergence():
    tickers = {"Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK"}
    data = yf.download(list(tickers.values()), period="1y", progress=False)
    if isinstance(data.columns, pd.MultiIndex): data = data['Close']
    data.columns = ['Bank Nifty', 'Nifty 50']
    data = data.dropna()
    
    normalized_prices = (data / data.iloc[0]) * 100
    log_returns = np.log(data / data.shift(1)).dropna()
    rolling_correlation = log_returns['Nifty 50'].rolling(window=20).corr(log_returns['Bank Nifty'])
    
    current_nifty = float(data['Nifty 50'].iloc[-1])
    current_bank = float(data['Bank Nifty'].iloc[-1])
    current_corr = float(rolling_correlation.iloc[-1])
    
    if current_corr > 0.80:
        regime, stat_property, color_theme = "HIGH CORRELATION", "Synchronized Movement", '#00FF00'
    elif current_corr < 0.50:
        regime, stat_property, color_theme = "SEVERE DIVERGENCE", "Sector Rotation", '#FF3333'
    else:
        regime, stat_property, color_theme = "MODERATE DIVERGENCE", "Decoupling Phase", '#FFA500'

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=120, gridspec_kw={'height_ratios': [1.5, 1]})
    
    ax1.plot(normalized_prices.index, normalized_prices['Nifty 50'], color='#00FFFF', linewidth=2, label='Nifty 50')
    ax1.plot(normalized_prices.index, normalized_prices['Bank Nifty'], color='#FFA500', linewidth=2, label='Bank Nifty')
    ax1.fill_between(normalized_prices.index, normalized_prices['Nifty 50'], normalized_prices['Bank Nifty'], color='gray', alpha=0.2)
    ax1.set_title('Inter-Index Correlation & Divergence', fontsize=18, color='white', pad=15, fontweight='bold')
    ax1.grid(True, color='#2A2A2A', linestyle=':')
    ax1.legend(loc='upper left', facecolor='black', edgecolor='gray')
    
    ax2.plot(rolling_correlation.index, rolling_correlation, color='white', linewidth=1.5)
    ax2.axhline(0.80, color='#00FF00', linestyle='--')
    ax2.axhline(0.50, color='#FF3333', linestyle='--')
    ax2.fill_between(rolling_correlation.index, 0.50, rolling_correlation, where=(rolling_correlation < 0.50), color='#FF3333', alpha=0.3)
    ax2.set_ylim(-0.2, 1.1)
    ax2.grid(True, color='#2A2A2A', linestyle=':')
    
    props = {'boxstyle': 'round,pad=0.5', 'facecolor': 'black', 'alpha': 0.9, 'edgecolor': color_theme, 'linewidth': 1.5}
    
    text_str = (
        f"Nifty 50: {current_nifty:.2f}\n"
        f"Bank Nifty: {current_bank:.2f}\n"
        f"20-Day Corr: {current_corr:.2f}\n"
        f"---------------------------\n"
        f"{regime}\n"
        f"{stat_property}"
    )
    
    ax1.text(0.02, 0.05, text_str, transform=ax1.transAxes, fontsize=12, verticalalignment='bottom', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_nifty_volatility():
    data = yf.download("^INDIAVIX", period="1y", progress=False)
    close_prices = data['Close'].squeeze()
    
    current_iv = float(close_prices.iloc[-1])
    high_52w = float(close_prices.max())
    low_52w = float(close_prices.min())
    
    ivr = ((current_iv - low_52w) / (high_52w - low_52w)) * 100
    ivp = ((close_prices < current_iv).sum() / len(close_prices)) * 100
    regime, color_theme = ("HIGH VOLATILITY: Net Short Premium", '#00FF00') if ivr > 50 else ("LOW VOLATILITY: Net Long Premium", '#FF3333')

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
    
    ax.plot(close_prices.index, close_prices.values, color='#00FFFF', linewidth=1.5)
    ax.axhline(high_52w, color='red', linestyle='--', alpha=0.5, label=f'52W High: {high_52w:.2f}')
    ax.axhline(low_52w, color='green', linestyle='--', alpha=0.5, label=f'52W Low: {low_52w:.2f}')
    ax.axhline(current_iv, color='white', linestyle='-', linewidth=2, label=f'Current: {current_iv:.2f}')
    ax.fill_between(close_prices.index, low_52w, current_iv, color='white', alpha=0.05)
    ax.set_title('NIFTY Implied Volatility (IVR & IVP)', fontsize=18, color='white', pad=20, fontweight='bold')
    ax.grid(True, color='#2A2A2A', linestyle=':')
    ax.legend(loc='upper right', facecolor='black', edgecolor='gray')
    
    props = {'boxstyle': 'round,pad=0.5', 'facecolor': 'black', 'alpha': 0.8, 'edgecolor': color_theme, 'linewidth': 1.5}
    
    text_str = (
        f"IV Rank (IVR): {ivr:.1f}%\n"
        f"IV Percentile (IVP): {ivp:.1f}%\n"
        f"Current VIX: {current_iv:.2f}\n"
        f"---------------------------\n"
        f"Edge: {regime}"
    )
    
    ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_crypto_volatility(ticker_name, ticker_symbol):
    data = yf.download(ticker_symbol, period="1y", progress=False)
    if data is None or data.empty: return None
    
    close_prices = data['Close'].squeeze()
    returns = np.log(close_prices / close_prices.shift(1)).dropna()
    
    current_price = float(close_prices.iloc[-1])
    current_vol = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
    high_52w = float(close_prices.max())
    low_52w = float(close_prices.min())
    
    vol_max = returns.rolling(252).std().max() * np.sqrt(252) * 100
    vol_min = returns.rolling(252).std().min() * np.sqrt(252) * 100
    
    vol_rank = ((current_vol - vol_min) / (vol_max - vol_min) * 100) if vol_max > vol_min else 50
    vol_rank = np.clip(vol_rank, 0, 100)
    
    regime, color_theme = ("HIGH VOLATILITY", '#00FF00') if vol_rank > 60 else ("LOW VOLATILITY", '#FF3333')

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
    
    ax.plot(close_prices.index, close_prices.values, color='#00FFFF', linewidth=1.5)
    ax.axhline(high_52w, color='red', linestyle='--', alpha=0.5, label=f'52W High: ${high_52w:.2f}')
    ax.axhline(low_52w, color='green', linestyle='--', alpha=0.5, label=f'52W Low: ${low_52w:.2f}')
    ax.axhline(current_price, color='white', linestyle='-', linewidth=2, label=f'Current: ${current_price:.2f}')
    ax.fill_between(close_prices.index, low_52w, current_price, color='white', alpha=0.05)
    ax.set_title(f'{ticker_name} Volatility Rank & Price Range', fontsize=18, color='white', pad=20, fontweight='bold')
    ax.grid(True, color='#2A2A2A', linestyle=':')
    ax.legend(loc='upper right', facecolor='black', edgecolor='gray')
    
    props = {'boxstyle': 'round,pad=0.5', 'facecolor': 'black', 'alpha': 0.8, 'edgecolor': color_theme, 'linewidth': 1.5}
    
    text_str = (
        f"Volatility Rank: {vol_rank:.1f}%\n"
        f"20-Day Vol: {current_vol:.2f}%\n"
        f"---------------------------\n"
        f"{regime}"
    )
    
    ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_expected_move():
    nifty_data = yf.download("^NSEI", period="1mo", progress=False)
    vix_data = yf.download("^INDIAVIX", period="5d", progress=False)
    
    nifty_close = nifty_data['Close'].squeeze()
    spot_price = float(nifty_close.iloc[-1])
    current_vix = float(vix_data['Close'].squeeze().iloc[-1])
    
    expected_move_points = spot_price * ((current_vix / 100) * np.sqrt(1/365))
    upper_bound = spot_price + expected_move_points
    lower_bound = spot_price - expected_move_points

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
    
    recent_nifty = nifty_close.tail(15)
    x_dates = np.arange(len(recent_nifty))
    
    ax.plot(x_dates, recent_nifty.values, color='#00FFFF', linewidth=2, marker='o')
    tomorrow_x = len(recent_nifty)
    
    ax.hlines(spot_price, xmin=x_dates[-1], xmax=tomorrow_x, color='white', linestyle='-', linewidth=2)
    ax.scatter(tomorrow_x, spot_price, color='white', s=70, zorder=5)
    ax.scatter(tomorrow_x, upper_bound, color='#00FF00', s=120, marker='^', zorder=5)
    ax.scatter(tomorrow_x, lower_bound, color='#FF3333', s=120, marker='v', zorder=5)
    ax.hlines(upper_bound, xmin=x_dates[-1], xmax=tomorrow_x, color='#00FF00', linestyle='--')
    ax.hlines(lower_bound, xmin=x_dates[-1], xmax=tomorrow_x, color='#FF3333', linestyle='--')
    ax.fill_between([x_dates[-1], tomorrow_x], [spot_price, lower_bound], [spot_price, upper_bound], color='gray', alpha=0.2)
    ax.set_title('NIFTY 50 Implied Daily Expected Move', fontsize=18, color='white', pad=20, fontweight='bold')
    ax.grid(True, color='#2A2A2A', linestyle=':')
    ax.set_xticks([])
    
    props = {'boxstyle': 'round,pad=0.5', 'facecolor': 'black', 'alpha': 0.8, 'edgecolor': 'white', 'linewidth': 1.5}
    
    text_str = (
        f"Current VIX: {current_vix:.2f}\n"
        f"Spot Price: {spot_price:.2f}\n"
        f"Expected Move: ± {expected_move_points:.1f} pts\n"
        f"---------------------------\n"
        f"Safe Short Call: > {upper_bound:.0f}\n"
        f"Safe Short Put: < {lower_bound:.0f}"
    )
    
    ax.text(0.02, 0.45, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='center', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_liquidity_sweep():
    data = yf.download("^NSEI", period="5d", interval="15m", progress=False)
    if isinstance(data.columns, pd.MultiIndex): data.columns = [col[0] for col in data.columns]
    
    data['Prev_High'] = data['High'].rolling(20).max().shift(1)
    data['Prev_Low'] = data['Low'].rolling(20).min().shift(1)
    data['Supply_Sweep'] = (data['High'] > data['Prev_High']) & (data['Close'] < data['Prev_High'])
    data['Demand_Sweep'] = (data['Low'] < data['Prev_Low']) & (data['Close'] > data['Prev_Low'])
    current_price = float(data['Close'].iloc[-1])
    
    if data['Supply_Sweep'].iloc[-1]: 
        regime, color_theme, stat_property = "SUPPLY LIQUIDITY SWEPT", '#FF3333', "Failed Breakout"
    elif data['Demand_Sweep'].iloc[-1]: 
        regime, color_theme, stat_property = "DEMAND LIQUIDITY SWEPT", '#00FF00', "Failed Breakdown"
    else: 
        regime, color_theme, stat_property = "PRICE DISCOVERY", '#00FFFF', "Inside Bounds"

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
    
    plot_data = data.tail(60).copy()
    plot_data['Index'] = np.arange(len(plot_data))
    
    up = plot_data[plot_data['Close'] >= plot_data['Open']]
    down = plot_data[plot_data['Close'] < plot_data['Open']]
    
    ax.vlines(up['Index'], up['Low'], up['High'], color='#00FF00', linewidth=1.5)
    ax.vlines(down['Index'], down['Low'], down['High'], color='#FF3333', linewidth=1.5)
    ax.bar(up['Index'], up['Close'] - up['Open'], bottom=up['Open'], color='#00FF00', width=0.6)
    ax.bar(down['Index'], down['Open'] - down['Close'], bottom=down['Close'], color='#FF3333', width=0.6)
    
    for idx, row in plot_data.iterrows():
        if row['Supply_Sweep']:
            ax.scatter(row['Index'], row['High'] + 10, marker='v', color='#FF3333', s=150, zorder=5)
            ax.axhline(row['Prev_High'], color='#FF3333', linestyle='--', alpha=0.5)
        if row['Demand_Sweep']:
            ax.scatter(row['Index'], row['Low'] - 10, marker='^', color='#00FF00', s=150, zorder=5)
            ax.axhline(row['Prev_Low'], color='#00FF00', linestyle='--', alpha=0.5)
            
    ax.set_title('NIFTY 50 Intraday Liquidity Sweep (15m)', fontsize=16, color='white', pad=15, fontweight='bold')
    ax.grid(True, color='#2A2A2A', linestyle=':')
    ax.set_xticks([])
    
    props = {'boxstyle': 'round,pad=0.5', 'facecolor': 'black', 'alpha': 0.9, 'edgecolor': color_theme, 'linewidth': 1.5}
    
    text_str = (
        f"Live Spot Price: {current_price:.2f}\n"
        f"---------------------------\n"
        f"{regime}\n"
        f"{stat_property}"
    )
    
    ax.text(0.02, 0.05, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_hurst_regime():
    data = yf.download("^NSEI", period="1y", progress=False)
    nifty_close = data['Close'].squeeze()
    
    def calculate_hurst(ts):
        if len(ts) < 20: return np.nan
        ts_arr = ts.values
        reg = [np.std(ts_arr[lag:] - ts_arr[:-lag]) for lag in range(2, 20)]
        return np.polyfit(np.log(range(2, 20)), np.log(reg), 1)[0]
        
    df = pd.DataFrame({'Close': nifty_close, 'Hurst': np.log(nifty_close).rolling(window=60).apply(calculate_hurst, raw=False)}).dropna()
    current_price = float(df['Close'].iloc[-1])
    current_hurst = float(df['Hurst'].iloc[-1])
    
    if current_hurst < 0.45: 
        regime, color_theme, stat_property = "MEAN REVERTING", '#FF3333', "Range-Bound Action"
    elif current_hurst > 0.55: 
        regime, color_theme, stat_property = "TRENDING", '#00FF00', "Directional Movement"
    else: 
        regime, color_theme, stat_property = "RANDOM WALK", '#FFA500', "Unpredictable Noise"

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=120, gridspec_kw={'height_ratios': [1.5, 1]})
    
    ax1.plot(df.index, df['Close'], color='white', linewidth=1.5)
    ax1.set_title('NIFTY 50 Market Regime (Hurst Exponent)', fontsize=18, color='white', pad=15, fontweight='bold')
    ax1.grid(True, color='#2A2A2A', linestyle=':')
    ax1.axvspan(df.index[-15], df.index[-1], color=color_theme, alpha=0.1)
    
    ax2.plot(df.index, df['Hurst'], color='#00FFFF', linewidth=2)
    ax2.axhline(0.55, color='#00FF00', linestyle='--')
    ax2.axhline(0.45, color='#FF3333', linestyle='--')
    ax2.fill_between(df.index, 0.55, df['Hurst'], where=(df['Hurst'] > 0.55), color='#00FF00', alpha=0.2)
    ax2.fill_between(df.index, 0.45, df['Hurst'], where=(df['Hurst'] < 0.45), color='#FF3333', alpha=0.2)
    ax2.set_ylim(0.3, 0.7)
    ax2.grid(True, color='#2A2A2A', linestyle=':')
    
    props = {'boxstyle': 'round,pad=0.5', 'facecolor': 'black', 'alpha': 0.9, 'edgecolor': color_theme, 'linewidth': 1.5}
    
    text_str = (
        f"Current Nifty: {current_price:.2f}\n"
        f"Hurst Exponent: {current_hurst:.3f}\n"
        f"Regime: {regime}\n"
        f"---------------------------\n"
        f"{stat_property}"
    )
    
    ax1.text(0.02, 0.05, text_str, transform=ax1.transAxes, fontsize=12, verticalalignment='bottom', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_volatility_cone():
    data = yf.download("^NSEI", period="10y", progress=False)
    data['Returns'] = np.log(data['Close'].squeeze() / data['Close'].squeeze().shift(1))
    windows = [10, 20, 30, 60, 90, 120, 180, 252]
    
    max_vol, min_vol, median_vol, current_vol = [], [], [], []
    for window in windows:
        rolling_vol = data['Returns'].rolling(window=window).std() * np.sqrt(252)
        max_vol.append(rolling_vol.max() * 100)
        min_vol.append(rolling_vol.min() * 100)
        median_vol.append(rolling_vol.median() * 100)
        current_vol.append(rolling_vol.dropna().iloc[-1] * 100)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 7))
    
    plt.plot(windows, max_vol, marker='o', color='red', linewidth=2, label='Maximum Volatility')
    plt.plot(windows, min_vol, marker='o', color='limegreen', linewidth=2, label='Minimum Volatility')
    plt.plot(windows, median_vol, marker='s', color='white', linewidth=1.5, linestyle='--', label='Median Volatility')
    plt.plot(windows, current_vol, marker='X', color='yellow', linewidth=3, markersize=10, label='Current Volatility')
    plt.fill_between(windows, min_vol, max_vol, color='gray', alpha=0.2)
    
    plt.title('Volatility Cone for Nifty 50', fontsize=18, fontweight='bold', color='white')
    plt.xlabel('Time Window (Trading Days)', fontsize=14)
    plt.ylabel('Annualized Volatility (%)', fontsize=14)
    plt.xticks(windows)
    plt.grid(color='gray', linestyle=':', alpha=0.5)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    return fig

def plot_vrp():
    nifty_data = yf.download("^NSEI", period="6mo", progress=False)
    vix_data = yf.download("^INDIAVIX", period="6mo", progress=False)
    
    df = pd.DataFrame({
        'VIX': vix_data['Close'].squeeze(), 
        'HV': (np.log(nifty_data['Close'].squeeze() / nifty_data['Close'].squeeze().shift(1)).rolling(20).std() * np.sqrt(252) * 100)
    }).dropna()
    
    df['VRP'] = df['VIX'] - df['HV']
    current_vrp = float(df['VRP'].iloc[-1])
    
    if current_vrp > 0:
        regime, color_theme, stat_property = "POSITIVE VRP", '#00FF00', "Implied > Realized"
    else:
        regime, color_theme, stat_property = "NEGATIVE VRP", '#FF3333', "Realized > Implied"

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=120, gridspec_kw={'height_ratios': [1.5, 1]})
    
    ax1.plot(df.index, df['VIX'], color='red', linewidth=2, label='India VIX (Expected)')
    ax1.plot(df.index, df['HV'], color='dodgerblue', linewidth=2, label='20-Day HV (Actual)')
    ax1.set_title('Volatility Risk Premium (VRP) & Variance Spread', fontsize=18, color='white', pad=15, fontweight='bold')
    ax1.grid(True, color='#2A2A2A', linestyle=':')
    ax1.legend(loc='upper left', facecolor='black', edgecolor='gray')
    
    ax2.bar(df.index, df['VRP'], color=np.where(df['VRP'] > 0, '#00FF00', '#FF3333'), alpha=0.7, width=1)
    ax2.axhline(0, color='white', linewidth=1)
    ax2.grid(True, color='#2A2A2A', linestyle=':')
    
    props = {'boxstyle': 'round,pad=0.5', 'facecolor': 'black', 'alpha': 0.9, 'edgecolor': color_theme, 'linewidth': 1.5}
    
    text_str = (
        f"VIX (Expected): {df['VIX'].iloc[-1]:.2f}%\n"
        f"20-Day HV (Actual): {df['HV'].iloc[-1]:.2f}%\n"
        f"VRP Spread: {current_vrp:+.2f}%\n"
        f"---------------------------\n"
        f"{regime}\n"
        f"{stat_property}"
    )
    
    ax1.text(0.02, 0.05, text_str, transform=ax1.transAxes, fontsize=12, verticalalignment='bottom', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig

def calc_parkinson_vol():
    data = yf.download('^NSEI', period='1y', progress=False)
    log_hl = np.log(data['High'].squeeze() / data['Low'].squeeze())
    parkinson_vol = np.sqrt((1 / (4 * len(data) * np.log(2))) * (log_hl ** 2).sum()) * np.sqrt(252)
    c2c_vol = np.log(data['Close'].squeeze() / data['Close'].squeeze().shift(1)).std() * np.sqrt(252)
    return len(data), float(parkinson_vol) * 100, float(c2c_vol) * 100

# ─────────────────────────────────────────────
# MULTI-TIMEFRAME CHARTING ENGINE
# ─────────────────────────────────────────────
def bollinger_bands(close, period=20, std=2):
    sma = close.rolling(period).mean()
    return sma + (std * close.rolling(period).std()), sma, sma - (std * close.rolling(period).std())

def create_clean_ta_chart(chart_data, ticker_name, timeframe="1D"):
    df = chart_data.copy()
    bb_up, _, bb_lo = bollinger_bands(df['Close'], 20, 2.0)
    
    vol_sum = df['Volume'].replace(0, np.nan).ffill().cumsum()
    vwap_line = ((df['High'] + df['Low'] + df['Close']) / 3) * vol_sum / vol_sum
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC', increasing_line_color='#00c878', decreasing_line_color='#ff4d6d'), row=1, col=1)
    fig.add_trace(go.Scattergl(x=df.index, y=bb_up, line=dict(color='rgba(0,170,255,0.25)', dash='dot'), name='BB Up', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scattergl(x=df.index, y=bb_lo, line=dict(color='rgba(0,170,255,0.25)', dash='dot'), fill='tonexty', fillcolor='rgba(0,170,255,0.03)', name='BB Low', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scattergl(x=df.index, y=vwap_line, line=dict(color='#00e5ff', dash='dashdot', width=1.5), name='VWAP', showlegend=False), row=1, col=1)
    
    vol_colors = ['rgba(0,200,120,0.4)' if c >= o else 'rgba(255,77,109,0.4)' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_colors, name='Volume', showlegend=False), row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018', 
        title=dict(text=f"<b>{ticker_name} — {timeframe} Technical Canvas</b>", font=dict(family='Space Mono', color='#0af')), 
        height=650, xaxis_rangeslider_visible=False, hovermode='x unified', 
        margin=dict(l=60, r=20, t=50, b=40)
    )
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.03)', row=1, col=1)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.03)', row=2, col=1)
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.03)')
    
    return fig

# ─────────────────────────────────────────────
# CONTROL SIDEBAR & AUTHENTICATION ENGINE
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 AlphaQuant Pro")
    market = st.radio("Market Selection", ["Indian Market", "Crypto"], horizontal=True)

    if market == "Crypto":
        assets = {'Bitcoin': 'BTC-USD', 'Ethereum': 'ETH-USD'}
        currency = "$"
    else:
        assets = {'Nifty 50': '^NSEI', 'Bank Nifty': '^NSEBANK'}
        currency = "₹"

    selected_asset = st.selectbox("Active Asset", list(assets.keys()))
    ticker = assets[selected_asset]
    st.markdown("---")
    
    client_id = ""
    secret_key = ""

    if market == "Indian Market":
        st.markdown("### 🔐 Fyers Access Gateway")
        try:
            client_id = st.secrets["fyers"].get("client_id", "1429ZQANUF-100")
            secret_key = st.secrets["fyers"].get("secret_key", "KLD2AMQAQD")
            if not st.session_state.access_token:
                st.session_state.access_token = st.secrets["fyers"].get("access_token", "")
        except Exception:
            client_id = "1429ZQANUF-100"
            secret_key = "KLD2AMQAQD"

        client_id = st.text_input("App ID (client_id)", value=client_id, type="password")
        secret_key = st.text_input("Secret Key", value=secret_key, type="password")
        
        session = fyersModel.SessionModel(
            client_id=client_id, 
            secret_key=secret_key, 
            redirect_uri="https://127.0.0.1", 
            response_type="code", 
            grant_type="authorization_code"
        )
        
        if st.session_state.access_token:
            try:
                fyers_test = fyersModel.FyersModel(client_id=client_id, is_async=False, token=st.session_state.access_token, log_path="")
                if fyers_test.get_profile().get("s") == "ok":
                    st.session_state.fyers_authenticated = True
                    st.success("✅ Fyers API Active")
                else:
                    st.session_state.fyers_authenticated = False
                    st.error("❌ Token Expired or Invalid")
            except Exception:
                st.session_state.fyers_authenticated = False
                st.error("❌ Token Authentication Failure")

        if not st.session_state.fyers_authenticated:
            st.markdown(f'[🔗 Click Here to Authenticate via Fyers]({session.generate_authcode()})')
            raw_input_code = st.text_input("Paste Auth Code / Redirect URL:")
            if st.button("🔄 Complete Daily Activation", use_container_width=True):
                if raw_input_code:
                    parsed_code = raw_input_code.split("auth_code=")[1].split("&")[0] if "auth_code=" in raw_input_code else raw_input_code
                    try:
                        session.set_token(parsed_code)
                        token_response = session.generate_token()
                        if "access_token" in token_response:
                            st.session_state.access_token = token_response["access_token"]
                            save_token_to_secrets(client_id, secret_key, st.session_state.access_token)
                            st.session_state.fyers_authenticated = True
                            st.rerun()
                    except Exception as ex: 
                        st.error(f"Error: {ex}")
        st.markdown("---")

    if st.button("⚡ Purge Cache & Re-pull", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────────
# DASHBOARD ORCHESTRATOR
# ─────────────────────────────────────────────
hist_1y = fetch_data(ticker, period="1y", interval="1d")

if hist_1y is None or hist_1y.empty:
    st.error("❌ Upstream Engine Disconnected. Failed to resolve asset ticker history.")
    st.stop()

live = get_live_price(ticker)
spot = live['price'] if live else float(hist_1y['Close'].iloc[-1])
change_pct = live['pct'] if live else 0.0

main_tab1, main_tab2, main_tab3 = st.tabs(["📈 Terminal Hub", "🔥 Derivatives & Options", "🔬 Volatility & Structure"])

with main_tab1:
    st.metric(f"LAST TRADED PRICE", f"{currency}{spot:,.2f}", f"{change_pct:+.2f}%")
    
    # Multi-Timeframe Selection
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("1D", use_container_width=True, key="tf_1d"):
            st.session_state.selected_tf = "1D"
    with col2:
        if st.button("1W", use_container_width=True, key="tf_1w"):
            st.session_state.selected_tf = "1W"
    with col3:
        if st.button("1M", use_container_width=True, key="tf_1m"):
            st.session_state.selected_tf = "1M"
    with col4:
        if st.button("3M", use_container_width=True, key="tf_3m"):
            st.session_state.selected_tf = "3M"
    
    # Fetch data for selected timeframe
    tf_map = {"1D": ("1y", "1d"), "1W": ("5y", "1wk"), "1M": ("10y", "1mo"), "3M": ("10y", "3mo")}
    period, interval = tf_map[st.session_state.selected_tf]
    hist_tf = fetch_data(ticker, period=period, interval=interval)
    
    if hist_tf is not None and not hist_tf.empty:
        fig_comprehensive = create_clean_ta_chart(hist_tf, selected_asset, timeframe=st.session_state.selected_tf)
        if fig_comprehensive: 
            st.plotly_chart(fig_comprehensive, use_container_width=True)

with main_tab2:
    if market == "Indian Market" and st.session_state.fyers_authenticated:
        index_target = "NIFTY" if "NSEI" in ticker else "BANKNIFTY"
        with st.spinner("Hunting for Active Liquidity Chain..."):
            oi_data, status_msg = fetch_real_fyers_oi(client_id, st.session_state.access_token, index_target, spot)
        
        if oi_data is not None and not oi_data.empty:
            st.pyplot(plot_fyers_oi_profile(oi_data, spot, status_msg))
        else:
            st.warning(f"⚠️ API Diagnostics: Spot={spot:.2f}. Status: '{status_msg}'")
    else:
        st.info("Activate Fyers API in the sidebar to view Derivatives Analytics.")

with main_tab3:
    if market == "Indian Market":
        with st.spinner("Processing Volatility & Structural Diagnostics..."):
            try:
                N, p_vol, c2c_vol = calc_parkinson_vol()
                st.markdown("### Parkinson Estimator (Intraday Risk)")
                m1, m2, m3 = st.columns(3)
                m1.metric("Trading Days Analyzed", N)
                m2.metric("Parkinson Volatility", f"{p_vol:.2f}%")
                m3.metric("Close-to-Close Vol", f"{c2c_vol:.2f}%")
                st.markdown("---")
            except Exception: 
                pass
            
            c1, c2 = st.columns(2)
            with c1: 
                st.pyplot(plot_nifty_volatility())
                st.pyplot(plot_volatility_cone())
                st.pyplot(plot_liquidity_sweep())
                st.pyplot(plot_index_divergence())
            with c2: 
                st.pyplot(plot_expected_move())
                st.pyplot(plot_vrp())
                st.pyplot(plot_hurst_regime())
    
    else:  # CRYPTO VOLATILITY STUDIES
        st.markdown("### 🔐 Crypto Volatility & Risk Analysis")
        with st.spinner("Computing crypto volatility metrics..."):
            c1, c2 = st.columns(2)
            
            with c1:
                fig_btc_vol = plot_crypto_volatility("Bitcoin", "BTC-USD")
                if fig_btc_vol:
                    st.pyplot(fig_btc_vol)
                
                btc_data = yf.download("BTC-USD", period="1y", progress=False)
                if btc_data is not None and not btc_data.empty:
                    btc_close = btc_data['Close'].squeeze()
                    st.markdown("#### Bitcoin Metrics")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("52W High", f"${btc_close.max():.2f}")
                    m2.metric("52W Low", f"${btc_close.min():.2f}")
                    m3.metric("Current", f"${btc_close.iloc[-1]:.2f}")
            
            with c2:
                fig_eth_vol = plot_crypto_volatility("Ethereum", "ETH-USD")
                if fig_eth_vol:
                    st.pyplot(fig_eth_vol)
                
                eth_data = yf.download("ETH-USD", period="1y", progress=False)
                if eth_data is not None and not eth_data.empty:
                    eth_close = eth_data['Close'].squeeze()
                    st.markdown("#### Ethereum Metrics")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("52W High", f"${eth_close.max():.2f}")
                    m2.metric("52W Low", f"${eth_close.min():.2f}")
                    m3.metric("Current", f"${eth_close.iloc[-1]:.2f}")