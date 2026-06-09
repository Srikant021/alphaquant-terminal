# crypto_alphaquant_with_rolling_metrics_and_explanation.py
# AlphaQuant Terminal — merged, hardened, ML-explainable version with rolling precision/recall
# and practical explanation text added to the ML tab.

import logging
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

# Optional ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

# Basic logging to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("alphaquant")

st.set_page_config(page_title="AlphaQuant Terminal", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background: #080d12; }
    .metric-box {
        background: linear-gradient(135deg, #0d1520 0%, #111c2b 100%);
        padding: 16px 18px; border-radius: 8px; border-left: 3px solid #0af;
        margin: 5px 0; border-top: 1px solid rgba(0,170,255,0.08);
    }
    .explanation-box {
        background: rgba(0,170,255,0.05); border-left: 3px solid #0af;
        padding: 10px 14px; border-radius: 4px; font-size: 12px; margin: 8px 0;
        color: rgba(255,255,255,0.75);
    }
    .section-header {
        font-family: 'Space Mono', monospace; font-size: 13px; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase; color: #0af;
        margin: 24px 0 12px 0; padding-bottom: 6px;
        border-bottom: 1px solid rgba(0,170,255,0.2);
    }
    .sweep-badge {
        display: inline-block; padding: 5px 14px; border-radius: 20px;
        font-family: 'Space Mono', monospace; font-size: 11px; font-weight: 700;
        letter-spacing: 0.08em; text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))


def _month_end_alias() -> str:
    major, minor = (int(x) for x in pd.__version__.split(".")[:2])
    return "ME" if (major, minor) >= (2, 2) else "M"


# ─────────────────────────────────────────────
# SAFE YFINANCE DOWNLOAD WITH RETRY
# ─────────────────────────────────────────────
def _download_with_retry(ticker, period, interval, attempts=3, backoff=1.5):
    last_exc = None
    for i in range(attempts):
        try:
            logger.info(f"Fetching {ticker} period={period} interval={interval} (attempt {i+1})")
            data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            return data
        except Exception as e:
            last_exc = e
            wait = backoff ** i
            logger.warning(f"Fetch failed for {ticker} (attempt {i+1}): {e}. Retrying in {wait:.1f}s")
            time.sleep(wait)
    logger.error(f"All fetch attempts failed for {ticker}: {last_exc}")
    raise last_exc


# ─────────────────────────────────────────────
# TECHNICAL INDICATORS & MATH HELPERS
# ─────────────────────────────────────────────
def compute_iv_rank(iv_series, window=252):
    """Calculates Implied Volatility Rank (IVR) over a rolling window."""
    rolling_min = iv_series.rolling(window=window, min_periods=1).min()
    rolling_max = iv_series.rolling(window=window, min_periods=1).max()
    denominator = (rolling_max - rolling_min).replace(0, np.nan)
    ivr = ((iv_series - rolling_min) / denominator) * 100
    return ivr.fillna(0)


def compute_rsi(series, period=14):
    """Calculates the Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs)).fillna(50)


def bollinger_bands(series, period=20, std=2.0):
    """Calculates upper, middle, and lower Bollinger Bands."""
    sma = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper_band = sma + (rolling_std * std)
    lower_band = sma - (rolling_std * std)
    return upper_band, sma, lower_band


def compute_atr(df, period=14):
    """Calculates the Average True Range."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()


def compute_macd(series, fast=12, slow=26, signal=9):
    """Calculates MACD Line, Signal Line, and Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def hurst_exponent(price_series):
    price = np.asarray(price_series.squeeze().dropna(), dtype=float)
    n = len(price)
    if n < 100:
        return 0.5, "Insufficient data", "low"
    log_prices = np.log(price)
    max_lag = min(n // 2, 200)
    lags = np.unique(np.logspace(1, np.log10(max_lag), num=30).astype(int))
    lags = lags[lags >= 10]
    rs_values, valid_lags = [], []
    for lag in lags:
        n_windows = n // lag
        if n_windows < 3:
            continue
        rs_window = []
        for i in range(n_windows):
            window = log_prices[i * lag:(i + 1) * lag]
            mean_adj = window - window.mean()
            cumsum = np.cumsum(mean_adj)
            R = cumsum.max() - cumsum.min()
            S = window.std(ddof=1)
            if S > 1e-10:
                rs_window.append(R / S)
        if len(rs_window) >= 3:
            rs_values.append(np.mean(rs_window))
            valid_lags.append(lag)
    if len(valid_lags) < 8:
        return 0.5, "Insufficient data", "low"
    log_lags = np.log(valid_lags)
    log_rs = np.log(rs_values)
    coeffs = np.polyfit(log_lags, log_rs, 1)
    hurst = float(np.clip(coeffs[0], 0.05, 0.95))
    predicted = np.polyval(coeffs, log_lags)
    ss_res = np.sum((log_rs - predicted) ** 2)
    ss_tot = np.sum((log_rs - np.mean(log_rs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    confidence = "high" if (r2 > 0.97 and len(valid_lags) >= 8) else "medium" if r2 > 0.90 else "low"
    if hurst > 0.58:
        interp = "Strong Trend (Persistent)"
    elif hurst > 0.53:
        interp = "Weak Trend (Mildly Persistent)"
    elif hurst >= 0.47:
        interp = "Random Walk"
    elif hurst >= 0.42:
        interp = "Weak Mean-Reversion"
    else:
        interp = "Strong Mean-Reversion (Anti-Persistent)"
    return hurst, interp, confidence


# ─────────────────────────────────────────────
# PLOTTING FUNCTIONS
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
# DATA FETCHING (with caching)
# ─────────────────────────────────────────────
def _flatten_multiindex(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


@st.cache_data(ttl=300)
def fetch_data(ticker, period="1y", interval="1d"):
    try:
        raw = _download_with_retry(ticker, period, interval)
        if raw is None or raw.empty:
            return None
        return _flatten_multiindex(raw)
    except Exception as e:
        logger.exception(f"Error fetching {ticker}: {e}")
        return None


@st.cache_data(ttl=60)
def get_live_price(ticker):
    try:
        data = fetch_data(ticker, period="5d", interval="1d")
        if data is None or len(data) < 2:
            return None
        last = float(data['Close'].iloc[-1])
        prev = float(data['Close'].iloc[-2])
        return {
            'price': last,
            'change': last - prev,
            'pct': ((last - prev) / prev) * 100 if prev else 0.0,
            'high': float(data['High'].iloc[-1]),
            'low': float(data['Low'].iloc[-1]),
            'volume': float(data['Volume'].iloc[-1]),
        }
    except Exception as e:
        logger.exception(f"Error getting live price for {ticker}: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_live_vix(market: str) -> float:
    try:
        ticker = "^INDIAVIX" if market == "Indian Market" else "^VIX"
        default = 18.0 if market == "Indian Market" else 60.0
        data = fetch_data(ticker, period="5d", interval="1d")
        if data is None or data.empty:
            return default
        val = data['Close'].dropna().iloc[-1]
        return float(val) if not np.isnan(val) else default
    except Exception as e:
        logger.exception(f"Error fetching VIX for market {market}: {e}")
        return 60.0


# ─────────────────────────────────────────────
# ML PIPELINE & METRICS INTERPRETATION
# ─────────────────────────────────────────────
def build_ml_features(df, rsi_period=14, boll_period=20, boll_std=2.0, atr_period=14):
    feat = pd.DataFrame(index=df.index)
    close = df['Close'].squeeze()
    feat['rsi'] = compute_rsi(close, period=rsi_period)
    feat['returns'] = close.pct_change()
    feat['vol_20'] = feat['returns'].rolling(20).std()
    bb_up, _, bb_lo = bollinger_bands(close, period=boll_period, std=boll_std)
    feat['bb_pos'] = (close - bb_lo) / (bb_up - bb_lo + 1e-9)
    feat['atr'] = compute_atr(df, period=atr_period)
    feat['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    macd, sig, _ = compute_macd(close)
    feat['macd_diff'] = macd - sig
    return feat.dropna()


def explain_ml_prediction(model, feat_df, prob_pos):
    """Generates explicit human-readable reasons behind the machine learning model's output signal."""
    latest = feat_df.iloc[-1]
    importances = pd.Series(model.feature_importances_, index=feat_df.columns).sort_values(ascending=False)
    top_feats = importances.head(4).index.tolist()

    supporting = []
    opposing = []
    neutral = []

    for f in top_feats:
        v = latest.get(f, np.nan)
        if pd.isna(v):
            neutral.append(f)
            continue

        if f == 'rsi':
            if v < 40:
                supporting.append(f"RSI is low ({v:.1f}), indicating oversold conditions which often precede bounces")
            elif v > 60:
                opposing.append(f"RSI is high ({v:.1f}), indicating overbought conditions which often precede pullbacks")
            else:
                neutral.append(f"RSI is neutral ({v:.1f})")
        elif f == 'macd_diff':
            if v > 0:
                supporting.append(f"MACD diff is positive ({v:.4f}), showing bullish momentum")
            elif v < 0:
                opposing.append(f"MACD diff is negative ({v:.4f}), showing bearish momentum")
            else:
                neutral.append(f"MACD diff is near zero ({v:.4f})")
        elif f == 'bb_pos':
            if v < 0.3:
                supporting.append(f"Price is near the lower Bollinger band (bb_pos={v:.2f}), which can signal mean-reversion upside")
            elif v > 0.7:
                opposing.append(f"Price is near the upper Bollinger band (bb_pos={v:.2f}), which can signal mean-reversion downside")
            else:
                neutral.append(f"Bollinger position is mid-range (bb_pos={v:.2f})")
        elif f == 'atr':
            if 'atr' in feat_df.columns:
                med_atr = feat_df['atr'].median()
                if v > med_atr:
                    neutral.append(f"ATR is elevated ({v:.3f}), implying higher volatility and larger potential moves")
                else:
                    neutral.append(f"ATR is subdued ({v:.3f}), implying lower volatility")
            else:
                neutral.append(f"ATR = {v:.3f}")
        elif f == 'vol_ratio':
            if v > 1.5:
                supporting.append(f"Volume is elevated (vol_ratio={v:.2f}), which tends to confirm directional moves")
            elif v < 0.7:
                opposing.append(f"Volume is low (vol_ratio={v:.2f}), which can make breakouts less reliable")
            else:
                neutral.append(f"Volume is normal (vol_ratio={v:.2f})")
        elif f == 'returns':
            if v > 0:
                supporting.append(f"Recent return is positive ({v:.3%}), which supports short-term upside")
            elif v < 0:
                opposing.append(f"Recent return is negative ({v:.3%}), which signals short-term downside")
            else:
                neutral.append(f"Recent return is flat ({v:.3%})")

    return {
        "supporting": supporting,
        "opposing": opposing,
        "neutral": neutral,
        "probability": prob_pos
    }


# ─────────────────────────────────────────────
# MAIN STREAMLIT APPLICATION INTERFACE
# ─────────────────────────────────────────────
def main():
    st.title("⚡ AlphaQuant Terminal")
    
    # Global Sidebar
    st.sidebar.header("Navigation & Assets")
    market_selection = st.sidebar.radio("Select Target Infrastructure Context", ["Indian Market", "Crypto Assets"])
    
    if market_selection == "Indian Market":
        st.sidebar.markdown("### Institutional Suite")
        ticker = "^NSEI"
        name = "Nifty 50"
    else:
        st.sidebar.markdown("### Crypto Liquidity Matrix")
        crypto_ticker = st.sidebar.selectbox("Select Derivative Asset Pool", ["BTC-USD", "ETH-USD", "SOL-USD"])
        ticker = crypto_ticker
        name = crypto_ticker.split("-")[0]

    # Layout Workspace Divisions
    tab_vol, tab_struct, tab_ml = st.tabs(["📊 Volatility Engine", "📐 Structural Analysis", "🤖 ML Explanations & Rolling Metrics"])

    # --- Volatility Tab ---
    with tab_vol:
        st.header("Volatility Spread Matrix")
        if market_selection == "Indian Market":
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_nifty_volatility())
            with col2:
                st.pyplot(plot_vrp())
            
            st.markdown("### Volatility Cone Scaling Metrics")
            st.pyplot(plot_volatility_cone())
        else:
            st.pyplot(plot_crypto_volatility(name, ticker))

    # --- Structural Tab ---
    with tab_struct:
        st.header("Market Structure & Microstructure Profiles")
        if market_selection == "Indian Market":
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_expected_move())
            with col2:
                st.pyplot(plot_liquidity_sweep())
            
            st.markdown("### Inter-Index Dynamics & Long Memory Indices")
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_index_divergence())
            with c2:
                st.pyplot(plot_hurst_regime())
        else:
            raw_data = fetch_data(ticker, period="1y", interval="1d")
            if raw_data is not None:
                h_val, interp, conf = hurst_exponent(raw_data['Close'])
                st.metric("Asset Memory State (Hurst Exponent)", f"{h_val:.3f}", delta=interp)
                st.info(f"Mathematical regime profile confidence level evaluated as **{conf.upper()}**.")

    # --- Machine Learning Engine Tab ---
    with tab_ml:
        st.header("ML Engine Inference Engine")
        if not ML_AVAILABLE:
            st.error("ML Dependencies unavailable in context interpreter environment.")
            return

        raw_df = fetch_data(ticker, period="2y", interval="1d")
        if raw_df is None or raw_df.empty:
            st.warning("Data matrices unavailable for ML generation pipelines.")
            return

        features = build_ml_features(raw_df)
        # Create target shifted look-ahead label
        target = (raw_df['Close'].pct_change().shift(-1) > 0).astype(int).loc[features.index]
        
        # Align target arrays
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]

        if len(X) > 100:
            # Simple walk-forward split window train/test
            split = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            base_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            calibrated_model = CalibratedClassifierCV(base_rf, method='sigmoid', cv=3)
            calibrated_model.fit(X_train_scaled, y_train)

            # Pull raw trained tree for feature importances
            base_rf.fit(X_train_scaled, y_train)

            # Latest live asset vector prediction
            latest_vector = scaler.transform(X.iloc[[-1]])
            prob = calibrated_model.predict_proba(latest_vector)[0][1]

            explanation = explain_ml_prediction(base_rf, X, prob)

            st.subheader(f"Live Real-Time Prediction Signal Analysis: {name}")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Directional Upside Probability Factor", f"{prob * 100:.2f}%")
            with col_m2:
                signal_type = "BULLISH EDGE" if prob > 0.55 else "BEARISH EDGE" if prob < 0.45 else "NEUTRAL AMBIGUITY"
                st.markdown(f"#### Calculated Regime Bias: **{signal_type}**")

            st.markdown("<div class='section-header'>Explainable AI Rationale Breakdown</div>", unsafe_allow_html=True)
            
            c_sup, c_opp = st.columns(2)
            with c_sup:
                st.markdown("### 👍 Supporting Forces")
                for item in explanation["supporting"]:
                    st.markdown(f"* {item}")
            with c_opp:
                st.markdown("### 👎 Opposing Pressures")
                for item in explanation["opposing"]:
                    st.markdown(f"* {item}")
        else:
            st.warning("Insufficient structural row density in dataframe matrices to compute robust backforward pipeline walks.")


if __name__ == '__main__':
    main()