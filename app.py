import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Use Streamlit's wide layout for cinematic charts
st.set_page_config(page_title="Quant ML Master Dashboard", layout="wide", initial_sidebar_state="expanded")

warnings.filterwarnings('ignore')
plt.style.use('dark_background')

# ==========================================
# 1. IVR & IVP
# ==========================================
def plot_nifty_volatility():
    ticker = "^INDIAVIX"
    data = yf.download(ticker, period="1y", progress=False)
    if data.empty: return None

    close_prices = data['Close'].squeeze()
    current_iv = float(close_prices.iloc[-1])
    high_52w = float(close_prices.max())
    low_52w = float(close_prices.min())

    ivr = ((current_iv - low_52w) / (high_52w - low_52w)) * 100
    days_below = (close_prices < current_iv).sum()
    total_days = len(close_prices)
    ivp = (days_below / total_days) * 100

    if ivr > 50:
        regime = "HIGH VOLATILITY: Net Short Premium"
        color_theme = '#00FF00' 
    else:
        regime = "LOW VOLATILITY: Net Long Premium"
        color_theme = '#FF3333' 

    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
    ax.plot(close_prices.index, close_prices.values, color='#00FFFF', linewidth=1.5, label='India VIX (1Y)')
    ax.axhline(high_52w, color='red', linestyle='--', alpha=0.5, label=f'52W High: {high_52w:.2f}')
    ax.axhline(low_52w, color='green', linestyle='--', alpha=0.5, label=f'52W Low: {low_52w:.2f}')
    ax.axhline(current_iv, color='white', linestyle='-', linewidth=2, label=f'Current: {current_iv:.2f}')
    ax.fill_between(close_prices.index, low_52w, current_iv, color='white', alpha=0.05)

    ax.set_title('NIFTY Implied Volatility (IVR & IVP)', fontsize=18, color='white', pad=20, fontweight='bold')
    ax.set_ylabel('VIX Level', color='gray', fontsize=12)
    ax.grid(True, color='#2A2A2A', linestyle=':')
    ax.legend(loc='upper right', facecolor='black', edgecolor='gray', fontsize=10)

    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8, edgecolor=color_theme, linewidth=1.5)
    text_str = (f"🎯 IV Rank (IVR): {ivr:.1f}%\n📊 IV Percentile (IVP): {ivp:.1f}%\n"
                f"⚡ Current VIX: {current_iv:.2f}\n---------------------------\n🤖 Edge: {regime}")
    ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig

# ==========================================
# 2. Expected Move
# ==========================================
def plot_expected_move():
    nifty_data = yf.download("^NSEi", period="1mo", progress=False)
    vix_data = yf.download("^INDIAVIX", period="5d", progress=False)
    if nifty_data.empty or vix_data.empty: return None

    nifty_close = nifty_data['Close'].squeeze()
    spot_price = float(nifty_close.iloc[-1])
    current_vix = float(vix_data['Close'].squeeze().iloc[-1])

    daily_volatility = (current_vix / 100) * np.sqrt(1/365)
    expected_move_points = spot_price * daily_volatility
    upper_bound = spot_price + expected_move_points
    lower_bound = spot_price - expected_move_points

    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
    recent_nifty = nifty_close.tail(15)
    x_dates = np.arange(len(recent_nifty))

    ax.plot(x_dates, recent_nifty.values, color='#00FFFF', linewidth=2, marker='o', label='Nifty 50 Close')
    tomorrow_x = len(recent_nifty)
    ax.hlines(spot_price, xmin=x_dates[-1], xmax=tomorrow_x, color='white', linestyle='-', linewidth=2)
    ax.scatter(tomorrow_x, spot_price, color='white', s=70, zorder=5)
    ax.scatter(tomorrow_x, upper_bound, color='#00FF00', s=120, marker='^', zorder=5)
    ax.scatter(tomorrow_x, lower_bound, color='#FF3333', s=120, marker='v', zorder=5)
    ax.hlines(upper_bound, xmin=x_dates[-1], xmax=tomorrow_x, color='#00FF00', linestyle='--', linewidth=1.5)
    ax.hlines(lower_bound, xmin=x_dates[-1], xmax=tomorrow_x, color='#FF3333', linestyle='--', linewidth=1.5)
    ax.fill_between([x_dates[-1], tomorrow_x], [spot_price, lower_bound], [spot_price, upper_bound], color='gray', alpha=0.2)

    ax.set_title('NIFTY 50 Implied Daily Expected Move', fontsize=18, color='white', pad=20, fontweight='bold')
    ax.set_ylabel('Nifty 50 Price', color='gray', fontsize=12)
    ax.grid(True, color='#2A2A2A', linestyle=':')
    ax.set_xticks([]) 
    ax.legend(['Nifty 50 Close', 'Upper Bound (+1 SD)', 'Lower Bound (-1 SD)'], loc='upper left', facecolor='black', edgecolor='gray', fontsize=10)

    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8, edgecolor='white', linewidth=1.5)
    text_str = (f"⚡ Current VIX: {current_vix:.2f}\n🎯 Spot Price: {spot_price:.2f}\n"
                f"📏 Expected Move: ± {expected_move_points:.1f} points\n---------------------------\n"
                f"🟢 Safe Call Strike: > {upper_bound:.0f}\n🔴 Safe Put Strike: < {lower_bound:.0f}")
    ax.text(0.02, 0.45, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig

# ==========================================
# 3. Correlation
# ==========================================
def plot_index_divergence():
    tickers = {"Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK"}
    data = yf.download(list(tickers.values()), period="1y", progress=False)['Close']
    if data.empty: return None

    data.columns = ['Bank Nifty', 'Nifty 50']
    data = data.dropna()
    normalized_prices = (data / data.iloc[0]) * 100
    log_returns = np.log(data / data.shift(1)).dropna()
    rolling_correlation = log_returns['Nifty 50'].rolling(window=20).corr(log_returns['Bank Nifty'])

    current_nifty = float(data['Nifty 50'].iloc[-1])
    current_bank = float(data['Bank Nifty'].iloc[-1])
    current_corr = float(rolling_correlation.iloc[-1])

    if current_corr > 0.80:
        regime = "HIGH CORRELATION"
        stat_property = "Synchronized Movement / Low Dispersion"
        color_theme = '#00FF00' 
    elif current_corr < 0.50:
        regime = "SEVERE DIVERGENCE"
        stat_property = "Sector Rotation / High Dispersion Warning"
        color_theme = '#FF3333' 
    else:
        regime = "MODERATE DIVERGENCE"
        stat_property = "Decoupling / Monitoring Phase"
        color_theme = '#FFA500' 

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=120, gridspec_kw={'height_ratios': [1.5, 1]})
    ax1.plot(normalized_prices.index, normalized_prices['Nifty 50'], color='#00FFFF', linewidth=2, label='Nifty 50 (Normalized)')
    ax1.plot(normalized_prices.index, normalized_prices['Bank Nifty'], color='#FFA500', linewidth=2, label='Bank Nifty (Normalized)')
    ax1.fill_between(normalized_prices.index, normalized_prices['Nifty 50'], normalized_prices['Bank Nifty'], color='gray', alpha=0.2, label='Performance Spread')
    ax1.set_title('Inter-Index Correlation & Divergence', fontsize=18, color='white', pad=15, fontweight='bold')
    ax1.set_ylabel('Normalized Price', color='gray')
    ax1.grid(True, color='#2A2A2A', linestyle=':')
    ax1.legend(loc='upper left', facecolor='black', edgecolor='gray')

    ax2.plot(rolling_correlation.index, rolling_correlation, color='white', linewidth=1.5, label='20-Day Rolling Correlation')
    ax2.axhline(0.80, color='#00FF00', linestyle='--', linewidth=1.5, label='High Correlation (>0.80)')
    ax2.axhline(0.50, color='#FF3333', linestyle='--', linewidth=1.5, label='Severe Divergence (<0.50)')
    ax2.fill_between(rolling_correlation.index, 0.50, rolling_correlation, where=(rolling_correlation < 0.50), color='#FF3333', alpha=0.3, interpolate=True)
    ax2.set_ylabel('Pearson Correlation (r)', color='gray')
    ax2.set_ylim(-0.2, 1.1)
    ax2.grid(True, color='#2A2A2A', linestyle=':')
    ax2.legend(loc='lower left', facecolor='black', edgecolor='gray')

    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9, edgecolor=color_theme, linewidth=1.5)
    text_str = (f"Nifty 50: {current_nifty:.2f}\nBank Nifty: {current_bank:.2f}\n"
                f"20-Day Correlation: {current_corr:.2f}\n---------------------------\nStat Property: {stat_property}")
    ax1.text(0.02, 0.05, text_str, transform=ax1.transAxes, fontsize=12, verticalalignment='bottom', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig

# ==========================================
# 4. Volatility Cone
# ==========================================
def plot_volatility_cone():
    ticker = "^NSEI"
    data = yf.download(ticker, start="2015-01-01", progress=False)
    data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    windows = [10, 20, 30, 60, 90, 120, 180, 252]
    max_vol, min_vol, median_vol, current_vol = [], [], [], []

    for window in windows:
        rolling_vol = data['Returns'].rolling(window=window).std() * np.sqrt(252)
        max_vol.append(rolling_vol.max() * 100)       
        min_vol.append(rolling_vol.min() * 100)
        median_vol.append(rolling_vol.median() * 100)
        current_vol.append(rolling_vol.iloc[-1] * 100) 

    fig = plt.figure(figsize=(12, 7))
    plt.plot(windows, max_vol, marker='o', color='red', linewidth=2, label='Maximum Volatility')
    plt.plot(windows, min_vol, marker='o', color='limegreen', linewidth=2, label='Minimum Volatility')
    plt.plot(windows, median_vol, marker='s', color='white', linewidth=1.5, linestyle='--', label='Median Volatility')
    plt.plot(windows, current_vol, marker='X', color='yellow', linewidth=3, markersize=10, label='Current Volatility')
    plt.fill_between(windows, min_vol, max_vol, color='gray', alpha=0.2)
    plt.title(f'Volatility Cone for Nifty 50 ({ticker})', fontsize=18, fontweight='bold', color='white')
    plt.xlabel('Time Window (Trading Days)', fontsize=14)
    plt.ylabel('Annualized Volatility (%)', fontsize=14)
    plt.xticks(windows)
    plt.grid(color='gray', linestyle=':', alpha=0.5)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    return fig

# ==========================================
# 5. Volatility Risk Premium (VRP)
# ==========================================
def plot_vrp():
    nifty_data = yf.download("^NSEI", period="6mo", progress=False)
    vix_data = yf.download("^INDIAVIX", period="6mo", progress=False)
    if nifty_data.empty or vix_data.empty: return None

    nifty_close = nifty_data['Close'].squeeze()
    vix_close = vix_data['Close'].squeeze()
    
    log_returns = np.log(nifty_close / nifty_close.shift(1))
    historical_vol = log_returns.rolling(window=20).std() * np.sqrt(252) * 100

    df = pd.DataFrame({'VIX': vix_close, 'HV': historical_vol}).dropna()
    df['VRP'] = df['VIX'] - df['HV']

    current_vix = float(df['VIX'].iloc[-1])
    current_hv = float(df['HV'].iloc[-1])
    current_vrp = float(df['VRP'].iloc[-1])

    if current_vrp > 0:
        regime, stat_property, color_theme = "POSITIVE VRP", "Implied Risk > Realized Risk", '#00FF00'
    else:
        regime, stat_property, color_theme = "NEGATIVE VRP", "Realized Risk > Implied Risk", '#FF3333'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=120, gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(df.index, df['VIX'], color='#00FFFF', linewidth=2, label='Implied Volatility (India VIX)')
    ax1.plot(df.index, df['HV'], color='#FFA500', linewidth=2, label='Realized Volatility (20-Day HV)')
    ax1.fill_between(df.index, df['HV'], df['VIX'], where=(df['VIX'] > df['HV']), color='green', alpha=0.3)
    ax1.fill_between(df.index, df['HV'], df['VIX'], where=(df['VIX'] <= df['HV']), color='red', alpha=0.3)
    ax1.set_title('NIFTY 50 Volatility Risk Premium (VRP)', fontsize=18, color='white', pad=15, fontweight='bold')
    ax1.grid(True, color='#2A2A2A', linestyle=':')
    ax1.legend(loc='upper left', facecolor='black', edgecolor='gray')

    bar_colors = np.where(df['VRP'] > 0, '#00FF00', '#FF3333')
    ax2.bar(df.index, df['VRP'], color=bar_colors, alpha=0.7, width=1)
    ax2.axhline(0, color='white', linewidth=1)
    ax2.set_ylabel('VRP Spread (%)', color='gray')
    ax2.grid(True, color='#2A2A2A', linestyle=':')

    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9, edgecolor=color_theme, linewidth=1.5)
    text_str = (f"VIX (Expected): {current_vix:.2f}%\n20-Day HV (Actual): {current_hv:.2f}%\n"
                f"VRP Spread: {current_vrp:+.2f}%\n---------------------------\nRegime: {regime}\nProp: {stat_property}")
    ax1.text(0.02, 0.05, text_str, transform=ax1.transAxes, fontsize=12, verticalalignment='bottom', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig

# ==========================================
# 6. Hurst Exponent
# ==========================================
def calculate_hurst(ts):
    if len(ts) < 20: return np.nan
    lags = range(2, 20)
    tau = [lag for lag in lags]
    ts_arr = ts.values
    reg = [np.std(ts_arr[lag:] - ts_arr[:-lag]) for lag in lags]
    poly = np.polyfit(np.log(tau), np.log(reg), 1)
    return poly[0]

def plot_hurst_regime():
    data = yf.download("^NSEi", period="1y", progress=False)
    if data.empty: return None
    nifty_close = data['Close'].squeeze()
    log_prices = np.log(nifty_close)
    hurst_series = log_prices.rolling(window=60).apply(calculate_hurst, raw=False)
    df = pd.DataFrame({'Close': nifty_close, 'Hurst': hurst_series}).dropna()

    current_price = float(df['Close'].iloc[-1])
    current_hurst = float(df['Hurst'].iloc[-1])

    if current_hurst < 0.45: regime, stat_property, color_theme = "MEAN REVERTING", "Range-Bound Action", '#FF3333'
    elif current_hurst > 0.55: regime, stat_property, color_theme = "TRENDING", "Directional Movement", '#00FF00'
    else: regime, stat_property, color_theme = "RANDOM WALK", "Unpredictable Noise", '#FFA500'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=120, gridspec_kw={'height_ratios': [1.5, 1]})
    ax1.plot(df.index, df['Close'], color='white', linewidth=1.5, label='Nifty 50 Close')
    ax1.set_title('NIFTY 50 Market Regime (Hurst Exponent)', fontsize=18, color='white', pad=15, fontweight='bold')
    ax1.grid(True, color='#2A2A2A', linestyle=':')
    ax1.axvspan(df.index[-15], df.index[-1], color=color_theme, alpha=0.1)

    ax2.plot(df.index, df['Hurst'], color='#00FFFF', linewidth=2, label='60-Day Hurst Exponent')
    ax2.axhline(0.55, color='#00FF00', linestyle='--', linewidth=1.5)
    ax2.axhline(0.45, color='#FF3333', linestyle='--', linewidth=1.5)
    ax2.fill_between(df.index, 0.55, df['Hurst'], where=(df['Hurst'] > 0.55), color='#00FF00', alpha=0.2)
    ax2.fill_between(df.index, 0.45, df['Hurst'], where=(df['Hurst'] < 0.45), color='#FF3333', alpha=0.2)
    ax2.set_ylabel('Hurst Value (H)', color='gray')
    ax2.grid(True, color='#2A2A2A', linestyle=':')
    ax2.set_ylim(0.3, 0.7)

    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9, edgecolor=color_theme, linewidth=1.5)
    text_str = (f"Current Nifty: {current_price:.2f}\nHurst (H): {current_hurst:.3f}\n"
                f"Regime: {regime}\n---------------------------\nStat Property: {stat_property}")
    ax1.text(0.02, 0.05, text_str, transform=ax1.transAxes, fontsize=12, verticalalignment='bottom', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig

# ==========================================
# 7. Liquidity Detector
# ==========================================
def plot_liquidity_sweep():
    data = yf.download("^NSEi", period="30d", interval="15m", progress=False)
    if data.empty: return None
    if isinstance(data.columns, pd.MultiIndex): data.columns = [col[0] for col in data.columns]

    window = 20 
    data['Prev_High'] = data['High'].rolling(window=window).max().shift(1)
    data['Prev_Low'] = data['Low'].rolling(window=window).min().shift(1)
    data['Supply_Sweep'] = (data['High'] > data['Prev_High']) & (data['Close'] < data['Prev_High'])
    data['Demand_Sweep'] = (data['Low'] < data['Prev_Low']) & (data['Close'] > data['Prev_Low'])
    current_price = float(data['Close'].iloc[-1])

    if data['Supply_Sweep'].iloc[-1]: regime, stat_property, color_theme = "SUPPLY LIQUIDITY SWEPT", "Failed Breakout", '#FF3333'
    elif data['Demand_Sweep'].iloc[-1]: regime, stat_property, color_theme = "DEMAND LIQUIDITY SWEPT", "Failed Breakdown", '#00FF00'
    else: regime, stat_property, color_theme = "PRICE DISCOVERY PHASE", "Inside Structural Bounds", '#00FFFF'

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
            ax.axhline(row['Prev_High'], color='#FF3333', linestyle='--', alpha=0.5, linewidth=1)
        if row['Demand_Sweep']:
            ax.scatter(row['Index'], row['Low'] - 10, marker='^', color='#00FF00', s=150, zorder=5)
            ax.axhline(row['Prev_Low'], color='#00FF00', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_title('NIFTY 50 Intraday Liquidity Sweep (15m)', fontsize=16, color='white', pad=15, fontweight='bold')
    ax.grid(True, color='#2A2A2A', linestyle=':')
    ax.set_xticks([]) 

    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9, edgecolor=color_theme, linewidth=1.5)
    text_str = f"Live Spot Price: {current_price:.2f}\n---------------------------\nMicrostructure: {regime}\nProp: {stat_property}"
    ax.text(0.02, 0.05, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', bbox=props, color='white', fontweight='bold')
    plt.tight_layout()
    return fig

# ==========================================
# 8. Parkinson Estimator
# ==========================================
def calculate_parkinson():
    data = yf.download('^NSEi', period='1y', progress=False)
    h, l, c = data['High'].squeeze(), data['Low'].squeeze(), data['Close'].squeeze()
    N = len(data)

    log_hl = np.log(h / l)
    log_hl_squared = log_hl ** 2
    constant = 1 / (4 * N * np.log(2))
    parkinson_variance = constant * log_hl_squared.sum()
    parkinson_vol = np.sqrt(parkinson_variance) * np.sqrt(252)

    log_returns = np.log(c / c.shift(1))
    c2c_vol = log_returns.std() * np.sqrt(252)
    
    return N, float(parkinson_vol)*100, float(c2c_vol)*100

# ==========================================
# STREAMLIT UI LAYOUT
# ==========================================
st.title("⚡ Quantitative Financial Models Dashboard")
st.markdown("Select a mathematical model from the sidebar to visualize institutional risk metrics.")

# Sidebar Navigation
st.sidebar.header("Navigation")
tools = [
    "1. IVR & IVP", 
    "2. Expected Move", 
    "3. Correlation Divergence", 
    "4. Volatility Cone", 
    "5. Volatility Risk Premium", 
    "6. Hurst Exponent", 
    "7. Liquidity Detector", 
    "8. Parkinson Volatility",
    "9. FYERS Live Profile (Requires API)"
]
selected_tool = st.sidebar.radio("Choose Analysis Tool:", tools)

st.sidebar.markdown("---")
st.sidebar.header("FYERS API Settings (For Tool 9)")
fyers_token = st.sidebar.text_input("Access Token", type="password")
fyers_client = st.sidebar.text_input("Client ID")

# Main Display Logic
if selected_tool == "1. IVR & IVP":
    st.subheader("📊 IV Rank & IV Percentile")
    with st.spinner("Fetching data..."):
        st.pyplot(plot_nifty_volatility())

elif selected_tool == "2. Expected Move":
    st.subheader("📏 Daily Expected Move (Market Maker Bounds)")
    with st.spinner("Calculating probability cones..."):
        st.pyplot(plot_expected_move())

elif selected_tool == "3. Correlation Divergence":
    st.subheader("🔄 Inter-Index Correlation (Nifty vs Bank Nifty)")
    with st.spinner("Analyzing dispersion..."):
        st.pyplot(plot_index_divergence())

elif selected_tool == "4. Volatility Cone":
    st.subheader("🌪️ Historical Volatility Cone")
    with st.spinner("Mapping volatility term structure..."):
        st.pyplot(plot_volatility_cone())

elif selected_tool == "5. Volatility Risk Premium":
    st.subheader("💰 Volatility Risk Premium (VRP)")
    with st.spinner("Calculating Implied vs Realized spreads..."):
        st.pyplot(plot_vrp())

elif selected_tool == "6. Hurst Exponent":
    st.subheader("📈 Hurst Exponent (Market Regime Indicator)")
    with st.spinner("Analyzing fractional brownian motion..."):
        st.pyplot(plot_hurst_regime())

elif selected_tool == "7. Liquidity Detector":
    st.subheader("💧 Intraday Liquidity Sweeps")
    with st.spinner("Hunting order blocks..."):
        st.pyplot(plot_liquidity_sweep())

elif selected_tool == "8. Parkinson Volatility":
    st.subheader("📉 Parkinson Estimator (Intraday Risk)")
    with st.spinner("Calculating variance..."):
        n, park, c2c = calculate_parkinson()
        col1, col2, col3 = st.columns(3)
        col1.metric("Trading Days Analyzed", n)
        col2.metric("Parkinson Volatility", f"{park:.2f}%", help="Captures intraday high/low swings.")
        col3.metric("Standard C2C Volatility", f"{c2c:.2f}%", delta=f"{(park-c2c):.2f}% Difference", delta_color="inverse")

elif selected_tool == "9. FYERS Live Profile (Requires API)":
    st.subheader("🔥 FYERS Live Option Profile")
    if fyers_token and fyers_client:
        st.info("API Credentials Received. Add the `plot_fyers_oi_profile()` execution here using the credentials.")
        # Fyers execution logic goes here (commented out for safety as it requires live local connection)
    else:
        st.warning("⚠️ Please enter your FYERS Access Token and Client ID in the sidebar to use this module.")

st.markdown("---")
st.markdown("*Built for Machine Learning Feature Engineering and Quantitative Trading.*")