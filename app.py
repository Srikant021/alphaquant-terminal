import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from fyers_apiv3 import fyersModel
from urllib.parse import urlparse, parse_qs

warnings.filterwarnings('ignore')
plt.style.use('dark_background')

# --- Fyers API Configuration (Use Streamlit Secrets for production) ---
# For local testing, you can use the values from your Colab notebook
# In a deployed Streamlit app, you would use:
# CLIENT_ID = st.secrets["FYERS_CLIENT_ID"]
# SECRET_KEY = st.secrets["FYERS_SECRET_KEY"]
# REDIRECT_URI = st.secrets["FYERS_REDIRECT_URI"]

CLIENT_ID = "A8MXI4LH6N-200"
SECRET_KEY = "SwNNcmILk5Zo4UE8"
REDIRECT_URI = "https://google.com"

def generate_fyers_token_streamlit():
    # This function needs to be adapted for Streamlit if direct interactive login is desired.
    # For simplicity, we'll assume access_token is set via secrets or environment for Streamlit deployment.
    # In a real Streamlit app, you might have a button to initiate login and then a text input for the redirected URL.
    st.warning("Fyers API authentication is interactive. For Streamlit deployment, consider setting `access_token` as a secret or environment variable.")
    st.info("Using a placeholder/pre-generated token for demonstration.")
    # Placeholder: In a real app, this would come from user input or secrets
    return "YOUR_PRE_GENERATED_FYERS_ACCESS_TOKEN" # Replace with actual token or use Streamlit secrets


# --- Re-define functions from the notebook for Streamlit context ---

# 1. IVR & IVP (WITH EDGE READOUT BOX)
def plot_nifty_volatility_st():
    ticker = "^INDIAVIX"
    data = yf.download(ticker, period="1y", progress=False)

    if data.empty:
        st.error("Error: No data retrieved from Yahoo Finance for India VIX.")
        return

    close_prices = data['Close'].squeeze()

    current_iv = float(close_prices.iloc[-1])
    high_52w = float(close_prices.max())
    low_52w = float(close_prices.min())

    ivr = ((current_iv - low_52w) / (high_52w - low_52w)) * 100
    days_below = (close_prices < current_iv).sum()
    total_days = len(close_prices)
    ivp = (days_below / total_days) * 100

    if ivr > 50:
        regime = "HIGH VOLATILITY: Net Short Premium (Credit Spreads)"
        color_theme = '#00FF00'
    else:
        regime = "LOW VOLATILITY: Net Long Premium (Debit Spreads)"
        color_theme = '#FF3333'

    st.subheader("Market Structure Math")
    st.write(f"**Current VIX:** {current_iv:.2f}")
    st.write(f"**IV Rank (IVR):** {ivr:.1f}%")
    st.write(f"**IV Percentile (IVP):** {ivp:.1f}%")
    st.markdown(f"**Regime:** <span style='color:{color_theme}'>**{regime}**</span>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot(close_prices.index, close_prices.values, color='#00FFFF', linewidth=1.5, label='India VIX (1Y)')
    ax.axhline(high_52w, color='red', linestyle='--', alpha=0.5, label=f'52W High: {high_52w:.2f}')
    ax.axhline(low_52w, color='green', linestyle='--', alpha=0.5, label=f'52W Low: {low_52w:.2f}')
    ax.axhline(current_iv, color='white', linestyle='-', linewidth=2, label=f'Current: {current_iv:.2f}')
    ax.fill_between(close_prices.index, low_52w, current_iv, color='white', alpha=0.05)
    ax.set_title('NIFTY Implied Volatility (IVR & IVP)', fontsize=18, color='white', pad=20, fontweight='bold')
    ax.set_ylabel('VIX Level', color='gray', fontsize=12)
    ax.grid(True, color='#2A2A2A', linestyle=':')
    ax.legend(loc='upper right', facecolor='black', edgecolor='gray', fontsize=10)

    st.pyplot(fig)
    plt.close(fig)

# 2. Expected Move
def plot_expected_move_st():
    st.write("Fetching Nifty 50 and India VIX data...")
    nifty_data = yf.download("^NSEI", period="1mo", progress=False)
    vix_data = yf.download("^INDIAVIX", period="5d", progress=False)

    if nifty_data.empty or vix_data.empty:
        st.error("Error: Yahoo Finance returned no data for Nifty or VIX.")
        return

    nifty_close = nifty_data['Close'].squeeze()
    spot_price = float(nifty_close.iloc[-1])
    current_vix = float(vix_data['Close'].squeeze().iloc[-1])

    daily_volatility = (current_vix / 100) * np.sqrt(1/365)
    expected_move_points = spot_price * daily_volatility

    upper_bound = spot_price + expected_move_points
    lower_bound = spot_price - expected_move_points

    st.subheader("Daily Expected Move Math")
    st.write(f"**Nifty Spot Price:** {spot_price:.2f}")
    st.write(f"**Current VIX:** {current_vix:.2f}")
    st.write(f"**Daily Implied Move:** ± {expected_move_points:.1f} points")
    st.write(f"**Upper Target (+1 SD):** {upper_bound:.2f}")
    st.write(f"**Lower Target (-1 SD):** {lower_bound:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

    recent_nifty = nifty_close.tail(15)
    x_dates = np.arange(len(recent_nifty))

    ax.plot(x_dates, recent_nifty.values, color='#00FFFF', linewidth=2, marker='o', label='Nifty 50 Close')

    tomorrow_x = len(recent_nifty)

    ax.hlines(spot_price, xmin=x_dates[-1], xmax=tomorrow_x, color='white', linestyle='-', linewidth=2)
    ax.scatter(tomorrow_x, spot_price, color='white', s=70, zorder=5)

    ax.scatter(tomorrow_x, upper_bound, color='#00FF00', s=120, marker='^', zorder=5)
    ax.scatter(tomorrow_x, lower_bound, color='#FF3333', s=120, marker='v', zorder=5)

    ax.hlines(upper_bound, xmin=x_dates[-1], xmax=tomorrow_x, color='#00FF00', linestyle='--', linewidth=1.5, label='Upper Bound (+1 SD)')
    ax.hlines(lower_bound, xmin=x_dates[-1], xmax=tomorrow_x, color='#FF3333', linestyle='--', linewidth=1.5, label='Lower Bound (-1 SD)')

    ax.fill_between([x_dates[-1], tomorrow_x], [spot_price, lower_bound], [spot_price, upper_bound], color='gray', alpha=0.2)

    ax.set_title('NIFTY 50 Implied Daily Expected Move', fontsize=18, color='white', pad=20, fontweight='bold')
    ax.set_ylabel('Nifty 50 Price', color='gray', fontsize=12)
    ax.grid(True, color='#2A2A2A', linestyle=':')
    ax.set_xticks([])
    ax.legend(loc='upper left', facecolor='black', edgecolor='gray', fontsize=10)

    st.pyplot(fig)
    plt.close(fig)

# 3. Correlation
def plot_index_divergence_st():
    st.write("Fetching Nifty 50 and Bank Nifty data...")
    tickers = {"Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK"}
    data = yf.download(list(tickers.values()), period="1y", progress=False)['Close']

    if data.empty:
        st.error("Error: Yahoo Finance returned no data for Nifty or Bank Nifty.")
        return

    data.columns = ['Bank Nifty', 'Nifty 50']
    data = data.dropna()

    normalized_prices = (data / data.iloc[0]) * 100
    log_returns = np.log(data / data.shift(1)).dropna()

    rolling_window = 20
    rolling_correlation = log_returns['Nifty 50'].rolling(window=rolling_window).corr(log_returns['Bank Nifty'])

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

    st.subheader("Correlation Math")
    st.write(f"**Current Nifty 50:** {current_nifty:.2f}")
    st.write(f"**Current Bank Nifty:** {current_bank:.2f}")
    st.write(f"**20-Day Correlation:** {current_corr:.2f}")
    st.markdown(f"**Stat Property:** <span style='color:{color_theme}'>**{stat_property}**</span>", unsafe_allow_html=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=120, gridspec_kw={'height_ratios': [1.5, 1]})

    ax1.plot(normalized_prices.index, normalized_prices['Nifty 50'], color='#00FFFF', linewidth=2, label='Nifty 50 (Normalized)')
    ax1.plot(normalized_prices.index, normalized_prices['Bank Nifty'], color='#FFA500', linewidth=2, label='Bank Nifty (Normalized)')
    ax1.fill_between(normalized_prices.index, normalized_prices['Nifty 50'], normalized_prices['Bank Nifty'],
                     color='gray', alpha=0.2, label='Performance Spread')
    ax1.set_title('Inter-Index Correlation & Divergence (Nifty vs Bank Nifty)', fontsize=18, color='white', pad=15, fontweight='bold')
    ax1.set_ylabel('Normalized Price (Base 100)', color='gray')
    ax1.grid(True, color='#2A2A2A', linestyle=':')
    ax1.legend(loc='upper left', facecolor='black', edgecolor='gray')

    ax2.plot(rolling_correlation.index, rolling_correlation, color='white', linewidth=1.5, label='20-Day Rolling Correlation')
    ax2.axhline(0.80, color='#00FF00', linestyle='--', linewidth=1.5, label='High Correlation (>0.80)')
    ax2.axhline(0.50, color='#FF3333', linestyle='--', linewidth=1.5, label='Severe Divergence (<0.50)')
    ax2.fill_between(rolling_correlation.index, 0.50, rolling_correlation,
                     where=(rolling_correlation < 0.50), color='#FF3333', alpha=0.3, interpolate=True)
    ax2.set_ylabel('Pearson Correlation (r)', color='gray')
    ax2.set_ylim(-0.2, 1.1)
    ax2.grid(True, color='#2A2A2A', linestyle=':')
    ax2.legend(loc='lower left', facecolor='black', edgecolor='gray')

    st.pyplot(fig)
    plt.close(fig)

# 4. Volatility Cone
def plot_volatility_cone_st():
    st.write("Fetching data for Nifty 50...")
    ticker = "^NSEI"
    data = yf.download(ticker, start="2015-01-01", end="2026-04-30")

    if data.empty:
        st.error("Error: Yahoo Finance returned no data for Nifty 50.")
        return

    data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))

    windows = [10, 20, 30, 60, 90, 120, 180, 252]
    max_vol = []
    min_vol = []
    median_vol = []
    current_vol = []

    for window in windows:
        rolling_vol = data['Returns'].rolling(window=window).std() * np.sqrt(252)
        max_vol.append(rolling_vol.max() * 100)
        min_vol.append(rolling_vol.min() * 100)
        median_vol.append(rolling_vol.median() * 100)
        current_vol.append(rolling_vol.iloc[-1] * 100)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(windows, max_vol, marker='o', color='red', linewidth=2, label='Maximum Volatility')
    ax.plot(windows, min_vol, marker='o', color='limegreen', linewidth=2, label='Minimum Volatility')
    ax.plot(windows, median_vol, marker='s', color='white', linewidth=1.5, linestyle='--', label='Median Volatility')
    ax.plot(windows, current_vol, marker='X', color='yellow', linewidth=3, markersize=10, label='Current Volatility')

    ax.fill_between(windows, min_vol, max_vol, color='gray', alpha=0.2)

    ax.set_title(f'Volatility Cone for Nifty 50 ({ticker})', fontsize=18, fontweight='bold', color='white')
    ax.set_xlabel('Time Window (Trading Days)', fontsize=14)
    ax.set_ylabel('Annualized Volatility (%)', fontsize=14)
    ax.set_xticks(windows)
    ax.grid(color='gray', linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=12)

    st.pyplot(fig)
    plt.close(fig)

# 5. FYERS Smart Profile (OI & Volume Fallback)
def plot_fyers_oi_profile_st(access_token_st, client_id_st):
    if not access_token_st or not client_id_st:
        st.error("Fyers API Client ID or Access Token is missing. Please configure Streamlit secrets or provide them.")
        return

    st.subheader("Fyers Smart Profile (OI & Volume Fallback)")
    st.write("Enter parameters for Fyers OI Profile:")

    underlying = st.selectbox("Underlying Index", ["NIFTY", "BANKNIFTY"], index=0) # Add other indices as needed
    expiry_str = st.text_input("Expiry (e.g., 26JUN or 26618)", "26JUN") # User input for expiry
    spot_price_input = st.number_input("Closest Live Spot Price", min_value=10000.0, max_value=40000.0, value=23300.0, step=100.0)
    strike_step = st.selectbox("Strike Step", [50, 100, 25, 200], index=0)

    spot_price = float(spot_price_input)

    if st.button("Fetch Fyers OI Profile"):
        with st.spinner(f"Fetching Live Data from Fyers for {underlying} (Expiry: {expiry_str})..."):
            fyers = fyersModel.FyersModel(client_id=client_id_st, is_async=False, token=access_token_st, log_path="")

            center_strike = int(round(spot_price / strike_step) * strike_step)
            strikes = [center_strike + (i * strike_step) for i in range(-15, 16)]

            symbols = []
            for strike in strikes:
                symbols.append(f"NSE:{underlying}{expiry_str}{strike}CE")
                symbols.append(f"NSE:{underlying}{expiry_str}{strike}PE")

            response = fyers.quotes(data={"symbols": ",".join(symbols)})

            if response.get('s') != 'ok':
                st.error(f"Error fetching data from Fyers: {response}")
                return

            df_oi = pd.DataFrame({'Strike': strikes, 'Call_Data': np.zeros(len(strikes)), 'Put_Data': np.zeros(len(strikes))}).set_index('Strike')
            valid_data_found = False
            metric_used = "Open Interest"

            for item in response['d']:
                sym = item['n']
                if item.get('s') == 'ok' and 'v' in item:
                    if 'open_interest' in item['v']:
                        val = item['v']['open_interest']
                    elif 'volume' in item['v']:
                        val = item['v']['volume']
                        metric_used = "Volume"
                    else:
                        continue

                    valid_data_found = True
                    try:
                        strike = int(''.join(filter(str.isdigit, sym.split(expiry_str)[-1])))
                        if sym.endswith('CE'): df_oi.loc[strike, 'Call_Data'] = val
                        elif sym.endswith('PE'): df_oi.loc[strike, 'Put_Data'] = val
                    except: pass
                else:
                    if sym == symbols[0]:
                        st.warning(f"❌ FYERS REJECTED THE SYMBOL: {sym}")

            if not valid_data_found:
                st.warning("Could not plot chart: No valid data returned for these strikes.")
                return

            st.write(f"-> Success! Generating {metric_used} Profile...")

            fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
            ax.barh(df_oi.index, df_oi['Call_Data'], height=strike_step*0.6, color='#FF3333', alpha=0.8, label=f'Call {metric_used} (Resistance)')
            ax.barh(df_oi.index, -df_oi['Put_Data'], height=strike_step*0.6, color='#00FF00', alpha=0.8, label=f'Put {metric_used} (Support)')

            ax.axvline(0, color='white', linewidth=1)
            ax.axhline(spot_price, color='#00FFFF', linestyle='--', linewidth=2, label=f'Current Spot (~{spot_price})')

            max_pain = (df_oi['Call_Data'] + df_oi['Put_Data']).idxmax()
            ax.axhline(max_pain, color='yellow', linestyle=':', linewidth=2, label=f'Highest {metric_used} Concentration: {max_pain}')

            ax.set_title(f'{underlying} {metric_used} Profile (Expiry: {expiry_str})', fontsize=16, fontweight='bold')
            ax.set_xticklabels([int(abs(tick)) for tick in ax.get_xticks()])
            ax.legend(facecolor='black'); ax.grid(True, linestyle=':')
            st.pyplot(fig)
            plt.close(fig)

# 6. Volatility Risk Premium (VRP)
def plot_vrp_st():
    st.write("Fetching Nifty 50 and India VIX data (Last 6 Months)...")
    try:
        nifty_data = yf.download("^NSEI", period="6mo", progress=False)
        vix_data = yf.download("^INDIAVIX", period="6mo", progress=False)

        if nifty_data.empty or vix_data.empty:
            st.error("Error: Yahoo Finance returned no data for VRP calculation.")
            return
    except Exception as e:
        st.error(f"Download failed for VRP: {e}")
        return

    nifty_close = nifty_data['Close'].squeeze()
    vix_close = vix_data['Close'].squeeze()

    st.write("Calculating 20-Day Realized Volatility and VRP Spread...")
    log_returns = np.log(nifty_close / nifty_close.shift(1))
    historical_vol = log_returns.rolling(window=20).std() * np.sqrt(252) * 100

    df = pd.DataFrame({
        'VIX': vix_close,
        'HV': historical_vol
    }).dropna()

    df['VRP'] = df['VIX'] - df['HV']

    current_vix = float(df['VIX'].iloc[-1])
    current_hv = float(df['HV'].iloc[-1])
    current_vrp = float(df['VRP'].iloc[-1])

    if current_vrp > 0:
        regime = "POSITIVE VRP"
        stat_property = "Implied Risk > Realized Risk (Premium Expansion)"
        color_theme = '#00FF00'
    else:
        regime = "NEGATIVE VRP"
        stat_property = "Realized Risk > Implied Risk (Premium Compression)"
        color_theme = '#FF3333'

    st.subheader("Volatility Risk Premium (VRP)")
    st.write(f"**VIX (Expected):** {current_vix:.2f}%")
    st.write(f"**20-Day HV (Actual):** {current_hv:.2f}%")
    st.write(f"**VRP Spread:** {current_vrp:+.2f}%")
    st.markdown(f"**Regime:** <span style='color:{color_theme}'>**{regime}**</span><br>**Stat Property:** {stat_property}", unsafe_allow_html=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=120, gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(df.index, df['VIX'], color='#00FFFF', linewidth=2, label='Implied Volatility (India VIX)')
    ax1.plot(df.index, df['HV'], color='#FFA500', linewidth=2, label='Realized Volatility (20-Day HV)')
    ax1.fill_between(df.index, df['HV'], df['VIX'], where=(df['VIX'] > df['HV']), color='green', alpha=0.3, interpolate=True)
    ax1.fill_between(df.index, df['HV'], df['VIX'], where=(df['VIX'] <= df['HV']), color='red', alpha=0.3, interpolate=True)
    ax1.set_title('NIFTY 50 Volatility Risk Premium (VRP)', fontsize=18, color='white', pad=15, fontweight='bold')
    ax1.set_ylabel('Annualized Volatility (%)', color='gray')
    ax1.grid(True, color='#2A2A2A', linestyle=':')
    ax1.legend(loc='upper left', facecolor='black', edgecolor='gray')

    bar_colors = np.where(df['VRP'] > 0, '#00FF00', '#FF3333')
    ax2.bar(df.index, df['VRP'], color=bar_colors, alpha=0.7, width=1)
    ax2.axhline(0, color='white', linewidth=1)
    ax2.set_ylabel('VRP Spread (%)', color='gray')
    ax2.grid(True, color='#2A2A2A', linestyle=':')

    st.pyplot(fig)
    plt.close(fig)

# 7. Hurst Exponent
def calculate_hurst_st(ts):
    if len(ts) < 20:
        return np.nan
    lags = range(2, 20)
    tau = [lag for lag in lags]
    ts_arr = ts.values
    reg = [np.std(ts_arr[lag:] - ts_arr[:-lag]) for lag in lags]
    poly = np.polyfit(np.log(tau), np.log(reg), 1)
    return poly[0]

def plot_hurst_regime_st():
    st.write("Fetching Nifty 50 data...")
    data = yf.download("^NSEI", period="1y", progress=False)

    if data.empty:
        st.error("Error: Yahoo Finance returned no data for Hurst Exponent.")
        return

    nifty_close = data['Close'].squeeze()

    st.write("Calculating 60-Day Rolling Hurst Exponent...")
    log_prices = np.log(nifty_close)
    hurst_series = log_prices.rolling(window=60).apply(calculate_hurst_st, raw=False)

    df = pd.DataFrame({'Close': nifty_close, 'Hurst': hurst_series}).dropna()

    current_price = float(df['Close'].iloc[-1])
    current_hurst = float(df['Hurst'].iloc[-1])

    if current_hurst < 0.45:
        regime = "MEAN REVERTING"
        stat_property = "Range-Bound Action / Volatility Compression"
        color_theme = '#FF3333'
    elif current_hurst > 0.55:
        regime = "TRENDING"
        stat_property = "Directional Movement / Momentum Expansion"
        color_theme = '#00FF00'
    else:
        regime = "RANDOM WALK"
        stat_property = "Unpredictable Noise / Transition Phase"
        color_theme = '#FFA500'

    st.subheader("Hurst Exponent (Market Regime)")
    st.write(f"**Current Nifty:** {current_price:.2f}")
    st.write(f"**Hurst Exponent (H):** {current_hurst:.3f}")
    st.markdown(f"**Regime:** <span style='color:{color_theme}'>**{regime}**</span><br>**Stat Property:** {stat_property}", unsafe_allow_html=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=120, gridspec_kw={'height_ratios': [1.5, 1]})

    ax1.plot(df.index, df['Close'], color='white', linewidth=1.5, label='Nifty 50 Close')
    ax1.set_title('NIFTY 50 Market Regime (Hurst Exponent)', fontsize=18, color='white', pad=15, fontweight='bold')
    ax1.set_ylabel('Nifty 50 Price', color='gray')
    ax1.grid(True, color='#2A2A2A', linestyle=':')
    ax1.legend(loc='upper left', facecolor='black', edgecolor='gray')

    ax1.axvspan(df.index[-15], df.index[-1], color=color_theme, alpha=0.1)

    ax2.plot(df.index, df['Hurst'], color='#00FFFF', linewidth=2, label='60-Day Hurst Exponent')
    ax2.axhline(0.55, color='#00FF00', linestyle='--', linewidth=1.5, label='Trending Threshold (>0.55)')
    ax2.axhline(0.45, color='#FF3333', linestyle='--', linewidth=1.5, label='Mean Reverting Threshold (<0.45)')
    ax2.axhline(0.50, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    ax2.fill_between(df.index, 0.55, df['Hurst'], where=(df['Hurst'] > 0.55), color='#00FF00', alpha=0.2, interpolate=True)
    ax2.fill_between(df.index, 0.45, df['Hurst'], where=(df['Hurst'] < 0.45), color='#FF3333', alpha=0.2, interpolate=True)

    ax2.set_ylabel('Hurst Value (H)', color='gray')
    ax2.grid(True, color='#2A2A2A', linestyle=':')
    ax2.set_ylim(0.3, 0.7)
    ax2.legend(loc='upper right', facecolor='black', edgecolor='gray')

    st.pyplot(fig)
    plt.close(fig)

# 8. Liquidity Detector
def plot_liquidity_sweep_st():
    st.write("Fetching 15-minute intraday Nifty 50 data...")
    try:
        data = yf.download("^NSEI", period="30d", interval="15m", progress=False)
        if data.empty:
            st.error("Error: Yahoo Finance returned no data for Liquidity Detector.")
            return
    except Exception as e:
        st.error(f"Download failed for Liquidity Detector: {e}")
        return

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    st.write("Calculating Institutional Order Blocks...")

    window = 20
    data['Prev_High'] = data['High'].rolling(window=window).max().shift(1)
    data['Prev_Low'] = data['Low'].rolling(window=window).min().shift(1)

    data['Supply_Sweep'] = (data['High'] > data['Prev_High']) & (data['Close'] < data['Prev_High'])
    data['Demand_Sweep'] = (data['Low'] < data['Prev_Low']) & (data['Close'] > data['Prev_Low'])

    current_price = float(data['Close'].iloc[-1])

    if data['Supply_Sweep'].iloc[-1]:
        regime = "SUPPLY LIQUIDITY SWEPT"
        stat_property = "Failed Breakout / Institutional Absorption at Highs"
        color_theme = '#FF3333'
    elif data['Demand_Sweep'].iloc[-1]:
        regime = "DEMAND LIQUIDITY SWEPT"
        stat_property = "Failed Breakdown / Institutional Absorption at Lows"
        color_theme = '#00FF00'
    else:
        regime = "PRICE DISCOVERY PHASE"
        stat_property = "Trading Inside Established Structural Bounds"
        color_theme = '#00FFFF'

    st.subheader("Liquidity Sweep & Order Block Detector")
    st.write(f"**Live Spot Price:** {current_price:.2f}")
    st.markdown(f"**Microstructure:** <span style='color:{color_theme}'>**{regime}**</span><br>**Stat Property:** {stat_property}", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

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

    ax.set_title('NIFTY 50 Intraday Liquidity Sweep & Order Block Detector (15m)', fontsize=16, color='white', pad=15, fontweight='bold')
    ax.set_ylabel('Nifty 50 Price', color='gray')
    ax.grid(True, color='#2A2A2A', linestyle=':')
    ax.set_xticks([])

    st.pyplot(fig)
    plt.close(fig)

# 9. Parkinson Estimator
def plot_parkinson_estimator_st():
    st.write("Downloading Nifty 50 data for Parkinson Estimator...")
    data = yf.download('^NSEi', period='1y')

    if data.empty:
        st.error("Error: Yahoo Finance returned no data for Parkinson Estimator.")
        return

    h = data['High'].squeeze()
    l = data['Low'].squeeze()
    c = data['Close'].squeeze()

    N = len(data)

    log_hl = np.log(h / l)
    log_hl_squared = log_hl ** 2
    constant = 1 / (4 * N * np.log(2))
    parkinson_variance = constant * log_hl_squared.sum()
    parkinson_vol = np.sqrt(parkinson_variance) * np.sqrt(252)

    log_returns = np.log(c / c.shift(1))
    c2c_vol = log_returns.std() * np.sqrt(252)

    st.subheader("Parkinson Estimator")
    st.write(f"**Total Trading Days Analyzed (N):** {N}")
    st.write(f"**Parkinson Volatility (True Intraday Risk):** {float(parkinson_vol) * 100:.2f}%")
    st.write(f"**Close-to-Close Volatility (Standard Risk):** {float(c2c_vol) * 100:.2f}%")


# --- Streamlit App Layout (Single Page) ---
st.set_page_config(layout="wide", page_title="Financial Market Tools")
st.title("Financial Market Insights Dashboard")

# Call each tool function sequentially

st.header("1. NIFTY Implied Volatility (IVR & IVP)")
plot_nifty_volatility_st()

st.header("2. NIFTY 50 Implied Daily Expected Move")
plot_expected_move_st()

st.header("3. Inter-Index Correlation & Divergence")
plot_index_divergence_st()

st.header("4. Volatility Cone for Nifty 50")
plot_volatility_cone_st()

st.header("5. Fyers Smart OI/Volume Profile")
# Fyers API authentication logic
fyers_access_token = generate_fyers_token_streamlit()
plot_fyers_oi_profile_st(fyers_access_token, CLIENT_ID)

st.header("6. NIFTY 50 Volatility Risk Premium (VRP)")
plot_vrp_st()

st.header("7. NIFTY 50 Market Regime (Hurst Exponent)")
plot_hurst_regime_st()

st.header("8. NIFTY 50 Intraday Liquidity Sweep & Order Block Detector")
plot_liquidity_sweep_st()

st.header("9. Parkinson Estimator")
plot_parkinson_estimator_st()

# --- Streamlit App Layout (Single Page) ---
st.set_page_config(layout="wide", page_title="Financial Market Tools")
st.title("Financial Market Insights Dashboard")

# Call each tool function sequentially

st.header("1. NIFTY Implied Volatility (IVR & IVP)")
plot_nifty_volatility_st()

st.header("2. NIFTY 50 Implied Daily Expected Move")
plot_expected_move_st()

st.header("3. Inter-Index Correlation & Divergence")
plot_index_divergence_st()

st.header("4. Volatility Cone for Nifty 50")
plot_volatility_cone_st()

st.header("5. Fyers Smart OI/Volume Profile")
# Fyers API authentication logic
fyers_access_token = generate_fyers_token_streamlit()
plot_fyers_oi_profile_st(fyers_access_token, CLIENT_ID)

st.header("6. NIFTY 50 Volatility Risk Premium (VRP)")
plot_vrp_st()

st.header("7. NIFTY 50 Market Regime (Hurst Exponent)")
plot_hurst_regime_st()

st.header("8. NIFTY 50 Intraday Liquidity Sweep & Order Block Detector")
plot_liquidity_sweep_st()

st.header("9. Parkinson Estimator")
plot_parkinson_estimator_st()