# ==========================================
# CELL 3: MASTER FINANCIAL SCRIPTS (ALL 9 TOOLS)
# ==========================================
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from fyers_apiv3 import fyersModel

warnings.filterwarnings('ignore')
plt.style.use('dark_background')

# ------------------------------------------
# 1. IVR & IVP (WITH EDGE READOUT BOX)
# ------------------------------------------
def plot_nifty_volatility():
    # 1. Fetch 1 Year of India VIX Data
    ticker = "^INDIAVIX"
    print(f"Downloading 1-year data for {ticker}...")
    data = yf.download(ticker, period="1y", progress=False)

    if data.empty:
        print("Error: No data retrieved from Yahoo Finance.")
        return

    # Clean the data to a 1D array
    close_prices = data['Close'].squeeze()

    # 2. Calculate Quantitative Metrics
    current_iv = float(close_prices.iloc[-1])
    high_52w = float(close_prices.max())
    low_52w = float(close_prices.min())

    # IV Rank (IVR) Formula
    ivr = ((current_iv - low_52w) / (high_52w - low_52w)) * 100

    # IV Percentile (IVP) Formula
    days_below = (close_prices < current_iv).sum()
    total_days = len(close_prices)
    ivp = (days_below / total_days) * 100

    # Determine Trade Logic based on IVR
    if ivr > 50:
        regime = "HIGH VOLATILITY: Net Short Premium (Credit Spreads)"
        color_theme = '#00FF00' # Neon Green for Go (Selling)
    else:
        regime = "LOW VOLATILITY: Net Long Premium (Debit Spreads)"
        color_theme = '#FF3333' # Neon Red for Stop (Buying)

    print("\n--- Market Structure Math ---")
    print(f"Current VIX: {current_iv:.2f}")
    print(f"IV Rank (IVR): {ivr:.1f}%")
    print(f"IV Percentile (IVP): {ivp:.1f}%")

    # 3. Create the Cinematic Chart
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)

    # Plot the VIX curve
    ax.plot(close_prices.index, close_prices.values, color='#00FFFF', linewidth=1.5, label='India VIX (1Y)')

    # Plot High, Low, and Current lines
    ax.axhline(high_52w, color='red', linestyle='--', alpha=0.5, label=f'52W High: {high_52w:.2f}')
    ax.axhline(low_52w, color='green', linestyle='--', alpha=0.5, label=f'52W Low: {low_52w:.2f}')
    ax.axhline(current_iv, color='white', linestyle='-', linewidth=2, label=f'Current: {current_iv:.2f}')

    # Fill between current and low for visual depth
    ax.fill_between(close_prices.index, low_52w, current_iv, color='white', alpha=0.05)

    # Format the Chart
    ax.set_title('NIFTY Implied Volatility (IVR & IVP)', fontsize=18, color='white', pad=20, fontweight='bold')
    ax.set_ylabel('VIX Level', color='gray', fontsize=12)
    ax.grid(True, color='#2A2A2A', linestyle=':')
    ax.legend(loc='upper right', facecolor='black', edgecolor='gray', fontsize=10)

    # Add the Data Readout Box
    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8, edgecolor=color_theme, linewidth=1.5)
    text_str = (
        f"🎯 IV Rank (IVR): {ivr:.1f}%\n"
        f"📊 IV Percentile (IVP): {ivp:.1f}%\n"
        f"⚡ Current VIX: {current_iv:.2f}\n"
        f"---------------------------\n"
        f"🤖 Edge: {regime}"
    )
    ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, color='white', fontweight='bold')

    plt.tight_layout()



# Run the function
plot_nifty_volatility()

# ------------------------------------------
# 2. Expected Move
# ------------------------------------------
def plot_expected_move():
    print("Fetching Nifty 50 and India VIX data...")
    # 1. Fetch Nifty and VIX Data (Last 1 Month for visual context)
    nifty_data = yf.download("^NSEi", period="1mo", progress=False)
    vix_data = yf.download("^INDIAVIX", period="5d", progress=False)

    if nifty_data.empty or vix_data.empty:
        print("Error: Yahoo Finance returned no data.")
        return

    # Clean data safely
    nifty_close = nifty_data['Close'].squeeze()
    spot_price = float(nifty_close.iloc[-1])
    current_vix = float(vix_data['Close'].squeeze().iloc[-1])

    # 2. Calculate Market Maker Bounds (1 Standard Deviation)
    # VIX is annualized. We divide by sqrt(365) for the daily expected move.
    daily_volatility = (current_vix / 100) * np.sqrt(1/365)
    expected_move_points = spot_price * daily_volatility

    upper_bound = spot_price + expected_move_points
    lower_bound = spot_price - expected_move_points

    print("\n--- Daily Expected Move Math ---")
    print(f"Nifty Spot Price: {spot_price:.2f}")
    print(f"Current VIX: {current_vix:.2f}")
    print(f"Daily Implied Move: ± {expected_move_points:.1f} points")
    print(f"Upper Target (+1 SD): {upper_bound:.2f}")
    print(f"Lower Target (-1 SD): {lower_bound:.2f}")

    # 3. Create the Cinematic Chart
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)

    # Plot the last 15 days of Nifty to show the current trend
    recent_nifty = nifty_close.tail(15)
    x_dates = np.arange(len(recent_nifty))

    ax.plot(x_dates, recent_nifty.values, color='#00FFFF', linewidth=2, marker='o', label='Nifty 50 Close')

    # Define 'Tomorrow' on the x-axis
    tomorrow_x = len(recent_nifty)

    # Plot the Spot Price line extending to tomorrow
    ax.hlines(spot_price, xmin=x_dates[-1], xmax=tomorrow_x, color='white', linestyle='-', linewidth=2)
    ax.scatter(tomorrow_x, spot_price, color='white', s=70, zorder=5)

    # Plot the Expected Move Bounds for Tomorrow
    ax.scatter(tomorrow_x, upper_bound, color='#00FF00', s=120, marker='^', zorder=5)
    ax.scatter(tomorrow_x, lower_bound, color='#FF3333', s=120, marker='v', zorder=5)

    ax.hlines(upper_bound, xmin=x_dates[-1], xmax=tomorrow_x, color='#00FF00', linestyle='--', linewidth=1.5, label='Upper Bound (+1 SD)')
    ax.hlines(lower_bound, xmin=x_dates[-1], xmax=tomorrow_x, color='#FF3333', linestyle='--', linewidth=1.5, label='Lower Bound (-1 SD)')

    # Fill the 'Cone' of probability
    ax.fill_between([x_dates[-1], tomorrow_x], [spot_price, lower_bound], [spot_price, upper_bound], color='gray', alpha=0.2)

    # Chart Formatting
    ax.set_title('NIFTY 50 Implied Daily Expected Move', fontsize=18, color='white', pad=20, fontweight='bold')
    ax.set_ylabel('Nifty 50 Price', color='gray', fontsize=12)
    ax.grid(True, color='#2A2A2A', linestyle=':')
    ax.set_xticks([]) # Hide standard date ticks for clean look
    ax.legend(loc='upper left', facecolor='black', edgecolor='gray', fontsize=10)

    # Data Readout Box
    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8, edgecolor='white', linewidth=1.5)
    text_str = (
        f"⚡ Current VIX: {current_vix:.2f}\n"
        f"🎯 Spot Price: {spot_price:.2f}\n"
        f"📏 Expected Move: ± {expected_move_points:.1f} points\n"
        f"---------------------------\n"
        f"🟢 Safe Short Call Strike: > {upper_bound:.0f}\n"
        f"🔴 Safe Short Put Strike: < {lower_bound:.0f}"
    )
    ax.text(0.02, 0.45, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', bbox=props, color='white', fontweight='bold')

    plt.tight_layout()

    # Render the plot directly in the notebook
    plt.show()

# Run the function
plot_expected_move()

# ------------------------------------------
# 3. Correlation
# ------------------------------------------
def plot_index_divergence():
    print("Fetching Nifty 50 and Bank Nifty data...")
    # 1. Fetch 1 Year of Data for both indices
    tickers = {"Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK"}
    data = yf.download(list(tickers.values()), period="1y", progress=False)['Close']

    if data.empty:
        print("Error: Yahoo Finance returned no data.")
        return

    # Rename columns for easier access
    data.columns = ['Bank Nifty', 'Nifty 50']

    # Drop any missing days to ensure alignment
    data = data.dropna()

    # 2. Quantitative Math
    # Normalize prices to 100 at the start of the timeframe to visualize relative performance
    normalized_prices = (data / data.iloc[0]) * 100

    # Calculate daily logarithmic returns
    log_returns = np.log(data / data.shift(1)).dropna()

    # Calculate the 20-Day Rolling Pearson Correlation
    rolling_window = 20
    rolling_correlation = log_returns['Nifty 50'].rolling(window=rolling_window).corr(log_returns['Bank Nifty'])

    # Get current metrics
    current_nifty = float(data['Nifty 50'].iloc[-1])
    current_bank = float(data['Bank Nifty'].iloc[-1])
    current_corr = float(rolling_correlation.iloc[-1])

    # SEBI-Compliant Statistical Logic
    if current_corr > 0.80:
        regime = "HIGH CORRELATION"
        stat_property = "Synchronized Movement / Low Dispersion"
        color_theme = '#00FF00' # Green (Normal)
    elif current_corr < 0.50:
        regime = "SEVERE DIVERGENCE"
        stat_property = "Sector Rotation / High Dispersion Warning"
        color_theme = '#FF3333' # Red (Warning)
    else:
        regime = "MODERATE DIVERGENCE"
        stat_property = "Decoupling / Monitoring Phase"
        color_theme = '#FFA500' # Orange (Caution)

    print("\n--- Correlation Math ---")
    print(f"Current Nifty 50: {current_nifty:.2f}")
    print(f"Current Bank Nifty: {current_bank:.2f}")
    print(f"20-Day Correlation: {current_corr:.2f}")

    # 3. Create the Cinematic Chart
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=120, gridspec_kw={'height_ratios': [1.5, 1]})

    # --- Top Panel: Normalized Relative Performance ---
    ax1.plot(normalized_prices.index, normalized_prices['Nifty 50'], color='#00FFFF', linewidth=2, label='Nifty 50 (Normalized)')
    ax1.plot(normalized_prices.index, normalized_prices['Bank Nifty'], color='#FFA500', linewidth=2, label='Bank Nifty (Normalized)')

    # Fill the spread between them to visualize the divergence gap
    ax1.fill_between(normalized_prices.index, normalized_prices['Nifty 50'], normalized_prices['Bank Nifty'],
                     color='gray', alpha=0.2, label='Performance Spread')

    ax1.set_title('Inter-Index Correlation & Divergence (Nifty vs Bank Nifty)', fontsize=18, color='white', pad=15, fontweight='bold')
    ax1.set_ylabel('Normalized Price (Base 100)', color='gray')
    ax1.grid(True, color='#2A2A2A', linestyle=':')
    ax1.legend(loc='upper left', facecolor='black', edgecolor='gray')

    # --- Bottom Panel: Rolling 20-Day Correlation ---
    ax2.plot(rolling_correlation.index, rolling_correlation, color='white', linewidth=1.5, label='20-Day Rolling Correlation')

    # Draw Regime Threshold Boundaries
    ax2.axhline(0.80, color='#00FF00', linestyle='--', linewidth=1.5, label='High Correlation (>0.80)')
    ax2.axhline(0.50, color='#FF3333', linestyle='--', linewidth=1.5, label='Severe Divergence (<0.50)')

    # Fill the danger zone (Correlation dropping below 0.50)
    ax2.fill_between(rolling_correlation.index, 0.50, rolling_correlation,
                     where=(rolling_correlation < 0.50), color='#FF3333', alpha=0.3, interpolate=True)

    ax2.set_ylabel('Pearson Correlation (r)', color='gray')
    ax2.set_ylim(-0.2, 1.1)
    ax2.grid(True, color='#2A2A2A', linestyle=':')
    ax2.legend(loc='lower left', facecolor='black', edgecolor='gray')

    # 4. Data Readout Box (SEBI Compliant)
    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9, edgecolor=color_theme, linewidth=1.5)
    text_str = (
        f"Nifty 50: {current_nifty:.2f}\n"
        f"Bank Nifty: {current_bank:.2f}\n"
        f"20-Day Correlation: {current_corr:.2f}\n"
        f"---------------------------\n"
        f"Stat Property: {stat_property}"
    )

    # Place text box on the top panel
    ax1.text(0.02, 0.05, text_str, transform=ax1.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props, color='white', fontweight='bold')

    plt.tight_layout()

    # Display the plot directly in the notebook output
    plt.show()

# Run the function
plot_index_divergence()
# ------------------------------------------
# 4. Volatility Cone (PRO GLOW AESTHETIC)
# ------------------------------------------

# STEP 2: Fetch Data (We will use the Nifty 50 Index as an example)
ticker = "^NSEI"
print(f"Fetching data for {ticker}...")
data = yf.download(ticker, start="2015-01-01", end="2026-04-30")

# STEP 3: Calculate Daily Log Returns
# Formula: ln(Today's Close / Yesterday's Close)
data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))

# STEP 4: Define our Time Windows (in trading days)
windows = [10, 20, 30, 60, 90, 120, 180, 252]

# Create empty lists to store our data points for the cone
max_vol = []
min_vol = []
median_vol = []
current_vol = []

# STEP 5: The Engine - Calculate Volatility for each window
for window in windows:
    # Calculate rolling standard deviation and annualize it (* sqrt(252))
    rolling_vol = data['Returns'].rolling(window=window).std() * np.sqrt(252)

    # Extract the required data points
    max_vol.append(rolling_vol.max() * 100)       # Convert to percentage
    min_vol.append(rolling_vol.min() * 100)
    median_vol.append(rolling_vol.median() * 100)
    current_vol.append(rolling_vol.iloc[-1] * 100) # The very last value

# STEP 6: Plotting the Volatility Cone (Cinematic Dark Mode)
plt.style.use('dark_background')
plt.figure(figsize=(12, 7))

# Plot the lines with high-contrast colors
plt.plot(windows, max_vol, marker='o', color='red', linewidth=2, label='Maximum Volatility')
plt.plot(windows, min_vol, marker='o', color='limegreen', linewidth=2, label='Minimum Volatility')
plt.plot(windows, median_vol, marker='s', color='white', linewidth=1.5, linestyle='--', label='Median Volatility')
plt.plot(windows, current_vol, marker='X', color='yellow', linewidth=3, markersize=10, label='Current Volatility')

# Fill the area inside the cone for better visualization
plt.fill_between(windows, min_vol, max_vol, color='gray', alpha=0.2)

# Formatting the Chart
plt.title(f'Volatility Cone for Nifty 50 ({ticker})', fontsize=18, fontweight='bold', color='white')
plt.xlabel('Time Window (Trading Days)', fontsize=14)
plt.ylabel('Annualized Volatility (%)', fontsize=14)
plt.xticks(windows)
plt.grid(color='gray', linestyle=':', alpha=0.5)
plt.legend(loc='upper right', fontsize=12)

# Display the chart
plt.tight_layout()
plt.show()

# ------------------------------------------
# 5. FYERS Smart Profile (OI & Volume Fallback)
# ------------------------------------------
def plot_fyers_oi_profile(access_token, client_id, underlying="NIFTY", expiry_str="26JUN", spot_price=23300, strike_step=50):
    if not access_token:
        print("Missing Access Token!")
        return

    print(f"Fetching Live Data from Fyers for {underlying} (Expiry: {expiry_str})...")
    fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")

    center_strike = int(round(spot_price / strike_step) * strike_step)
    strikes = [center_strike + (i * strike_step) for i in range(-15, 16)]

    symbols = []
    for strike in strikes:
        symbols.append(f"NSE:{underlying}{expiry_str}{strike}CE")
        symbols.append(f"NSE:{underlying}{expiry_str}{strike}PE")

    response = fyers.quotes(data={"symbols": ",".join(symbols)})

    if response.get('s') != 'ok':
        print(f"Error fetching data: {response}")
        return

    df_oi = pd.DataFrame({'Strike': strikes, 'Call_Data': np.zeros(len(strikes)), 'Put_Data': np.zeros(len(strikes))}).set_index('Strike')
    valid_data_found = False
    metric_used = "Open Interest" # Default assumption

    for item in response['d']:
        sym = item['n']
        if item.get('s') == 'ok' and 'v' in item:

            # THE SMART FALLBACK LOGIC
            if 'open_interest' in item['v']:
                val = item['v']['open_interest']
            elif 'volume' in item['v']:
                val = item['v']['volume']
                metric_used = "Volume" # Fyers hid OI, falling back to Volume!
            else:
                continue # Neither exists, skip

            valid_data_found = True
            try:
                strike = int(''.join(filter(str.isdigit, sym.split(expiry_str)[-1])))
                if sym.endswith('CE'): df_oi.loc[strike, 'Call_Data'] = val
                elif sym.endswith('PE'): df_oi.loc[strike, 'Put_Data'] = val
            except: pass
        else:
            if sym == symbols[0]:
                print(f"\n❌ FYERS REJECTED THE SYMBOL: {sym}")

    if not valid_data_found:
        print("\nCould not plot chart: No valid data returned for these strikes.")
        return

    print(f"-> Success! Generating {metric_used} Profile...")

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    ax.barh(df_oi.index, df_oi['Call_Data'], height=strike_step*0.6, color='#FF3333', alpha=0.8, label=f'Call {metric_used} (Resistance)')
    ax.barh(df_oi.index, -df_oi['Put_Data'], height=strike_step*0.6, color='#00FF00', alpha=0.8, label=f'Put {metric_used} (Support)')

    ax.axvline(0, color='white', linewidth=1)
    ax.axhline(spot_price, color='#00FFFF', linestyle='--', linewidth=2, label=f'Current Spot (~{spot_price})')

    max_pain = (df_oi['Call_Data'] + df_oi['Put_Data']).idxmax()
    ax.axhline(max_pain, color='yellow', linestyle=':', linewidth=2, label=f'Highest {metric_used} Concentration: {max_pain}')

    ax.set_title(f'{underlying} {metric_used} Profile (Expiry: {expiry_str})', fontsize=16, fontweight='bold')
    ax.set_xticklabels([int(abs(tick)) for tick in ax.get_xticks()])
    ax.legend(facecolor='black'); ax.grid(True, linestyle=':')
    plt.tight_layout(); plt.show()

    # Display the chart
plt.tight_layout()
plt.show()
# ------------------------------------------
# 6. Volatility Risk Premium (VRP)
# ------------------------------------------
def plot_vrp():
    print("Fetching Nifty 50 and India VIX data (Last 6 Months)...")

    # 1. Fetch Nifty and VIX Data
    try:
        nifty_data = yf.download("^NSEI", period="6mo", progress=False)
        vix_data = yf.download("^INDIAVIX", period="6mo", progress=False)

        if nifty_data.empty or vix_data.empty:
            print("Error: Yahoo Finance returned no data.")
            return
    except Exception as e:
        print(f"Download failed: {e}")
        return

    # Clean the data to 1D arrays
    nifty_close = nifty_data['Close'].squeeze()
    vix_close = vix_data['Close'].squeeze()

    print("Calculating 20-Day Realized Volatility and VRP Spread...")

    # 2. Calculate 20-Day Historical Volatility (Realized Volatility)
    # Step A: Calculate daily logarithmic returns
    log_returns = np.log(nifty_close / nifty_close.shift(1))

    # Step B: Calculate 20-day rolling standard deviation and annualize it (252 trading days)
    historical_vol = log_returns.rolling(window=20).std() * np.sqrt(252) * 100

    # 3. Align the Data & Calculate VRP
    # Combine into one DataFrame to drop the first 20 days of NaNs cleanly
    df = pd.DataFrame({
        'VIX': vix_close,
        'HV': historical_vol
    }).dropna()

    # The Core VRP Formula
    df['VRP'] = df['VIX'] - df['HV']

    # Get current metrics
    current_vix = float(df['VIX'].iloc[-1])
    current_hv = float(df['HV'].iloc[-1])
    current_vrp = float(df['VRP'].iloc[-1])

    # 4. SEBI-Compliant Educational Logic
    if current_vrp > 0:
        regime = "POSITIVE VRP"
        stat_property = "Implied Risk > Realized Risk (Premium Expansion)"
        color_theme = '#00FF00' # Green
    else:
        regime = "NEGATIVE VRP"
        stat_property = "Realized Risk > Implied Risk (Premium Compression)"
        color_theme = '#FF3333' # Red

    # 5. Create the Cinematic Chart
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=120, gridspec_kw={'height_ratios': [2, 1]})

    # --- Top Panel: VIX vs Realized Volatility ---
    ax1.plot(df.index, df['VIX'], color='#00FFFF', linewidth=2, label='Implied Volatility (India VIX)')
    ax1.plot(df.index, df['HV'], color='#FFA500', linewidth=2, label='Realized Volatility (20-Day HV)')

    # Fill the premium/discount gaps
    ax1.fill_between(df.index, df['HV'], df['VIX'], where=(df['VIX'] > df['HV']), color='green', alpha=0.3, interpolate=True)
    ax1.fill_between(df.index, df['HV'], df['VIX'], where=(df['VIX'] <= df['HV']), color='red', alpha=0.3, interpolate=True)

    ax1.set_title('NIFTY 50 Volatility Risk Premium (VRP)', fontsize=18, color='white', pad=15, fontweight='bold')
    ax1.set_ylabel('Annualized Volatility (%)', color='gray')
    ax1.grid(True, color='#2A2A2A', linestyle=':')
    ax1.legend(loc='upper left', facecolor='black', edgecolor='gray')

    # --- Bottom Panel: The VRP Histogram ---
    # Create an array of colors based on the condition
    bar_colors = np.where(df['VRP'] > 0, '#00FF00', '#FF3333')
    ax2.bar(df.index, df['VRP'], color=bar_colors, alpha=0.7, width=1)
    ax2.axhline(0, color='white', linewidth=1)

    ax2.set_ylabel('VRP Spread (%)', color='gray')
    ax2.grid(True, color='#2A2A2A', linestyle=':')

    # 6. Data Readout Box (SEBI Compliant)
    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9, edgecolor=color_theme, linewidth=1.5)
    text_str = (
        f"VIX (Expected): {current_vix:.2f}%\n"
        f"20-Day HV (Actual): {current_hv:.2f}%\n"
        f"VRP Spread: {current_vrp:+.2f}%\n"
        f"---------------------------\n"
        f"Regime: {regime}\n"
        f"Stat Property: {stat_property}"
    )

    # Place text box on the top panel
    ax1.text(0.02, 0.05, text_str, transform=ax1.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props, color='white', fontweight='bold')

    plt.tight_layout()

    # Render the plot directly in the notebook
    plt.show()

# Run the function
plot_vrp()
# ------------------------------------------
# 7. Hurst Exponent
# ------------------------------------------
# Institutional Math: Calculate the Hurst Exponent using Rescaled Variance
def calculate_hurst(ts):
    if len(ts) < 20:
        return np.nan
    lags = range(2, 20)
    tau = [lag for lag in lags]
    ts_arr = ts.values

    # Standard deviation of the price differences
    reg = [np.std(ts_arr[lag:] - ts_arr[:-lag]) for lag in lags]

    # Linear fit of log(lags) vs log(std_dev) - the slope is the Hurst Exponent
    poly = np.polyfit(np.log(tau), np.log(reg), 1)
    return poly[0]

def plot_hurst_regime():
    print("Fetching Nifty 50 data...")
    # Fetch Nifty Data (1 Year to get a good rolling average)
    data = yf.download("^NSEi", period="1y", progress=False)

    if data.empty:
        print("Error: Yahoo Finance returned no data.")
        return

    nifty_close = data['Close'].squeeze()

    print("Calculating 60-Day Rolling Hurst Exponent...")
    # Calculate the Rolling 60-Day Hurst Exponent
    # We use log prices for mathematical accuracy
    log_prices = np.log(nifty_close)
    hurst_series = log_prices.rolling(window=60).apply(calculate_hurst, raw=False)

    # Drop NaNs to clean up the plotting timeline
    df = pd.DataFrame({'Close': nifty_close, 'Hurst': hurst_series}).dropna()

    current_price = float(df['Close'].iloc[-1])
    current_hurst = float(df['Hurst'].iloc[-1])

    # Educational/Statistical Logic (SEBI Compliant)
    if current_hurst < 0.45:
        regime = "MEAN REVERTING"
        stat_property = "Range-Bound Action / Volatility Compression"
        color_theme = '#FF3333' # Red (Stop trending)
    elif current_hurst > 0.55:
        regime = "TRENDING"
        stat_property = "Directional Movement / Momentum Expansion"
        color_theme = '#00FF00' # Green (Go with trend)
    else:
        regime = "RANDOM WALK"
        stat_property = "Unpredictable Noise / Transition Phase"
        color_theme = '#FFA500' # Orange (Caution)

    # Create the Cinematic Chart
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=120, gridspec_kw={'height_ratios': [1.5, 1]})

    # --- Top Panel: Nifty 50 Price ---
    ax1.plot(df.index, df['Close'], color='white', linewidth=1.5, label='Nifty 50 Close')
    ax1.set_title('NIFTY 50 Market Regime (Hurst Exponent)', fontsize=18, color='white', pad=15, fontweight='bold')
    ax1.set_ylabel('Nifty 50 Price', color='gray')
    ax1.grid(True, color='#2A2A2A', linestyle=':')
    ax1.legend(loc='upper left', facecolor='black', edgecolor='gray')

    # Color-code the price background based on regime for the last 15 days
    ax1.axvspan(df.index[-15], df.index[-1], color=color_theme, alpha=0.1)

    # --- Bottom Panel: The Rolling Hurst Exponent ---
    ax2.plot(df.index, df['Hurst'], color='#00FFFF', linewidth=2, label='60-Day Hurst Exponent')

    # Draw Regime Boundaries
    ax2.axhline(0.55, color='#00FF00', linestyle='--', linewidth=1.5, label='Trending Threshold (>0.55)')
    ax2.axhline(0.45, color='#FF3333', linestyle='--', linewidth=1.5, label='Mean Reverting Threshold (<0.45)')
    ax2.axhline(0.50, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    # Fill regimes for visual clarity
    ax2.fill_between(df.index, 0.55, df['Hurst'], where=(df['Hurst'] > 0.55), color='#00FF00', alpha=0.2, interpolate=True)
    ax2.fill_between(df.index, 0.45, df['Hurst'], where=(df['Hurst'] < 0.45), color='#FF3333', alpha=0.2, interpolate=True)

    ax2.set_ylabel('Hurst Value (H)', color='gray')
    ax2.grid(True, color='#2A2A2A', linestyle=':')
    ax2.set_ylim(0.3, 0.7)
    ax2.legend(loc='upper right', facecolor='black', edgecolor='gray')

    # Data Readout Box
    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9, edgecolor=color_theme, linewidth=1.5)
    text_str = (
        f"Current Nifty: {current_price:.2f}\n"
        f"Hurst Exponent (H): {current_hurst:.3f}\n"
        f"Regime: {regime}\n"
        f"---------------------------\n"
        f"Stat Property: {stat_property}"
    )

    # Place text box on the top panel
    ax1.text(0.02, 0.05, text_str, transform=ax1.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props, color='white', fontweight='bold')

    plt.tight_layout()

    # Render the plot directly in the notebook
    plt.show()

# Execute the function
plot_hurst_regime()
# ------------------------------------------
# 7. Liquidity Detector (PRO CANDLESTICK VERSION)
# ------------------------------------------
def plot_liquidity_sweep():
    print("Fetching 15-minute intraday Nifty 50 data...")

    # 1. Fetch data for the last 5 days
    try:
        data = yf.download("^NSEi", period="30d", interval="15m", progress=False)
        if data.empty:
            print("Error: Yahoo Finance returned no data.")
            return
    except Exception as e:
        print(f"Download failed: {e}")
        return

    # Flatten columns if yfinance returns a multi-index structure
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    print("Calculating Institutional Order Blocks...")

    # 2. Quantitative Math: Liquidity Sweep Logic
    window = 20 # 20-period rolling swing highs/lows

    # Define previous structural extremes
    data['Prev_High'] = data['High'].rolling(window=window).max().shift(1)
    data['Prev_Low'] = data['Low'].rolling(window=window).min().shift(1)

    # Supply Sweep (Bearish rejection): Pierced the structural high, but closed below it.
    data['Supply_Sweep'] = (data['High'] > data['Prev_High']) & (data['Close'] < data['Prev_High'])

    # Demand Sweep (Bullish rejection): Pierced the structural low, but closed above it.
    data['Demand_Sweep'] = (data['Low'] < data['Prev_Low']) & (data['Close'] > data['Prev_Low'])

    current_price = float(data['Close'].iloc[-1])

    # Determine SEBI-Compliant Statistical Regime
    if data['Supply_Sweep'].iloc[-1]:
        regime = "SUPPLY LIQUIDITY SWEPT"
        stat_property = "Failed Breakout / Institutional Absorption at Highs"
        color_theme = '#FF3333' # Red
    elif data['Demand_Sweep'].iloc[-1]:
        regime = "DEMAND LIQUIDITY SWEPT"
        stat_property = "Failed Breakdown / Institutional Absorption at Lows"
        color_theme = '#00FF00' # Green
    else:
        regime = "PRICE DISCOVERY PHASE"
        stat_property = "Trading Inside Established Structural Bounds"
        color_theme = '#00FFFF' # Cyan

    # 3. Create the Cinematic Candlestick Chart
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)

    # Filter to the last 60 candles for visual clarity (about 2.5 trading days)
    plot_data = data.tail(60).copy()
    plot_data['Index'] = np.arange(len(plot_data)) # Numerical x-axis for clean plotting

    # Custom Candlestick Plotting Logic
    up = plot_data[plot_data['Close'] >= plot_data['Open']]
    down = plot_data[plot_data['Close'] < plot_data['Open']]

    # Plot wicks (High to Low)
    ax.vlines(up['Index'], up['Low'], up['High'], color='#00FF00', linewidth=1.5)
    ax.vlines(down['Index'], down['Low'], down['High'], color='#FF3333', linewidth=1.5)

    # Plot bodies (Open to Close)
    ax.bar(up['Index'], up['Close'] - up['Open'], bottom=up['Open'], color='#00FF00', width=0.6)
    ax.bar(down['Index'], down['Open'] - down['Close'], bottom=down['Close'], color='#FF3333', width=0.6)

    # Highlight Sweeps with visual markers
    for idx, row in plot_data.iterrows():
        if row['Supply_Sweep']:
            ax.scatter(row['Index'], row['High'] + 10, marker='v', color='#FF3333', s=150, zorder=5)
            ax.axhline(row['Prev_High'], color='#FF3333', linestyle='--', alpha=0.5, linewidth=1)
        if row['Demand_Sweep']:
            ax.scatter(row['Index'], row['Low'] - 10, marker='^', color='#00FF00', s=150, zorder=5)
            ax.axhline(row['Prev_Low'], color='#00FF00', linestyle='--', alpha=0.5, linewidth=1)

    # Chart Formatting
    ax.set_title('NIFTY 50 Intraday Liquidity Sweep & Order Block Detector (15m)', fontsize=16, color='white', pad=15, fontweight='bold')
    ax.set_ylabel('Nifty 50 Price', color='gray')
    ax.grid(True, color='#2A2A2A', linestyle=':')
    ax.set_xticks([]) # Hide numerical x-ticks for a cleaner aesthetic

    # SEBI-Compliant Data Readout Box
    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9, edgecolor=color_theme, linewidth=1.5)
    text_str = (
        f"Live Spot Price: {current_price:.2f}\n"
        f"---------------------------\n"
        f"Microstructure: {regime}\n"
        f"Stat Property: {stat_property}"
    )

    ax.text(0.02, 0.05, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props, color='white', fontweight='bold')

    plt.tight_layout()

    # Render the plot directly in the notebook output
    plt.show()

# Run the function to generate the chart
plot_liquidity_sweep()
# ------------------------------------------
# 9. Parkinson Estimator
# ------------------------------------------
# 1. Download 1 year of daily data for the NIFTY 50 Index
print("Downloading Nifty 50 data...")
data = yf.download('^NSEi', period='1y')

# yfinance recently updated how it structures data.
# We use .squeeze() to ensure we get a simple list of numbers.
h = data['High'].squeeze()
l = data['Low'].squeeze()
c = data['Close'].squeeze()

N = len(data)

# ---------------------------------------------------------
# 2. PARKINSON VOLATILITY (Intraday High/Low Swings)
# ---------------------------------------------------------
log_hl = np.log(h / l)
log_hl_squared = log_hl ** 2
constant = 1 / (4 * N * np.log(2))
parkinson_variance = constant * log_hl_squared.sum()
parkinson_vol = np.sqrt(parkinson_variance) * np.sqrt(252)

# ---------------------------------------------------------
# 3. CLOSE-TO-CLOSE VOLATILITY (The Retail Standard)
# ---------------------------------------------------------
# Calculate the daily percentage changes (log returns) and find their standard deviation
log_returns = np.log(c / c.shift(1))
c2c_vol = log_returns.std() * np.sqrt(252)

# ---------------------------------------------------------
# 4. Display the Results (Using float() to prevent the error!)
# ---------------------------------------------------------
print("-" * 40)
print(f"Total Trading Days Analyzed (N): {N}")
print(f"Parkinson Volatility (True Intraday Risk):  {float(parkinson_vol) * 100:.2f}%")
print(f"Close-to-Close Volatility (Standard Risk):  {float(c2c_vol) * 100:.2f}%")
print("-" * 40)


# ==========================================
# CELL 4: EXECUTE ALL 9 FINANCIAL MODELS
# ==========================================

print("🚀 INITIALIZING ALL 9 FINANCIAL MODELS...\n" + "="*50)



# 7. Liquidity Detector
print("\n[7/9] Running Liquidity Sweep Detector...")
plot_liquidity_detector()



# 9. Fyers Live Profile (OI / Volume)
print("\n[9/9] Running Live Fyers Smart Profile...")
plot_fyers_oi_profile(
    access_token=access_token,
    client_id=CLIENT_ID,
    underlying="NIFTY",
    expiry_str="26JUN",   # Change to "26618" if Fyers rejects "26JUN" again
    spot_price=23300,     # Update this to the closest live Nifty Spot Price
    strike_step=50
)

print("\n" + "="*50 + "\n✅ ALL 9 MODELS EXECUTED SUCCESSFULLY!")
