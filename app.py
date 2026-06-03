In 1 st code expected move and hust component  showing different and in below code its showing different why..?





```notebook-python

import yfinance as yf

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



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

```