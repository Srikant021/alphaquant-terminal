# nifty_tools.py
"""
Plotting utilities for Nifty Live Tools.
Each function returns a BytesIO PNG buffer (ready for Streamlit display/download).
Uses yfinance for market data.
"""
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

plt.style.use("dark_background")


def _fig_to_bytes(fig, dpi=120):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf


def fetch_yf(ticker, **kwargs):
    """Wrapper to fetch data and raise a clear error if empty."""
    df = yf.download(ticker, **kwargs)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    return df


def plot_nifty_volatility_buf():
    data = fetch_yf("^INDIAVIX", period="1y", progress=False)
    close = data["Close"].squeeze()
    current_iv = float(close.iloc[-1])
    high_52w = float(close.max())
    low_52w = float(close.min())

    ivr = ((current_iv - low_52w) / (high_52w - low_52w)) * 100 if high_52w != low_52w else 0.0
    days_below = (close < current_iv).sum()
    total_days = len(close)
    ivp = (days_below / total_days) * 100 if total_days > 0 else 0.0

    regime = "HIGH VOLATILITY: Net Short Premium" if ivr > 50 else "LOW VOLATILITY: Net Long Premium"
    color_theme = "#00FF00" if ivr > 50 else "#FF3333"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(close.index, close.values, color="#00FFFF", linewidth=1.5, label="India VIX (1Y)")
    ax.axhline(high_52w, color="red", linestyle="--", alpha=0.6, label=f"52W High: {high_52w:.2f}")
    ax.axhline(low_52w, color="green", linestyle="--", alpha=0.6, label=f"52W Low: {low_52w:.2f}")
    ax.axhline(current_iv, color="white", linestyle="-", linewidth=1.5, label=f"Current: {current_iv:.2f}")
    ax.fill_between(close.index, low_52w, current_iv, color="white", alpha=0.03)
    ax.set_title("NIFTY Implied Volatility (IVR & IVP)", fontsize=14)
    ax.set_ylabel("VIX Level")
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper right", facecolor="black", edgecolor="gray", fontsize=9)

    props = dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.85, edgecolor=color_theme, linewidth=1.2)
    text_str = (
        f"IV Rank (IVR): {ivr:.1f}%\n"
        f"IV Percentile (IVP): {ivp:.1f}%\n"
        f"Current VIX: {current_iv:.2f}\n"
        f"Edge: {regime}"
    )
    ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=props, color="white")

    return _fig_to_bytes(fig)


def plot_expected_move_buf():
    nifty = fetch_yf("^NSEI", period="1mo", progress=False)
    vix = fetch_yf("^INDIAVIX", period="5d", progress=False)

    nifty_close = nifty["Close"].squeeze()
    spot = float(nifty_close.iloc[-1])
    current_vix = float(vix["Close"].squeeze().iloc[-1])

    # Use trading-day annualization consistently (sqrt(252)) for realized vol; for daily expected move we keep original approach
    daily_volatility = (current_vix / 100) * np.sqrt(1 / 365)
    expected_move = spot * daily_volatility
    upper = spot + expected_move
    lower = spot - expected_move

    recent = nifty_close.tail(15)
    x = np.arange(len(recent))
    tomorrow_x = len(recent)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, recent.values, color="#00FFFF", linewidth=2, marker="o", label="Nifty 50 Close")
    ax.hlines(spot, xmin=x[-1], xmax=tomorrow_x, color="white", linestyle="-", linewidth=1.5)
    ax.scatter(tomorrow_x, spot, color="white", s=50, zorder=5)
    ax.scatter(tomorrow_x, upper, color="#00FF00", s=80, marker="^", zorder=5)
    ax.scatter(tomorrow_x, lower, color="#FF3333", s=80, marker="v", zorder=5)
    ax.hlines(upper, xmin=x[-1], xmax=tomorrow_x, color="#00FF00", linestyle="--", linewidth=1)
    ax.hlines(lower, xmin=x[-1], xmax=tomorrow_x, color="#FF3333", linestyle="--", linewidth=1)
    ax.fill_between([x[-1], tomorrow_x], [spot, lower], [spot, upper], color="gray", alpha=0.18)

    ax.set_title("NIFTY 50 Implied Daily Expected Move", fontsize=14)
    ax.set_ylabel("Nifty 50 Price")
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper left", facecolor="black")

    props = dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.85, edgecolor="white", linewidth=1.2)
    text_str = (
        f"Current VIX: {current_vix:.2f}\n"
        f"Spot Price: {spot:.2f}\n"
        f"Expected Move: ± {expected_move:.1f} points\n"
        f"Upper: {upper:.2f}\nLower: {lower:.2f}"
    )
    ax.text(0.02, 0.45, text_str, transform=ax.transAxes, fontsize=10, verticalalignment="center", bbox=props, color="white")

    return _fig_to_bytes(fig)


# Additional plotting functions (volatility cone, VRP, Hurst, liquidity, Parkinson)
# Implemented similarly and returning BytesIO. For brevity they are omitted here but included in the repo.
# (In the repo these functions are present: plot_index_divergence_buf, plot_volatility_cone_buf,
#  plot_vrp_buf, plot_hurst_buf, plot_liquidity_sweep_buf, plot_parkinson_buf)
