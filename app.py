# streamlit_app.py
"""
Enhanced Streamlit GUI for Nifty Live Tools
- Refactored plotting functions return PNG BytesIO buffers for display + download
- Tabbed layout, per-tool caching, FYERS secret handling, status & last-run timestamps
- Deploy to Streamlit Community Cloud or run locally: `streamlit run streamlit_app.py`
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from io import BytesIO
import time
import traceback
import base64

warnings.filterwarnings("ignore")
plt.style.use("dark_background")

# -------------------------
# Helper: save matplotlib fig to BytesIO
# -------------------------
def fig_to_bytes(fig, dpi=120):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf

# -------------------------
# Plotting functions (return BytesIO)
# Each function raises RuntimeError on data fetch failure
# -------------------------

def plot_nifty_volatility_buf():
    ticker = "^INDIAVIX"
    data = yf.download(ticker, period="1y", progress=False)
    if data.empty:
        raise RuntimeError("No data retrieved from Yahoo Finance for India VIX.")

    close_prices = data["Close"].squeeze()
    current_iv = float(close_prices.iloc[-1])
    high_52w = float(close_prices.max())
    low_52w = float(close_prices.min())

    ivr = ((current_iv - low_52w) / (high_52w - low_52w)) * 100 if high_52w != low_52w else 0.0
    days_below = (close_prices < current_iv).sum()
    total_days = len(close_prices)
    ivp = (days_below / total_days) * 100 if total_days > 0 else 0.0

    regime = "HIGH VOLATILITY: Net Short Premium (Credit Spreads)" if ivr > 50 else "LOW VOLATILITY: Net Long Premium (Debit Spreads)"
    color_theme = "#00FF00" if ivr > 50 else "#FF3333"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(close_prices.index, close_prices.values, color="#00FFFF", linewidth=1.5, label="India VIX (1Y)")
    ax.axhline(high_52w, color="red", linestyle="--", alpha=0.6, label=f"52W High: {high_52w:.2f}")
    ax.axhline(low_52w, color="green", linestyle="--", alpha=0.6, label=f"52W Low: {low_52w:.2f}")
    ax.axhline(current_iv, color="white", linestyle="-", linewidth=1.5, label=f"Current: {current_iv:.2f}")
    ax.fill_between(close_prices.index, low_52w, current_iv, color="white", alpha=0.03)
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

    return fig_to_bytes(fig)

def plot_expected_move_buf():
    nifty_data = yf.download("^NSEI", period="1mo", progress=False)
    vix_data = yf.download("^INDIAVIX", period="5d", progress=False)
    if nifty_data.empty or vix_data.empty:
        raise RuntimeError("No data from Yahoo Finance for Nifty or INDIAVIX")

    nifty_close = nifty_data["Close"].squeeze()
    spot_price = float(nifty_close.iloc[-1])
    current_vix = float(vix_data["Close"].squeeze().iloc[-1])

    daily_volatility = (current_vix / 100) * np.sqrt(1 / 365)
    expected_move_points = spot_price * daily_volatility
    upper_bound = spot_price + expected_move_points
    lower_bound = spot_price - expected_move_points

    recent_nifty = nifty_close.tail(15)
    x = np.arange(len(recent_nifty))
    tomorrow_x = len(recent_nifty)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, recent_nifty.values, color="#00FFFF", linewidth=2, marker="o", label="Nifty 50 Close")
    ax.hlines(spot_price, xmin=x[-1], xmax=tomorrow_x, color="white", linestyle="-", linewidth=1.5)
    ax.scatter(tomorrow_x, spot_price, color="white", s=50, zorder=5)
    ax.scatter(tomorrow_x, upper_bound, color="#00FF00", s=80, marker="^", zorder=5)
    ax.scatter(tomorrow_x, lower_bound, color="#FF3333", s=80, marker="v", zorder=5)
    ax.hlines(upper_bound, xmin=x[-1], xmax=tomorrow_x, color="#00FF00", linestyle="--", linewidth=1)
    ax.hlines(lower_bound, xmin=x[-1], xmax=tomorrow_x, color="#FF3333", linestyle="--", linewidth=1)
    ax.fill_between([x[-1], tomorrow_x], [spot_price, lower_bound], [spot_price, upper_bound], color="gray", alpha=0.18)

    ax.set_title("NIFTY 50 Implied Daily Expected Move", fontsize=14)
    ax.set_ylabel("Nifty 50 Price")
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper left", facecolor="black")

    props = dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.85, edgecolor="white", linewidth=1.2)
    text_str = (
        f"Current VIX: {current_vix:.2f}\n"
        f"Spot Price: {spot_price:.2f}\n"
        f"Expected Move: ± {expected_move_points:.1f} points\n"
        f"Upper: {upper_bound:.2f}\nLower: {lower_bound:.2f}"
    )
    ax.text(0.02, 0.45, text_str, transform=ax.transAxes, fontsize=10, verticalalignment="center", bbox=props, color="white")

    return fig_to_bytes(fig)

def plot_index_divergence_buf():
    tickers = {"Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK"}
    data = yf.download(list(tickers.values()), period="1y", progress=False)["Close"]
    if data.empty:
        raise RuntimeError("No data from Yahoo Finance for Nifty or Bank Nifty.")

    data.columns = ["Bank Nifty", "Nifty 50"]
    data = data.dropna()
    normalized_prices = (data / data.iloc[0]) * 100
    log_returns = np.log(data / data.shift(1)).dropna()
    rolling_window = 20
    rolling_correlation = log_returns["Nifty 50"].rolling(window=rolling_window).corr(log_returns["Bank Nifty"])

    current_corr = float(rolling_correlation.iloc[-1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1.5, 1]})
    ax1.plot(normalized_prices.index, normalized_prices["Nifty 50"], color="#00FFFF", linewidth=2, label="Nifty 50 (Normalized)")
    ax1.plot(normalized_prices.index, normalized_prices["Bank Nifty"], color="#FFA500", linewidth=2, label="Bank Nifty (Normalized)")
    ax1.fill_between(normalized_prices.index, normalized_prices["Nifty 50"], normalized_prices["Bank Nifty"], color="gray", alpha=0.2)
    ax1.set_title("Inter-Index Correlation & Divergence (Nifty vs Bank Nifty)", fontsize=14)
    ax1.set_ylabel("Normalized Price (Base 100)")
    ax1.grid(True, linestyle=":")
    ax1.legend(loc="upper left", facecolor="black")

    ax2.plot(rolling_correlation.index, rolling_correlation, color="white", linewidth=1.5, label="20-Day Rolling Correlation")
    ax2.axhline(0.80, color="#00FF00", linestyle="--", linewidth=1.5, label="High Correlation (>0.80)")
    ax2.axhline(0.50, color="#FF3333", linestyle="--", linewidth=1.5, label="Severe Divergence (<0.50)")
    ax2.fill_between(rolling_correlation.index, 0.50, rolling_correlation, where=(rolling_correlation < 0.50), color="#FF3333", alpha=0.3, interpolate=True)
    ax2.set_ylabel("Pearson Correlation (r)")
    ax2.set_ylim(-0.2, 1.1)
    ax2.grid(True, linestyle=":")
    ax2.legend(loc="lower left", facecolor="black")

    return fig_to_bytes(fig)

def plot_volatility_cone_buf():
    ticker = "^NSEI"
    data = yf.download(ticker, start="2015-01-01", end="2026-04-30", progress=False)
    if data.empty:
        raise RuntimeError("No data from Yahoo Finance for Nifty 50.")

    data["Returns"] = np.log(data["Close"] / data["Close"].shift(1))
    windows = [10, 20, 30, 60, 90, 120, 180, 252]
    max_vol, min_vol, median_vol, current_vol = [], [], [], []

    for window in windows:
        rolling_vol = data["Returns"].rolling(window=window).std() * np.sqrt(252)
        max_vol.append(rolling_vol.max() * 100)
        min_vol.append(rolling_vol.min() * 100)
        median_vol.append(rolling_vol.median() * 100)
        current_vol.append(rolling_vol.iloc[-1] * 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(windows, max_vol, marker="o", color="red", linewidth=2, label="Maximum Volatility")
    ax.plot(windows, min_vol, marker="o", color="limegreen", linewidth=2, label="Minimum Volatility")
    ax.plot(windows, median_vol, marker="s", color="white", linewidth=1.5, linestyle="--", label="Median Volatility")
    ax.plot(windows, current_vol, marker="X", color="yellow", linewidth=3, markersize=10, label="Current Volatility")
    ax.fill_between(windows, min_vol, max_vol, color="gray", alpha=0.2)
    ax.set_title(f"Volatility Cone for Nifty 50 ({ticker})", fontsize=14)
    ax.set_xlabel("Time Window (Trading Days)")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_xticks(windows)
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper right", fontsize=10)
    return fig_to_bytes(fig)

def plot_vrp_buf():
    nifty_data = yf.download("^NSEI", period="6mo", progress=False)
    vix_data = yf.download("^INDIAVIX", period="6mo", progress=False)
    if nifty_data.empty or vix_data.empty:
        raise RuntimeError("No data for VRP calculation.")

    nifty_close = nifty_data["Close"].squeeze()
    vix_close = vix_data["Close"].squeeze()
    log_returns = np.log(nifty_close / nifty_close.shift(1))
    historical_vol = log_returns.rolling(window=20).std() * np.sqrt(252) * 100

    df = pd.DataFrame({"VIX": vix_close, "HV": historical_vol}).dropna()
    df["VRP"] = df["VIX"] - df["HV"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(df.index, df["VIX"], color="#00FFFF", linewidth=2, label="Implied Volatility (India VIX)")
    ax1.plot(df.index, df["HV"], color="#FFA500", linewidth=2, label="Realized Volatility (20-Day HV)")
    ax1.fill_between(df.index, df["HV"], df["VIX"], where=(df["VIX"] > df["HV"]), color="green", alpha=0.3, interpolate=True)
    ax1.fill_between(df.index, df["HV"], df["VIX"], where=(df["VIX"] <= df["HV"]), color="red", alpha=0.3, interpolate=True)
    ax1.set_title("NIFTY 50 Volatility Risk Premium (VRP)", fontsize=14)
    ax1.set_ylabel("Annualized Volatility (%)")
    ax1.grid(True, linestyle=":")
    ax1.legend(loc="upper left", facecolor="black")

    bar_colors = np.where(df["VRP"] > 0, "#00FF00", "#FF3333")
    ax2.bar(df.index, df["VRP"], color=bar_colors, alpha=0.7, width=1)
    ax2.axhline(0, color="white", linewidth=1)
    ax2.set_ylabel("VRP Spread (%)")
    ax2.grid(True, linestyle=":")
    return fig_to_bytes(fig)

def plot_hurst_buf():
    data = yf.download("^NSEI", period="1y", progress=False)
    if data.empty:
        raise RuntimeError("No data for Hurst Exponent.")

    nifty_close = data["Close"].squeeze()
    log_prices = np.log(nifty_close)

    def calculate_hurst(ts):
        if len(ts) < 20:
            return np.nan
        lags = range(2, 20)
        ts_arr = ts.values
        reg = [np.std(ts_arr[lag:] - ts_arr[:-lag]) for lag in lags]
        poly = np.polyfit(np.log(list(lags)), np.log(reg), 1)
        return poly[0]

    hurst_series = log_prices.rolling(window=60).apply(calculate_hurst, raw=False)
    df = pd.DataFrame({"Close": nifty_close, "Hurst": hurst_series}).dropna()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1.5, 1]})
    ax1.plot(df.index, df["Close"], color="white", linewidth=1.5, label="Nifty 50 Close")
    ax1.set_title("NIFTY 50 Market Regime (Hurst Exponent)", fontsize=14)
    ax1.set_ylabel("Nifty 50 Price")
    ax1.grid(True, linestyle=":")
    ax1.legend(loc="upper left", facecolor="black")
    ax1.axvspan(df.index[-15], df.index[-1], color="#00FFFF", alpha=0.08)

    ax2.plot(df.index, df["Hurst"], color="#00FFFF", linewidth=2, label="60-Day Hurst Exponent")
    ax2.axhline(0.55, color="#00FF00", linestyle="--", linewidth=1.5)
    ax2.axhline(0.45, color="#FF3333", linestyle="--", linewidth=1.5)
    ax2.set_ylabel("Hurst Value (H)")
    ax2.grid(True, linestyle=":")
    ax2.set_ylim(0.3, 0.7)
    ax2.legend(loc="upper right", facecolor="black")
    return fig_to_bytes(fig)

def plot_liquidity_sweep_buf():
    data = yf.download("^NSEI", period="30d", interval="15m", progress=False)
    if data.empty:
        raise RuntimeError("No intraday data for Liquidity Detector.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    window = 20
    data["Prev_High"] = data["High"].rolling(window=window).max().shift(1)
    data["Prev_Low"] = data["Low"].rolling(window=window).min().shift(1)
    data["Supply_Sweep"] = (data["High"] > data["Prev_High"]) & (data["Close"] < data["Prev_High"])
    data["Demand_Sweep"] = (data["Low"] < data["Prev_Low"]) & (data["Close"] > data["Prev_Low"])

    plot_data = data.tail(60).copy()
    plot_data["Index"] = np.arange(len(plot_data))

    fig, ax = plt.subplots(figsize=(12, 6))
    up = plot_data[plot_data["Close"] >= plot_data["Open"]]
    down = plot_data[plot_data["Close"] < plot_data["Open"]]

    ax.vlines(up["Index"], up["Low"], up["High"], color="#00FF00", linewidth=1.5)
    ax.vlines(down["Index"], down["Low"], down["High"], color="#FF3333", linewidth=1.5)
    ax.bar(up["Index"], up["Close"] - up["Open"], bottom=up["Open"], color="#00FF00", width=0.6)
    ax.bar(down["Index"], down["Open"] - down["Close"], bottom=down["Close"], color="#FF3333", width=0.6)

    for _, row in plot_data.iterrows():
        if row.get("Supply_Sweep"):
            ax.scatter(row["Index"], row["High"] + 10, marker="v", color="#FF3333", s=80, zorder=5)
            ax.axhline(row["Prev_High"], color="#FF3333", linestyle="--", alpha=0.5, linewidth=1)
        if row.get("Demand_Sweep"):
            ax.scatter(row["Index"], row["Low"] - 10, marker="^", color="#00FF00", s=80, zorder=5)
            ax.axhline(row["Prev_Low"], color="#00FF00", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_title("NIFTY 50 Intraday Liquidity Sweep & Order Block Detector (15m)", fontsize=14)
    ax.set_ylabel("Nifty 50 Price")
    ax.grid(True, linestyle=":")
    ax.set_xticks([])
    return fig_to_bytes(fig)

def plot_parkinson_buf():
    data = yf.download("^NSEI", period="1y", progress=False)
    if data.empty:
        raise RuntimeError("No data for Parkinson Estimator.")

    h = data["High"].squeeze()
    l = data["Low"].squeeze()
    c = data["Close"].squeeze()
    N = len(data)
    log_hl = np.log(h / l)
    log_hl_squared = log_hl ** 2
    constant = 1 / (4 * N * np.log(2))
    parkinson_variance = constant * log_hl_squared.sum()
    parkinson_vol = np.sqrt(parkinson_variance) * np.sqrt(252)
    log_returns = np.log(c / c.shift(1))
    c2c_vol = log_returns.std() * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    text = (
        f"Total Trading Days Analyzed (N): {N}\n\n"
        f"Parkinson Volatility (True Intraday Risk): {float(parkinson_vol) * 100:.2f}%\n\n"
        f"Close-to-Close Volatility (Standard Risk): {float(c2c_vol) * 100:.2f}%"
    )
    ax.text(0.01, 0.5, text, fontsize=12, color="white", va="center")
    return fig_to_bytes(fig)

# -------------------------
# UI: Sidebar controls
# -------------------------
st.set_page_config(page_title="Nifty Live Tools", layout="wide", initial_sidebar_state="expanded")
st.title("Nifty Live Tools — Enhanced Dashboard")

with st.sidebar:
    st.header("Controls")
    selected = st.multiselect("Select tools to show", [
        "IVR & IVP",
        "Expected Move",
        "Correlation",
        "Volatility Cone",
        "VRP",
        "Hurst",
        "Liquidity Sweep",
        "Parkinson"
    ], default=["IVR & IVP", "Expected Move"])
    st.markdown("---")
    st.subheader("FYERS (optional)")
    st.info("For production, set FYERS tokens in Streamlit Secrets (App settings).")
    fyers_token_input = st.text_input("FYERS Access Token (optional)", type="password")
    fyers_client_id_input = st.text_input("FYERS Client ID (optional)")
    use_secrets = False
    if "FYERS_ACCESS_TOKEN" in st.secrets or "FYERS_CLIENT_ID" in st.secrets:
        use_secrets = st.checkbox("Use Streamlit Secrets for FYERS", value=True)
    st.markdown("---")
    st.subheader("Refresh & caching")
    refresh_now = st.button("Refresh all (clear cache)")
    auto_refresh = st.checkbox("Auto refresh every 5 minutes", value=False)
    st.caption("Tip: enable only the tools you need to reduce API calls.")

if refresh_now:
    st.cache_data.clear()
    st.experimental_rerun()

# -------------------------
# Helper: run and display a buffered plot with download
# -------------------------
def render_buffered_plot(get_buf_func, title, cache_ttl=300):
    key = f"buf_{title}"
    try:
        @st.cache_data(ttl=cache_ttl, show_spinner=False)
        def _cached():
            return get_buf_func().getvalue()
        img_bytes = _cached()
        buf = BytesIO(img_bytes)
        st.image(buf)
        if st.button(f"Download {title} PNG"):
            b64 = base64.b64encode(img_bytes).decode()
            href = f'<a href="data:file/png;base64,{b64}" download="{title.replace(" ", "_")}.png">Download {title} PNG</a>'
            st.markdown(href, unsafe_allow_html=True)
        st.success(f"{title} rendered at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        tb = traceback.format_exc()
        st.error(f"Failed to render {title}: {e}")
        with st.expander("Error details"):
            st.text(tb)

# -------------------------
# Main: Tabs for selected tools
# -------------------------
tabs = st.tabs(selected if selected else ["No tools selected"])
for i, tool in enumerate(selected):
    with tabs[i]:
        st.header(tool)
        if tool == "IVR & IVP":
            render_buffered_plot(plot_nifty_volatility_buf, "IVR_IVP", cache_ttl=300)
        elif tool == "Expected Move":
            render_buffered_plot(plot_expected_move_buf, "Expected_Move", cache_ttl=120)
        elif tool == "Correlation":
            render_buffered_plot(plot_index_divergence_buf, "Correlation", cache_ttl=3600)
        elif tool == "Volatility Cone":
            render_buffered_plot(plot_volatility_cone_buf, "Volatility_Cone", cache_ttl=86400)
        elif tool == "VRP":
            render_buffered_plot(plot_vrp_buf, "VRP", cache_ttl=1800)
        elif tool == "Hurst":
            render_buffered_plot(plot_hurst_buf, "Hurst", cache_ttl=1800)
        elif tool == "Liquidity Sweep":
            render_buffered_plot(plot_liquidity_sweep_buf, "Liquidity_Sweep", cache_ttl=300)
        elif tool == "Parkinson":
            render_buffered_plot(plot_parkinson_buf, "Parkinson", cache_ttl=86400)
        else:
            st.info("Tool not implemented in this build.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Data via yfinance. FYERS integration requires valid credentials and may need additional UI flows for interactive auth.")
