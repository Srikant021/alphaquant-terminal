import math
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Quant Dashboard",
    layout="wide",
)


# -----------------------------
# Shared utilities
# -----------------------------
@st.cache_data(ttl=60 * 60)
def yf_download(
    tickers,
    *,
    period=None,
    start=None,
    end=None,
    interval=None,
    threads: bool = True,
    proxy: str | None = None,
):
    """
    Cached wrapper over yfinance.download with stable defaults.
    """
    def _call(_threads: bool):
        return yf.download(
            tickers,
            period=period,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=_threads,
            proxy=proxy or None,
        )

    # Sometimes Yahoo blocks / rate-limits and yfinance can fail sporadically.
    # We try twice: (1) user-selected threads option, (2) fallback to threads=False.
    try:
        df = _call(threads)
        if getattr(df, "empty", True) and threads:
            df = _call(False)
        return df
    except Exception:
        if threads:
            return _call(False)
        raise


def _safe_squeeze(series_or_df):
    if isinstance(series_or_df, pd.DataFrame) and series_or_df.shape[1] == 1:
        return series_or_df.iloc[:, 0]
    return series_or_df.squeeze()


def _fmt_float(x, decimals=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):.{decimals}f}"


def _new_fig():
    plt.style.use("dark_background")
    fig = plt.figure(dpi=120)
    return fig


def _st_error_block(msg: str, exc: Exception | None = None):
    st.error(msg)
    if exc is not None:
        st.caption(f"Error: `{type(exc).__name__}: {exc}`")
        with st.expander("Details"):
            st.exception(exc)


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Inputs")
default_nifty = st.sidebar.text_input("Nifty ticker (Yahoo)", value="^NSEI")
default_bank = st.sidebar.text_input("Bank Nifty ticker (Yahoo)", value="^NSEBANK")
default_vix = st.sidebar.text_input("India VIX ticker (Yahoo)", value="^INDIAVIX")

st.sidebar.divider()
st.sidebar.subheader("Yahoo download settings")
YF_THREADS = st.sidebar.toggle("Use multi-thread download", value=True)
YF_PROXY = st.sidebar.text_input("Proxy (optional)", value="", help="Example: http://user:pass@host:port")
SHOW_DIAGNOSTICS = st.sidebar.toggle("Show diagnostics", value=False)

st.sidebar.divider()
st.sidebar.caption(
    "Note: Most tabs use Yahoo Finance via yfinance. "
    "Open Interest tab needs Zerodha Kite API key + access token."
)


# -----------------------------
# Module implementations
# -----------------------------
def module_correlation(nifty_ticker: str, bank_ticker: str):
    st.subheader("Correlation / Divergence (Nifty vs Bank Nifty)")

    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("Data period", ["6mo", "1y", "2y"], index=1, key="corr_period")
    with col2:
        rolling_window = st.number_input("Rolling correlation window (days)", 5, 120, 20, 1, key="corr_window")

    try:
        data = yf_download([nifty_ticker, bank_ticker], period=period, threads=YF_THREADS, proxy=YF_PROXY or None)
        if SHOW_DIAGNOSTICS:
            st.write("Raw columns:", data.columns)
            st.dataframe(data.tail(5))
        close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
        close = close.dropna()
        close = close.rename(columns={nifty_ticker: "Nifty 50", bank_ticker: "Bank Nifty"})
    except Exception as e:
        _st_error_block("Failed to download data from Yahoo Finance.", e)
        return

    if close.empty or "Nifty 50" not in close.columns or "Bank Nifty" not in close.columns:
        _st_error_block("No usable data returned. Try a different ticker/period.")
        return

    normalized_prices = (close / close.iloc[0]) * 100
    log_returns = np.log(close / close.shift(1)).dropna()
    rolling_corr = log_returns["Nifty 50"].rolling(window=int(rolling_window)).corr(log_returns["Bank Nifty"])

    current_nifty = float(close["Nifty 50"].iloc[-1])
    current_bank = float(close["Bank Nifty"].iloc[-1])
    current_corr = float(rolling_corr.dropna().iloc[-1]) if rolling_corr.dropna().shape[0] else np.nan

    if not np.isnan(current_corr) and current_corr > 0.80:
        regime = "HIGH CORRELATION"
        stat_property = "Synchronized Movement / Low Dispersion"
        color_theme = "#00FF00"
    elif not np.isnan(current_corr) and current_corr < 0.50:
        regime = "SEVERE DIVERGENCE"
        stat_property = "Sector Rotation / High Dispersion Warning"
        color_theme = "#FF3333"
    else:
        regime = "MODERATE DIVERGENCE"
        stat_property = "Decoupling / Monitoring Phase"
        color_theme = "#FFA500"

    m1, m2, m3 = st.columns(3)
    m1.metric("Nifty 50", _fmt_float(current_nifty, 2))
    m2.metric("Bank Nifty", _fmt_float(current_bank, 2))
    m3.metric(f"{rolling_window}-day Corr", _fmt_float(current_corr, 2))
    st.caption(f"Regime: **{regime}** — {stat_property}")

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), dpi=120, gridspec_kw={"height_ratios": [1.5, 1]}
    )

    ax1.plot(normalized_prices.index, normalized_prices["Nifty 50"], color="#00FFFF", linewidth=2, label="Nifty 50")
    ax1.plot(
        normalized_prices.index, normalized_prices["Bank Nifty"], color="#FFA500", linewidth=2, label="Bank Nifty"
    )
    ax1.fill_between(
        normalized_prices.index,
        normalized_prices["Nifty 50"],
        normalized_prices["Bank Nifty"],
        color="gray",
        alpha=0.2,
    )
    ax1.set_ylabel("Normalized Price (Base 100)", color="gray")
    ax1.grid(True, color="#2A2A2A", linestyle=":")
    ax1.legend(loc="upper left", facecolor="black", edgecolor="gray")
    ax1.set_title(
        "Inter-Index Correlation & Divergence (Nifty vs Bank Nifty)",
        fontsize=16,
        color="white",
        pad=10,
        fontweight="bold",
    )

    ax2.plot(rolling_corr.index, rolling_corr, color="white", linewidth=1.5, label="Rolling Correlation")
    ax2.axhline(0.80, color="#00FF00", linestyle="--", linewidth=1.5, label="High Corr (>0.80)")
    ax2.axhline(0.50, color="#FF3333", linestyle="--", linewidth=1.5, label="Low Corr (<0.50)")
    ax2.fill_between(
        rolling_corr.index,
        0.50,
        rolling_corr,
        where=(rolling_corr < 0.50),
        color="#FF3333",
        alpha=0.3,
        interpolate=True,
    )
    ax2.set_ylabel("Pearson r", color="gray")
    ax2.set_ylim(-0.2, 1.1)
    ax2.grid(True, color="#2A2A2A", linestyle=":")
    ax2.legend(loc="lower left", facecolor="black", edgecolor="gray")

    props = dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.9, edgecolor=color_theme, linewidth=1.5)
    text_str = (
        f"Nifty 50: {current_nifty:.2f}\n"
        f"Bank Nifty: {current_bank:.2f}\n"
        f"{rolling_window}-Day Corr: {current_corr:.2f}\n"
        f"---------------------------\n"
        f"Stat Property: {stat_property}"
    )
    ax1.text(
        0.02,
        0.05,
        text_str,
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        bbox=props,
        color="white",
        fontweight="bold",
    )
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def module_expected_move(nifty_ticker: str, vix_ticker: str):
    st.subheader("Expected Move (Implied daily range via VIX)")

    col1, col2 = st.columns(2)
    with col1:
        nifty_period = st.selectbox("Nifty period (for context)", ["1mo", "3mo", "6mo"], index=0, key="em_n_period")
    with col2:
        vix_period = st.selectbox("VIX period", ["5d", "1mo"], index=0, key="em_v_period")

    try:
        nifty_data = yf_download(nifty_ticker, period=nifty_period, threads=YF_THREADS, proxy=YF_PROXY or None)
        vix_data = yf_download(vix_ticker, period=vix_period, threads=YF_THREADS, proxy=YF_PROXY or None)
        if SHOW_DIAGNOSTICS:
            st.write("Nifty columns:", nifty_data.columns)
            st.write("VIX columns:", vix_data.columns)
    except Exception as e:
        _st_error_block("Failed to download Nifty/VIX data.", e)
        return

    if nifty_data.empty or vix_data.empty:
        _st_error_block("Yahoo Finance returned no data.")
        return

    nifty_close = _safe_squeeze(nifty_data["Close"])
    spot_price = float(nifty_close.iloc[-1])
    current_vix = float(_safe_squeeze(vix_data["Close"]).iloc[-1])

    daily_vol = (current_vix / 100.0) * math.sqrt(1 / 365)
    expected_move_points = spot_price * daily_vol
    upper_bound = spot_price + expected_move_points
    lower_bound = spot_price - expected_move_points

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spot", _fmt_float(spot_price, 2))
    m2.metric("VIX", _fmt_float(current_vix, 2))
    m3.metric("Expected move (pts)", _fmt_float(expected_move_points, 1))
    m4.metric("Range", f"{_fmt_float(lower_bound,0)}  →  {_fmt_float(upper_bound,0)}")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)

    recent_nifty = nifty_close.tail(15)
    x_dates = np.arange(len(recent_nifty))
    ax.plot(x_dates, recent_nifty.values, color="#00FFFF", linewidth=2, marker="o", label="Nifty Close")
    tomorrow_x = len(recent_nifty)

    ax.hlines(spot_price, xmin=x_dates[-1], xmax=tomorrow_x, color="white", linestyle="-", linewidth=2)
    ax.scatter(tomorrow_x, spot_price, color="white", s=70, zorder=5)

    ax.scatter(tomorrow_x, upper_bound, color="#00FF00", s=120, marker="^", zorder=5)
    ax.scatter(tomorrow_x, lower_bound, color="#FF3333", s=120, marker="v", zorder=5)
    ax.hlines(upper_bound, xmin=x_dates[-1], xmax=tomorrow_x, color="#00FF00", linestyle="--", linewidth=1.5)
    ax.hlines(lower_bound, xmin=x_dates[-1], xmax=tomorrow_x, color="#FF3333", linestyle="--", linewidth=1.5)
    ax.fill_between([x_dates[-1], tomorrow_x], [spot_price, lower_bound], [spot_price, upper_bound], color="gray", alpha=0.2)

    ax.set_title("NIFTY Implied Daily Expected Move", fontsize=16, color="white", pad=15, fontweight="bold")
    ax.set_ylabel("Nifty Price", color="gray", fontsize=12)
    ax.grid(True, color="#2A2A2A", linestyle=":")
    ax.set_xticks([])
    ax.legend(loc="upper left", facecolor="black", edgecolor="gray", fontsize=10)

    props = dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8, edgecolor="white", linewidth=1.5)
    text_str = (
        f"Current VIX: {current_vix:.2f}\n"
        f"Spot: {spot_price:.2f}\n"
        f"Expected Move: ± {expected_move_points:.1f} pts\n"
        f"---------------------------\n"
        f"Upper (+1 SD): {upper_bound:.0f}\n"
        f"Lower (-1 SD): {lower_bound:.0f}"
    )
    ax.text(0.02, 0.45, text_str, transform=ax.transAxes, fontsize=11, verticalalignment="center", bbox=props, color="white")

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def calculate_hurst(ts: pd.Series) -> float:
    if len(ts) < 20:
        return np.nan
    lags = range(2, 20)
    tau = [lag for lag in lags]
    ts_arr = ts.values
    reg = [np.std(ts_arr[lag:] - ts_arr[:-lag]) for lag in lags]
    poly = np.polyfit(np.log(tau), np.log(reg), 1)
    return float(poly[0])


def module_hurst(nifty_ticker: str):
    st.subheader("Hurst Exponent (Market Regime)")

    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("Data period", ["6mo", "1y", "2y"], index=1, key="hurst_period")
    with col2:
        window = st.number_input("Rolling window (days)", 30, 200, 60, 5, key="hurst_window")

    try:
        data = yf_download(nifty_ticker, period=period, threads=YF_THREADS, proxy=YF_PROXY or None)
        if SHOW_DIAGNOSTICS:
            st.write("Raw columns:", data.columns)
    except Exception as e:
        _st_error_block("Failed to download Nifty data.", e)
        return

    if data.empty:
        _st_error_block("Yahoo Finance returned no data.")
        return

    close = _safe_squeeze(data["Close"])
    log_prices = np.log(close)
    hurst_series = log_prices.rolling(window=int(window)).apply(calculate_hurst, raw=False)
    df = pd.DataFrame({"Close": close, "Hurst": hurst_series}).dropna()

    if df.empty:
        _st_error_block("Not enough data to compute the rolling Hurst series.")
        return

    current_price = float(df["Close"].iloc[-1])
    current_hurst = float(df["Hurst"].iloc[-1])

    if current_hurst < 0.45:
        regime = "MEAN REVERTING"
        stat_property = "Range-Bound Action / Volatility Compression"
        color_theme = "#FF3333"
    elif current_hurst > 0.55:
        regime = "TRENDING"
        stat_property = "Directional Movement / Momentum Expansion"
        color_theme = "#00FF00"
    else:
        regime = "RANDOM WALK"
        stat_property = "Unpredictable Noise / Transition Phase"
        color_theme = "#FFA500"

    m1, m2 = st.columns(2)
    m1.metric("Nifty", _fmt_float(current_price, 2))
    m2.metric("Hurst (H)", _fmt_float(current_hurst, 3))
    st.caption(f"Regime: **{regime}** — {stat_property}")

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=120, gridspec_kw={"height_ratios": [1.5, 1]})
    ax1.plot(df.index, df["Close"], color="white", linewidth=1.5, label="Nifty Close")
    ax1.set_title("NIFTY Market Regime (Hurst Exponent)", fontsize=16, color="white", pad=10, fontweight="bold")
    ax1.set_ylabel("Nifty", color="gray")
    ax1.grid(True, color="#2A2A2A", linestyle=":")
    ax1.legend(loc="upper left", facecolor="black", edgecolor="gray")
    if len(df.index) >= 15:
        ax1.axvspan(df.index[-15], df.index[-1], color=color_theme, alpha=0.1)

    ax2.plot(df.index, df["Hurst"], color="#00FFFF", linewidth=2, label=f"{window}-Day Hurst")
    ax2.axhline(0.55, color="#00FF00", linestyle="--", linewidth=1.5, label="Trending (>0.55)")
    ax2.axhline(0.45, color="#FF3333", linestyle="--", linewidth=1.5, label="Mean Reverting (<0.45)")
    ax2.axhline(0.50, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    ax2.fill_between(df.index, 0.55, df["Hurst"], where=(df["Hurst"] > 0.55), color="#00FF00", alpha=0.2, interpolate=True)
    ax2.fill_between(df.index, 0.45, df["Hurst"], where=(df["Hurst"] < 0.45), color="#FF3333", alpha=0.2, interpolate=True)
    ax2.set_ylabel("H", color="gray")
    ax2.set_ylim(0.3, 0.7)
    ax2.grid(True, color="#2A2A2A", linestyle=":")
    ax2.legend(loc="upper right", facecolor="black", edgecolor="gray")

    props = dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.9, edgecolor=color_theme, linewidth=1.5)
    text_str = (
        f"Current Nifty: {current_price:.2f}\n"
        f"Hurst (H): {current_hurst:.3f}\n"
        f"Regime: {regime}\n"
        f"---------------------------\n"
        f"Stat Property: {stat_property}"
    )
    ax1.text(0.02, 0.05, text_str, transform=ax1.transAxes, fontsize=11, verticalalignment="bottom", bbox=props, color="white")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def module_ivr_ivp(vix_ticker: str):
    st.subheader("IV Rank (IVR) & IV Percentile (IVP) – India VIX")

    period = st.selectbox("Data period", ["6mo", "1y", "2y"], index=1, key="ivr_period")

    try:
        data = yf_download(vix_ticker, period=period, threads=YF_THREADS, proxy=YF_PROXY or None)
        if SHOW_DIAGNOSTICS:
            st.write("Raw columns:", data.columns)
    except Exception as e:
        _st_error_block("Failed to download India VIX data.", e)
        return

    if data.empty:
        _st_error_block("Yahoo Finance returned no data.")
        return

    close = _safe_squeeze(data["Close"])
    current_iv = float(close.iloc[-1])
    high = float(close.max())
    low = float(close.min())

    if high == low:
        _st_error_block("Cannot compute IVR/IVP (high == low in selected period).")
        return

    ivr = ((current_iv - low) / (high - low)) * 100
    ivp = (float((close < current_iv).sum()) / float(len(close))) * 100

    if ivr > 50:
        regime = "HIGH VOLATILITY"
        edge = "Net Short Premium (Credit Spreads)"
        color_theme = "#00FF00"
    else:
        regime = "LOW VOLATILITY"
        edge = "Net Long Premium (Debit Spreads)"
        color_theme = "#FF3333"

    m1, m2, m3 = st.columns(3)
    m1.metric("Current VIX", _fmt_float(current_iv, 2))
    m2.metric("IV Rank (IVR)", f"{_fmt_float(ivr,1)}%")
    m3.metric("IV Percentile (IVP)", f"{_fmt_float(ivp,1)}%")
    st.caption(f"Regime: **{regime}** — {edge}")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
    ax.plot(close.index, close.values, color="#00FFFF", linewidth=1.5, label=f"{vix_ticker} Close")
    ax.axhline(high, color="red", linestyle="--", alpha=0.5, label=f"High: {high:.2f}")
    ax.axhline(low, color="green", linestyle="--", alpha=0.5, label=f"Low: {low:.2f}")
    ax.axhline(current_iv, color="white", linestyle="-", linewidth=2, label=f"Current: {current_iv:.2f}")
    ax.fill_between(close.index, low, current_iv, color="white", alpha=0.05)
    ax.set_title("Implied Volatility (IVR & IVP)", fontsize=16, color="white", pad=15, fontweight="bold")
    ax.set_ylabel("VIX", color="gray")
    ax.grid(True, color="#2A2A2A", linestyle=":")
    ax.legend(loc="upper right", facecolor="black", edgecolor="gray", fontsize=10)

    props = dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8, edgecolor=color_theme, linewidth=1.5)
    text_str = f"IVR: {ivr:.1f}%\nIVP: {ivp:.1f}%\nCurrent: {current_iv:.2f}\n---------------------------\nEdge: {edge}"
    ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=11, verticalalignment="top", bbox=props, color="white")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def module_liquidity_sweep(nifty_ticker: str):
    st.subheader("Liquidity Sweep / Order Block Detector (15m)")

    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("Period", ["5d", "10d"], index=0, key="liq_period")
    with col2:
        interval = st.selectbox("Interval", ["15m", "30m"], index=0, key="liq_interval")

    try:
        data = yf_download(nifty_ticker, period=period, interval=interval, threads=YF_THREADS, proxy=YF_PROXY or None)
        if SHOW_DIAGNOSTICS:
            st.write("Raw columns:", data.columns)
    except Exception as e:
        _st_error_block("Failed to download intraday data.", e)
        return

    if data.empty:
        _st_error_block("Yahoo Finance returned no intraday data (try different interval/period).")
        return

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]

    window = st.number_input("Structure window (bars)", 5, 100, 20, 1, key="liq_window")

    df = data.copy()
    df["Prev_High"] = df["High"].rolling(window=int(window)).max().shift(1)
    df["Prev_Low"] = df["Low"].rolling(window=int(window)).min().shift(1)
    df["Supply_Sweep"] = (df["High"] > df["Prev_High"]) & (df["Close"] < df["Prev_High"])
    df["Demand_Sweep"] = (df["Low"] < df["Prev_Low"]) & (df["Close"] > df["Prev_Low"])

    current_price = float(df["Close"].iloc[-1])
    if bool(df["Supply_Sweep"].iloc[-1]):
        regime = "SUPPLY LIQUIDITY SWEPT"
        stat_property = "Failed Breakout / Institutional Absorption at Highs"
        color_theme = "#FF3333"
    elif bool(df["Demand_Sweep"].iloc[-1]):
        regime = "DEMAND LIQUIDITY SWEPT"
        stat_property = "Failed Breakdown / Institutional Absorption at Lows"
        color_theme = "#00FF00"
    else:
        regime = "PRICE DISCOVERY"
        stat_property = "Trading Inside Established Structural Bounds"
        color_theme = "#00FFFF"

    st.metric("Last price", _fmt_float(current_price, 2))
    st.caption(f"Microstructure: **{regime}** — {stat_property}")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
    plot_data = df.tail(60).copy()
    plot_data["Index"] = np.arange(len(plot_data))

    up = plot_data[plot_data["Close"] >= plot_data["Open"]]
    down = plot_data[plot_data["Close"] < plot_data["Open"]]

    ax.vlines(up["Index"], up["Low"], up["High"], color="#00FF00", linewidth=1.5)
    ax.vlines(down["Index"], down["Low"], down["High"], color="#FF3333", linewidth=1.5)
    ax.bar(up["Index"], up["Close"] - up["Open"], bottom=up["Open"], color="#00FF00", width=0.6)
    ax.bar(down["Index"], down["Open"] - down["Close"], bottom=down["Close"], color="#FF3333", width=0.6)

    for _, row in plot_data.iterrows():
        if bool(row["Supply_Sweep"]):
            ax.scatter(row["Index"], row["High"] + 10, marker="v", color="#FF3333", s=150, zorder=5)
            ax.axhline(row["Prev_High"], color="#FF3333", linestyle="--", alpha=0.5, linewidth=1)
        if bool(row["Demand_Sweep"]):
            ax.scatter(row["Index"], row["Low"] - 10, marker="^", color="#00FF00", s=150, zorder=5)
            ax.axhline(row["Prev_Low"], color="#00FF00", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_title("Intraday Liquidity Sweep Detector", fontsize=16, color="white", pad=12, fontweight="bold")
    ax.set_ylabel("Price", color="gray")
    ax.grid(True, color="#2A2A2A", linestyle=":")
    ax.set_xticks([])

    props = dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.9, edgecolor=color_theme, linewidth=1.5)
    text_str = f"Live Spot: {current_price:.2f}\n---------------------------\n{regime}\n{stat_property}"
    ax.text(0.02, 0.05, text_str, transform=ax.transAxes, fontsize=11, verticalalignment="bottom", bbox=props, color="white")

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


@st.cache_data(ttl=60 * 60 * 24)
def _kite_instruments_nfo(api_key: str, access_token: str):
    from kiteconnect import KiteConnect

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    instruments = kite.instruments("NFO")
    return pd.DataFrame(instruments)


def module_open_interest_profile():
    st.subheader("Open Interest Profile (Zerodha Kite) + Max Pain")

    st.info(
        "This tab needs Zerodha Kite credentials. Enter your API key + daily access token in the sidebar below and click Fetch."
    )

    api_key = st.text_input("Kite API key", value="", key="kite_api")
    access_token = st.text_input("Kite access token", value="", type="password", key="kite_token")
    strike_range = st.number_input("Strike range around spot (points)", 200, 3000, 1000, 50, key="oi_range")
    plot_range = st.number_input("Plot range around spot (y-axis)", 200, 2000, 600, 50, key="oi_plot_range")

    if not st.button("Fetch live OI", type="primary"):
        return

    if not api_key or not access_token:
        _st_error_block("Please enter Kite API key and access token.")
        return

    try:
        from kiteconnect import KiteConnect

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)

        spot_quote = kite.quote(["NSE:NIFTY 50"])
        spot_price = float(spot_quote["NSE:NIFTY 50"]["last_price"])

        inst_df = _kite_instruments_nfo(api_key, access_token)
        nifty_opt = inst_df[(inst_df["name"] == "NIFTY") & (inst_df["segment"] == "NFO-OPT")].copy()
        nifty_opt["expiry"] = pd.to_datetime(nifty_opt["expiry"])
        current_expiry = nifty_opt["expiry"].min()
        options = nifty_opt[nifty_opt["expiry"] == current_expiry].copy()

        lower_bound = spot_price - float(strike_range)
        upper_bound = spot_price + float(strike_range)
        options = options[(options["strike"] >= lower_bound) & (options["strike"] <= upper_bound)].copy()

        symbols = ["NFO:" + sym for sym in options["tradingsymbol"].tolist()]
        quotes = kite.quote(symbols)

        oi_data = []
        for _, row in options.iterrows():
            sym = "NFO:" + row["tradingsymbol"]
            q = quotes.get(sym)
            if not q:
                continue
            oi_data.append({"Strike": float(row["strike"]), "Type": row["instrument_type"], "OI": float(q.get("oi", 0))})

        oi_df = pd.DataFrame(oi_data)
        if oi_df.empty:
            _st_error_block("No OI data returned from Kite quote API (check token/permissions).")
            return

        calls = oi_df[oi_df["Type"] == "CE"].set_index("Strike")["OI"].sort_index()
        puts = oi_df[oi_df["Type"] == "PE"].set_index("Strike")["OI"].sort_index()

        strikes = np.sort(list(set(calls.index).union(set(puts.index))))
        calls = calls.reindex(strikes, fill_value=0)
        puts = puts.reindex(strikes, fill_value=0)

        pain_values = {}
        for test_strike in strikes:
            call_loss = np.maximum(0, test_strike - strikes) * calls.values
            put_loss = np.maximum(0, strikes - test_strike) * puts.values
            pain_values[test_strike] = float(np.sum(call_loss) + np.sum(put_loss))
        max_pain_strike = float(min(pain_values, key=pain_values.get))

    except Exception as e:
        _st_error_block("Failed to fetch/process Kite data.", e)
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Spot", _fmt_float(spot_price, 2))
    m2.metric("Max Pain", _fmt_float(max_pain_strike, 0))
    m3.metric("Expiry", current_expiry.strftime("%d %b %Y"))

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 8), dpi=120)
    ax.barh(strikes, calls.values / 100000, height=25, color="#FF3333", alpha=0.8, label="Call OI (Resistance)")
    ax.barh(strikes, -puts.values / 100000, height=25, color="#00FF00", alpha=0.8, label="Put OI (Support)")
    ax.axhline(spot_price, color="#00FFFF", linestyle="-", linewidth=2, label=f"Spot: {spot_price:.2f}")
    ax.axhline(max_pain_strike, color="white", linestyle="--", linewidth=2.5, label=f"Max Pain: {max_pain_strike:.0f}")

    ax.set_title(f"NIFTY OI Profile (Expiry: {current_expiry.strftime('%d %b %Y')})", fontsize=16, color="white", pad=15, fontweight="bold")
    ax.set_xlabel("Open Interest (Lakhs)", color="gray")
    ax.set_ylabel("Strike", color="gray")

    ticks = ax.get_xticks()
    ax.set_xticklabels([str(abs(int(tick))) for tick in ticks])
    ax.set_ylim(spot_price - float(plot_range), spot_price + float(plot_range))
    ax.grid(True, color="#2A2A2A", linestyle=":")
    ax.legend(loc="upper right", facecolor="black", edgecolor="gray", fontsize=10)

    props = dict(boxstyle="round,pad=0.6", facecolor="black", alpha=0.9, edgecolor="gray", linewidth=1.5)
    text_str = (
        f"Spot: {spot_price:.2f}\n"
        f"Max Pain: {max_pain_strike:.0f}\n"
        f"---------------------------\n"
        f"Highest Put Wall: {float(puts.idxmax()):.0f}\n"
        f"Highest Call Wall: {float(calls.idxmax()):.0f}"
    )
    ax.text(0.02, 0.05, text_str, transform=ax.transAxes, fontsize=11, verticalalignment="bottom", bbox=props, color="white")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def module_parkinson(nifty_ticker: str):
    st.subheader("Parkinson Volatility Estimator (vs Close-to-Close)")

    period = st.selectbox("Data period", ["6mo", "1y", "2y"], index=1, key="park_period")
    try:
        data = yf_download(nifty_ticker, period=period, threads=YF_THREADS, proxy=YF_PROXY or None)
        if SHOW_DIAGNOSTICS:
            st.write("Raw columns:", data.columns)
    except Exception as e:
        _st_error_block("Failed to download Nifty data.", e)
        return

    if data.empty:
        _st_error_block("Yahoo Finance returned no data.")
        return

    h = _safe_squeeze(data["High"])
    l = _safe_squeeze(data["Low"])
    c = _safe_squeeze(data["Close"])
    n = int(len(data))

    log_hl = np.log(h / l)
    parkinson_var = (1 / (4 * n * np.log(2))) * (log_hl**2).sum()
    parkinson_vol = float(np.sqrt(parkinson_var) * np.sqrt(252))

    log_returns = np.log(c / c.shift(1))
    c2c_vol = float(log_returns.std() * np.sqrt(252))

    m1, m2, m3 = st.columns(3)
    m1.metric("Trading days", str(n))
    m2.metric("Parkinson vol", f"{parkinson_vol*100:.2f}%")
    m3.metric("Close-to-close vol", f"{c2c_vol*100:.2f}%")

    st.caption("Parkinson volatility uses intraday high/low range; close-to-close uses only closing returns.")


def module_volatility_cone(nifty_ticker: str):
    st.subheader("Volatility Cone")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=date(2015, 1, 1), key="cone_start")
    with col2:
        end_date = st.date_input("End date", value=date.today(), key="cone_end")

    windows = [10, 20, 30, 60, 90, 120, 180, 252]

    try:
        data = yf_download(
            nifty_ticker,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            threads=YF_THREADS,
            proxy=YF_PROXY or None,
        )
        if SHOW_DIAGNOSTICS:
            st.write("Raw columns:", data.columns)
    except Exception as e:
        _st_error_block("Failed to download historical data.", e)
        return

    if data.empty or "Close" not in data:
        _st_error_block("Yahoo Finance returned no data for the selected dates.")
        return

    df = data.copy()
    df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))

    max_vol, min_vol, median_vol, current_vol = [], [], [], []
    for w in windows:
        rolling = df["Returns"].rolling(window=w).std() * np.sqrt(252)
        max_vol.append(float(rolling.max() * 100))
        min_vol.append(float(rolling.min() * 100))
        median_vol.append(float(rolling.median() * 100))
        current_vol.append(float(rolling.iloc[-1] * 100))

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
    ax.plot(windows, max_vol, marker="o", color="red", linewidth=2, label="Maximum")
    ax.plot(windows, min_vol, marker="o", color="limegreen", linewidth=2, label="Minimum")
    ax.plot(windows, median_vol, marker="s", color="white", linewidth=1.5, linestyle="--", label="Median")
    ax.plot(windows, current_vol, marker="X", color="yellow", linewidth=3, markersize=10, label="Current")
    ax.fill_between(windows, min_vol, max_vol, color="gray", alpha=0.2)
    ax.set_title(f"Volatility Cone ({nifty_ticker})", fontsize=16, fontweight="bold", color="white", pad=10)
    ax.set_xlabel("Window (trading days)")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_xticks(windows)
    ax.grid(color="gray", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def module_vrp(nifty_ticker: str, vix_ticker: str):
    st.subheader("Volatility Risk Premium (VRP)")

    period = st.selectbox("Data period", ["3mo", "6mo", "1y"], index=1, key="vrp_period")
    hv_window = st.number_input("Realized vol window (days)", 5, 60, 20, 1, key="vrp_window")

    try:
        nifty_data = yf_download(nifty_ticker, period=period, threads=YF_THREADS, proxy=YF_PROXY or None)
        vix_data = yf_download(vix_ticker, period=period, threads=YF_THREADS, proxy=YF_PROXY or None)
        if SHOW_DIAGNOSTICS:
            st.write("Nifty columns:", nifty_data.columns)
            st.write("VIX columns:", vix_data.columns)
    except Exception as e:
        _st_error_block("Failed to download data.", e)
        return

    if nifty_data.empty or vix_data.empty:
        _st_error_block("Yahoo Finance returned no data.")
        return

    nifty_close = _safe_squeeze(nifty_data["Close"])
    vix_close = _safe_squeeze(vix_data["Close"])

    log_returns = np.log(nifty_close / nifty_close.shift(1))
    hv = log_returns.rolling(window=int(hv_window)).std() * np.sqrt(252) * 100

    df = pd.DataFrame({"VIX": vix_close, "HV": hv}).dropna()
    if df.empty:
        _st_error_block("Not enough data to compute VRP.")
        return

    df["VRP"] = df["VIX"] - df["HV"]
    current_vix = float(df["VIX"].iloc[-1])
    current_hv = float(df["HV"].iloc[-1])
    current_vrp = float(df["VRP"].iloc[-1])

    if current_vrp > 0:
        regime = "POSITIVE VRP"
        stat_property = "Implied Risk > Realized Risk (Premium Expansion)"
        color_theme = "#00FF00"
    else:
        regime = "NEGATIVE VRP"
        stat_property = "Realized Risk > Implied Risk (Premium Compression)"
        color_theme = "#FF3333"

    m1, m2, m3 = st.columns(3)
    m1.metric("VIX (Implied)", f"{_fmt_float(current_vix,2)}%")
    m2.metric(f"{hv_window}-day HV (Realized)", f"{_fmt_float(current_hv,2)}%")
    m3.metric("VRP", f"{current_vrp:+.2f}%")
    st.caption(f"Regime: **{regime}** — {stat_property}")

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=120, gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(df.index, df["VIX"], color="#00FFFF", linewidth=2, label="Implied (VIX)")
    ax1.plot(df.index, df["HV"], color="#FFA500", linewidth=2, label="Realized (HV)")
    ax1.fill_between(df.index, df["HV"], df["VIX"], where=(df["VIX"] > df["HV"]), color="green", alpha=0.3, interpolate=True)
    ax1.fill_between(df.index, df["HV"], df["VIX"], where=(df["VIX"] <= df["HV"]), color="red", alpha=0.3, interpolate=True)
    ax1.set_title("Volatility Risk Premium (VRP)", fontsize=16, color="white", pad=10, fontweight="bold")
    ax1.set_ylabel("Vol (%)", color="gray")
    ax1.grid(True, color="#2A2A2A", linestyle=":")
    ax1.legend(loc="upper left", facecolor="black", edgecolor="gray")

    bar_colors = np.where(df["VRP"] > 0, "#00FF00", "#FF3333")
    ax2.bar(df.index, df["VRP"], color=bar_colors, alpha=0.7, width=1)
    ax2.axhline(0, color="white", linewidth=1)
    ax2.set_ylabel("VRP (%)", color="gray")
    ax2.grid(True, color="#2A2A2A", linestyle=":")

    props = dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.9, edgecolor=color_theme, linewidth=1.5)
    text_str = (
        f"VIX: {current_vix:.2f}%\n"
        f"{hv_window}-D HV: {current_hv:.2f}%\n"
        f"VRP: {current_vrp:+.2f}%\n"
        f"---------------------------\n"
        f"{regime}\n{stat_property}"
    )
    ax1.text(0.02, 0.05, text_str, transform=ax1.transAxes, fontsize=11, verticalalignment="bottom", bbox=props, color="white")

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# App layout
# -----------------------------
st.title("Single Dashboard (Streamlit)")
st.caption("Modules: Correlation, Expected Move, Hurst, IVR/IVP, Liquidity Sweeps, OI Profile, Parkinson, Vol Cone, VRP.")

tabs = st.tabs(
    [
        "Correlation",
        "Expected Move",
        "Hurst",
        "IVR & IVP",
        "Liquidity Detector",
        "Open Interest",
        "Parkinson",
        "Volatility Cone",
        "VRP",
    ]
)

with tabs[0]:
    module_correlation(default_nifty, default_bank)
with tabs[1]:
    module_expected_move(default_nifty, default_vix)
with tabs[2]:
    module_hurst(default_nifty)
with tabs[3]:
    module_ivr_ivp(default_vix)
with tabs[4]:
    module_liquidity_sweep(default_nifty)
with tabs[5]:
    module_open_interest_profile()
with tabs[6]:
    module_parkinson(default_nifty)
with tabs[7]:
    module_volatility_cone(default_nifty)
with tabs[8]:
    module_vrp(default_nifty, default_vix)
