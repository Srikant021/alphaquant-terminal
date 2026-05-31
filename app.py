"""
AlphaQuant Terminal Pro — Rewritten for clarity, correctness, and learning.
Run with:  streamlit run alphaquant_terminal.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import time
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import scipy.stats as si
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# Optional dependencies — degrade gracefully if missing
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlphaQuant Terminal Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  html, body, .stApp {
    font-family: 'Syne', sans-serif;
    background: #0a0a0f;
    color: #e0e0f0;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #12122a 100%);
    border-right: 1px solid #1e1e3a;
  }

  /* Cards */
  .aq-card {
    background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s ease;
  }
  .aq-card:hover { border-color: #4a4aaa; }

  .aq-card-label {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6666aa;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
  }
  .aq-card-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
  }
  .aq-card-delta {
    font-size: 0.82rem;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
  }
  .aq-positive { color: #00e5a0; }
  .aq-negative { color: #ff4d6a; }
  .aq-neutral  { color: #aaaacc; }

  /* Quick stat tiles */
  .aq-tile {
    background: rgba(255,255,255,0.03);
    border: 1px solid #1e1e3a;
    border-radius: 10px;
    padding: 12px 14px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .aq-tile:hover {
    background: rgba(100,100,255,0.08);
    border-color: #4a4aaa;
    transform: translateY(-2px);
  }
  .aq-tile-key   { font-size: 0.7rem; color: #6666aa; letter-spacing: 0.08em; text-transform: uppercase; }
  .aq-tile-val   { font-size: 1.05rem; font-weight: 700; color: #fff; margin: 4px 0; font-family: 'JetBrains Mono', monospace; }
  .aq-tile-sub   { font-size: 0.68rem; color: #555588; }

  /* Section headers */
  .aq-section {
    font-size: 1.1rem;
    font-weight: 800;
    color: #c0c0ff;
    letter-spacing: 0.05em;
    margin: 22px 0 10px;
    border-left: 3px solid #5555ff;
    padding-left: 10px;
  }

  /* Info boxes */
  .aq-info {
    background: rgba(85,85,255,0.08);
    border: 1px solid rgba(85,85,255,0.25);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #b0b0dd;
    margin: 8px 0;
  }

  /* Signal badge */
  .aq-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.06em;
  }
  .aq-badge-bull { background: rgba(0,229,160,0.15); color: #00e5a0; border: 1px solid #00e5a0; }
  .aq-badge-bear { background: rgba(255,77,106,0.15); color: #ff4d6a; border: 1px solid #ff4d6a; }
  .aq-badge-neut { background: rgba(170,170,200,0.15); color: #aaaacc; border: 1px solid #aaaacc; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #3333aa, #5555ff);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.04em;
    transition: all 0.2s ease;
  }
  .stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 16px rgba(85,85,255,0.4);
  }

  /* Matplotlib dark patch */
  .element-container { background: transparent !important; }

  @media (max-width: 768px) {
    .aq-card-value { font-size: 1.2rem; }
  }
</style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CRYPTO_ASSETS = {
    "Bitcoin":  "BTC-USD",
    "Ethereum": "ETH-USD",
    "Dogecoin": "DOGE-USD",
    "XRP":      "XRP-USD",
}

INDIAN_ASSETS = {
    "Nifty 50":    "^NSEI",
    "Sensex":      "^BSESN",
    "Bank Nifty":  "^NSEBANK",
    "Gold (MCX)":  "GOLDM.NS",
    "Silver (MCX)":"SILVERM.NS",
}

BINANCE_MAP = {
    "BTC-USD":  "BTCUSDT",
    "ETH-USD":  "ETHUSDT",
    "DOGE-USD": "DOGEUSDT",
    "XRP-USD":  "XRPUSDT",
}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "market":           "Crypto",
    "live_mode":        False,
    "refresh_sec":      120,
    "asset_choice":     "Bitcoin",
    "active_tab":       "📊 Dashboard",
    "analysis_module":  "Hurst Exponent",
    "hist_data":        {},
    "paper_balance":    100_000.0,
    "paper_positions":  [],
    "paper_history":    [],
    "trade_journal":    [],
    "snapshots":        [],
    "alert_price":      0.0,
    "alert_vol":        0.0,
    "auto_exit":        True,
    "ml_model":         None,
    "ml_trained":       False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# ── DATA UTILITIES ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def yf_download(ticker: str, period: str = "1y", interval: str = "1d",
                max_retries: int = 3) -> pd.DataFrame:
    """
    Robust yfinance download with retries and MultiIndex flattening.
    Returns empty DataFrame on failure.
    """
    for attempt in range(max_retries):
        try:
            raw = yf.download(ticker, period=period, interval=interval,
                              progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                raise ValueError("Empty response")
            # Flatten MultiIndex columns produced by yfinance >= 0.2
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            return raw
        except Exception as exc:
            logger.warning("yf_download attempt %d failed for %s: %s",
                           attempt + 1, ticker, exc)
            time.sleep(2 ** attempt)
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker_dict: dict) -> dict[str, pd.DataFrame]:
    """Download 2-year daily history for all assets in ticker_dict."""
    bundle: dict[str, pd.DataFrame] = {}
    for name, ticker in ticker_dict.items():
        df = yf_download(ticker, period="2y")
        if not df.empty:
            bundle[name] = df
    return bundle


@st.cache_data(ttl=120, show_spinner=False)
def fetch_live_price(ticker: str) -> dict | None:
    """
    Return latest price info dict:
      {spot, prev_close, change, pct, ts}
    or None on failure.
    """
    df = yf_download(ticker, period="5d")
    if df.empty or len(df) < 2:
        return None
    close = df["Close"].squeeze()
    last  = float(close.iloc[-1])
    prev  = float(close.iloc[-2])
    chg   = last - prev
    pct   = (chg / prev) * 100 if prev else 0.0
    return {"spot": last, "prev_close": prev, "change": chg,
            "pct": pct, "ts": datetime.now().strftime("%H:%M:%S")}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday(ticker: str, interval: str = "15m") -> pd.DataFrame | None:
    """Intraday OHLCV (last 5 days)."""
    df = yf_download(ticker, period="5d", interval=interval)
    required = {"Open", "High", "Low", "Close"}
    if df.empty or not required.issubset(df.columns):
        return None
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_deribit_chain(coin: str = "BTC") -> pd.DataFrame | None:
    """Fetch live option chain from Deribit public API."""
    base = "https://www.deribit.com/api/v2/public/"
    try:
        r = requests.get(
            base + "get_instruments",
            params={"currency": coin, "kind": "option", "expired": "false"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        records = []
        for inst in r.json().get("result", [])[:40]:
            name = inst["instrument_name"]
            r2 = requests.get(base + "get_order_book",
                              params={"instrument_name": name, "depth": 1}, timeout=5)
            if r2.status_code != 200:
                continue
            book = r2.json().get("result", {})
            greeks = book.get("greeks", {})
            records.append({
                "instrument":       name,
                "strike":           inst["strike"],
                "option_type":      "call" if name.split("-")[-1] == "C" else "put",
                "expiry_ts":        inst["expiration_timestamp"],
                "mark_iv":          book.get("mark_iv", 0.0),
                "underlying_price": book.get("underlying_price", 0.0),
                "open_interest":    book.get("open_interest", 0.0),
                "delta":            greeks.get("delta", 0.0),
                "gamma":            greeks.get("gamma", 0.0),
                "theta":            greeks.get("theta", 0.0),
                "vega":             greeks.get("vega", 0.0),
            })
        return pd.DataFrame(records) if records else None
    except Exception as exc:
        logger.warning("Deribit fetch failed: %s", exc)
        return None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_binance_funding(symbol: str = "BTCUSDT") -> float | None:
    """Current funding rate (%) from Binance Futures."""
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": symbol}, timeout=5,
        )
        if r.status_code == 200:
            return float(r.json()["lastFundingRate"]) * 100
    except Exception:
        pass
    return None


@st.cache_data(ttl=10, show_spinner=False)
def fetch_binance_orderbook(symbol: str = "BTCUSDT",
                             limit: int = 20) -> tuple[pd.DataFrame | None,
                                                       pd.DataFrame | None]:
    """Live order book from Binance."""
    try:
        r = requests.get("https://api.binance.com/api/v3/depth",
                         params={"symbol": symbol, "limit": limit}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            bids = pd.DataFrame(data["bids"], columns=["Price", "Size"],
                                dtype=float)
            asks = pd.DataFrame(data["asks"], columns=["Price", "Size"],
                                dtype=float)
            return bids, asks
    except Exception:
        pass
    return None, None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_india_vix(period: str = "1y") -> pd.Series | None:
    """India VIX closing series."""
    df = yf_download("^INDIAVIX", period=period)
    if df.empty:
        return None
    return df["Close"].squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# ── QUANTITATIVE FUNCTIONS ───────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI using Wilder's smoothing (exponential moving average).
    This is the correct method — the original code used mean/std which is wrong.
    """
    delta = prices.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    # Wilder smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_hurst(series: np.ndarray, max_lags: int = 20) -> float:
    """
    Hurst exponent via R/S analysis.
    H > 0.55 → trending
    H < 0.45 → mean-reverting
    H ≈ 0.50 → random walk
    Returns np.nan when data is insufficient.
    """
    n = len(series)
    if n < 20:
        return np.nan
    lags = range(2, min(max_lags, n // 4))
    if len(lags) < 3:
        return np.nan
    try:
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag])))
               for lag in lags]
        if any(t <= 0 for t in tau):
            return np.nan
        slope, _ = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return slope * 2.0
    except Exception:
        return np.nan


def calculate_parkinson_vol(high: pd.Series, low: pd.Series,
                             periods_per_year: int = 252) -> float:
    """
    Parkinson (1980) volatility estimator using daily high-low range.
    More efficient than close-to-close for liquid markets.
    """
    if len(high) != len(low) or len(high) < 2:
        return 0.0
    log_hl = (np.log(high / low) ** 2)
    n = len(log_hl)
    return float(np.sqrt((log_hl.sum() / (4 * n * np.log(2))) * periods_per_year) * 100)


def calculate_iv_rank_percentile(close: pd.Series,
                                  window: int = 20) -> tuple[float, float]:
    """
    IV Rank  = (current_vol - 1y_low) / (1y_high - 1y_low) × 100
    IV Percentile = % of days current_vol > rolling_vol

    High IVR (>50) → options are expensive → prefer selling premium.
    Low  IVR (<30) → options are cheap     → prefer buying  premium.
    """
    if len(close) < window + 5:
        return 50.0, 50.0
    log_ret     = np.log(close / close.shift(1)).dropna()
    rolling_vol = log_ret.rolling(window).std() * np.sqrt(252) * 100
    rolling_vol = rolling_vol.dropna()
    if rolling_vol.empty:
        return 50.0, 50.0
    cur  = float(rolling_vol.iloc[-1])
    vmin = float(rolling_vol.min())
    vmax = float(rolling_vol.max())
    ivr  = ((cur - vmin) / (vmax - vmin)) * 100 if vmax != vmin else 50.0
    ivp  = float((rolling_vol < cur).mean()) * 100
    return ivr, ivp


def garch_forecast(ticker: str,
                   trading_days: int = 252) -> tuple[float, float]:
    """
    Fit GARCH(1,1) and GJR-GARCH(1,1,1) models.
    Returns annualised volatility forecasts (%) for both.
    GJR-GARCH adds an asymmetric term that captures the leverage effect
    (bad news increases vol more than good news).
    Falls back to simple historical vol if arch is unavailable.
    """
    df = yf_download(ticker, period="1y")
    if df.empty:
        return 80.0, 80.0

    close = df["Close"].squeeze()
    ret   = (100 * close.pct_change().dropna())

    if len(ret) < 100 or not ARCH_AVAILABLE:
        # Fallback: annualised historical vol
        hv = float(ret.std() * np.sqrt(trading_days))
        return hv, hv

    def _fit_and_forecast(o: int) -> float:
        try:
            model = arch_model(ret, vol="GARCH", p=1, o=o, q=1, rescale=True)
            res   = model.fit(disp="off")
            var   = res.forecast(horizon=1).variance.iloc[-1, 0]
            return float(np.sqrt(var) * np.sqrt(trading_days))
        except Exception:
            return float(ret.std() * np.sqrt(trading_days))

    return _fit_and_forecast(0), _fit_and_forecast(1)


def black_scholes_greeks(S: float, K: float, T: float, r: float,
                          sigma: float,
                          option_type: str = "call") -> dict[str, float]:
    """
    Black-Scholes option pricing and Greeks.
    S     = spot price
    K     = strike price
    T     = time to expiry in years
    r     = risk-free rate
    sigma = implied volatility (annualised, decimal)
    Returns price, delta, gamma, theta (per day), vega (per 1% move).
    """
    T = max(T, 1e-6)
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        pdf_d1 = si.norm.pdf(d1)
        cdf_d1 = si.norm.cdf(d1)
        cdf_d2 = si.norm.cdf(d2)
        disc   = K * np.exp(-r * T)

        if option_type == "call":
            price = S * cdf_d1 - disc * cdf_d2
            delta = cdf_d1
            theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                     - r * disc * cdf_d2) / 365
        else:
            price = disc * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
            delta = cdf_d1 - 1
            theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                     + r * disc * si.norm.cdf(-d2)) / 365

        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega  = (S * np.sqrt(T) * pdf_d1) / 100   # per 1% IV move

        return {
            "price": max(0.01, price),
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega":  vega,
        }
    except Exception:
        return {"price": 0.0, "delta": 0.0, "gamma": 0.0,
                "theta": 0.0, "vega": 0.0}


def portfolio_greeks(positions: list[dict],
                      spot: float,
                      sigma: float,
                      r: float = 0.05) -> dict[str, float]:
    """
    Aggregate Greeks across all open paper positions.
    Spot positions contribute only delta (±1).
    """
    agg = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    for pos in positions:
        sign = 1 if pos["Direction"] == "Long" else -1
        if pos.get("Type", "Spot") == "Spot":
            agg["delta"] += sign
        else:
            K   = pos.get("Strike", spot)
            T   = pos.get("Expiry", 1) / 252
            g   = black_scholes_greeks(spot, K, T, r, sigma,
                                        pos.get("Type", "call"))
            for key in agg:
                agg[key] += sign * g[key]
    margin = abs(agg["delta"]) * spot * 0.10
    return {**agg, "margin_est": margin}


# ─────────────────────────────────────────────────────────────────────────────
# ── ML MODEL ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def build_features(close: pd.Series) -> pd.DataFrame:
    """
    Create a tidy feature matrix from a price series.
    All features are computed correctly:
      - Returns: pct change
      - Vol: 10-period rolling std of returns
      - RSI: Wilder method (correct)
      - MACD: 12/26 EMA diff
      - Target: 1 if next bar closes higher
    """
    df = pd.DataFrame({"close": close})
    df["ret"]    = df["close"].pct_change()
    df["vol10"]  = df["ret"].rolling(10).std()
    df["rsi14"]  = calculate_rsi(df["close"], 14)
    ema12        = df["close"].ewm(span=12, adjust=False).mean()
    ema26        = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]   = ema12 - ema26
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna()


def train_ml_model(close: pd.Series) -> bool:
    """Train XGBoost classifier on last 500 bars. Returns success flag."""
    if not ML_AVAILABLE or len(close) < 150:
        return False
    df = build_features(close).tail(500)
    if len(df) < 100:
        return False
    features = ["ret", "vol10", "rsi14", "macd"]
    X, y = df[features].values, df["target"].values
    try:
        model = XGBClassifier(n_estimators=100, max_depth=3,
                              use_label_encoder=False, eval_metric="logloss")
        model.fit(X, y)
        st.session_state["ml_model"]   = model
        st.session_state["ml_trained"] = True
        return True
    except Exception as exc:
        logger.warning("ML training failed: %s", exc)
        return False


def ml_predict(close: pd.Series) -> int | None:
    """
    Predict direction for next bar.
    Returns 1 (bullish), 0 (bearish), or None.
    """
    if not st.session_state["ml_trained"] or st.session_state["ml_model"] is None:
        return None
    if len(close) < 30:
        return None
    try:
        df = build_features(close.tail(100))
        if df.empty:
            return None
        features = ["ret", "vol10", "rsi14", "macd"]
        X = df[features].iloc[[-1]].values
        return int(st.session_state["ml_model"].predict(X)[0])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ── SIGNAL ENGINE ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def generate_signal(
    asset_name: str,
    ticker: str,
    spot: float,
    spot_change: float,
    garch_vol: float,
    trading_days: int,
    park_vol: float | None = None,
    ivr: float | None = None,
    ivp: float | None = None,
    deribit_iv: float | None = None,
    funding_rate: float | None = None,
) -> dict:
    """
    Unified signal generator.  All inputs passed explicitly — no globals.

    Returns a dict with:
      regime, vol_environment, strategy, direction,
      confidence, risk_level, reasoning, daily_move
    """
    hist = st.session_state["hist_data"].get(asset_name)
    if hist is None or hist.empty:
        return {"error": "No historical data available."}

    close = hist["Close"].squeeze()
    if len(close) < 60:
        return {"error": "Insufficient price history."}

    # ── Hurst regime ─────────────────────────────────────────────────────────
    hurst = calculate_hurst(np.log(close.values[-200:]))
    if np.isnan(hurst):
        regime = "unknown"
    elif hurst > 0.55:
        regime = "trending"
    elif hurst < 0.45:
        regime = "mean_reverting"
    else:
        regime = "random_walk"

    # ── Historical vol (20-day) ───────────────────────────────────────────────
    log_ret = np.log(close / close.shift(1)).dropna()
    hist_vol_20 = float(log_ret.tail(21).std() * np.sqrt(trading_days) * 100)

    # ── Daily expected move ───────────────────────────────────────────────────
    daily_move = spot * (garch_vol / 100) / np.sqrt(trading_days)

    # ── Liquidity sweeps (intraday microstructure) ────────────────────────────
    intra = fetch_intraday(ticker, "30m")
    sweep_up = sweep_down = False
    if intra is not None and not intra.empty:
        h = intra["High"]; l = intra["Low"]; c = intra["Close"]
        prev_high = h.rolling(20).max().shift(1)
        prev_low  = l.rolling(20).min().shift(1)
        sweep_up   = bool(((h > prev_high) & (c < prev_high)).any())
        sweep_down = bool(((l < prev_low)  & (c > prev_low)).any())

    # ── Volatility environment ────────────────────────────────────────────────
    vol_votes = 0
    vol_env   = "neutral"

    def _vote(condition_expensive: bool, condition_cheap: bool) -> None:
        nonlocal vol_votes, vol_env
        if condition_expensive:
            vol_votes += 1
            vol_env = "expensive" if vol_env != "cheap" else "mixed"
        elif condition_cheap:
            vol_votes += 1
            vol_env = "cheap" if vol_env != "expensive" else "mixed"

    _vote(garch_vol > hist_vol_20 * 1.2, garch_vol < hist_vol_20 * 0.8)
    if ivr is not None:
        _vote(ivr > 65, ivr < 30)
    if park_vol is not None:
        _vote(park_vol > garch_vol * 1.2, park_vol < garch_vol * 0.8)
    if deribit_iv is not None:
        _vote(deribit_iv > garch_vol * 1.15, deribit_iv < garch_vol * 0.85)
    if funding_rate is not None:
        _vote(funding_rate > 0.10, funding_rate < -0.10)

    if vol_votes == 0:
        vol_env = "neutral"

    # ── Strategy selection ────────────────────────────────────────────────────
    direction  = "neutral"
    strategy   = "Iron Condor"
    confidence = 50
    risk_level = "low"

    if regime == "trending":
        direction = "bullish" if spot_change > 0 else "bearish"
        if vol_env == "cheap":
            strategy   = "Long Call" if direction == "bullish" else "Long Put"
            confidence += 15
            risk_level  = "high"
        elif direction == "bullish":
            strategy   = "Bull Call Spread"
            confidence += 8
            risk_level  = "medium"
        else:
            strategy   = "Bear Put Spread"
            confidence += 8
            risk_level  = "medium"

    elif regime == "mean_reverting":
        if sweep_up:                          # price shot up and failed
            direction = "bearish"
            strategy  = "Bear Put Spread"
        elif sweep_down:                      # price dipped and recovered
            direction = "bullish"
            strategy  = "Bull Call Spread"
        else:
            strategy  = "Iron Condor"
        confidence += 10
        risk_level  = "low" if strategy == "Iron Condor" else "medium"

    else:   # random walk
        if vol_env == "expensive":
            strategy   = "Short Strangle"
            risk_level = "medium"
        elif sweep_up:
            direction  = "bearish"
            strategy   = "Bear Put Spread"
            risk_level = "medium"
        elif sweep_down:
            direction  = "bullish"
            strategy   = "Bull Call Spread"
            risk_level = "medium"

    # ── ML overlay ───────────────────────────────────────────────────────────
    ml_pred = ml_predict(close)
    if ml_pred is not None:
        if (ml_pred == 1 and direction == "bullish") or \
           (ml_pred == 0 and direction == "bearish"):
            confidence += 10
        else:
            confidence -= 5
    confidence = max(0, min(100, confidence))

    # ── Reasoning string ─────────────────────────────────────────────────────
    parts = [
        f"Hurst={hurst:.3f}({regime})",
        f"GARCH={garch_vol:.1f}%",
        f"HV20={hist_vol_20:.1f}%",
        f"Vol={vol_env}",
        f"Sweep={'↑' if sweep_up else '↓' if sweep_down else '—'}",
        f"Conf={confidence}%",
        f"Risk={risk_level}",
    ]
    if park_vol:     parts.append(f"Park={park_vol:.1f}%")
    if ivr:          parts.append(f"IVR={ivr:.0f}%")
    if deribit_iv:   parts.append(f"DerIV={deribit_iv:.1f}%")
    if funding_rate: parts.append(f"Fund={funding_rate:.3f}%")
    if ml_pred is not None:
        parts.append(f"ML={'↑' if ml_pred == 1 else '↓'}")

    return {
        "regime":      regime,
        "vol_env":     vol_env,
        "strategy":    strategy,
        "direction":   direction,
        "confidence":  confidence,
        "risk_level":  risk_level,
        "daily_move":  daily_move,
        "garch_vol":   garch_vol,
        "hist_vol_20": hist_vol_20,
        "hurst":       hurst,
        "sweep_up":    sweep_up,
        "sweep_down":  sweep_down,
        "ml_pred":     ml_pred,
        "reasoning":   " · ".join(parts),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ── CHARTING FUNCTIONS ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def _fig_base(title: str, figsize=(12, 6)) -> tuple:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#0a0a0f")
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_title(title, color="#c0c0ff", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#666688")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e1e3a")
    ax.grid(True, color="#14142a", linestyle=":")
    return fig, ax


def chart_hurst(close: pd.Series) -> plt.Figure | None:
    if len(close) < 120:
        return None
    log_p    = np.log(close)
    h_series = log_p.rolling(60).apply(
        lambda x: calculate_hurst(x.values), raw=False
    )
    df = pd.DataFrame({"Close": close, "Hurst": h_series}).dropna()
    if df.empty:
        return None

    h_now = df["Hurst"].iloc[-1]
    if h_now > 0.55:   color, label = "#00e5a0", "TRENDING"
    elif h_now < 0.45: color, label = "#ff4d6a", "MEAN REVERTING"
    else:              color, label = "#ffaa33", "RANDOM WALK"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={"height_ratios": [2, 1]})
    for ax in (ax1, ax2):
        ax.set_facecolor("#0a0a0f")
        ax.grid(True, color="#14142a", linestyle=":")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e1e3a")
        ax.tick_params(colors="#666688")
    fig.patch.set_facecolor("#0a0a0f")

    ax1.plot(df.index, df["Close"], color="#7777ff", linewidth=1.5, label="Price")
    ax1.axvspan(df.index[-15], df.index[-1], color=color, alpha=0.08)
    ax1.set_title("Market Regime — Hurst Exponent", color="#c0c0ff",
                  fontsize=13, fontweight="bold")
    ax1.legend(facecolor="#0f0f1e", edgecolor="#1e1e3a")

    ax2.plot(df.index, df["Hurst"], color="#00ccff", linewidth=1.8)
    ax2.axhline(0.55, color="#00e5a0", linestyle="--", linewidth=1.2, label="Trending >0.55")
    ax2.axhline(0.45, color="#ff4d6a", linestyle="--", linewidth=1.2, label="Mean Rev. <0.45")
    ax2.axhline(0.50, color="#333355", linewidth=1.0)
    ax2.fill_between(df.index, 0.55, df["Hurst"],
                     where=df["Hurst"] > 0.55, color="#00e5a0", alpha=0.15, interpolate=True)
    ax2.fill_between(df.index, 0.45, df["Hurst"],
                     where=df["Hurst"] < 0.45, color="#ff4d6a", alpha=0.15, interpolate=True)
    ax2.set_ylim(0.3, 0.7)
    ax2.legend(facecolor="#0f0f1e", edgecolor="#1e1e3a", fontsize=9)

    bbox = dict(boxstyle="round,pad=0.5", facecolor="#0f0f1e",
                edgecolor=color, linewidth=1.5, alpha=0.9)
    ax1.text(0.02, 0.05,
             f"H = {h_now:.3f}  →  {label}",
             transform=ax1.transAxes, color=color,
             fontsize=11, fontweight="bold", bbox=bbox, va="bottom")
    plt.tight_layout()
    return fig


def chart_volatility_cone(close: pd.Series,
                           trading_days: int = 252) -> plt.Figure | None:
    log_ret = np.log(close / close.shift(1)).dropna()
    windows = [5, 10, 20, 30, 60, 90, 120, 180, 252]
    stats   = {"max": [], "min": [], "median": [], "current": []}

    for w in windows:
        rv = log_ret.rolling(w).std() * np.sqrt(trading_days) * 100
        rv = rv.dropna()
        if rv.empty:
            for v in stats.values():
                v.append(np.nan)
        else:
            stats["max"].append(rv.max())
            stats["min"].append(rv.min())
            stats["median"].append(rv.median())
            stats["current"].append(rv.iloc[-1])

    fig, ax = _fig_base("Volatility Cone")
    ax.plot(windows, stats["max"],    "o-", color="#ff4d6a", label="Max",     linewidth=1.5)
    ax.plot(windows, stats["min"],    "o-", color="#00e5a0", label="Min",     linewidth=1.5)
    ax.plot(windows, stats["median"], "s--", color="#aaaacc", label="Median",  linewidth=1.2)
    ax.plot(windows, stats["current"],"X-", color="#ffdd44", label="Current", linewidth=2,
            markersize=10)
    ax.fill_between(windows, stats["min"], stats["max"],
                    color="#7777ff", alpha=0.07)
    ax.set_xlabel("Window (days)", color="#666688")
    ax.set_ylabel("Annualised Vol (%)", color="#666688")
    ax.legend(facecolor="#0f0f1e", edgecolor="#1e1e3a")
    plt.tight_layout()
    return fig


def chart_vrp(close: pd.Series,
              iv_series: pd.Series | None,
              trading_days: int = 252) -> plt.Figure | None:
    log_ret = np.log(close / close.shift(1)).dropna()
    hv      = log_ret.rolling(20).std() * np.sqrt(trading_days) * 100
    hv      = hv.dropna()

    if iv_series is None:
        return None

    common = hv.index.intersection(iv_series.index)
    if len(common) < 10:
        return None

    hv_c = hv[common]; iv_c = iv_series[common]
    vrp   = iv_c - hv_c
    label = "Positive VRP → Sell Premium" if float(vrp.iloc[-1]) > 0 \
            else "Negative VRP → Buy Premium"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                    gridspec_kw={"height_ratios": [2, 1]})
    for ax in (ax1, ax2):
        ax.set_facecolor("#0a0a0f")
        ax.grid(True, color="#14142a", linestyle=":")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e1e3a")
        ax.tick_params(colors="#666688")
    fig.patch.set_facecolor("#0a0a0f")

    ax1.plot(common, iv_c, color="#00ccff", linewidth=1.5, label="Implied Vol")
    ax1.plot(common, hv_c, color="#ffaa33", linewidth=1.5, label="20d Realised Vol")
    ax1.fill_between(common, hv_c, iv_c,
                     where=(iv_c > hv_c), color="#00e5a0", alpha=0.15, interpolate=True)
    ax1.fill_between(common, hv_c, iv_c,
                     where=(iv_c <= hv_c), color="#ff4d6a", alpha=0.15, interpolate=True)
    ax1.set_title("Volatility Risk Premium (VRP)", color="#c0c0ff",
                  fontsize=13, fontweight="bold")
    ax1.legend(facecolor="#0f0f1e", edgecolor="#1e1e3a")

    colors = ["#00e5a0" if v > 0 else "#ff4d6a" for v in vrp]
    ax2.bar(common, vrp, color=colors, alpha=0.7)
    ax2.axhline(0, color="#444466")
    ax2.set_ylabel("VRP (%)", color="#666688")

    bbox = dict(boxstyle="round", facecolor="#0f0f1e", edgecolor="#5555ff")
    ax1.text(0.02, 0.05, f"VRP: {float(vrp.iloc[-1]):+.1f}%  {label}",
             transform=ax1.transAxes, color="#c0c0ff", fontsize=10, bbox=bbox, va="bottom")
    plt.tight_layout()
    return fig


def chart_ivr(close: pd.Series,
              iv_series: pd.Series | None,
              label: str = "Vol") -> plt.Figure | None:
    log_ret     = np.log(close / close.shift(1)).dropna()
    rolling_vol = log_ret.rolling(20).std() * np.sqrt(252) * 100
    series      = iv_series if iv_series is not None else rolling_vol.dropna()

    cur  = float(series.iloc[-1])
    vmin = float(series.min()); vmax = float(series.max())
    ivr  = ((cur - vmin) / (vmax - vmin)) * 100 if vmax != vmin else 50.0
    ivp  = float((series < cur).mean()) * 100

    fig, ax = _fig_base(f"IV Rank & IV Percentile — {label}")
    ax.plot(series.index, series, color="#00ccff", linewidth=1.5, label=label)
    ax.axhline(vmax, color="#ff4d6a", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(vmin, color="#00e5a0", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(cur,  color="#ffffff", linestyle="-",  linewidth=1.5)
    bbox = dict(boxstyle="round", facecolor="#0f0f1e", edgecolor="#5555ff")
    ax.text(0.02, 0.92,
            f"IVR: {ivr:.1f}%   IVP: {ivp:.1f}%\n"
            f"{'High IV → Sell Premium' if ivr > 50 else 'Low IV → Buy Premium'}",
            transform=ax.transAxes, color="#c0c0ff", fontsize=10,
            bbox=bbox, va="top")
    ax.legend(facecolor="#0f0f1e", edgecolor="#1e1e3a")
    plt.tight_layout()
    return fig


def chart_expected_move(spot: float,
                         iv_pct: float,
                         recent_close: pd.Series,
                         currency: str = "$",
                         trading_days: int = 252) -> plt.Figure:
    daily_vol  = iv_pct / 100 / np.sqrt(trading_days)
    daily_move = spot * daily_vol
    upper      = spot + daily_move
    lower      = spot - daily_move

    fig, ax = _fig_base("Expected Daily Move (±1σ)", figsize=(12, 6))
    x = np.arange(len(recent_close))
    ax.plot(x, recent_close.values, color="#7777ff", linewidth=2, marker="o",
            markersize=4, label="Price")
    t = len(recent_close)
    ax.scatter(t, upper, color="#00e5a0", s=150, zorder=5, marker="^")
    ax.scatter(t, lower, color="#ff4d6a", s=150, zorder=5, marker="v")
    ax.hlines([upper, spot, lower], xmin=t - 1, xmax=t,
              colors=["#00e5a0", "#ffffff", "#ff4d6a"],
              linestyles="--", linewidth=1.4)
    ax.fill_between([t - 1, t], [spot, lower], [spot, upper],
                    color="#5555ff", alpha=0.12)
    ax.set_xticks([])
    bbox = dict(boxstyle="round,pad=0.6", facecolor="#0f0f1e",
                edgecolor="#5555ff", linewidth=1.5)
    ax.text(
        0.02, 0.5,
        f"Spot:  {currency}{spot:,.2f}\n"
        f"IV:    {iv_pct:.1f}%\n"
        f"±Move: {currency}{daily_move:,.1f}\n"
        f"Upper: {currency}{upper:,.2f}\n"
        f"Lower: {currency}{lower:,.2f}",
        transform=ax.transAxes, color="#c0c0ff",
        fontsize=11, fontweight="bold", bbox=bbox, va="center",
    )
    ax.legend(facecolor="#0f0f1e", edgecolor="#1e1e3a")
    plt.tight_layout()
    return fig


def chart_payoff(legs: list[dict], spot: float) -> plt.Figure:
    """
    legs: list of {"label": str, "K": float, "premium": float, "sign": int}
    sign = +1 for Long, -1 for Short
    """
    prices  = np.linspace(spot * 0.80, spot * 1.20, 300)
    payoff  = np.zeros_like(prices)
    for leg in legs:
        K      = leg["K"]
        prem   = leg["premium"]
        sign   = leg["sign"]      # +1 long, -1 short
        is_call = "Call" in leg["label"]
        for i, p in enumerate(prices):
            intrinsic = max(0.0, p - K) if is_call else max(0.0, K - p)
            payoff[i] += sign * (intrinsic - prem)

    fig, ax = _fig_base("Strategy Payoff at Expiry", figsize=(10, 5))
    ax.plot(prices, payoff, color="#00ccff", linewidth=2.5)
    ax.axhline(0, color="#444466", linewidth=1)
    ax.axvline(spot, color="#ffaa33", linewidth=1.5,
               linestyle="--", label=f"Spot {spot:,.2f}")
    ax.fill_between(prices, 0, payoff, where=(payoff > 0),
                    color="#00e5a0", alpha=0.15, interpolate=True)
    ax.fill_between(prices, 0, payoff, where=(payoff <= 0),
                    color="#ff4d6a", alpha=0.15, interpolate=True)
    ax.set_xlabel("Underlying Price at Expiry", color="#666688")
    ax.set_ylabel("Profit / Loss", color="#666688")
    ax.legend(facecolor="#0f0f1e", edgecolor="#1e1e3a")
    plt.tight_layout()
    return fig


def chart_rsi(close: pd.Series) -> plt.Figure:
    rsi = calculate_rsi(close).dropna()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                    gridspec_kw={"height_ratios": [2, 1]})
    for ax in (ax1, ax2):
        ax.set_facecolor("#0a0a0f")
        ax.grid(True, color="#14142a", linestyle=":")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e1e3a")
        ax.tick_params(colors="#666688")
    fig.patch.set_facecolor("#0a0a0f")

    ax1.plot(close.index, close, color="#7777ff", linewidth=1.5, label="Price")
    ax1.set_title("Price & RSI (Wilder, 14)", color="#c0c0ff",
                  fontsize=13, fontweight="bold")
    ax1.legend(facecolor="#0f0f1e", edgecolor="#1e1e3a")

    ax2.plot(rsi.index, rsi, color="#00ccff", linewidth=1.5)
    ax2.axhline(70, color="#ff4d6a", linestyle="--", linewidth=1.2, label="OB 70")
    ax2.axhline(30, color="#00e5a0", linestyle="--", linewidth=1.2, label="OS 30")
    ax2.fill_between(rsi.index, 70, rsi, where=(rsi > 70),
                     color="#ff4d6a", alpha=0.15, interpolate=True)
    ax2.fill_between(rsi.index, 30, rsi, where=(rsi < 30),
                     color="#00e5a0", alpha=0.15, interpolate=True)
    ax2.set_ylim(0, 100)
    ax2.legend(facecolor="#0f0f1e", edgecolor="#1e1e3a", fontsize=9)
    plt.tight_layout()
    return fig


def chart_correlation(df_pair: pd.DataFrame,
                       names: tuple[str, str]) -> plt.Figure | None:
    df = df_pair.dropna()
    if df.shape[1] != 2 or len(df) < 25:
        return None
    df.columns = list(names)
    normalised  = df / df.iloc[0] * 100
    log_ret     = np.log(df / df.shift(1)).dropna()
    roll_corr   = log_ret[names[0]].rolling(20).corr(log_ret[names[1]])
    cur_corr    = float(roll_corr.iloc[-1])

    if cur_corr > 0.8:   corr_color, corr_label = "#00e5a0", "High Correlation"
    elif cur_corr < 0.5: corr_color, corr_label = "#ff4d6a", "Decoupling"
    else:                corr_color, corr_label = "#ffaa33", "Moderate"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                    gridspec_kw={"height_ratios": [2, 1]})
    for ax in (ax1, ax2):
        ax.set_facecolor("#0a0a0f")
        ax.grid(True, color="#14142a", linestyle=":")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e1e3a")
        ax.tick_params(colors="#666688")
    fig.patch.set_facecolor("#0a0a0f")

    ax1.plot(normalised.index, normalised[names[0]],
             color="#7777ff", linewidth=1.5, label=names[0])
    ax1.plot(normalised.index, normalised[names[1]],
             color="#00ccff", linewidth=1.5, label=names[1])
    ax1.fill_between(normalised.index,
                     normalised[names[0]], normalised[names[1]],
                     alpha=0.10, color="#5555ff")
    ax1.set_title(f"Correlation: {names[0]} vs {names[1]}",
                  color="#c0c0ff", fontsize=13, fontweight="bold")
    ax1.legend(facecolor="#0f0f1e", edgecolor="#1e1e3a")

    ax2.plot(roll_corr.index, roll_corr, color="#ffaa33", linewidth=1.8)
    ax2.axhline(0.8, color="#00e5a0", linestyle="--", linewidth=1)
    ax2.axhline(0.5, color="#ff4d6a", linestyle="--", linewidth=1)
    ax2.set_ylim(-0.2, 1.1); ax2.set_ylabel("20-Day Corr", color="#666688")

    bbox = dict(boxstyle="round", facecolor="#0f0f1e", edgecolor=corr_color)
    ax1.text(0.02, 0.05,
             f"Corr: {cur_corr:.2f}  →  {corr_label}",
             transform=ax1.transAxes, color=corr_color,
             fontsize=10, fontweight="bold", bbox=bbox)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ── UI HELPERS ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, delta: str = "",
                positive: bool | None = None) -> None:
    delta_class = ("aq-positive" if positive is True
                   else "aq-negative" if positive is False
                   else "aq-neutral")
    st.markdown(
        f"""<div class="aq-card">
              <div class="aq-card-label">{label}</div>
              <div class="aq-card-value">{value}</div>
              <div class="aq-card-delta {delta_class}">{delta}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="aq-section">{title}</div>',
                unsafe_allow_html=True)


def info_box(text: str) -> None:
    st.markdown(f'<div class="aq-info">{text}</div>', unsafe_allow_html=True)


def direction_badge(direction: str) -> str:
    if direction == "bullish":
        return '<span class="aq-badge aq-badge-bull">▲ BULLISH</span>'
    if direction == "bearish":
        return '<span class="aq-badge aq-badge-bear">▼ BEARISH</span>'
    return '<span class="aq-badge aq-badge-neut">◆ NEUTRAL</span>'


# ─────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ AlphaQuant Pro")
    st.markdown("---")

    market = st.radio("Market", ["Crypto", "Indian Market"],
                      index=0 if st.session_state["market"] == "Crypto" else 1,
                      horizontal=True)
    if market != st.session_state["market"]:
        st.session_state["market"]      = market
        st.session_state["hist_data"]   = {}
        st.session_state["ml_trained"]  = False
        st.cache_data.clear()
        st.rerun()

    TICKER_DICT = CRYPTO_ASSETS if market == "Crypto" else INDIAN_ASSETS
    trading_days = 365 if market == "Crypto" else 252
    currency     = "$"  if market == "Crypto" else "₹"

    asset_choice = st.selectbox("Asset", list(TICKER_DICT.keys()))
    ticker       = TICKER_DICT[asset_choice]

    st.markdown("---")
    tab = st.radio("Navigate", [
        "📊 Dashboard",
        "📈 Technical",
        "📄 Paper Trading",
        "🧙 Strategy Wizard",
        "📓 Journal",
    ])
    st.session_state["active_tab"] = tab

    st.markdown("---")

    live_mode = st.checkbox("🟢 Live Mode", value=st.session_state["live_mode"])
    st.session_state["live_mode"] = live_mode

    with st.expander("🔔 Price Alerts"):
        alert_price = st.number_input("Price Alert", value=st.session_state["alert_price"], step=100.0)
        alert_vol   = st.number_input("GARCH Vol Alert (%)", value=st.session_state["alert_vol"], step=1.0)
        if st.button("Save Alerts"):
            st.session_state["alert_price"] = alert_price
            st.session_state["alert_vol"]   = alert_vol
            st.success("Saved!")

    if ML_AVAILABLE:
        with st.expander("🧠 ML Model"):
            st.caption("Train an XGBoost classifier on the selected asset.")
            if st.button("Train Model"):
                hist = st.session_state["hist_data"].get(asset_choice)
                if hist is not None and not hist.empty:
                    ok = train_ml_model(hist["Close"].squeeze())
                    st.success("Trained!" if ok else "Insufficient data.")
                else:
                    st.warning("Load data first.")
            if st.session_state["ml_trained"]:
                st.success("Model ready ✓")

    st.markdown("---")
    if st.button("🔄 Refresh All Data"):
        st.cache_data.clear()
        st.session_state["hist_data"] = {}
        st.rerun()
    st.caption("AlphaQuant Terminal Pro · For learning purposes only.")


# ─────────────────────────────────────────────────────────────────────────────
# ── DATA LOADING ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state["hist_data"] or \
        set(st.session_state["hist_data"].keys()) != set(TICKER_DICT.keys()):
    with st.spinner("Loading market data…"):
        st.session_state["hist_data"] = fetch_history(TICKER_DICT)

# Live price
lp = fetch_live_price(ticker)
if lp is None:
    # fallback to last two rows of history
    h = st.session_state["hist_data"].get(asset_choice)
    if h is not None and len(h) >= 2:
        last = float(h["Close"].squeeze().iloc[-1])
        prev = float(h["Close"].squeeze().iloc[-2])
        lp   = {"spot": last, "prev_close": prev,
                "change": last - prev,
                "pct": (last - prev) / prev * 100 if prev else 0.0,
                "ts": "hist"}
    else:
        lp = {"spot": 0.0, "prev_close": 0.0,
              "change": 0.0, "pct": 0.0, "ts": "unavailable"}

asset_spot   = lp["spot"]
asset_change = lp["change"]
asset_pct    = lp["pct"]

# GARCH
with st.spinner("Fitting volatility models…"):
    garch_vol, gjr_vol = garch_forecast(ticker, trading_days)

# Derived vol metrics
park_vol = ivr_val = ivp_val = None
asset_df = st.session_state["hist_data"].get(asset_choice)
if asset_df is not None and not asset_df.empty:
    close_px = asset_df["Close"].squeeze()
    if {"High", "Low"}.issubset(asset_df.columns):
        park_vol = calculate_parkinson_vol(
            asset_df["High"].squeeze().tail(60),
            asset_df["Low"].squeeze().tail(60),
            trading_days,
        )
    ivr_val, ivp_val = calculate_iv_rank_percentile(close_px, 20)

# Alerts
if st.session_state["alert_price"] > 0 and asset_spot >= st.session_state["alert_price"]:
    st.toast(f"🔔 Price alert: {asset_choice} hit {currency}{st.session_state['alert_price']:,.0f}",
             icon="🔔")
if st.session_state["alert_vol"] > 0 and garch_vol >= st.session_state["alert_vol"]:
    st.toast(f"🔔 Vol alert: GARCH {garch_vol:.1f}% ≥ {st.session_state['alert_vol']:.0f}%",
             icon="⚠️")

# Correlation pair data
if market == "Crypto":
    corr_pair_tickers = ["BTC-USD", "ETH-USD"]
    corr_names        = ("Bitcoin", "Ethereum")
else:
    corr_pair_tickers = ["^NSEI", "^NSEBANK"]
    corr_names        = ("Nifty 50", "Bank Nifty")

@st.cache_data(ttl=1800, show_spinner=False)
def _corr_data(tickers, market_key):
    raw = yf.download(tickers, period="1y", progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Close"]
    return raw[["Close"]] if "Close" in raw.columns else raw

corr_df = _corr_data(corr_pair_tickers, market)


# ─────────────────────────────────────────────────────────────────────────────
# ══ TAB: DASHBOARD ═══════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
active_tab = st.session_state["active_tab"]

if active_tab == "📊 Dashboard":
    st.title("📊 Market Intelligence Dashboard")

    # ── Market Overview ──────────────────────────────────────────────────────
    section("Market Overview")
    if market == "Crypto":
        btc_lp = fetch_live_price("BTC-USD")
        eth_lp = fetch_live_price("ETH-USD")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if btc_lp:
                metric_card("Bitcoin", f"${btc_lp['spot']:,.0f}",
                            f"{btc_lp['change']:+,.0f} ({btc_lp['pct']:+.2f}%)",
                            btc_lp["pct"] >= 0)
        with c2:
            if eth_lp:
                metric_card("Ethereum", f"${eth_lp['spot']:,.2f}",
                            f"{eth_lp['change']:+,.2f} ({eth_lp['pct']:+.2f}%)",
                            eth_lp["pct"] >= 0)
        with c3:
            metric_card("GARCH Vol", f"{garch_vol:.1f}%",
                        "Annualised 1-day ahead")
        with c4:
            metric_card("GJR-GARCH Vol", f"{gjr_vol:.1f}%",
                        "Leverage-adjusted")
    else:
        nifty_lp  = fetch_live_price("^NSEI")
        sensex_lp = fetch_live_price("^BSESN")
        vix_s     = fetch_india_vix("5d")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if nifty_lp:
                metric_card("Nifty 50", f"₹{nifty_lp['spot']:,.2f}",
                            f"{nifty_lp['pct']:+.2f}%", nifty_lp["pct"] >= 0)
        with c2:
            if sensex_lp:
                metric_card("Sensex", f"₹{sensex_lp['spot']:,.2f}",
                            f"{sensex_lp['pct']:+.2f}%", sensex_lp["pct"] >= 0)
        with c3:
            if vix_s is not None:
                metric_card("India VIX", f"{float(vix_s.iloc[-1]):.2f}",
                            "Implied vol index")
        with c4:
            metric_card("GARCH Vol", f"{garch_vol:.1f}%", "1-day ahead forecast")

    # ── Active Asset ─────────────────────────────────────────────────────────
    section(f"Active Asset — {asset_choice}")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        metric_card("Spot Price",
                    f"{currency}{asset_spot:,.2f}",
                    f"{asset_change:+,.2f} ({asset_pct:+.2f}%)",
                    asset_pct >= 0)
    with col_b:
        metric_card("Parkinson Vol",
                    f"{park_vol:.1f}%" if park_vol else "N/A",
                    "High-Low estimator")
    with col_c:
        metric_card("IV Rank",
                    f"{ivr_val:.0f}%" if ivr_val else "N/A",
                    "Sell >65 · Buy <30")
    with col_d:
        metric_card("IV Percentile",
                    f"{ivp_val:.0f}%" if ivp_val else "N/A",
                    f"Last update: {lp['ts']}")

    # ── Signal Box ───────────────────────────────────────────────────────────
    section("Signal Engine")
    with st.spinner("Computing signal…"):
        vix_now = None
        if market == "Indian Market":
            vix_s2 = fetch_india_vix("5d")
            if vix_s2 is not None:
                vix_now = float(vix_s2.iloc[-1])

        signal = generate_signal(
            asset_name   = asset_choice,
            ticker       = ticker,
            spot         = asset_spot,
            spot_change  = asset_change,
            garch_vol    = garch_vol,
            trading_days = trading_days,
            park_vol     = park_vol,
            ivr          = ivr_val,
            ivp          = ivp_val,
            deribit_iv   = vix_now,
        )

    if "error" not in signal:
        sc1, sc2, sc3 = st.columns([2, 2, 3])
        with sc1:
            st.markdown(f"""
            <div class="aq-card">
              <div class="aq-card-label">Suggested Strategy</div>
              <div class="aq-card-value" style="font-size:1.2rem">{signal['strategy']}</div>
              <div style="margin-top:8px">{direction_badge(signal['direction'])}</div>
            </div>""", unsafe_allow_html=True)
        with sc2:
            conf   = signal["confidence"]
            bar_w  = conf
            bar_c  = "#00e5a0" if conf >= 65 else "#ffaa33" if conf >= 45 else "#ff4d6a"
            st.markdown(f"""
            <div class="aq-card">
              <div class="aq-card-label">Confidence</div>
              <div class="aq-card-value">{conf}%</div>
              <div style="background:#1e1e3a;border-radius:4px;height:6px;margin-top:8px">
                <div style="background:{bar_c};width:{bar_w}%;height:6px;border-radius:4px"></div>
              </div>
              <div class="aq-card-delta aq-neutral" style="margin-top:4px">
                Risk: {signal['risk_level'].upper()}
              </div>
            </div>""", unsafe_allow_html=True)
        with sc3:
            st.markdown(f"""
            <div class="aq-card">
              <div class="aq-card-label">Regime & Vol Environment</div>
              <div class="aq-card-value" style="font-size:1rem">{signal['regime'].replace('_',' ').title()}</div>
              <div class="aq-card-delta aq-neutral">
                Vol: {signal['vol_env']} · Daily ±{currency}{signal['daily_move']:,.0f}
              </div>
            </div>""", unsafe_allow_html=True)

        with st.expander("📋 Full Reasoning"):
            st.code(signal["reasoning"], language=None)
            daily = signal["daily_move"]
            info_box(
                f"Strike zones based on daily ±{currency}{daily:,.0f}:<br>"
                f"• Directional OTM: {currency}{asset_spot - daily:,.0f} — "
                f"{currency}{asset_spot + daily:,.0f}<br>"
                f"• Short gamma sell zone: {currency}{asset_spot - daily*1.5:,.0f} / "
                f"{currency}{asset_spot + daily*1.5:,.0f}"
            )
    else:
        st.warning(signal["error"])

    # ── Live Terminal ─────────────────────────────────────────────────────────
    with st.expander("💹 Live Chart & Order Book", expanded=False):
        live_df = yf_download(ticker, period="1d", interval="5m")
        if not live_df.empty:
            fig_live = go.Figure(data=[go.Candlestick(
                x=live_df.index,
                open=live_df["Open"], high=live_df["High"],
                low=live_df["Low"],   close=live_df["Close"],
            )])
            fig_live.update_layout(
                title=f"{asset_choice} — 5m",
                xaxis_rangeslider_visible=False,
                template="plotly_dark", height=380,
                paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
            )
            st.plotly_chart(fig_live, use_container_width=True)
        else:
            st.info("Intraday chart unavailable.")

        if market == "Crypto":
            bin_sym = BINANCE_MAP.get(ticker, "BTCUSDT")
            bids, asks = fetch_binance_orderbook(bin_sym)
            if bids is not None and asks is not None:
                best_bid = float(bids["Price"].iloc[0])
                best_ask = float(asks["Price"].iloc[0])
                mid      = (best_bid + best_ask) / 2
                spread   = best_ask - best_bid
                imbal    = ((bids["Size"].sum() - asks["Size"].sum())
                            / (bids["Size"].sum() + asks["Size"].sum()))

                ob1, ob2, ob3, ob4 = st.columns(4)
                ob1.metric("Best Bid",  f"{currency}{best_bid:,.2f}")
                ob2.metric("Best Ask",  f"{currency}{best_ask:,.2f}")
                ob3.metric("Spread",    f"{currency}{spread:,.2f}")
                ob4.metric("Imbalance", f"{imbal:+.3f}",
                           "Bids heavy" if imbal > 0.1
                           else "Asks heavy" if imbal < -0.1 else "Neutral")

                fig_depth = go.Figure()
                fig_depth.add_trace(go.Scatter(
                    x=bids["Price"], y=bids["Size"].cumsum(),
                    mode="lines", name="Bids", line=dict(color="#00e5a0", width=2),
                    fill="tozeroy", fillcolor="rgba(0,229,160,0.08)",
                ))
                fig_depth.add_trace(go.Scatter(
                    x=asks["Price"], y=asks["Size"].cumsum(),
                    mode="lines", name="Asks", line=dict(color="#ff4d6a", width=2),
                    fill="tozeroy", fillcolor="rgba(255,77,106,0.08)",
                ))
                fig_depth.add_vline(x=mid, line_dash="dot",
                                    annotation_text="Mid", line_color="#7777ff")
                fig_depth.update_layout(
                    template="plotly_dark", height=280,
                    paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
                    title="Order Book Depth",
                )
                st.plotly_chart(fig_depth, use_container_width=True)

                # Funding rate
                funding = fetch_binance_funding(bin_sym)
                if funding is not None:
                    color = "🟢" if funding < 0 else "🔴"
                    st.caption(f"{color} Funding Rate: {funding:.4f}%  "
                               f"({'Longs pay shorts' if funding > 0 else 'Shorts pay longs'})")

    # ── Portfolio Greeks snapshot ─────────────────────────────────────────────
    with st.expander("📐 Portfolio Greeks"):
        risk = portfolio_greeks(
            st.session_state["paper_positions"], asset_spot, garch_vol / 100
        )
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Delta",  f"{risk['delta']:+.3f}")
        g2.metric("Gamma",  f"{risk['gamma']:+.4f}")
        g3.metric("Theta",  f"{risk['theta']:+.3f}")
        g4.metric("Vega",   f"{risk['vega']:+.3f}")
        g5.metric("Margin", f"{currency}{risk['margin_est']:,.0f}")
        info_box("Theta is daily decay. Vega is per 1% IV move. "
                 "Margin is a rough 10% delta-notional estimate.")

    # ── Correlation ───────────────────────────────────────────────────────────
    with st.expander("📊 Correlation Analysis"):
        fig_corr = chart_correlation(corr_df, corr_names)
        if fig_corr:
            st.pyplot(fig_corr)
            info_box("20-day rolling correlation. Above 0.8 = high lockstep. "
                     "Below 0.5 = decoupling, consider pair trades or neutral strategies.")
        else:
            st.info("Correlation data unavailable.")


# ─────────────────────────────────────────────────────────────────────────────
# ══ TAB: TECHNICAL ═══════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif active_tab == "📈 Technical":
    st.title("📈 Technical Analysis")

    if asset_df is None or asset_df.empty:
        st.error("No historical data. Try refreshing.")
        st.stop()

    close_px = asset_df["Close"].squeeze()

    MODULES = [
        "Hurst Exponent",
        "RSI (Wilder)",
        "Volatility Cone",
        "Volatility Risk Premium",
        "IV Rank & Percentile",
        "Expected Daily Move",
        "Payoff Builder",
    ]
    module = st.selectbox("Select Module", MODULES)

    if module == "Hurst Exponent":
        fig = chart_hurst(close_px)
        if fig:
            st.pyplot(fig)
        else:
            st.warning("Insufficient data (need ≥ 120 bars).")
        info_box(
            "<b>How to read:</b> H > 0.55 = trending (use momentum strategies). "
            "H < 0.45 = mean-reverting (fade breakouts, use Iron Condors). "
            "H ≈ 0.50 = random walk (reduce position size, wait for clarity)."
        )

    elif module == "RSI (Wilder)":
        fig = chart_rsi(close_px)
        st.pyplot(fig)
        cur_rsi = float(calculate_rsi(close_px).dropna().iloc[-1])
        if cur_rsi > 70:
            st.warning(f"RSI {cur_rsi:.1f} — Overbought. Consider reducing longs.")
        elif cur_rsi < 30:
            st.success(f"RSI {cur_rsi:.1f} — Oversold. Potential long opportunity.")
        else:
            st.info(f"RSI {cur_rsi:.1f} — Neutral zone.")
        info_box(
            "<b>RSI uses Wilder smoothing</b> (EMA with α=1/14). "
            "The original code's mean/std method is incorrect — it's fixed here."
        )

    elif module == "Volatility Cone":
        fig = chart_volatility_cone(close_px, trading_days)
        if fig:
            st.pyplot(fig)
        info_box(
            "<b>Volatility Cone:</b> Shows historical vol ranges for different lookback windows. "
            "If current vol (yellow X) is near the top → expensive, sell premium. "
            "If near the bottom → cheap, buy premium."
        )

    elif module == "Volatility Risk Premium":
        iv_for_vrp = None
        if market == "Indian Market":
            iv_for_vrp = fetch_india_vix("6mo")
            if iv_for_vrp is not None:
                iv_for_vrp = iv_for_vrp.squeeze()
        fig = chart_vrp(close_px, iv_for_vrp, trading_days)
        if fig:
            st.pyplot(fig)
        else:
            st.info("VRP chart requires India VIX data (Indian Market mode).")
        info_box(
            "<b>VRP = Implied Vol − Realised Vol.</b> "
            "Positive VRP (green) → sellers of premium collect the risk premium on average. "
            "Negative VRP (red) → realised moves exceeded implied — buying premium was correct."
        )

    elif module == "IV Rank & Percentile":
        iv_series_ivr = None
        label_ivr     = "Historical Vol (20d)"
        if market == "Indian Market":
            iv_series_ivr = fetch_india_vix("1y")
            label_ivr     = "India VIX"
        fig = chart_ivr(close_px, iv_series_ivr, label_ivr)
        if fig:
            st.pyplot(fig)
        info_box(
            "<b>IVR > 65</b> → sell premium (Iron Condor, Short Strangle, Credit Spreads). "
            "<b>IVR < 30</b> → buy premium (Long Calls/Puts, Debit Spreads, Calendars)."
        )

    elif module == "Expected Daily Move":
        iv_em = garch_vol
        if market == "Indian Market":
            vix_em = fetch_india_vix("5d")
            if vix_em is not None:
                iv_em = float(vix_em.iloc[-1])
        recent = close_px.tail(15)
        fig = chart_expected_move(asset_spot, iv_em, recent, currency, trading_days)
        st.pyplot(fig)
        info_box(
            "The ±1σ daily move is the strike distance where options have roughly "
            "16% probability of expiring in-the-money. "
            "Safe short strikes should be beyond this range."
        )

    elif module == "Payoff Builder":
        st.markdown("#### Build a multi-leg strategy and visualise payoff at expiry")
        n_legs = st.number_input("Number of legs", 1, 4, 1)
        legs   = []
        step   = max(1.0, round(asset_spot * 0.01, 0))
        for i in range(n_legs):
            c1, c2, c3, c4 = st.columns(4)
            ltype   = c1.selectbox(f"Leg {i+1}", ["Long Call", "Short Call", "Long Put", "Short Put"], key=f"lt{i}")
            strike  = c2.number_input(f"Strike {i+1}", value=float(asset_spot), step=step, key=f"sk{i}")
            premium = c3.number_input(f"Premium {i+1}", value=10.0, step=0.5, key=f"pm{i}")
            sign    = 1 if "Long" in ltype else -1
            legs.append({"label": ltype, "K": strike, "premium": premium, "sign": sign})
        if st.button("Plot Payoff"):
            fig = chart_payoff(legs, asset_spot)
            st.pyplot(fig)
        info_box(
            "Premium values should be the actual option price paid (Long) or received (Short). "
            "The payoff shows intrinsic value minus premium at expiry — no time value."
        )


# ─────────────────────────────────────────────────────────────────────────────
# ══ TAB: PAPER TRADING ═══════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif active_tab == "📄 Paper Trading":
    st.title("📄 Paper Trading")
    st.caption(f"Virtual account — starting balance {currency}100,000")

    # Balance & unrealised P&L
    unrealised = 0.0
    for pos in st.session_state["paper_positions"]:
        e    = pos["Entry"]
        q    = pos["Qty"]
        sign = 1 if pos["Direction"] == "Long" else -1
        unrealised += sign * (asset_spot - e) * q

    equity = st.session_state["paper_balance"] + unrealised
    c1, c2, c3 = st.columns(3)
    c1.metric("Cash Balance",  f"{currency}{st.session_state['paper_balance']:,.2f}")
    c2.metric("Unrealised P&L", f"{currency}{unrealised:,.2f}", delta=f"{unrealised:+,.2f}")
    c3.metric("Total Equity",   f"{currency}{equity:,.2f}")

    st.markdown("---")

    # Trade form
    with st.expander("➕ Open a Trade", expanded=True):
        with st.form("paper_form", clear_on_submit=True):
            f1, f2, f3, f4 = st.columns(4)
            f_asset = f1.selectbox("Asset",     list(TICKER_DICT.keys()))
            f_dir   = f2.selectbox("Direction", ["Long", "Short"])
            f_qty   = f3.number_input("Quantity", min_value=0.001, value=0.01, step=0.001, format="%.3f")
            f_price = f4.number_input("Price",   value=float(asset_spot), step=1.0)
            submitted = st.form_submit_button("Execute Trade")
            if submitted:
                cost = f_qty * f_price
                if cost > st.session_state["paper_balance"]:
                    st.error("Insufficient balance.")
                else:
                    st.session_state["paper_balance"] -= cost
                    st.session_state["paper_positions"].append({
                        "Asset":     f_asset,
                        "Direction": f_dir,
                        "Qty":       f_qty,
                        "Entry":     f_price,
                        "Type":      "Spot",
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    st.session_state["paper_history"].append({
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Asset":     f_asset,
                        "Direction": f_dir,
                        "Qty":       f_qty,
                        "Price":     f_price,
                        "Cost":      cost,
                        "Action":    "Open",
                        "PnL":       None,
                    })
                    st.success(f"Opened {f_dir} {f_qty:.3f} {f_asset} @ {currency}{f_price:,.2f}")
                    st.rerun()

    # Open positions
    if st.session_state["paper_positions"]:
        section("Open Positions")
        pos_df = pd.DataFrame(st.session_state["paper_positions"])
        pos_df.index = range(1, len(pos_df) + 1)

        # Live P&L column
        pos_df["Current"] = asset_spot
        pos_df["P&L"] = pos_df.apply(
            lambda r: (asset_spot - r["Entry"]) * r["Qty"]
                       if r["Direction"] == "Long"
                       else (r["Entry"] - asset_spot) * r["Qty"],
            axis=1,
        )
        st.dataframe(pos_df[["Asset", "Direction", "Qty", "Entry", "Current", "P&L"]]
                     .style.format({
                         "Entry":   f"{currency}{{:,.2f}}",
                         "Current": f"{currency}{{:,.2f}}",
                         "P&L":     f"{currency}{{:,.2f}}",
                     })
                     .applymap(lambda v: "color:#00e5a0" if isinstance(v, float) and v > 0
                               else ("color:#ff4d6a" if isinstance(v, float) and v < 0 else ""),
                               subset=["P&L"]),
                     use_container_width=True)

        close_idx  = st.selectbox("Select position to close", pos_df.index)
        close_price = st.number_input("Close at", value=float(asset_spot), step=1.0)
        if st.button("Close Position"):
            pos  = pos_df.loc[close_idx]
            sign = 1 if pos["Direction"] == "Long" else -1
            pnl  = sign * (close_price - pos["Entry"]) * pos["Qty"]
            st.session_state["paper_balance"] += pos["Entry"] * pos["Qty"] + pnl
            st.session_state["paper_history"].append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Asset":     pos["Asset"],
                "Direction": pos["Direction"],
                "Qty":       pos["Qty"],
                "Price":     close_price,
                "Action":    "Close",
                "PnL":       pnl,
            })
            st.session_state["paper_positions"].pop(int(close_idx) - 1)
            st.success(f"Closed — P&L: {currency}{pnl:+,.2f}")
            st.rerun()

    # History
    if st.session_state["paper_history"]:
        section("Trade History")
        hist_df = pd.DataFrame(st.session_state["paper_history"])
        st.dataframe(hist_df, use_container_width=True)
        closed = hist_df.dropna(subset=["PnL"])
        if not closed.empty:
            total_pnl = closed["PnL"].sum()
            win_rate  = (closed["PnL"] > 0).mean() * 100
            col_r1, col_r2 = st.columns(2)
            col_r1.metric("Realised P&L",  f"{currency}{total_pnl:+,.2f}")
            col_r2.metric("Win Rate",       f"{win_rate:.1f}%")

    if st.button("🗑️ Reset Account"):
        st.session_state["paper_balance"]   = 100_000.0
        st.session_state["paper_positions"] = []
        st.session_state["paper_history"]   = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ══ TAB: STRATEGY WIZARD ═════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif active_tab == "🧙 Strategy Wizard":
    st.title("🧙 Strategy Wizard")
    st.caption("Quantitative signal + guided execution.")

    with st.spinner("Generating signal…"):
        sig = generate_signal(
            asset_name   = asset_choice,
            ticker       = ticker,
            spot         = asset_spot,
            spot_change  = asset_change,
            garch_vol    = garch_vol,
            trading_days = trading_days,
            park_vol     = park_vol,
            ivr          = ivr_val,
            ivp          = ivp_val,
        )

    if "error" in sig:
        st.warning(sig["error"])
        st.stop()

    section("Current Signal")
    w1, w2, w3 = st.columns(3)
    w1.markdown(f"""
    <div class="aq-card">
      <div class="aq-card-label">Strategy</div>
      <div class="aq-card-value" style="font-size:1.1rem">{sig['strategy']}</div>
      <div style="margin-top:8px">{direction_badge(sig['direction'])}</div>
    </div>""", unsafe_allow_html=True)
    w2.markdown(f"""
    <div class="aq-card">
      <div class="aq-card-label">Confidence / Risk</div>
      <div class="aq-card-value">{sig['confidence']}%</div>
      <div class="aq-card-delta aq-neutral">{sig['risk_level'].upper()}</div>
    </div>""", unsafe_allow_html=True)
    w3.markdown(f"""
    <div class="aq-card">
      <div class="aq-card-label">Daily ±1σ Move</div>
      <div class="aq-card-value">{currency}{sig['daily_move']:,.0f}</div>
      <div class="aq-card-delta aq-neutral">Regime: {sig['regime'].replace('_',' ')}</div>
    </div>""", unsafe_allow_html=True)

    st.code(sig["reasoning"], language=None)

    section("Execute via Paper Account")
    risk_pct = st.slider("Risk % of balance", 0.5, 5.0, 1.0, 0.5)
    max_risk  = st.session_state["paper_balance"] * risk_pct / 100
    qty_auto  = max_risk / asset_spot if asset_spot > 0 else 0.0
    direction = ("Long"  if sig["direction"] == "bullish"
                 else "Short" if sig["direction"] == "bearish" else "Long")

    st.info(
        f"Direction: **{direction}** | Qty: **{qty_auto:.4f}** "
        f"{asset_choice} | Cost: **{currency}{max_risk:,.2f}**"
    )

    if st.button("Execute Signal Trade"):
        cost = qty_auto * asset_spot
        if cost > st.session_state["paper_balance"]:
            st.error("Insufficient balance.")
        else:
            st.session_state["paper_balance"] -= cost
            pos = {
                "Asset":     asset_choice,
                "Direction": direction,
                "Qty":       qty_auto,
                "Entry":     asset_spot,
                "Type":      "Spot",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state["paper_positions"].append(pos)
            st.session_state["paper_history"].append({
                **pos, "Price": asset_spot,
                "Cost": cost, "Action": "Open", "PnL": None,
            })
            st.success(f"Opened {direction} {qty_auto:.4f} {asset_choice} "
                       f"@ {currency}{asset_spot:,.2f}")

    # Quick backtest of signal bias
    section("Signal Backtest (Last 6 Months)")
    if st.button("Run Quick Backtest"):
        if asset_df is not None:
            close_bt = asset_df["Close"].squeeze().tail(130)
            ret_bt   = close_bt.pct_change().dropna()
            d_sign   = (1 if sig["direction"] == "bullish"
                        else -1 if sig["direction"] == "bearish" else 0)
            if d_sign != 0:
                strat_ret  = ret_bt * d_sign
                cumulative = (1 + strat_ret).cumprod()
                sharpe     = (np.sqrt(trading_days) * strat_ret.mean()
                              / strat_ret.std() if strat_ret.std() else 0)
                max_dd     = float((cumulative / cumulative.cummax() - 1).min())
                win_rate   = float((strat_ret > 0).mean() * 100)

                b1, b2, b3 = st.columns(3)
                b1.metric("Win Rate",     f"{win_rate:.1f}%")
                b2.metric("Sharpe",       f"{sharpe:.2f}")
                b3.metric("Max Drawdown", f"{max_dd:.2%}")

                fig_bt, ax_bt = _fig_base("Cumulative Return (Signal Bias)", figsize=(10, 4))
                ax_bt.plot(cumulative.index, cumulative, color="#7777ff", linewidth=2)
                ax_bt.axhline(1.0, color="#333355")
                st.pyplot(fig_bt)
            else:
                st.info("Signal is neutral — no directional backtest performed.")


# ─────────────────────────────────────────────────────────────────────────────
# ══ TAB: JOURNAL ════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif active_tab == "📓 Journal":
    st.title("📓 Trading Journal")

    # Snapshot
    if st.button("📸 Save Market Snapshot"):
        snap = {
            "timestamp":   datetime.now().isoformat(),
            "asset":       asset_choice,
            "spot":        asset_spot,
            "garch_vol":   garch_vol,
            "gjr_vol":     gjr_vol,
            "park_vol":    park_vol,
            "ivr":         ivr_val,
            "ivp":         ivp_val,
        }
        st.session_state["snapshots"].append(snap)
        st.success("Snapshot saved!")

    if st.session_state["snapshots"]:
        with st.expander("📸 Saved Snapshots"):
            st.dataframe(pd.DataFrame(st.session_state["snapshots"]),
                         use_container_width=True)

    section("Log a Trade")
    with st.form("journal_form"):
        j1, j2, j3 = st.columns(3)
        j_asset = j1.selectbox("Asset",     list(TICKER_DICT.keys()), key="j_asset")
        j_dir   = j2.selectbox("Direction", ["Long", "Short"])
        j_date  = j3.date_input("Date",     datetime.today())
        j2c, j3c, j4c = st.columns(3)
        j_entry = j2c.number_input("Entry", min_value=0.0, value=float(asset_spot), step=0.01)
        j_exit  = j3c.number_input("Exit",  min_value=0.0, value=float(asset_spot), step=0.01)
        j_qty   = j4c.number_input("Qty",   min_value=0.0, value=1.0, step=0.001, format="%.3f")
        j_notes = st.text_area("Notes / Rationale")
        if st.form_submit_button("Log Trade"):
            if j_entry > 0 and j_exit > 0 and j_qty > 0:
                sign = 1 if j_dir == "Long" else -1
                pnl  = sign * (j_exit - j_entry) * j_qty
                st.session_state["trade_journal"].append({
                    "Date":      j_date.strftime("%Y-%m-%d"),
                    "Asset":     j_asset,
                    "Direction": j_dir,
                    "Entry":     j_entry,
                    "Exit":      j_exit,
                    "Qty":       j_qty,
                    "P&L":       round(pnl, 2),
                    "Notes":     j_notes,
                    "GARCH Vol": f"{garch_vol:.1f}%",
                    "IVR":       f"{ivr_val:.0f}%" if ivr_val else "N/A",
                })
                st.success(f"Logged! P&L = {currency}{pnl:+,.2f}")
            else:
                st.error("All fields must be > 0.")

    if st.session_state["trade_journal"]:
        section("Performance Analytics")
        jdf = pd.DataFrame(st.session_state["trade_journal"])
        jdf["Date"]           = pd.to_datetime(jdf["Date"])
        jdf                   = jdf.sort_values("Date")
        jdf["Cumulative P&L"] = jdf["P&L"].cumsum()
        st.dataframe(jdf, use_container_width=True)

        fig_j, ax_j = _fig_base("Cumulative P&L", figsize=(12, 5))
        colors_j = ["#00e5a0" if v >= 0 else "#ff4d6a" for v in jdf["P&L"]]
        ax_j.bar(jdf["Date"], jdf["P&L"], color=colors_j, alpha=0.7)
        ax_j.plot(jdf["Date"], jdf["Cumulative P&L"],
                  color="#7777ff", linewidth=2.5, label="Cumulative P&L")
        ax_j.axhline(0, color="#333355")
        ax_j.legend(facecolor="#0f0f1e", edgecolor="#1e1e3a")
        st.pyplot(fig_j)

        total = jdf["P&L"].sum()
        wins  = jdf[jdf["P&L"] > 0]
        loss  = jdf[jdf["P&L"] < 0]
        wr    = len(wins) / len(jdf) * 100
        avg_w = wins["P&L"].mean() if not wins.empty else 0
        avg_l = abs(loss["P&L"].mean()) if not loss.empty else 1
        rr    = avg_w / avg_l if avg_l > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total P&L",    f"{currency}{total:+,.2f}")
        m2.metric("Win Rate",      f"{wr:.1f}%")
        m3.metric("Avg Win",       f"{currency}{avg_w:,.2f}")
        m4.metric("Risk:Reward",   f"{rr:.2f}")

        # Kelly criterion
        if avg_l > 0 and wr > 0:
            p     = wr / 100
            b     = avg_w / avg_l
            kelly = p - (1 - p) / b
            kelly = max(0.0, min(kelly, 0.25))   # cap at 25%
            info_box(
                f"<b>Kelly Criterion:</b> Risk {kelly:.1%} of capital per trade. "
                f"Based on {len(jdf)} trades with win rate {wr:.1f}% and R:R {rr:.2f}. "
                f"Use half-Kelly ({kelly/2:.1%}) for a safer approach."
            )

# ─────────────────────────────────────────────────────────────────────────────
# ── AUTO REFRESH ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["live_mode"]:
    refresh_sec = st.sidebar.slider("Refresh (s)", 30, 600, 120, 10)
    st.sidebar.caption(f"Next refresh in ~{refresh_sec}s")
    time.sleep(refresh_sec)
    st.rerun()

st.markdown("---")
st.caption("AlphaQuant Terminal Pro · Educational use only · Not financial advice")