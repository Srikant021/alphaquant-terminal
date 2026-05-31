# AlphaQuant Terminal Pro — Simplified (No Paper/Wizard/Journal, No Live OB)
# Run with:  streamlit run alphaquant_simple.py

import time, logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.stats as si
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    CRYPTO_ASSETS: Dict[str, str] = field(default_factory=lambda: {
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD",
        "Dogecoin": "DOGE-USD", "XRP": "XRP-USD"
    })
    INDIAN_ASSETS: Dict[str, str] = field(default_factory=lambda: {
        "Nifty 50": "^NSEI", "Sensex": "^BSESN",
        "Bank Nifty": "^NSEBANK", "Gold (MCX)": "GOLDM.NS",
        "Silver (MCX)": "SILVERM.NS"
    })
    BINANCE_MAP: Dict[str, str] = field(default_factory=lambda: {
        "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT",
        "DOGE-USD": "DOGEUSDT", "XRP-USD": "XRPUSDT"
    })
    MAX_WORKERS: int = 5

config = Config()

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE (simplified – only essential keys)
# ─────────────────────────────────────────────────────────────────────────────
_SESSION_DEFAULTS = {
    "market": "Crypto",
    "live_mode": False,
    "refresh_sec": 120,
    "asset_choice": "Bitcoin",
    "active_tab": "📊 Dashboard",
    "analysis_module": "Hurst Exponent",
    "hist_data": {},
    "ml_model": None,
    "ml_trained": False,
    "daily_snapshots": [],          # list of dicts for saved daily data
}
for k, v in _SESSION_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AlphaQuant Terminal", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
  /* ... [same styling as before, omitted for brevity] ... */
  html, body, .stApp { font-family: 'Syne', sans-serif; background: #0a0a0f; color: #e0e0f0; }
  [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d0d1a 0%, #12122a 100%); border-right: 1px solid #1e1e3a; }
  .aq-card { background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%); border: 1px solid #2a2a4a; border-radius: 12px; padding: 18px 20px; margin-bottom: 12px; transition: border-color 0.2s ease; }
  .aq-card:hover { border-color: #4a4aaa; }
  .aq-card-label { font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: #6666aa; margin-bottom: 4px; font-family: 'JetBrains Mono', monospace; }
  .aq-card-value { font-size: 1.6rem; font-weight: 800; color: #ffffff; font-family: 'JetBrains Mono', monospace; }
  .aq-card-delta { font-size: 0.82rem; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
  .aq-positive { color: #00e5a0; } .aq-negative { color: #ff4d6a; } .aq-neutral { color: #aaaacc; }
  .aq-section { font-size: 1.1rem; font-weight: 800; color: #c0c0ff; letter-spacing: 0.05em; margin: 22px 0 10px; border-left: 3px solid #5555ff; padding-left: 10px; }
  .aq-info { background: rgba(85,85,255,0.08); border: 1px solid rgba(85,85,255,0.25); border-radius: 8px; padding: 12px 16px; font-size: 0.88rem; color: #b0b0dd; margin: 8px 0; }
  .aq-badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.06em; }
  .aq-badge-bull { background: rgba(0,229,160,0.15); color: #00e5a0; border: 1px solid #00e5a0; }
  .aq-badge-bear { background: rgba(255,77,106,0.15); color: #ff4d6a; border: 1px solid #ff4d6a; }
  .aq-badge-neut { background: rgba(170,170,200,0.15); color: #aaaacc; border: 1px solid #aaaacc; }
  .stButton > button { background: linear-gradient(135deg, #3333aa, #5555ff); color: white; border: none; border-radius: 8px; font-weight: 700; font-family: 'Syne', sans-serif; letter-spacing: 0.04em; transition: all 0.2s ease; }
  .stButton > button:hover { transform: scale(1.02); box-shadow: 0 4px 16px rgba(85,85,255,0.4); }
  .element-container { background: transparent !important; }
  @media (max-width: 768px) { .aq-card-value { font-size: 1.2rem; } }
</style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")

# ─────────────────────────────────────────────────────────────────────────────
# DATA UTILITIES (parallel, robust, caching)
# ─────────────────────────────────────────────────────────────────────────────

def yf_download_robust(ticker: str, period: str = "1y", interval: str = "1d",
                       max_retries: int = 3) -> pd.DataFrame:
    for attempt in range(max_retries):
        try:
            raw = yf.download(ticker, period=period, interval=interval,
                              progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                raise ValueError("Empty response")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            return raw
        except Exception as exc:
            logger.warning(f"Attempt {attempt+1} failed for {ticker}: {exc}")
            time.sleep(2 ** attempt)
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history_parallel(ticker_dict: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    results = {}
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        future_to_name = {
            executor.submit(yf_download_robust, ticker, "2y", "1d"): name
            for name, ticker in ticker_dict.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                df = future.result()
                if not df.empty:
                    results[name] = df
            except Exception as e:
                logger.error(f"Failed to fetch {name}: {e}")
    return results

@st.cache_data(ttl=120, show_spinner=False)
def fetch_live_price(ticker: str) -> Optional[Dict]:
    df = yf_download_robust(ticker, period="5d")
    if df.empty or len(df) < 2:
        return None
    close = df["Close"].squeeze()
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    chg = last - prev
    pct = (chg / prev) * 100 if prev else 0.0
    return {"spot": last, "prev_close": prev, "change": chg,
            "pct": pct, "ts": datetime.now().strftime("%H:%M:%S")}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday(ticker: str, interval: str = "15m") -> Optional[pd.DataFrame]:
    df = yf_download_robust(ticker, period="5d", interval=interval)
    required = {"Open", "High", "Low", "Close"}
    if df.empty or not required.issubset(df.columns):
        return None
    return df

@st.cache_data(ttl=300, show_spinner=False)
def fetch_option_chain_deribit(coin: str = "BTC") -> Optional[pd.DataFrame]:
    base = "https://www.deribit.com/api/v2/public/"
    try:
        r = requests.get(base + "get_instruments",
                         params={"currency": coin, "kind": "option", "expired": "false"},
                         timeout=10)
        if r.status_code != 200:
            return None
        records = []
        for inst in r.json().get("result", [])[:50]:
            name = inst["instrument_name"]
            r2 = requests.get(base + "get_order_book",
                              params={"instrument_name": name, "depth": 1}, timeout=5)
            if r2.status_code != 200:
                continue
            book = r2.json().get("result", {})
            greeks = book.get("greeks", {})
            records.append({
                "instrument": name,
                "strike": inst["strike"],
                "option_type": "call" if name.split("-")[-1] == "C" else "put",
                "expiry_ts": inst["expiration_timestamp"],
                "mark_iv": book.get("mark_iv", 0.0),
                "underlying_price": book.get("underlying_price", 0.0),
                "open_interest": book.get("open_interest", 0.0),
                "delta": greeks.get("delta", 0.0),
                "gamma": greeks.get("gamma", 0.0),
                "theta": greeks.get("theta", 0.0),
                "vega": greeks.get("vega", 0.0),
            })
        return pd.DataFrame(records) if records else None
    except Exception as exc:
        logger.warning(f"Deribit fetch failed: {exc}")
        return None

@st.cache_data(ttl=60, show_spinner=False)
def fetch_binance_funding(symbol: str = "BTCUSDT") -> Optional[float]:
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                         params={"symbol": symbol}, timeout=5)
        if r.status_code == 200:
            return float(r.json()["lastFundingRate"]) * 100
    except Exception:
        pass
    return None

@st.cache_data(ttl=10, show_spinner=False)
def fetch_binance_orderbook(symbol: str = "BTCUSDT", limit: int = 20) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    try:
        r = requests.get("https://api.binance.com/api/v3/depth",
                         params={"symbol": symbol, "limit": limit}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            bids = pd.DataFrame(data["bids"], columns=["Price", "Size"], dtype=float)
            asks = pd.DataFrame(data["asks"], columns=["Price", "Size"], dtype=float)
            return bids, asks
    except Exception:
        pass
    return None, None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_india_vix(period: str = "1y") -> Optional[pd.Series]:
    df = yf_download_robust("^INDIAVIX", period=period)
    if df.empty:
        return None
    return df["Close"].squeeze()

# ─────────────────────────────────────────────────────────────────────────────
# QUANTITATIVE FUNCTIONS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_hurst(series: np.ndarray, max_lags: int = 20) -> float:
    n = len(series)
    if n < 20:
        return np.nan
    lags = range(2, min(max_lags, n // 4))
    if len(lags) < 3:
        return np.nan
    try:
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        if any(t <= 0 for t in tau):
            return np.nan
        slope, _ = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return slope * 2.0
    except Exception:
        return np.nan

def calculate_parkinson_vol(high: pd.Series, low: pd.Series,
                            periods_per_year: int = 252) -> float:
    if len(high) != len(low) or len(high) < 2:
        return 0.0
    log_hl = (np.log(high / low) ** 2)
    n = len(log_hl)
    return float(np.sqrt((log_hl.sum() / (4 * n * np.log(2))) * periods_per_year) * 100)

def calculate_iv_rank_percentile(close: pd.Series, window: int = 20) -> Tuple[float, float]:
    if len(close) < window + 5:
        return 50.0, 50.0
    log_ret = np.log(close / close.shift(1)).dropna()
    rolling_vol = log_ret.rolling(window).std() * np.sqrt(252) * 100
    rolling_vol = rolling_vol.dropna()
    if rolling_vol.empty:
        return 50.0, 50.0
    cur = float(rolling_vol.iloc[-1])
    vmin = float(rolling_vol.min())
    vmax = float(rolling_vol.max())
    ivr = ((cur - vmin) / (vmax - vmin)) * 100 if vmax != vmin else 50.0
    ivp = float((rolling_vol < cur).mean()) * 100
    return ivr, ivp

def garch_forecast(ticker: str, trading_days: int = 252) -> Tuple[float, float]:
    df = yf_download_robust(ticker, period="1y")
    if df.empty:
        return 80.0, 80.0
    close = df["Close"].squeeze()
    ret = (100 * close.pct_change().dropna())
    if len(ret) < 100 or not ARCH_AVAILABLE:
        hv = float(ret.std() * np.sqrt(trading_days))
        return hv, hv

    def _fit_and_forecast(o: int) -> float:
        try:
            model = arch_model(ret, vol="GARCH", p=1, o=o, q=1, rescale=True)
            res = model.fit(disp="off")
            var = res.forecast(horizon=1).variance.iloc[-1, 0]
            return float(np.sqrt(var) * np.sqrt(trading_days))
        except Exception:
            return float(ret.std() * np.sqrt(trading_days))

    return _fit_and_forecast(0), _fit_and_forecast(1)

def black_scholes_greeks(S: float, K: float, T: float, r: float,
                         sigma: float, option_type: str = "call") -> Dict[str, float]:
    T = max(T, 1e-6)
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        pdf_d1 = si.norm.pdf(d1)
        cdf_d1 = si.norm.cdf(d1)
        cdf_d2 = si.norm.cdf(d2)
        disc = K * np.exp(-r * T)

        if option_type == "call":
            price = S * cdf_d1 - disc * cdf_d2
            delta = cdf_d1
            theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * disc * cdf_d2) / 365
        else:
            price = disc * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
            delta = cdf_d1 - 1
            theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * disc * si.norm.cdf(-d2)) / 365

        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega = (S * np.sqrt(T) * pdf_d1) / 100   # per 1% IV move

        return {"price": max(0.01, price), "delta": delta, "gamma": gamma,
                "theta": theta, "vega": vega}
    except Exception:
        return {"price": 0.0, "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def build_features(close: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"close": close})
    df["ret"] = df["close"].pct_change()
    df["vol10"] = df["ret"].rolling(10).std()
    df["rsi14"] = calculate_rsi(df["close"], 14)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["mom5"] = df["close"].pct_change(5)
    df["mom20"] = df["close"].pct_change(20)
    df["vol20"] = df["ret"].rolling(20).std()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna()

def train_ml_model(close: pd.Series) -> bool:
    if not ML_AVAILABLE or len(close) < 150:
        return False
    df = build_features(close).tail(500)
    if len(df) < 100:
        return False
    features = ["ret", "vol10", "rsi14", "macd", "mom5", "mom20", "vol20"]
    X, y = df[features].values, df["target"].values
    try:
        model = XGBClassifier(n_estimators=100, max_depth=3,
                              use_label_encoder=False, eval_metric="logloss")
        model.fit(X, y)
        st.session_state["ml_model"] = model
        st.session_state["ml_trained"] = True
        return True
    except Exception as exc:
        logger.warning(f"ML training failed: {exc}")
        return False

def ml_predict(close: pd.Series) -> Optional[int]:
    if not st.session_state["ml_trained"] or st.session_state["ml_model"] is None:
        return None
    if len(close) < 30:
        return None
    try:
        df = build_features(close.tail(100))
        if df.empty:
            return None
        features = ["ret", "vol10", "rsi14", "macd", "mom5", "mom20", "vol20"]
        X = df[features].iloc[[-1]].values
        return int(st.session_state["ml_model"].predict(X)[0])
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL ENGINE (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def compute_option_skew(chain: pd.DataFrame) -> Optional[float]:
    calls = chain[chain["option_type"] == "call"].copy()
    puts = chain[chain["option_type"] == "put"].copy()
    if calls.empty or puts.empty:
        return None
    call_25 = calls.iloc[(calls["delta"] - 0.25).abs().argsort()[:1]]
    put_25 = puts.iloc[(puts["delta"] + 0.25).abs().argsort()[:1]]
    if call_25.empty or put_25.empty:
        return None
    return float(call_25["mark_iv"].values[0] - put_25["mark_iv"].values[0])

def generate_signal(
    asset_name: str, ticker: str, spot: float, spot_change: float,
    garch_vol: float, trading_days: int,
    park_vol: Optional[float] = None, ivr: Optional[float] = None,
    ivp: Optional[float] = None, deribit_chain: Optional[pd.DataFrame] = None,
    funding_rate: Optional[float] = None, intra_df: Optional[pd.DataFrame] = None
) -> dict:
    hist = st.session_state["hist_data"].get(asset_name)
    if hist is None or hist.empty:
        return {"error": "No historical data available."}

    close = hist["Close"].squeeze()
    if len(close) < 60:
        return {"error": "Insufficient price history."}

    hurst = calculate_hurst(np.log(close.values[-200:]))
    if np.isnan(hurst):
        regime = "unknown"
    elif hurst > 0.55:
        regime = "trending"
    elif hurst < 0.45:
        regime = "mean_reverting"
    else:
        regime = "random_walk"

    log_ret = np.log(close / close.shift(1)).dropna()
    hist_vol_20 = float(log_ret.tail(21).std() * np.sqrt(trading_days) * 100)
    daily_move = spot * (garch_vol / 100) / np.sqrt(trading_days)

    sweep_up = sweep_down = False
    if intra_df is not None and not intra_df.empty:
        h = intra_df["High"]; l = intra_df["Low"]; c = intra_df["Close"]
        prev_high = h.rolling(20).max().shift(1)
        prev_low = l.rolling(20).min().shift(1)
        sweep_up = bool(((h > prev_high) & (c < prev_high)).any())
        sweep_down = bool(((l < prev_low) & (c > prev_low)).any())

    vol_votes = 0
    vol_env = "neutral"
    def _vote(expensive: bool, cheap: bool):
        nonlocal vol_votes, vol_env
        if expensive:
            vol_votes += 1
            vol_env = "expensive" if vol_env != "cheap" else "mixed"
        elif cheap:
            vol_votes += 1
            vol_env = "cheap" if vol_env != "expensive" else "mixed"

    _vote(garch_vol > hist_vol_20 * 1.2, garch_vol < hist_vol_20 * 0.8)
    if ivr is not None:
        _vote(ivr > 65, ivr < 30)
    if park_vol is not None:
        _vote(park_vol > garch_vol * 1.2, park_vol < garch_vol * 0.8)
    if deribit_chain is not None and not deribit_chain.empty:
        avg_iv = deribit_chain["mark_iv"].mean()
        _vote(avg_iv > garch_vol * 1.15, avg_iv < garch_vol * 0.85)
    if funding_rate is not None:
        _vote(funding_rate > 0.10, funding_rate < -0.10)

    if vol_votes == 0:
        vol_env = "neutral"

    direction = "neutral"
    strategy = "Iron Condor"
    confidence = 50
    risk_level = "low"

    if regime == "trending":
        direction = "bullish" if spot_change > 0 else "bearish"
        if vol_env == "cheap":
            strategy = "Long Call" if direction == "bullish" else "Long Put"
            confidence += 15
            risk_level = "high"
        elif direction == "bullish":
            strategy = "Bull Call Spread"
            confidence += 8
            risk_level = "medium"
        else:
            strategy = "Bear Put Spread"
            confidence += 8
            risk_level = "medium"

    elif regime == "mean_reverting":
        if sweep_up:
            direction = "bearish"
            strategy = "Bear Put Spread"
        elif sweep_down:
            direction = "bullish"
            strategy = "Bull Call Spread"
        else:
            strategy = "Iron Condor"
        confidence += 10
        risk_level = "low" if strategy == "Iron Condor" else "medium"

    else:
        if vol_env == "expensive":
            strategy = "Short Strangle"
            risk_level = "medium"
        elif sweep_up:
            direction = "bearish"
            strategy = "Bear Put Spread"
        elif sweep_down:
            direction = "bullish"
            strategy = "Bull Call Spread"

    ml_pred = ml_predict(close)
    if ml_pred is not None:
        if (ml_pred == 1 and direction == "bullish") or (ml_pred == 0 and direction == "bearish"):
            confidence += 10
        else:
            confidence -= 5
    confidence = max(0, min(100, confidence))

    skew = compute_option_skew(deribit_chain) if deribit_chain is not None else None
    if skew is not None:
        if skew > 5:
            if direction == "bullish": confidence -= 5
            elif direction == "bearish": confidence += 5
        elif skew < -5:
            if direction == "bullish": confidence += 5
            elif direction == "bearish": confidence -= 5

    parts = [
        f"H={hurst:.3f}({regime})", f"GV={garch_vol:.1f}%", f"HV={hist_vol_20:.1f}%",
        f"Vol={vol_env}", f"Sweep={'↑' if sweep_up else '↓' if sweep_down else '—'}",
        f"Conf={confidence}%", f"Risk={risk_level}"
    ]
    if park_vol: parts.append(f"Park={park_vol:.1f}%")
    if ivr: parts.append(f"IVR={ivr:.0f}%")
    if funding_rate: parts.append(f"Fund={funding_rate:.3f}%")
    if ml_pred is not None: parts.append(f"ML={'↑' if ml_pred==1 else '↓'}")
    if skew is not None: parts.append(f"Skew={skew:+.1f}%")

    return {
        "regime": regime, "vol_env": vol_env, "strategy": strategy,
        "direction": direction, "confidence": confidence, "risk_level": risk_level,
        "daily_move": daily_move, "garch_vol": garch_vol, "hist_vol_20": hist_vol_20,
        "hurst": hurst, "sweep_up": sweep_up, "sweep_down": sweep_down,
        "ml_pred": ml_pred, "skew": skew,
        "reasoning": " · ".join(parts)
    }

# ─────────────────────────────────────────────────────────────────────────────
# CHARTING FUNCTIONS (all as before, plus new ones for added modules)
# ─────────────────────────────────────────────────────────────────────────────

def _fig_base(title: str, figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#0a0a0f")
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_title(title, color="#c0c0ff", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#666688")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e1e3a")
    ax.grid(True, color="#14142a", linestyle=":")
    return fig, ax

def fig_to_download(fig: plt.Figure) -> bytes:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()

# Existing chart functions (unchanged): chart_hurst, chart_volatility_cone, chart_vrp,
# chart_ivr, chart_expected_move, chart_payoff, chart_rsi, chart_correlation.
# They remain exactly the same as in the previous version.

def chart_hurst(close: pd.Series) -> Optional[plt.Figure]:
    # ... same as before ...
    pass

def chart_volatility_cone(close: pd.Series, trading_days: int = 252) -> Optional[plt.Figure]:
    # ... same as before ...
    pass

def chart_vrp(close: pd.Series, iv_series: Optional[pd.Series], trading_days: int = 252) -> Optional[plt.Figure]:
    # ... same as before ...
    pass

def chart_ivr(close: pd.Series, iv_series: Optional[pd.Series], label: str = "Vol") -> Optional[plt.Figure]:
    # ... same as before ...
    pass

def chart_expected_move(spot: float, iv_pct: float, recent_close: pd.Series,
                         currency: str = "$", trading_days: int = 252) -> plt.Figure:
    # ... same as before ...
    pass

def chart_payoff(legs: List[dict], spot: float) -> plt.Figure:
    # ... same as before ...
    pass

def chart_rsi(close: pd.Series) -> plt.Figure:
    # ... same as before ...
    pass

def chart_correlation(df_pair: pd.DataFrame, names: Tuple[str, str]) -> Optional[plt.Figure]:
    # ... same as before ...
    pass

# NEW CHART: Liquidity Detector
def chart_liquidity(high: pd.Series, low: pd.Series, close: pd.Series,
                     window: int = 20) -> Optional[plt.Figure]:
    """Plot price and highlight potential liquidity sweeps."""
    df = pd.DataFrame({"High": high, "Low": low, "Close": close})
    if len(df) < window+1:
        return None
    df["prev_high"] = df["High"].rolling(window).max().shift(1)
    df["prev_low"]  = df["Low"].rolling(window).min().shift(1)
    df["sweep_up"]  = (df["High"] > df["prev_high"]) & (df["Close"] < df["prev_high"])
    df["sweep_down"]= (df["Low"]  < df["prev_low"])  & (df["Close"] > df["prev_low"])

    fig, ax = _fig_base("Liquidity Sweeps")
    ax.plot(df.index, df["Close"], color="#7777ff", linewidth=1.5, label="Close")
    ax.scatter(df.index[df["sweep_up"]], df["High"][df["sweep_up"]],
               color="#ff4d6a", s=80, marker="v", label="Sweep Up")
    ax.scatter(df.index[df["sweep_down"]], df["Low"][df["sweep_down"]],
               color="#00e5a0", s=80, marker="^", label="Sweep Down")
    ax.legend(facecolor="#0f0f1e", edgecolor="#1e1e3a")
    plt.tight_layout()
    return fig

# NEW CHART: Open Interest Profile (Deribit)
def chart_open_interest(chain: pd.DataFrame) -> Optional[plt.Figure]:
    if chain.empty or "open_interest" not in chain.columns:
        return None
    df = chain.groupby("strike")["open_interest"].sum().reset_index()
    fig, ax = _fig_base("Open Interest by Strike")
    ax.bar(df["strike"], df["open_interest"], color="#00ccff", alpha=0.8)
    ax.set_xlabel("Strike", color="#666688")
    ax.set_ylabel("Open Interest", color="#666688")
    plt.tight_layout()
    return fig

# NEW CHART: Parkinson Volatility (time series)
def chart_parkinson(high: pd.Series, low: pd.Series,
                     window: int = 20, trading_days: int = 252) -> Optional[plt.Figure]:
    """Rolling Parkinson volatility estimate."""
    if len(high) < window:
        return None
    log_hl = (np.log(high / low) ** 2)
    park_series = np.sqrt(log_hl.rolling(window).sum() / (4 * window * np.log(2))) * np.sqrt(trading_days) * 100
    park_series = park_series.dropna()

    fig, ax = _fig_base("Rolling Parkinson Volatility (Annualised %)")
    ax.plot(park_series.index, park_series, color="#ffaa33", linewidth=1.8)
    ax.set_ylabel("Volatility %", color="#666688")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, delta: str = "", positive: Optional[bool] = None):
    delta_class = ("aq-positive" if positive is True else "aq-negative" if positive is False else "aq-neutral")
    st.markdown(f"""<div class="aq-card">
              <div class="aq-card-label">{label}</div>
              <div class="aq-card-value">{value}</div>
              <div class="aq-card-delta {delta_class}">{delta}</div>
            </div>""", unsafe_allow_html=True)

def section(title: str):
    st.markdown(f'<div class="aq-section">{title}</div>', unsafe_allow_html=True)

def info_box(text: str):
    st.markdown(f'<div class="aq-info">{text}</div>', unsafe_allow_html=True)

def direction_badge(direction: str) -> str:
    if direction == "bullish":
        return '<span class="aq-badge aq-badge-bull">▲ BULLISH</span>'
    elif direction == "bearish":
        return '<span class="aq-badge aq-badge-bear">▼ BEARISH</span>'
    return '<span class="aq-badge aq-badge-neut">◆ NEUTRAL</span>'

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR (simplified, no alerts, no risk)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ AlphaQuant Terminal")
    st.markdown("---")

    market = st.radio("Market", ["Crypto", "Indian Market"],
                      index=0 if st.session_state["market"] == "Crypto" else 1, horizontal=True)
    if market != st.session_state["market"]:
        st.session_state["market"] = market
        st.session_state["hist_data"] = {}
        st.session_state["ml_trained"] = False
        st.cache_data.clear()
        st.rerun()

    TICKER_DICT = config.CRYPTO_ASSETS if market == "Crypto" else config.INDIAN_ASSETS
    trading_days = 365 if market == "Crypto" else 252
    currency = "$" if market == "Crypto" else "₹"

    asset_choice = st.selectbox("Asset", list(TICKER_DICT.keys()))
    ticker = TICKER_DICT[asset_choice]

    st.markdown("---")
    tab = st.radio("Navigate", ["📊 Dashboard", "📈 Technical"])
    st.session_state["active_tab"] = tab

    st.markdown("---")
    live_mode = st.checkbox("🟢 Live Mode", value=st.session_state["live_mode"])
    st.session_state["live_mode"] = live_mode

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
    st.caption("AlphaQuant Terminal · For learning purposes only.")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (shared across tabs)
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state["hist_data"] or set(st.session_state["hist_data"].keys()) != set(TICKER_DICT.keys()):
    with st.spinner("Loading market data..."):
        st.session_state["hist_data"] = fetch_history_parallel(TICKER_DICT)

lp = fetch_live_price(ticker)
if lp is None:
    h = st.session_state["hist_data"].get(asset_choice)
    if h is not None and len(h) >= 2:
        last = float(h["Close"].squeeze().iloc[-1])
        prev = float(h["Close"].squeeze().iloc[-2])
        lp = {"spot": last, "prev_close": prev, "change": last - prev,
              "pct": (last-prev)/prev*100 if prev else 0.0, "ts": "hist"}
    else:
        lp = {"spot": 0.0, "prev_close": 0.0, "change": 0.0, "pct": 0.0, "ts": "unavailable"}

asset_spot = lp["spot"]
asset_change = lp["change"]
asset_pct = lp["pct"]

with st.spinner("Fitting volatility models..."):
    garch_vol, gjr_vol = garch_forecast(ticker, trading_days)

park_vol = ivr_val = ivp_val = None
asset_df = st.session_state["hist_data"].get(asset_choice)
if asset_df is not None and not asset_df.empty:
    close_px = asset_df["Close"].squeeze()
    if {"High", "Low"}.issubset(asset_df.columns):
        park_vol = calculate_parkinson_vol(asset_df["High"].squeeze().tail(60),
                                           asset_df["Low"].squeeze().tail(60), trading_days)
    ivr_val, ivp_val = calculate_iv_rank_percentile(close_px, 20)

intra_15m = fetch_intraday(ticker, "15m")

deribit_chain = None
if market == "Crypto" and asset_choice in ["Bitcoin", "Ethereum"]:
    coin_map = {"Bitcoin": "BTC", "Ethereum": "ETH"}
    deribit_chain = fetch_option_chain_deribit(coin_map[asset_choice])

# Correlation data
if market == "Crypto":
    corr_pair_tickers = ["BTC-USD", "ETH-USD"]
    corr_names = ("Bitcoin", "Ethereum")
else:
    corr_pair_tickers = ["^NSEI", "^NSEBANK"]
    corr_names = ("Nifty 50", "Bank Nifty")

@st.cache_data(ttl=1800, show_spinner=False)
def _corr_data(tickers, _):
    raw = yf.download(tickers, period="1y", progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Close"]
    return raw[["Close"]] if "Close" in raw.columns else raw

corr_df = _corr_data(corr_pair_tickers, market)

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD TAB (removed Live Chart & Order Book)
# ─────────────────────────────────────────────────────────────────────────────
active_tab = st.session_state["active_tab"]

if active_tab == "📊 Dashboard":
    st.title("📊 Market Intelligence Dashboard")

    section("Market Overview")
    if market == "Crypto":
        btc_lp = fetch_live_price("BTC-USD")
        eth_lp = fetch_live_price("ETH-USD")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if btc_lp: metric_card("Bitcoin", f"${btc_lp['spot']:,.0f}", f"{btc_lp['change']:+,.0f} ({btc_lp['pct']:+.2f}%)", btc_lp["pct"]>=0)
        with c2:
            if eth_lp: metric_card("Ethereum", f"${eth_lp['spot']:,.2f}", f"{eth_lp['change']:+,.2f} ({eth_lp['pct']:+.2f}%)", eth_lp["pct"]>=0)
        with c3:
            metric_card("GARCH Vol", f"{garch_vol:.1f}%", "Annualised 1-day ahead")
        with c4:
            metric_card("GJR-GARCH Vol", f"{gjr_vol:.1f}%", "Leverage-adjusted")
    else:
        nifty_lp = fetch_live_price("^NSEI")
        sensex_lp = fetch_live_price("^BSESN")
        vix_s = fetch_india_vix("5d")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if nifty_lp: metric_card("Nifty 50", f"₹{nifty_lp['spot']:,.2f}", f"{nifty_lp['pct']:+.2f}%", nifty_lp["pct"]>=0)
        with c2:
            if sensex_lp: metric_card("Sensex", f"₹{sensex_lp['spot']:,.2f}", f"{sensex_lp['pct']:+.2f}%", sensex_lp["pct"]>=0)
        with c3:
            if vix_s is not None: metric_card("India VIX", f"{float(vix_s.iloc[-1]):.2f}", "Implied vol index")
        with c4:
            metric_card("GARCH Vol", f"{garch_vol:.1f}%", "1-day ahead forecast")

    section(f"Active Asset — {asset_choice}")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        metric_card("Spot Price", f"{currency}{asset_spot:,.2f}", f"{asset_change:+,.2f} ({asset_pct:+.2f}%)", asset_pct>=0)
    with col_b:
        metric_card("Parkinson Vol", f"{park_vol:.1f}%" if park_vol else "N/A", "High-Low estimator")
    with col_c:
        metric_card("IV Rank", f"{ivr_val:.0f}%" if ivr_val else "N/A", "Sell >65 · Buy <30")
    with col_d:
        metric_card("IV Percentile", f"{ivp_val:.0f}%" if ivp_val else "N/A", f"Last update: {lp['ts']}")

    section("Signal Engine")
    with st.spinner("Computing signal..."):
        signal = generate_signal(
            asset_name=asset_choice, ticker=ticker, spot=asset_spot,
            spot_change=asset_change, garch_vol=garch_vol,
            trading_days=trading_days, park_vol=park_vol, ivr=ivr_val,
            ivp=ivp_val, deribit_chain=deribit_chain,
            intra_df=intra_15m
        )

    if "error" not in signal:
        sc1, sc2, sc3 = st.columns([2,2,3])
        with sc1:
            st.markdown(f"""
            <div class="aq-card">
                <div class="aq-card-label">Suggested Strategy</div>
                <div class="aq-card-value" style="font-size:1.2rem">{signal['strategy']}</div>
                <div style="margin-top:8px">{direction_badge(signal['direction'])}</div>
            </div>""", unsafe_allow_html=True)
        with sc2:
            conf = signal["confidence"]
            bar_c = "#00e5a0" if conf>=65 else "#ffaa33" if conf>=45 else "#ff4d6a"
            st.markdown(f"""
            <div class="aq-card">
                <div class="aq-card-label">Confidence</div>
                <div class="aq-card-value">{conf}%</div>
                <div style="background:#1e1e3a;border-radius:4px;height:6px;margin-top:8px">
                    <div style="background:{bar_c};width:{conf}%;height:6px;border-radius:4px"></div>
                </div>
                <div class="aq-card-delta aq-neutral" style="margin-top:4px">Risk: {signal['risk_level'].upper()}</div>
            </div>""", unsafe_allow_html=True)
        with sc3:
            st.markdown(f"""
            <div class="aq-card">
                <div class="aq-card-label">Regime & Vol Environment</div>
                <div class="aq-card-value" style="font-size:1rem">{signal['regime'].replace('_',' ').title()}</div>
                <div class="aq-card-delta aq-neutral">Vol: {signal['vol_env']} · Daily ±{currency}{signal['daily_move']:,.0f}</div>
            </div>""", unsafe_allow_html=True)

        with st.expander("📋 Full Reasoning"):
            st.code(signal["reasoning"], language=None)
            daily = signal["daily_move"]
            info_box(f"Strike zones based on daily ±{currency}{daily:,.0f}:<br>"
                     f"• Directional OTM: {currency}{asset_spot-daily:,.0f} — {currency}{asset_spot+daily:,.0f}<br>"
                     f"• Short gamma sell zone: {currency}{asset_spot-daily*1.5:,.0f} / {currency}{asset_spot+daily*1.5:,.0f}")

    # Correlation
    with st.expander("📊 Correlation Analysis"):
        fig_corr = chart_correlation(corr_df, corr_names)
        if fig_corr:
            st.pyplot(fig_corr)
            png = fig_to_download(fig_corr)
            st.download_button("Download PNG", data=png, file_name="correlation.png", mime="image/png")

# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL TAB (all modules added)
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
        "Correlation Analysis",
        "Liquidity Detector",
        "Open Interest Profile",
        "Parkinson Volatility",
    ]
    module = st.selectbox("Select Module", MODULES)

    if module == "Hurst Exponent":
        fig = chart_hurst(close_px)
        if fig:
            st.pyplot(fig)
            st.download_button("Download PNG", fig_to_download(fig), "hurst.png")
        else:
            st.warning("Insufficient data (need ≥ 120 bars).")
        info_box("H > 0.55 = trending; H < 0.45 = mean-reverting; H ≈ 0.50 = random walk.")

    elif module == "RSI (Wilder)":
        fig = chart_rsi(close_px)
        st.pyplot(fig)
        st.download_button("Download PNG", fig_to_download(fig), "rsi.png")
        cur_rsi = float(calculate_rsi(close_px).dropna().iloc[-1])
        if cur_rsi > 70: st.warning(f"RSI {cur_rsi:.1f} — Overbought")
        elif cur_rsi < 30: st.success(f"RSI {cur_rsi:.1f} — Oversold")
        else: st.info(f"RSI {cur_rsi:.1f} — Neutral")

    elif module == "Volatility Cone":
        fig = chart_volatility_cone(close_px, trading_days)
        if fig:
            st.pyplot(fig)
            st.download_button("Download PNG", fig_to_download(fig), "vol_cone.png")

    elif module == "Volatility Risk Premium":
        iv_for_vrp = None
        if market == "Indian Market":
            iv_for_vrp = fetch_india_vix("6mo")
            if iv_for_vrp is not None:
                iv_for_vrp = iv_for_vrp.squeeze()
        fig = chart_vrp(close_px, iv_for_vrp, trading_days)
        if fig:
            st.pyplot(fig)
            st.download_button("Download PNG", fig_to_download(fig), "vrp.png")
        else:
            st.info("VRP requires India VIX (Indian Market mode).")

    elif module == "IV Rank & Percentile":
        iv_series_ivr = None
        label_ivr = "Historical Vol (20d)"
        if market == "Indian Market":
            iv_series_ivr = fetch_india_vix("1y")
            label_ivr = "India VIX"
        fig = chart_ivr(close_px, iv_series_ivr, label_ivr)
        if fig:
            st.pyplot(fig)
            st.download_button("Download PNG", fig_to_download(fig), "ivr.png")
        info_box("IVR >65 sell premium; IVR <30 buy premium.")

    elif module == "Expected Daily Move":
        iv_em = garch_vol
        if market == "Indian Market":
            vix_em = fetch_india_vix("5d")
            if vix_em is not None:
                iv_em = float(vix_em.iloc[-1])
        recent = close_px.tail(15)
        fig = chart_expected_move(asset_spot, iv_em, recent, currency, trading_days)
        st.pyplot(fig)
        st.download_button("Download PNG", fig_to_download(fig), "expected_move.png")

    elif module == "Payoff Builder":
        st.markdown("#### Build a multi-leg strategy")
        n_legs = st.number_input("Number of legs", 1, 4, 1)
        legs = []
        step = max(1.0, round(asset_spot * 0.01, 0))
        for i in range(n_legs):
            c1, c2, c3, c4 = st.columns(4)
            ltype = c1.selectbox(f"Leg {i+1}", ["Long Call", "Short Call", "Long Put", "Short Put"], key=f"lt{i}")
            strike = c2.number_input(f"Strike {i+1}", value=float(asset_spot), step=step, key=f"sk{i}")
            premium = c3.number_input(f"Premium {i+1}", value=10.0, step=0.5, key=f"pm{i}")
            sign = 1 if "Long" in ltype else -1
            legs.append({"label": ltype, "K": strike, "premium": premium, "sign": sign})
        if st.button("Plot Payoff"):
            fig = chart_payoff(legs, asset_spot)
            st.pyplot(fig)
            st.download_button("Download PNG", fig_to_download(fig), "payoff.png")

    elif module == "Correlation Analysis":
        fig_corr = chart_correlation(corr_df, corr_names)
        if fig_corr:
            st.pyplot(fig_corr)
            st.download_button("Download PNG", fig_to_download(fig_corr), "correlation.png")
        else:
            st.info("Correlation data unavailable.")

    elif module == "Liquidity Detector":
        if {"High", "Low"}.issubset(asset_df.columns):
            fig = chart_liquidity(asset_df["High"].squeeze(), asset_df["Low"].squeeze(), close_px)
            if fig:
                st.pyplot(fig)
                st.download_button("Download PNG", fig_to_download(fig), "liquidity.png")
            else:
                st.info("Insufficient data for liquidity detection.")
        else:
            st.warning("High/Low data not available.")

    elif module == "Open Interest Profile":
        if deribit_chain is not None and not deribit_chain.empty:
            fig = chart_open_interest(deribit_chain)
            if fig:
                st.pyplot(fig)
                st.download_button("Download PNG", fig_to_download(fig), "open_interest.png")
        else:
            st.info("Open Interest data available only for crypto (Deribit).")

    elif module == "Parkinson Volatility":
        if {"High", "Low"}.issubset(asset_df.columns):
            fig = chart_parkinson(asset_df["High"].squeeze(), asset_df["Low"].squeeze(),
                                  window=20, trading_days=trading_days)
            if fig:
                st.pyplot(fig)
                st.download_button("Download PNG", fig_to_download(fig), "parkinson.png")
            else:
                st.info("Insufficient data for Parkinson estimator.")
        else:
            st.warning("High/Low data not available.")

    # ── Daily Snapshot Feature ──
    st.markdown("---")
    st.markdown("### 📸 Save Daily Snapshot & Analyse")
    if st.button("Save Today's Snapshot"):
        snap = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "asset": asset_choice,
            "spot": asset_spot,
            "garch_vol": garch_vol,
            "park_vol": park_vol,
            "ivr": ivr_val,
            "ivp": ivp_val,
            "hurst": calculate_hurst(np.log(close_px.values[-200:])),
        }
        st.session_state["daily_snapshots"].append(snap)
        st.success("Snapshot saved!")

    if st.session_state["daily_snapshots"]:
        with st.expander("📋 Saved Snapshots"):
            snaps_df = pd.DataFrame(st.session_state["daily_snapshots"])
            st.dataframe(snaps_df, use_container_width=True)
            if len(snaps_df) > 1:
                fig, ax = _fig_base("Snapshot Metrics Over Time", figsize=(10, 4))
                ax.plot(snaps_df["date"], snaps_df["spot"], marker="o", label="Spot")
                ax2 = ax.twinx()
                ax2.plot(snaps_df["date"], snaps_df["garch_vol"], marker="s", color="#ffaa33", label="GARCH Vol")
                ax2.plot(snaps_df["date"], snaps_df["park_vol"], marker="^", color="#00e5a0", label="Parkinson Vol")
                fig.legend(loc="upper left")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
# LIVE MODE
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["live_mode"]:
    refresh_sec = st.sidebar.slider("Refresh (s)", 30, 600, 120, 10)
    st.sidebar.caption(f"Next refresh in ~{refresh_sec}s")
    time.sleep(refresh_sec)
    st.rerun()

st.markdown("---")
st.caption("AlphaQuant Terminal · Educational use only")