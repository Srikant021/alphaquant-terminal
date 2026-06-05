#!/usr/bin/env python3
"""
crypto.py

Single-file AlphaQuant Streamlit app.

Usage:
  streamlit run crypto.py        # launches the Streamlit UI
  python crypto.py --test        # runs unit tests (unittest)

Notes:
- Requires: streamlit, pandas, numpy, yfinance, scikit-learn, plotly
- Optional: shap for SHAP explanations
"""

import logging
from logging.handlers import RotatingFileHandler
import time
from datetime import timezone, timedelta
from functools import lru_cache
import sys
import argparse
import math
import os

import numpy as np
import pandas as pd

# Optional imports
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ---------------------------
# Logging
# ---------------------------
LOGGER_NAME = "crypto_alphaquant"
logger = logging.getLogger(LOGGER_NAME)


def setup_logging(logfile: str = "crypto_alphaquant.log", level=logging.INFO):
    """
    Configure logging to console and rotating file.
    """
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        fh = RotatingFileHandler(logfile, maxBytes=5_000_000, backupCount=3)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(ch)
        logger.addHandler(fh)


setup_logging()

IST = timezone(timedelta(hours=5, minutes=30))

# ---------------------------
# Utilities
# ---------------------------
def _flatten_multiindex(data: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten yfinance MultiIndex columns if present.
    """
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def _download_with_retry(ticker: str, period: str, interval: str, attempts: int = 3, backoff: float = 1.5) -> pd.DataFrame:
    """
    Download data from yfinance with retry/backoff.
    Raises the last exception if all attempts fail.
    """
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance is not installed. Install with `pip install yfinance`.")
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

# ---------------------------
# Indicators
# ---------------------------
def _validate_series(series, name="series"):
    if series is None:
        raise ValueError(f"{name} is None")
    if not isinstance(series, (pd.Series, pd.DataFrame, np.ndarray)):
        raise TypeError(f"{name} must be a pandas Series/DataFrame or numpy array")
    return True


def bollinger_bands(close: pd.Series, period: int = 20, std: float = 2.0):
    """
    Compute Bollinger Bands (upper, middle, lower).
    """
    _validate_series(close, "close")
    sma = close.rolling(window=period, min_periods=period).mean()
    std_dev = close.rolling(window=period, min_periods=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing (EWMA with alpha=1/period).
    """
    _validate_series(series, "series")
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Compute MACD line, signal line, and histogram (macd - signal).
    """
    _validate_series(close, "close")
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute ATR using exponential moving average of True Range.
    """
    if df is None or not {'High', 'Low', 'Close'}.issubset(df.columns):
        raise ValueError("DataFrame must contain High, Low, Close columns")
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute VWAP from OHLCV DataFrame.
    """
    if df is None or not {'High', 'Low', 'Close', 'Volume'}.issubset(df.columns):
        raise ValueError("DataFrame must contain High, Low, Close, Volume columns")
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    vol = df['Volume'].replace(0, np.nan).ffill()
    return (typical * vol).cumsum() / vol.cumsum()


def compute_parkinson_vol(high: pd.Series, low: pd.Series, periods: int = 252) -> float:
    """
    Compute Parkinson volatility estimator annualized (percentage).
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    high = np.array(pd.Series(high).dropna())
    low = np.array(pd.Series(low).dropna())
    if len(high) < 2 or len(low) < 2:
        return 0.0
    log_hl = np.log(high / low) ** 2
    variance = log_hl.mean() / (4 * np.log(2))
    return float(np.sqrt(variance * periods) * 100)


def compute_iv_rank(close: pd.Series, window: int = 20):
    """
    Compute a simple IV rank proxy using historical volatility of log returns.
    Returns (ivr, ivp).
    """
    _validate_series(close, "close")
    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) < window:
        return 50.0, 50.0
    hv = log_ret.rolling(window).std() * np.sqrt(252) * 100
    hv = hv.dropna()
    if hv.empty:
        return 50.0, 50.0
    current = hv.iloc[-1]
    ivr = (current - hv.min()) / (hv.max() - hv.min()) * 100 if hv.max() != hv.min() else 50.0
    ivp = (hv < current).sum() / len(hv) * 100
    return float(ivr), float(ivp)


@lru_cache(maxsize=64)
def hurst_exponent_cached(price_tuple):
    """
    Wrapper to allow caching of hurst_exponent for immutable inputs.
    price_tuple should be a tuple of floats.
    """
    series = pd.Series(price_tuple)
    return hurst_exponent(series)


def hurst_exponent(price_series):
    """
    Estimate Hurst exponent using R/S method with log-log regression.
    Returns (hurst, interpretation, confidence).
    """
    price = np.asarray(pd.Series(price_series).dropna(), dtype=float).squeeze()
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

# ---------------------------
# ML helpers
# ---------------------------
def build_ml_features(df, rsi_period=14, boll_period=20, boll_std=2.0, atr_period=14):
    """
    Build a standard feature DataFrame from OHLCV DataFrame.
    """
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
    return feat


def train_ml_model(ticker: str,
                   rsi_period: int = 14,
                   boll_period: int = 20,
                   boll_std: float = 2.0,
                   atr_period: int = 14,
                   calibrate: bool = True,
                   target_threshold_pct: float = 0.0):
    """
    Train RandomForest on 2 years of daily data.
    target_threshold_pct: label as 'up' only if next-day return > target_threshold_pct (e.g., 0.01 for 1%).
    Returns (model, scaler, oos_acc_pct, feature_columns)
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available; cannot train model.")
        return None, None, 0.0, None
    try:
        raw = _download_with_retry(ticker, period="2y", interval="1d")
        data = _flatten_multiindex(raw)
    except Exception as e:
        logger.exception(f"Failed to fetch data for ML training: {e}")
        return None, None, 0.0, None

    if data is None or len(data) < 200:
        logger.warning("Insufficient data for ML training.")
        return None, None, 0.0, None

    feat = build_ml_features(data, rsi_period=rsi_period, boll_period=boll_period, boll_std=boll_std, atr_period=atr_period)
    next_ret = data['Close'].squeeze().shift(-1) / data['Close'].squeeze() - 1.0
    target = (next_ret > target_threshold_pct).astype(int).reindex(feat.index).dropna()
    feat = feat.reindex(target.index)
    if feat.empty:
        logger.warning("No aligned features for training.")
        return None, None, 0.0, None

    split = int(len(feat) * 0.8)
    X_tr, X_te = feat.iloc[:split], feat.iloc[split:]
    y_tr, y_te = target.iloc[:split], target.iloc[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_tr_s, y_tr)

    if calibrate:
        try:
            calib = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
            calib.fit(X_te_s, y_te)
            model = calib
            logger.info("Applied probability calibration (CalibratedClassifierCV).")
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")

    acc = model.score(X_te_s, y_te) * 100
    train_cols = list(feat.columns)
    logger.info(f"Trained RF for {ticker} — OOS acc: {acc:.2f}% — features: {train_cols}")
    return model, scaler, acc, train_cols


def explain_ml_prediction(model, scaler, feat_df: pd.DataFrame, use_shap: bool = False) -> str:
    """
    Provide a human-readable explanation for the latest row in feat_df.
    If SHAP is available and requested, include SHAP summary for the single prediction.
    """
    if model is None or feat_df is None or feat_df.empty:
        return "No model or features available to explain."

    latest = feat_df.iloc[-1:]
    Xs = scaler.transform(latest) if scaler is not None else latest.values
    try:
        prob_pos = float(model.predict_proba(Xs)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(Xs)[0])
    except Exception:
        prob_pos = 0.5

    # Basic feature importance explanation
    importances = None
    try:
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=feat_df.columns).sort_values(ascending=False)
        elif hasattr(model, "base_estimator_") and hasattr(model.base_estimator_, "feature_importances_"):
            importances = pd.Series(model.base_estimator_.feature_importances_, index=feat_df.columns).sort_values(ascending=False)
    except Exception:
        importances = None

    lines = []
    direction = "Bullish" if prob_pos >= 0.5 else "Bearish"
    lines.append(f"Model signal: {direction} — probability {prob_pos*100:.1f}%.")

    if use_shap and SHAP_AVAILABLE:
        try:
            explainer = shap.Explainer(model, feat_df, feature_names=feat_df.columns)
            shap_vals = explainer(latest)
            top = np.argsort(np.abs(shap_vals.values[0]))[::-1][:4]
            shap_lines = []
            for idx in top:
                name = feat_df.columns[idx]
                val = latest.iloc[0, idx]
                contrib = shap_vals.values[0, idx]
                sign = "supports" if contrib > 0 else "opposes"
                shap_lines.append(f"{name}={val:.3f} ({sign} prediction by {abs(contrib):.3f})")
            lines.append("SHAP top contributions: " + "; ".join(shap_lines) + ".")
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")

    if importances is not None:
        top_feats = importances.head(4).index.tolist()
        for f in top_feats:
            v = latest.iloc[0][f]
            med = feat_df[f].median() if f in feat_df.columns else None
            if med is not None:
                if v > med:
                    lines.append(f"{f} is above median ({v:.3f} > {med:.3f}).")
                elif v < med:
                    lines.append(f"{f} is below median ({v:.3f} < {med:.3f}).")
                else:
                    lines.append(f"{f} is near median ({v:.3f}).")
            else:
                lines.append(f"{f} = {v:.3f}.")
    else:
        lines.append("Feature importance not available; model type may not expose importances.")

    return " ".join(lines)

# ---------------------------
# Streamlit app
# ---------------------------
def run_streamlit_app():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit is not installed. Install with `pip install streamlit` and re-run `streamlit run crypto.py`.")
        return

    st.set_page_config(page_title="AlphaQuant Terminal", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
        html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
        .stApp { background: #080d12; color: #e6eef8; }
    </style>
    """, unsafe_allow_html=True)

    st.title("AlphaQuant Terminal — Single-file Edition")

    # Sidebar controls
    ticker = st.sidebar.text_input("Ticker", value="AAPL")
    period = st.sidebar.selectbox("Data period", ["6mo", "1y", "2y"], index=1)
    interval = st.sidebar.selectbox("Interval", ["1d", "1wk"], index=0)
    show_boll = st.sidebar.checkbox("Show Bollinger Bands", value=True)
    show_vwap = st.sidebar.checkbox("Show VWAP", value=True)
    boll_period = st.sidebar.slider("Bollinger period", 10, 50, 20)
    boll_std = st.sidebar.slider("Bollinger std", 1.0, 3.0, 2.0)

    # ML controls
    train_model_btn = st.sidebar.button("Train ML model")
    calibrate = st.sidebar.checkbox("Calibrate probabilities", value=True)
    target_threshold_pct = st.sidebar.slider("ML target threshold (%)", 0.0, 5.0, 0.0) / 100.0
    use_shap = st.sidebar.checkbox("Use SHAP explanations (if available)", value=False)

    @st.cache_data(ttl=300)
    def fetch_data_cached(ticker, period="1y", interval="1d"):
        try:
            raw = _download_with_retry(ticker, period, interval)
            if raw is None or raw.empty:
                return None
            return _flatten_multiindex(raw)
        except Exception as e:
            logger.exception(f"Error fetching {ticker}: {e}")
            return None

    data = fetch_data_cached(ticker, period=period, interval=interval)
    if data is None:
        st.error("No data available for ticker.")
        st.stop()

    st.subheader(f"{ticker} — Price & Indicators")
    col1, col2 = st.columns([3, 1])

    with col1:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.75, 0.25])
        df = data.copy()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)
        bb_up, bb_mid, bb_lo = bollinger_bands(df['Close'], period=boll_period, std=boll_std)
        if show_boll:
            fig.add_trace(go.Scatter(x=df.index, y=bb_up, line=dict(color='rgba(0,170,255,0.3)'), name='BB Up'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bb_lo, line=dict(color='rgba(0,170,255,0.3)'), name='BB Lo'), row=1, col=1)
        if show_vwap:
            try:
                fig.add_trace(go.Scatter(x=df.index, y=vwap(df), line=dict(color='#00e5ff'), name='VWAP'), row=1, col=1)
            except Exception:
                pass
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Last Close", f"{float(data['Close'].iloc[-1]):.2f}")
        st.metric("Change", f"{(data['Close'].iloc[-1] - data['Close'].iloc[-2]):.2f}")
        ivr, ivp = compute_iv_rank(data['Close'], window=20)
        st.write(f"IV Rank proxy: **{ivr:.1f}%** (position {ivp:.1f}%)")
        h, interp, conf = hurst_exponent(data['Close'].tail(252))
        st.write(f"Hurst (252): **{h:.3f}** — {interp} (confidence: {conf})")

    # ML section
    st.header("ML Model")
    feat = build_ml_features(data)
    st.write("Feature snapshot (latest):")
    st.dataframe(feat.tail(3))

    if train_model_btn:
        with st.spinner("Training model..."):
            model, scaler, acc, cols = train_ml_model(ticker, rsi_period=14, boll_period=boll_period, boll_std=boll_std, atr_period=14, calibrate=calibrate, target_threshold_pct=target_threshold_pct)
            if model is None:
                st.error("Model training failed or insufficient data.")
            else:
                st.success(f"Model trained — OOS accuracy: {acc:.2f}%")
                explanation = explain_ml_prediction(model, scaler, feat, use_shap=use_shap)
                st.markdown("**Model explanation:**")
                st.write(explanation)

    st.markdown("---")
    st.write("Interactive controls let you toggle indicators and windows in the sidebar.")

# ---------------------------
# Unit tests (unittest)
# ---------------------------
import unittest


class TestIndicators(unittest.TestCase):
    def test_bollinger_basic(self):
        s = pd.Series(np.arange(1, 51).astype(float))
        up, mid, lo = bollinger_bands(s, period=10, std=2.0)
        self.assertTrue(len(up.dropna()) > 0)
        self.assertTrue((up >= lo).all())

    def test_rsi_range(self):
        s = pd.Series(np.linspace(1, 100, 200))
        rsi = compute_rsi(s, period=14)
        self.assertGreaterEqual(rsi.min(), 0)
        self.assertLessEqual(rsi.max(), 100)

    def test_macd_shapes(self):
        s = pd.Series(np.random.random(200).cumsum())
        macd, sig, hist = compute_macd(s)
        self.assertEqual(len(macd), len(s))
        self.assertEqual(len(sig), len(s))
        self.assertEqual(len(hist), len(s))

    def test_atr_nonnegative(self):
        df = pd.DataFrame({
            'High': np.linspace(10, 20, 100) + np.random.random(100),
            'Low': np.linspace(9, 19, 100) + np.random.random(100),
            'Close': np.linspace(9.5, 19.5, 100) + np.random.random(100),
        })
        atr = compute_atr(df, period=14)
        self.assertTrue((atr >= 0).all())

    def test_hurst_small_series(self):
        s = pd.Series(np.ones(50))
        h, interp, conf = hurst_exponent(s)
        self.assertIsInstance(h, float)
        self.assertIn(conf, ("low", "medium", "high"))

    def test_iv_rank_default(self):
        s = pd.Series(np.cumprod(1 + np.random.normal(0, 0.01, 300)))
        ivr, ivp = compute_iv_rank(s, window=20)
        self.assertGreaterEqual(ivr, 0)
        self.assertLessEqual(ivr, 100)
        self.assertGreaterEqual(ivp, 0)
        self.assertLessEqual(ivp, 100)


def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestIndicators)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result

# ---------------------------
# Entrypoint behavior
# ---------------------------
def main_entry():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    known_args, _ = parser.parse_known_args()

    if known_args.test:
        res = run_tests()
        if not res.wasSuccessful():
            sys.exit(1)
        return

    # If Streamlit is running this script, Streamlit will import it and we should run the app.
    # Heuristic: if STREAMLIT_AVAILABLE and (running under streamlit or invoked directly with 'streamlit run'),
    # run the app. When streamlit runs, sys.argv often contains 'run' and the script name.
    argv_join = " ".join(sys.argv).lower()
    if STREAMLIT_AVAILABLE and ("streamlit" in argv_join or "run" in argv_join or os.environ.get("STREAMLIT_RUN", "") == "true"):
        # Run the Streamlit app immediately so `streamlit run crypto.py` launches the UI.
        try:
            run_streamlit_app()
        except Exception as e:
            logger.exception(f"Streamlit app failed: {e}")
            raise
    else:
        # If executed directly with python (not via streamlit), print short instructions.
        print("This file contains a Streamlit app. To run the UI, use:")
        print("  streamlit run crypto.py")
        print("To run unit tests:")
        print("  python crypto.py --test")


# If imported by Streamlit, run_streamlit_app will be invoked by main_entry below.
if __name__ == "__main__":
    main_entry()
