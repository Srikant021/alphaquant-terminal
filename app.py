# crypto_alphaquant.py
# AlphaQuant Terminal — merged, hardened, and ML-explainable version
# - Preserves original ticker selection behavior
# - Adds safe fetching, retry/backoff, prediction alignment, optional calibration, logging
# - Adds explain_ml_prediction(...) to provide human-readable reasons for ML signals

import logging
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# Optional ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

# Basic logging to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("alphaquant")

st.set_page_config(page_title="AlphaQuant Terminal", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background: #080d12; }
    .metric-box {
        background: linear-gradient(135deg, #0d1520 0%, #111c2b 100%);
        padding: 16px 18px; border-radius: 8px; border-left: 3px solid #0af;
        margin: 5px 0; border-top: 1px solid rgba(0,170,255,0.08);
    }
    .explanation-box {
        background: rgba(0,170,255,0.05); border-left: 3px solid #0af;
        padding: 10px 14px; border-radius: 4px; font-size: 12px; margin: 8px 0;
        color: rgba(255,255,255,0.75);
    }
    .section-header {
        font-family: 'Space Mono', monospace; font-size: 13px; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase; color: #0af;
        margin: 24px 0 12px 0; padding-bottom: 6px;
        border-bottom: 1px solid rgba(0,170,255,0.2);
    }
    .sweep-badge {
        display: inline-block; padding: 5px 14px; border-radius: 20px;
        font-family: 'Space Mono', monospace; font-size: 11px; font-weight: 700;
        letter-spacing: 0.08em; text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))


def _month_end_alias() -> str:
    major, minor = (int(x) for x in pd.__version__.split(".")[:2])
    return "ME" if (major, minor) >= (2, 2) else "M"


# ─────────────────────────────────────────────
# SAFE YFINANCE DOWNLOAD WITH RETRY
# ─────────────────────────────────────────────
def _download_with_retry(ticker, period, interval, attempts=3, backoff=1.5):
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


# ─────────────────────────────────────────────
# HURST EXPONENT
# ─────────────────────────────────────────────
def hurst_exponent(price_series):
    price = np.asarray(price_series.squeeze().dropna(), dtype=float)
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


# ─────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────
def bollinger_bands(close, period=20, std=2):
    sma = close.rolling(window=period).mean()
    std_dev = close.rolling(window=period).std()
    return sma + (std * std_dev), sma, sma - (std * std_dev)


def vwap(df):
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    vol = df['Volume'].replace(0, np.nan).ffill()
    return (typical * vol).cumsum() / vol.cumsum()


def compute_liquidity_sweeps(df, window=20):
    df = df.copy()
    df['Prev_High'] = df['High'].rolling(window=window).max().shift(1)
    df['Prev_Low'] = df['Low'].rolling(window=window).min().shift(1)
    df['Supply_Sweep'] = (df['High'] > df['Prev_High']) & (df['Close'] < df['Prev_High'])
    df['Demand_Sweep'] = (df['Low'] < df['Prev_Low']) & (df['Close'] > df['Prev_Low'])
    return df


def compute_parkinson_vol(high, low, periods=252):
    high = np.array(high.dropna())
    low = np.array(low.dropna())
    if len(high) < 2 or len(low) < 2:
        return 0.0
    log_hl = np.log(high / low) ** 2
    variance = log_hl.mean() / (4 * np.log(2))
    return float(np.sqrt(variance * periods) * 100)


def compute_iv_rank(close, window=20):
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


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def compute_atr(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ─────────────────────────────────────────────
# DATA FETCHING (with caching)
# ─────────────────────────────────────────────
def _flatten_multiindex(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


@st.cache_data(ttl=300)
def fetch_data(ticker, period="1y", interval="1d"):
    try:
        raw = _download_with_retry(ticker, period, interval)
        if raw is None or raw.empty:
            return None
        return _flatten_multiindex(raw)
    except Exception as e:
        logger.exception(f"Error fetching {ticker}: {e}")
        return None


@st.cache_data(ttl=60)
def get_live_price(ticker):
    try:
        data = fetch_data(ticker, period="5d", interval="1d")
        if data is None or len(data) < 2:
            return None
        last = float(data['Close'].iloc[-1])
        prev = float(data['Close'].iloc[-2])
        return {
            'price': last,
            'change': last - prev,
            'pct': ((last - prev) / prev) * 100 if prev else 0.0,
            'high': float(data['High'].iloc[-1]),
            'low': float(data['Low'].iloc[-1]),
            'volume': float(data['Volume'].iloc[-1]),
        }
    except Exception as e:
        logger.exception(f"Error getting live price for {ticker}: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_live_vix(market: str) -> float:
    try:
        ticker = "^INDIAVIX" if market == "Indian Market" else "^VIX"
        default = 18.0 if market == "Indian Market" else 60.0
        data = fetch_data(ticker, period="5d", interval="1d")
        if data is None or data.empty:
            return default
        val = data['Close'].dropna().iloc[-1]
        return float(val) if not np.isnan(val) else default
    except Exception as e:
        logger.exception(f"Error fetching VIX for market {market}: {e}")
        return 60.0


# ─────────────────────────────────────────────
# ML HELPERS
# ─────────────────────────────────────────────
def build_ml_features(df, rsi_period=14, boll_period=20, boll_std=2.0, atr_period=14):
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
    return feat.dropna()


def explain_ml_prediction(model, feat_df, prob_pos):
    latest = feat_df.iloc[-1]
    importances = pd.Series(model.feature_importances_, index=feat_df.columns).sort_values(ascending=False)
    top_feats = importances.head(4).index.tolist()

    supporting = []
    opposing = []
    neutral = []

    for f in top_feats:
        v = latest.get(f, np.nan)
        if pd.isna(v):
            neutral.append(f)
            continue

        if f == 'rsi':
            if v < 40:
                supporting.append(f"RSI is low ({v:.1f}), indicating oversold conditions which often precede bounces")
            elif v > 60:
                opposing.append(f"RSI is high ({v:.1f}), indicating overbought conditions which often precede pullbacks")
            else:
                neutral.append(f"RSI is neutral ({v:.1f})")
        elif f == 'macd_diff':
            if v > 0:
                supporting.append(f"MACD diff is positive ({v:.4f}), showing bullish momentum")
            elif v < 0:
                opposing.append(f"MACD diff is negative ({v:.4f}), showing bearish momentum")
            else:
                neutral.append(f"MACD diff is near zero ({v:.4f})")
        elif f == 'bb_pos':
            if v < 0.3:
                supporting.append(f"Price is near the lower Bollinger band (bb_pos={v:.2f}), which can signal mean-reversion upside")
            elif v > 0.7:
                opposing.append(f"Price is near the upper Bollinger band (bb_pos={v:.2f}), which can signal mean-reversion downside")
            else:
                neutral.append(f"Bollinger position is mid-range (bb_pos={v:.2f})")
        elif f == 'atr':
            if 'atr' in feat_df.columns:
                med_atr = feat_df['atr'].median()
                if v > med_atr:
                    neutral.append(f"ATR is elevated ({v:.3f}), implying higher volatility and larger potential moves")
                else:
                    neutral.append(f"ATR is subdued ({v:.3f}), implying lower volatility")
            else:
                neutral.append(f"ATR = {v:.3f}")
        elif f == 'vol_ratio':
            if v > 1.5:
                supporting.append(f"Volume is elevated (vol_ratio={v:.2f}), which tends to confirm directional moves")
            elif v < 0.7:
                opposing.append(f"Volume is low (vol_ratio={v:.2f}), which can make breakouts less reliable")
            else:
                neutral.append(f"Volume is normal (vol_ratio={v:.2f})")
        elif f == 'returns':
            if v > 0:
                supporting.append(f"Recent return is positive ({v:.3%}), which supports short-term upside")
            elif v < 0:
                opposing.append(f"Recent return is negative ({v:.3%}), which supports short-term downside")
            else:
                neutral.append("Recent return is flat")
        else:
            med = feat_df[f].median() if f in feat_df.columns else None
            if med is not None:
                if v > med:
                    supporting.append(f"{f} is above its median ({v:.3f} > {med:.3f})")
                elif v < med:
                    opposing.append(f"{f} is below its median ({v:.3f} < {med:.3f})")
                else:
                    neutral.append(f"{f} is near its median ({v:.3f})")
            else:
                neutral.append(f"{f} = {v:.3f}")

    prob_pct = prob_pos * 100
    direction = "Bullish" if prob_pos >= 0.5 else "Bearish"
    conf = "high" if prob_pct >= 65 else "medium" if prob_pct >= 55 else "low"

    lines = []
    lines.append(f"**Model signal: {direction}**  —  probability {prob_pct:.1f}% (confidence: {conf}).")

    if supporting:
        lines.append("**Supporting factors:** " + "; ".join(supporting) + ".")
    if opposing:
        lines.append("**Opposing factors:** " + "; ".join(opposing) + ".")
    if neutral:
        lines.append("**Neutral / context:** " + "; ".join(neutral) + ".")

    if direction == "Bullish":
        lines.append("Interpretation: the model sees more indicators favoring upside; consider risk sizing because market conditions (volatility/volume) may affect reliability.")
    else:
        lines.append("Interpretation: the model sees more indicators favoring downside; consider protective measures (stop, hedge) as appropriate.")

    return " ".join(lines)


@st.cache_data(ttl=3600, show_spinner=False)
def train_ml_model(ticker, rsi_period=14, boll_period=20, boll_std=2.0, atr_period=14, calibrate=True):
    """
    Train RandomForest on 2 years of daily data.
    Returns: model, scaler, acc, train_cols
    """
    if not ML_AVAILABLE:
        logger.warning("ML libraries not available.")
        return None, None, 0, None

    data = fetch_data(ticker, period="2y", interval="1d")
    if data is None or len(data) < 200:
        logger.warning("Insufficient data for ML training.")
        return None, None, 0, None

    feat = build_ml_features(data, rsi_period=rsi_period, boll_period=boll_period, boll_std=boll_std, atr_period=atr_period)
    target = (data['Close'].squeeze().shift(-1) > data['Close'].squeeze()).astype(int).reindex(feat.index).dropna()
    feat = feat.reindex(target.index)
    if feat.empty:
        logger.warning("No aligned features for training.")
        return None, None, 0, None

    split = int(len(feat) * 0.8)
    X_tr, X_te = feat.iloc[:split], feat.iloc[split:]
    y_tr, y_te = target.iloc[:split], target.iloc[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_tr_s, y_tr)

    # Optional calibration for better probabilities
    if calibrate:
        try:
            calib = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
            calib.fit(X_te_s, y_te)
            model = calib
            logger.info("Applied probability calibration (CalibratedClassifierCV).")
        except Exception as e:
            logger.warning(f"Calibration failed or not available: {e}")

    acc = model.score(X_te_s, y_te) * 100
    train_cols = list(feat.columns)
    logger.info(f"Trained RF for {ticker} — OOS acc: {acc:.2f}% — features: {train_cols}")
    return model, scaler, acc, train_cols


# ─────────────────────────────────────────────
# CHARTS (preserved)
# ─────────────────────────────────────────────
def create_price_chart(chart_data, ticker_name, currency, show_sweeps=True, sweep_window=20, boll_period=20, boll_std=2.0):
    if chart_data is None or chart_data.empty:
        return None
    df = compute_liquidity_sweeps(chart_data, window=sweep_window) if show_sweeps else chart_data.copy()
    bb_upper, bb_mid, bb_lower = bollinger_bands(df['Close'], period=boll_period, std=boll_std)
    vwap_line = vwap(df)
    vol_colors = [
        'rgba(0,200,120,0.35)' if float(df['Close'].iloc[i]) >= float(df['Open'].iloc[i])
        else 'rgba(255,77,109,0.35)'
        for i in range(len(df))
    ]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.78, 0.22])
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='OHLC',
        increasing=dict(line=dict(color='#00c878', width=1)),
        decreasing=dict(line=dict(color='#ff4d6d', width=1)),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_upper,
                             line=dict(color='rgba(0,170,255,0.30)', width=1, dash='dot'),
                             name='BB Upper', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_lower,
                             line=dict(color='rgba(0,170,255,0.30)', width=1, dash='dot'),
                             name='BB Lower', fill='tonexty', fillcolor='rgba(0,170,255,0.04)',
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_mid,
                             line=dict(color='rgba(0,170,255,0.55)', width=1),
                             name='BB Mid / SMA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=vwap_line,
                             line=dict(color='#00e5ff', width=1.5, dash='dashdot'),
                             name='VWAP'), row=1, col=1)
    if show_sweeps and 'Supply_Sweep' in df.columns:
        supply_rows = df[df['Supply_Sweep']]
        demand_rows = df[df['Demand_Sweep']]
        price_range = float(df['High'].max() - df['Low'].min()) + 1e-9
        offset = price_range * 0.008
        plotted_s, plotted_d = set(), set()
        for _, row in supply_rows.tail(5).iterrows():
            lv = round(float(row['Prev_High']), 2)
            if lv not in plotted_s:
                fig.add_hline(y=lv, line=dict(color='rgba(255,77,77,0.45)', width=1, dash='dash'),
                              row=1, col=1)
                plotted_s.add(lv)
        for _, row in demand_rows.tail(5).iterrows():
            lv = round(float(row['Prev_Low']), 2)
            if lv not in plotted_d:
                fig.add_hline(y=lv, line=dict(color='rgba(0,200,120,0.45)', width=1, dash='dash'),
                              row=1, col=1)
                plotted_d.add(lv)
        if not supply_rows.empty:
            fig.add_trace(go.Scatter(
                x=supply_rows.index, y=supply_rows['High'] + offset,
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='#ff4d6d',
                            line=dict(width=1, color='#ff0000')),
                name='Supply Sweep',
                hovertemplate='Supply Sweep<br>High: %{customdata:.2f}<extra></extra>',
                customdata=supply_rows['High'],
            ), row=1, col=1)
        if not demand_rows.empty:
            fig.add_trace(go.Scatter(
                x=demand_rows.index, y=demand_rows['Low'] - offset,
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='#00c878',
                            line=dict(width=1, color='#00ff88')),
                name='Demand Sweep',
                hovertemplate='Demand Sweep<br>Low: %{customdata:.2f}<extra></extra>',
                customdata=demand_rows['Low'],
            ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        marker_color=vol_colors, name='Volume', showlegend=False,
    ), row=2, col=1)
    vol_ma = df['Volume'].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=df.index, y=vol_ma,
        line=dict(color='#ffd700', width=1.5),
        name='Vol MA20',
    ), row=2, col=1)
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
        title=dict(text=f"<b>{ticker_name}</b> — Price Action",
                   font=dict(family='Space Mono, monospace', size=14, color='#0af'), x=0.01),
        height=700, xaxis_rangeslider_visible=False, hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0,
                    font=dict(size=11), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=20, t=50, b=40),
    )
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)', title_text=f"Price ({currency})",
                     row=1, col=1)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)', title_text="Volume", row=2, col=1)
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


def create_hurst_yearly_chart(daily_1y):
    close = daily_1y['Close'].squeeze()
    if len(close) < 120:
        return None, 0.5, "Insufficient data", "low"
    hurst_vals, dates = [], []
    for i in range(120, len(close), 5):
        h, _, _ = hurst_exponent(close.iloc[i - 120:i])
        hurst_vals.append(h)
        dates.append(close.index[i - 1])
    current_h, current_interp, confidence = hurst_exponent(close.tail(min(252, len(close))))
    if not hurst_vals:
        return None, current_h, current_interp, confidence
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.10, row_heights=[0.50, 0.50])
    fig.add_trace(go.Scatter(x=close.index, y=close.values,
                             name='Price (1Y Daily)', line=dict(color='rgba(255,255,255,0.75)', width=1.5)),
                  row=1, col=1)
    colors_h = ['#00c878' if h > 0.58 else '#ff4d6d' if h < 0.42 else '#ffd700'
                for h in hurst_vals]
    for i in range(1, len(hurst_vals)):
        fig.add_trace(go.Scatter(
            x=dates[i - 1:i + 1], y=hurst_vals[i - 1:i + 1],
            mode='lines', line=dict(color=colors_h[i], width=2),
            showlegend=False, hoverinfo='skip'), row=2, col=1)
    fig.add_hrect(y0=0.58, y1=0.90, fillcolor='rgba(0,200,100,0.06)', line_width=0, row=2, col=1)
    fig.add_hrect(y0=0.42, y1=0.58, fillcolor='rgba(200,200,200,0.03)', line_width=0, row=2, col=1)
    fig.add_hrect(y0=0.10, y1=0.42, fillcolor='rgba(255,80,80,0.06)', line_width=0, row=2, col=1)
    for y, color, label in [
        (0.58, '#00c878', 'Trending  H>0.58'),
        (0.50, 'rgba(200,200,200,0.4)', 'Random Walk'),
        (0.42, '#ff4d6d', 'Mean-Rev  H<0.42'),
    ]:
        fig.add_hline(y=y, line_dash='dash', line_color=color,
                      annotation_text=label,
                      annotation_font=dict(size=10, color=color), row=2, col=1)
    if hurst_vals:
        fig.add_trace(go.Scatter(
            x=[dates[-1]], y=[hurst_vals[-1]], mode='markers',
            marker=dict(size=11, color='#ffd700', symbol='diamond',
                        line=dict(width=2, color='white')),
            name=f'Current H={hurst_vals[-1]:.3f}'), row=2, col=1)
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
        height=540, hovermode='x unified',
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=20, t=30, b=40),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(title_text="Hurst (120D rolling)", range=[0.2, 0.8], row=2, col=1,
                     gridcolor='rgba(255,255,255,0.04)')
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig, current_h, current_interp, confidence


def create_hurst_loglog_chart(price_series):
    price = np.asarray(price_series.squeeze().dropna(), dtype=float)
    n = len(price)
    if n < 100:
        return None
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
    if len(valid_lags) < 4:
        return None
    log_lags = np.log(valid_lags)
    log_rs = np.log(rs_values)
    coeffs = np.polyfit(log_lags, log_rs, 1)
    hurst = float(np.clip(coeffs[0], 0.05, 0.95))
    fit_line = np.polyval(coeffs, log_lags)
    x_range = [min(log_lags), max(log_lags)]
    intercept = np.mean(log_rs) - 0.5 * np.mean(log_lags)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=log_lags, y=log_rs, mode='markers',
                             marker=dict(color='#00e5ff', size=8), name='R/S points'))
    fig.add_trace(go.Scatter(x=log_lags, y=fit_line, mode='lines',
                             line=dict(color='#ffd700', width=2, dash='dash'),
                             name=f'OLS slope = {hurst:.3f}'))
    fig.add_trace(go.Scatter(
        x=x_range, y=[0.5 * x + intercept for x in x_range],
        line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'),
        name='H=0.5 reference'))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
        title=f'<b>Hurst R/S Log-Log Plot</b> — H = {hurst:.4f}',
        xaxis_title='log(lag)', yaxis_title='log(R/S)',
        height=380, hovermode='x unified', margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
    )
    return fig


def create_correlation_chart(ticker1, ticker2, name1, name2):
    data1 = fetch_data(ticker1, period="1y")
    data2 = fetch_data(ticker2, period="1y")
    if data1 is None or data2 is None:
        return None
    merged = pd.DataFrame({name1: data1['Close'], name2: data2['Close']}).dropna()
    if len(merged) < 20:
        return None
    norm = merged / merged.iloc[0] * 100
    log_ret = np.log(merged / merged.shift(1)).dropna()
    corr = log_ret[name1].rolling(20).corr(log_ret[name2])
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12, row_heights=[0.6, 0.4])
    fig.add_trace(go.Scatter(x=norm.index, y=norm[name1], name=name1,
                             line=dict(color='#00c878', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=norm.index, y=norm[name2], name=name2,
                             line=dict(color='#ff4d6d', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=corr.index, y=corr, name='20D Rolling Corr',
                             line=dict(color='#00e5ff', width=2)), row=2, col=1)
    fig.add_hline(y=0.8, line_dash="dash", line_color="#00c878",
                  annotation_text="High (0.8)", row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="#ff4d6d",
                  annotation_text="Low (0.5)", row=2, col=1)
    fig.add_hline(y=0.0, line_dash="solid", line_color="rgba(255,255,255,0.2)", row=2, col=1)
    fig.update_layout(template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
                      height=500, hovermode='x unified', margin=dict(l=60, r=20, t=30, b=40),
                      legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
    fig.update_yaxes(title_text="Normalised Price", row=1, col=1,
                     gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(title_text="Correlation", range=[-1, 1], row=2, col=1,
                     gridcolor='rgba(255,255,255,0.04)')
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


def create_volatility_cone(close, trading_days=252):
    log_ret = np.log(close / close.shift(1)).dropna()
    windows = [10, 20, 30, 60, 90, 120, 180]
    max_vol, min_vol, med_vol, cur_vol = [], [], [], []
    for w in windows:
        rv = log_ret.rolling(w).std() * np.sqrt(trading_days) * 100
        rv = rv.dropna()
        if len(rv) > 0:
            max_vol.append(rv.max()); min_vol.append(rv.min())
            med_vol.append(rv.median()); cur_vol.append(rv.iloc[-1])
    if len(max_vol) < 3:
        return None
    w = windows[:len(max_vol)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w + w[::-1], y=max_vol + min_vol[::-1],
                             fill='toself', fillcolor='rgba(0,170,255,0.07)', line=dict(color='rgba(0,0,0,0)'),
                             name='Historical Range'))
    fig.add_trace(go.Scatter(x=w, y=max_vol, name='Max',
                             line=dict(color='#ff4d6d', width=1.5, dash='dot'), mode='lines+markers',
                             marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=w, y=min_vol, name='Min',
                             line=dict(color='#00c878', width=1.5, dash='dot'), mode='lines+markers',
                             marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=w, y=med_vol, name='Median',
                             line=dict(color='rgba(255,255,255,0.5)', width=1.5), mode='lines+markers',
                             marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=w, y=cur_vol, name='Current',
                             line=dict(color='#ffd700', width=3), mode='lines+markers',
                             marker=dict(size=9, symbol='diamond', color='#ffd700',
                                         line=dict(width=2, color='white'))))
    fig.update_layout(template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
                      title=dict(text='<b>Volatility Cone</b>',
                                 font=dict(family='Space Mono, monospace', size=13, color='#0af')),
                      xaxis_title='Window (Days)', yaxis_title='Annualised Volatility (%)',
                      height=420, hovermode='x unified', margin=dict(l=60, r=20, t=50, b=40),
                      legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


def create_iv_rank_chart(close, trading_days=252):
    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) < 30:
        return None
    hv = log_ret.rolling(20).std() * np.sqrt(trading_days) * 100
    hv = hv.dropna()
    if len(hv) < 20:
        return None
    current = hv.iloc[-1]
    ivr = (current - hv.min()) / (hv.max() - hv.min()) * 100 if hv.max() != hv.min() else 50.0
    ivp = (hv < current).sum() / len(hv) * 100
    high_thresh = hv.quantile(0.65)
    low_thresh = hv.quantile(0.30)
    colors = ['#ff4d6d' if v >= high_thresh else '#00c878' if v <= low_thresh else '#ffd700'
              for v in hv.values]
    fig = go.Figure()
    for i in range(1, len(hv)):
        fig.add_trace(go.Scatter(x=hv.index[i - 1:i + 1], y=hv.values[i - 1:i + 1],
                                 mode='lines', line=dict(color=colors[i], width=2),
                                 showlegend=False, hoverinfo='skip'))
    for y, color, label in [
        (hv.max(), '#ff4d6d', f"52W High: {hv.max():.1f}%"),
        (hv.min(), '#00c878', f"52W Low: {hv.min():.1f}%"),
        (current, '#ffd700', f"Current: {current:.1f}%"),
    ]:
        fig.add_hline(y=y, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_font=dict(color=color, size=10))
    fig.update_layout(template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
                      title=dict(text=f'<b>IV Rank {ivr:.0f}%</b>  |  IV Percentile {ivp:.0f}%',
                                 font=dict(family='Space Mono, monospace', size=13, color='#0af')),
                      yaxis_title='HV-20 (%)', height=420, hovermode='x unified',
                      margin=dict(l=60, r=20, t=50, b=40))
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


def create_expected_move_chart(spot, implied_vol, trading_days, currency="$"):
    daily_move = spot * (implied_vol / 100) / np.sqrt(trading_days)
    weekly_move = daily_move * np.sqrt(5)
    monthly_move = daily_move * np.sqrt(21)
    labels = ['Daily (±1σ)', 'Weekly (±1σ)', 'Monthly (±1σ)']
    values = [daily_move, weekly_move, monthly_move]
    pcts = [v / spot * 100 for v in values]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values,
                         marker_color=['#00e5ff', '#ffd700', '#ff9500'],
                         text=[f'{currency}{v:,.0f}<br>({p:.1f}%)' for v, p in zip(values, pcts)],
                         textposition='outside', textfont=dict(size=11)))
    fig.update_layout(template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
                      title=dict(text=f'<b>Expected Move</b>  (IV: {implied_vol:.1f}%, 1σ = 68% prob)',
                                 font=dict(family='Space Mono, monospace', size=13, color='#0af')),
                      yaxis_title=f'Move ({currency})', height=420, showlegend=False,
                      margin=dict(l=60, r=20, t=60, b=40))
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


def create_oi_profile(spot):
    step = 500 if spot > 10000 else 100 if spot > 2000 else 50
    base = round(spot / step) * step
    strikes = np.arange(base - 8 * step, base + 9 * step, step)
    rng = np.random.default_rng(int(spot * 100) % (2 ** 31))
    calls = rng.integers(10, 80, len(strikes)) * 50000
    puts = rng.integers(10, 80, len(strikes)) * 50000
    pain = {s: (np.sum(np.maximum(0, s - strikes) * calls) +
                np.sum(np.maximum(0, strikes - s) * puts)) for s in strikes}
    max_pain = min(pain, key=pain.get)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=strikes, x=calls / 1e5, orientation='h',
                         name='Call OI', marker_color='rgba(255,77,109,0.75)'))
    fig.add_trace(go.Bar(y=strikes, x=-puts / 1e5, orientation='h',
                         name='Put OI', marker_color='rgba(0,200,120,0.75)'))
    fig.add_hline(y=spot, line_color='#ffd700', line_dash='solid', line_width=2,
                  annotation_text=f'Spot {spot:,.0f}',
                  annotation_font=dict(color='#ffd700', size=11))
    fig.add_hline(y=max_pain, line_color='#c084fc', line_dash='dash', line_width=1.5,
                  annotation_text=f'Max Pain {max_pain:,.0f}',
                  annotation_font=dict(color='#c084fc', size=11))
    fig.add_vline(x=0, line_color='rgba(255,255,255,0.2)', line_width=1)
    fig.update_layout(template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
                      title=dict(text='<b>Open Interest Profile</b>',
                                 font=dict(family='Space Mono, monospace', size=13, color='#0af')),
                      xaxis_title='OI (Lakhs)', yaxis_title='Strike Price',
                      height=520, barmode='relative', hovermode='y unified',
                      margin=dict(l=80, r=20, t=50, b=40),
                      legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


# ─────────────────────────────────────────────
# Sidebar — original controls preserved; enhancements added
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 AlphaQuant")
    st.markdown("---")

    market = st.radio("Market", ["Crypto", "Indian Market"], horizontal=True)

    if market == "Crypto":
        assets = {
            'Bitcoin': 'BTC-USD',
            'Ethereum': 'ETH-USD',
            'Dogecoin': 'DOGE-USD',
            'XRP': 'XRP-USD',
            'Solana': 'SOL-USD',
            'BNB': 'BNB-USD',
        }
        trading_days = 365
        currency = "$"
    else:
        assets = {
            'Nifty 50': '^NSEI',
            'Sensex': '^BSESN',
            'Bank Nifty': '^NSEBANK',
            'Nifty IT': '^CNXIT',
        }
        trading_days = 252
        currency = "₹"

    selected_asset = st.selectbox("Asset", list(assets.keys()))
    ticker = assets[selected_asset]  # <--- ticker selection preserved exactly

    st.markdown("---")
    st.markdown("### Liquidity Sweep Settings")
    show_sweeps = st.checkbox("Show Liquidity Sweeps", value=True)
    sweep_window = st.slider("Sweep Detection Window", min_value=10, max_value=50,
                             value=20, step=5)

    st.markdown("---")
    st.markdown("### Indicator Parameters")
    rsi_period = st.slider("RSI Period", 5, 30, 14,
                           help="Relative Strength Index period. Higher = smoother RSI.")
    boll_period = st.number_input("Bollinger Period", min_value=5, max_value=60, value=20,
                                 help="SMA period used for Bollinger Bands.")
    boll_std = st.slider("Bollinger Std Dev", 1.0, 3.0, 2.0, step=0.1,
                         help="Number of standard deviations for Bollinger Bands.")
    atr_period = st.slider("ATR Period", 5, 30, 14,
                          help="ATR averaging period used for volatility.")

    st.markdown("---")
    st.markdown("### Correlation Pair")
    if market == "Crypto":
        corr_pair = st.selectbox("Pair", ["Bitcoin vs Ethereum", "Bitcoin vs Dogecoin"])
        corr_map = {
            "Bitcoin vs Ethereum": ("BTC-USD", "ETH-USD", "Bitcoin", "Ethereum"),
            "Bitcoin vs Dogecoin": ("BTC-USD", "DOGE-USD", "Bitcoin", "Dogecoin"),
        }
    else:
        corr_pair = st.selectbox("Pair", ["Nifty 50 vs Bank Nifty", "Nifty 50 vs Sensex"])
        corr_map = {
            "Nifty 50 vs Bank Nifty": ("^NSEI", "^NSEBANK", "Nifty 50", "Bank Nifty"),
            "Nifty 50 vs Sensex": ("^NSEI", "^BSESN", "Nifty 50", "Sensex"),
        }

    st.markdown("---")
    st.markdown("### Multi-Ticker Comparison")
    default_compare = [ticker]
    if market == "Crypto":
        default_compare = [ticker, "BTC-USD"] if ticker != "BTC-USD" else ["BTC-USD", "ETH-USD"]
    else:
        default_compare = [ticker, "^NSEI"] if ticker != "^NSEI" else ["^NSEI", "^NSEBANK"]

    all_options = list(dict.fromkeys(list(assets.values()) + ["AAPL", "MSFT", "^VIX", "^INDIAVIX"]))
    for d in default_compare:
        if d not in all_options:
            all_options.append(d)

    # Fixed multiselect signature (label, options=..., default=...)
    compare_tickers = st.multiselect(
        "Select tickers to compare (norm. price)",
        options=all_options,
        default=default_compare,
        help="Normalized price comparison across tickers. This does not change the main ticker."
    )

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────────
# Data load (preserve original behavior)
# ─────────────────────────────────────────────
hist_1y = fetch_data(ticker, period="1y", interval="1d")
hist_2y = fetch_data(ticker, period="2y", interval="1d")
live = get_live_price(ticker)
implied_vol = fetch_live_vix(market)

if hist_1y is None or hist_1y.empty:
    st.error("❌ Unable to load data. Please try again.")
    st.stop()

close_1y = hist_1y['Close'].squeeze()
high_1y = hist_1y['High'].squeeze()
low_1y = hist_1y['Low'].squeeze()

if live:
    spot = live['price']
    change_pct = live['pct']
else:
    spot = float(hist_1y['Close'].iloc[-1])
    change_pct = 0.0

iv_rank, iv_percentile = compute_iv_rank(close_1y, 20)
parkinson = compute_parkinson_vol(high_1y, low_1y, trading_days)
fig_hurst, current_h, current_interp, hurst_confidence = create_hurst_yearly_chart(hist_1y)

regime_color = '#00c878' if current_h > 0.58 else '#ff4d6d' if current_h < 0.42 else '#ffd700'

liq_df = compute_liquidity_sweeps(hist_1y, window=sweep_window)
last_supply = bool(liq_df['Supply_Sweep'].iloc[-1]) if 'Supply_Sweep' in liq_df.columns else False
last_demand = bool(liq_df['Demand_Sweep'].iloc[-1]) if 'Demand_Sweep' in liq_df.columns else False
total_supply = int(liq_df['Supply_Sweep'].sum()) if 'Supply_Sweep' in liq_df.columns else 0
total_demand = int(liq_df['Demand_Sweep'].sum()) if 'Demand_Sweep' in liq_df.columns else 0

if last_supply:
    sweep_regime = "SUPPLY SWEEP ACTIVE"
    sweep_desc = "Failed breakout / Institutional absorption at highs"
    sweep_badge_color = "#ff4d6d"
elif last_demand:
    sweep_regime = "DEMAND SWEEP ACTIVE"
    sweep_desc = "Failed breakdown / Institutional absorption at lows"
    sweep_badge_color = "#00c878"
else:
    sweep_regime = "PRICE DISCOVERY PHASE"
    sweep_desc = "Trading inside established structural bounds"
    sweep_badge_color = "#00e5ff"


# ─────────────────────────────────────────────
# Multi-ticker comparison helper (uses cached fetches)
# ─────────────────────────────────────────────
def plot_multi_ticker_comparison(ticker_list, period="1y"):
    data_map = {}
    for t in ticker_list:
        d = fetch_data(t, period=period)
        if d is not None and 'Close' in d.columns:
            data_map[t] = d['Close']
    if not data_map:
        return None
    merged = pd.DataFrame(data_map).dropna()
    norm = merged / merged.iloc[0] * 100
    fig = go.Figure()
    for t in norm.columns:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[t], name=t))
    fig.update_layout(template='plotly_dark', height=500, margin=dict(l=60, r=20, t=40, b=40))
    return fig


# ─────────────────────────────────────────────
# MAIN TABS (original layout preserved; ML uses indicator params)
# ─────────────────────────────────────────────
main_tab1, main_tab2 = st.tabs([
    "📈 Dashboard",
    "🤖 ML Signal",
])

# TAB 1 — DASHBOARD
with main_tab1:
    st.title("AlphaQuant Terminal")
    st.markdown(
        f"<span style='opacity:0.45;font-size:12px;font-family:Space Mono,monospace'>"
        f"LIVE · {selected_asset} ({ticker}) · {datetime.now(IST).strftime('%H:%M:%S IST')}"
        f"</span>", unsafe_allow_html=True,
    )
    st.markdown("---")

    # Key Metrics
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    price_color = '#00c878' if change_pct >= 0 else '#ff4d6d'

    with c1:
        st.markdown(f"""<div class="metric-box" style="border-left-color:{price_color}">
            <div style="font-size:10px;opacity:0.6;font-family:Space Mono,monospace">SPOT PRICE</div>
            <div style="font-size:20px;font-weight:600;color:{price_color};margin:3px 0">{currency}{spot:,.2f}</div>
            <div style="font-size:10px;color:{price_color}">{change_pct:+.2f}%</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div class="metric-box">
            <div style="font-size:10px;opacity:0.6;font-family:Space Mono,monospace">PARKINSON VOL</div>
            <div style="font-size:20px;font-weight:600;color:#0af;margin:3px 0">{parkinson:.1f}%</div>
            <div style="font-size:10px;opacity:0.6">Annualised</div>
        </div>""", unsafe_allow_html=True)

    ivr_color = '#ff4d6d' if iv_rank > 65 else '#00c878' if iv_rank < 30 else '#ffd700'
    with c3:
        st.markdown(f"""<div class="metric-box" style="border-left-color:{ivr_color}">
            <div style="font-size:10px;opacity:0.6;font-family:Space Mono,monospace">IV RANK</div>
            <div style="font-size:20px;font-weight:600;color:{ivr_color};margin:3px 0">{iv_rank:.0f}%</div>
            <div style="font-size:10px;opacity:0.6">{'Sell' if iv_rank>65 else 'Buy' if iv_rank<30 else 'Neutral'} premium</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""<div class="metric-box">
            <div style="font-size:10px;opacity:0.6;font-family:Space Mono,monospace">IV PERCENTILE</div>
            <div style="font-size:20px;font-weight:600;color:#c084fc;margin:3px 0">{iv_percentile:.0f}%</div>
            <div style="font-size:10px;opacity:0.6">Historical ctx</div>
        </div>""", unsafe_allow_html=True)

    conf_icon = '●' if hurst_confidence == 'high' else '◑' if hurst_confidence == 'medium' else '○'
    with c5:
        st.markdown(f"""<div class="metric-box" style="border-left-color:{regime_color}">
            <div style="font-size:10px;opacity:0.6;font-family:Space Mono,monospace">HURST (1Y)</div>
            <div style="font-size:20px;font-weight:600;color:{regime_color};margin:3px 0">{current_h:.3f}</div>
            <div style="font-size:10px;color:{regime_color}">{conf_icon} {hurst_confidence}</div>
        </div>""", unsafe_allow_html=True)

    with c6:
        st.markdown(f"""<div class="metric-box" style="border-left-color:{regime_color}">
            <div style="font-size:10px;opacity:0.6;font-family:Space Mono,monospace">MARKET REGIME</div>
            <div style="font-size:13px;font-weight:600;color:{regime_color};margin:3px 0;line-height:1.3">{current_interp}</div>
            <div style="font-size:10px;opacity:0.6">R/S (120D window)</div>
        </div>""", unsafe_allow_html=True)

    with c7:
        st.markdown(f"""<div class="metric-box" style="border-left-color:{sweep_badge_color}">
            <div style="font-size:10px;opacity:0.6;font-family:Space Mono,monospace">LIQUIDITY</div>
            <div style="font-size:11px;font-weight:700;color:{sweep_badge_color};margin:3px 0;line-height:1.4">{sweep_regime}</div>
            <div style="font-size:10px;opacity:0.6">{total_supply}↓ supply / {total_demand}↑ demand (1Y)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Multi-Timeframe Price Chart
    st.markdown('<div class="section-header">Price Action — Multi-Timeframe</div>', unsafe_allow_html=True)

    TF_MAP = {
        "15m  (5D)": ("5d", "15m"),
        "1h   (1M)": ("1mo", "1h"),
        "4h   (3M)": ("3mo", "60m"),
        "1D   (1Y)": ("1y", "1d"),
        "1W   (5Y)": ("5y", "1wk"),
    }
    tabs_tf = st.tabs(list(TF_MAP.keys()))
    for tab, (label, (period_tf, interval_tf)) in zip(tabs_tf, TF_MAP.items()):
        with tab:
            tf_data = fetch_data(ticker, period=period_tf, interval=interval_tf)
            if tf_data is not None and not tf_data.empty:
                fig_price = create_price_chart(
                    tf_data, selected_asset, currency,
                    show_sweeps=show_sweeps, sweep_window=sweep_window,
                    boll_period=boll_period, boll_std=boll_std,
                )
                if fig_price:
                    st.plotly_chart(fig_price, use_container_width=True)
                    sweep_note = (
                        " · **▼ Red triangle** = Supply Sweep · "
                        "**▲ Green triangle** = Demand Sweep · Dashed lines = structural levels"
                    ) if show_sweeps else ""
                    st.markdown(f"""<div class="explanation-box">
                        <b>Chart layers:</b> Candlestick · Bollinger Bands ({boll_period}, {boll_std}σ) ·
                        VWAP · Volume bar + 20-bar MA{sweep_note}
                    </div>""", unsafe_allow_html=True)
            else:
                st.info(f"No data available for {label} timeframe.")

    st.markdown("---")

    # Liquidity Sweep Section
    st.markdown('<div class="section-header">Liquidity Sweep Analysis (1Y Daily)</div>', unsafe_allow_html=True)
    lsw_col1, lsw_col2, lsw_col3 = st.columns([1, 1, 2])

    with lsw_col1:
        st.markdown(f"""<div class="metric-box" style="border-left-color:{sweep_badge_color};padding:20px">
            <div style="font-size:11px;opacity:0.6;font-family:Space Mono,monospace;margin-bottom:8px">CURRENT MICROSTRUCTURE</div>
            <div style="font-size:15px;font-weight:700;color:{sweep_badge_color};margin-bottom:6px">{sweep_regime}</div>
            <div style="font-size:11px;color:rgba(255,255,255,0.65);line-height:1.5">{sweep_desc}</div>
        </div>""", unsafe_allow_html=True)

    with lsw_col2:
        recent_supply = liq_df[liq_df['Supply_Sweep']].tail(3)
        recent_demand = liq_df[liq_df['Demand_Sweep']].tail(3)
        supply_levels = [f"{float(r['Prev_High']):.2f}" for _, r in recent_supply.iterrows()]
        demand_levels = [f"{float(r['Prev_Low']):.2f}" for _, r in recent_demand.iterrows()]
        st.markdown(f"""<div class="metric-box" style="padding:20px">
            <div style="font-size:11px;opacity:0.6;font-family:Space Mono,monospace;margin-bottom:8px">RECENT SWEPT LEVELS</div>
            <div style="font-size:11px;color:#ff4d6d;margin-bottom:4px">
                <b>Supply swept:</b><br>{'  ·  '.join(supply_levels) if supply_levels else 'None in window'}
            </div>
            <div style="font-size:11px;color:#00c878;margin-top:8px">
                <b>Demand swept:</b><br>{'  ·  '.join(demand_levels) if demand_levels else 'None in window'}
            </div>
            <div style="font-size:10px;opacity:0.5;margin-top:8px">Detection window: {sweep_window} bars</div>
        </div>""", unsafe_allow_html=True)

    with lsw_col3:
        liq_monthly = liq_df[['Supply_Sweep', 'Demand_Sweep']].resample(_month_end_alias()).sum()
        if not liq_monthly.empty:
            fig_liq = go.Figure()
            fig_liq.add_trace(go.Bar(x=liq_monthly.index, y=liq_monthly['Supply_Sweep'],
                                     name='Supply Sweeps', marker_color='rgba(255,77,109,0.75)'))
            fig_liq.add_trace(go.Bar(x=liq_monthly.index, y=liq_monthly['Demand_Sweep'],
                                     name='Demand Sweeps', marker_color='rgba(0,200,120,0.75)'))
            fig_liq.update_layout(template='plotly_dark', paper_bgcolor='#080d12',
                                  plot_bgcolor='#0a1018',
                                  title=dict(text='<b>Monthly Sweep Frequency (1Y)</b>',
                                             font=dict(family='Space Mono, monospace', size=12, color='#0af')),
                                  height=220, barmode='group', hovermode='x unified',
                                  margin=dict(l=40, r=20, t=40, b=30),
                                  legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)', orientation='h'),
                                  yaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                                  xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                                  )
            st.plotly_chart(fig_liq, use_container_width=True)

    st.markdown("""<div class="explanation-box">
        <b>Liquidity Sweep Logic:</b>
        A <span style="color:#ff4d6d"><b>Supply Sweep</b></span> occurs when price wicks above the
        rolling structural high but closes back below it — institutional absorption / failed breakout (bearish).
        A <span style="color:#00c878"><b>Demand Sweep</b></span> occurs when price wicks below the rolling
        structural low but closes back above it — smart-money accumulation / failed breakdown (bullish).
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Correlation + IV Rank
    st.markdown('<div class="section-header">Advanced Analysis</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Correlation Analysis")
        t1, t2, n1, n2 = corr_map[corr_pair]
        fig_corr = create_correlation_chart(t1, t2, n1, n2)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown("""<div class="explanation-box">
                <b>Rolling 20D Correlation:</b> >0.8 = in sync · <0.5 = diverging · Negative = inverse
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Correlation data unavailable")

    with col_r:
        st.markdown("#### IV Rank & Percentile")
        fig_iv = create_iv_rank_chart(close_1y, trading_days)
        if fig_iv:
            st.plotly_chart(fig_iv, use_container_width=True)
            st.markdown("""<div class="explanation-box">
                Red = expensive vol (sell premium) · Green = cheap vol (buy premium) ·
                IVR > 65% → short vega; IVR < 30% → long vega
            </div>""", unsafe_allow_html=True)
        else:
            st.info("IV data unavailable")

    st.markdown("---")

    # Vol Cone + Expected Move
    st.markdown('<div class="section-header">Volatility & Options Framework</div>', unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("#### Volatility Cone")
        fig_vc = create_volatility_cone(close_1y, trading_days)
        if fig_vc:
            st.plotly_chart(fig_vc, use_container_width=True)
            st.markdown("""<div class="explanation-box">
                <b>Gold diamond = current vol.</b> Above median → elevated; below → compressed.
            </div>""", unsafe_allow_html=True)
    with v2:
        st.markdown(f"#### Expected Move (1σ)  —  IV: {implied_vol:.1f}%")
        fig_em = create_expected_move_chart(spot, implied_vol, trading_days, currency)
        st.plotly_chart(fig_em, use_container_width=True)
        st.markdown("""<div class="explanation-box">
            <b>1σ range</b> covers ~68% of expected outcomes. IV sourced live from VIX feed.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # OI Profile
    st.markdown('<div class="section-header">Open Interest Profile</div>', unsafe_allow_html=True)
    fig_oi = create_oi_profile(spot)
    st.plotly_chart(fig_oi, use_container_width=True)
    st.markdown("""<div class="explanation-box">
        <b>OI Profile:</b> Call (red) vs Put (green) OI by strike.
        <b>Gold = Spot · Purple = Max Pain</b> — price gravitates toward max pain near expiry.
    </div>""", unsafe_allow_html=True)

    # Multi-ticker comparison (optional)
    if compare_tickers and len(compare_tickers) >= 2:
        st.markdown("---")
        st.markdown('<div class="section-header">Multi-Ticker Comparison</div>', unsafe_allow_html=True)
        mt_fig = plot_multi_ticker_comparison(compare_tickers, period="1y")
        if mt_fig:
            st.plotly_chart(mt_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("""<div style="text-align:center;opacity:0.35;margin:16px 0;
        font-family:Space Mono,monospace;font-size:10px;letter-spacing:.12em">
        ALPHAQUANT TERMINAL · QUANTITATIVE ANALYSIS · DATA REFRESHES EVERY 5 MIN
    </div>""", unsafe_allow_html=True)


    st.markdown("""
    The **Hurst Exponent (H)** quantifies long-range memory / self-similarity in a price series.

    | Range | Regime | Strategy |
    |---|---|---|
    | H > 0.58 | Trending (persistent) | Momentum, trend-following |
    | H ≈ 0.50 | Random Walk | No directional edge |
    | H < 0.42 | Mean-Reverting (anti-persistent) | Fade extremes, range strategies |
    """)

    conf_colors = {"high": "#00c878", "medium": "#ffd700", "low": "#ff4d6d"}
    conf_color = conf_colors.get(hurst_confidence, "#aaa")
    conf_icon = '●' if hurst_confidence == 'high' else '◑' if hurst_confidence == 'medium' else '○'

    st.markdown(
        f'<div class="metric-box" style="border-left: 4px solid #0af; padding: 20px; margin-bottom:16px">'
        f'<span style="font-size:30px;font-weight:700;font-family:Space Mono,monospace;'
        f'color:{regime_color}">H = {current_h:.4f}</span>'
        f'&nbsp;&nbsp;<span style="font-size:14px;color:{regime_color}">{current_interp}</span><br>'
        f'<span style="font-family:monospace;font-size:12px;color:{conf_color}">'
        f'{conf_icon} Confidence: {hurst_confidence.upper()}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    if fig_hurst:
        st.plotly_chart(fig_hurst, use_container_width=True)
        st.markdown(f"""<div class="explanation-box">
            <b>Top panel:</b> Price (1Y daily)  ·  <b>Bottom panel:</b> Rolling 120-bar Hurst —
            Green = trending, Yellow = random walk, Red = mean-reverting.<br>
            Current H = {current_h:.3f} ({current_interp}) · Confidence: {hurst_confidence}
            (based on R² of log-log OLS fit)
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Need ≥120 bars of daily data for Hurst analysis.")

    st.markdown('<div class="section-header">R/S Log-Log Regression Diagnostic</div>', unsafe_allow_html=True)
    fig_ll = create_hurst_loglog_chart(close_1y)
    if fig_ll:
        st.plotly_chart(fig_ll, use_container_width=True)
        st.markdown("""<div class="explanation-box">
            <b>How to read:</b> Each dot is the average R/S ratio at that lag horizon.
            The yellow OLS line's slope = Hurst exponent.
            A tight fit (high R²) = reliable estimate.
            The dotted white line shows H=0.5 (pure random walk) for reference.
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("Insufficient data for log-log diagnostic (need ≥100 data points).")

    st.markdown("""<div class="explanation-box">
        <b>Methodology:</b> Rescaled Range (R/S) analysis using log-spaced lags (30 points, 10→n/2).
        Minimum 3 windows per lag, minimum 8 valid lag points for the OLS regression.
        Confidence level = R² of the log(R/S) vs log(lag) fit.
        Uses <code>.squeeze()</code> to ensure 1D input and avoid MultiIndex artefacts.
    </div>""", unsafe_allow_html=True)

# TAB 3 — ML SIGNAL
with main_tab2:
    st.title("ML Signal — Random Forest")

    if not ML_AVAILABLE:
        st.error("scikit-learn not installed. Run: `pip install scikit-learn`")
    else:
        col_btn, col_acc = st.columns([1, 3])
        with col_btn:
            retrain = st.button("🔁 Train / Refresh Model")

        if retrain:
            train_ml_model.clear()

        with st.spinner("Training model on 2Y of daily data…"):
            model, scaler, acc, train_cols = train_ml_model(
                ticker,
                rsi_period=rsi_period,
                boll_period=boll_period,
                boll_std=boll_std,
                atr_period=atr_period,
                calibrate=True
            )

        if model is None or scaler is None:
            st.warning("Could not train model — insufficient data (need ≥200 bars) or ML libs missing.")
        else:
            with col_acc:
                acc_color = '#00c878' if acc >= 55 else '#ffd700' if acc >= 50 else '#ff4d6d'
                st.markdown(
                    f'<div class="metric-box" style="border-left-color:{acc_color}">'
                    f'<div style="font-size:10px;opacity:0.6;font-family:Space Mono,monospace">OUT-OF-SAMPLE ACCURACY</div>'
                    f'<div style="font-size:26px;font-weight:700;color:{acc_color}">{acc:.1f}%</div>'
                    f'<div style="font-size:10px;opacity:0.6">Random Forest · 2Y train · 20% OOS</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Latest prediction using features built with the chosen indicator params
            feat_df = build_ml_features(hist_1y, rsi_period=rsi_period, boll_period=boll_period, boll_std=boll_std, atr_period=atr_period)
            if feat_df.empty:
                st.info("Not enough feature rows to predict.")
            else:
                # Align columns to training columns to avoid shape mismatch
                try:
                    if train_cols is not None:
                        feat_df = feat_df.reindex(columns=train_cols)
                    latest_row = feat_df.iloc[[-1]]  # keep DataFrame shape and column names
                    # Check for NaNs
                    if latest_row.isna().any(axis=None):
                        st.warning("Latest feature row contains NaNs; prediction skipped.")
                    else:
                        try:
                            Xs = scaler.transform(latest_row)
                            pred = model.predict(Xs)[0]
                            proba = model.predict_proba(Xs)[0] if hasattr(model, "predict_proba") else None
                        except Exception as e:
                            logger.exception(f"Prediction error: {e}")
                            st.error(f"Prediction failed: {e}")
                            pred, proba = None, None

                        if pred is not None and proba is not None:
                            bull_p = proba[1] * 100
                            bear_p = proba[0] * 100

                            st.markdown('<div class="section-header">Next-Day Prediction</div>', unsafe_allow_html=True)

                            dir_color = '#00c878' if pred == 1 else '#ff4d6d'
                            dir_label = "🟢 BULLISH" if pred == 1 else "🔴 BEARISH"
                            conf_level = "HIGH" if max(bull_p, bear_p) > 65 else "MEDIUM" if max(bull_p, bear_p) > 55 else "LOW"

                            st.markdown(
                                f'<div class="metric-box" style="border-left-color:{dir_color};padding:20px">'
                                f'<div style="font-size:24px;font-weight:700;color:{dir_color}">{dir_label}</div>'
                                f'<div style="font-size:12px;color:{dir_color};margin-top:4px">Signal confidence: {conf_level}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                            p1, p2 = st.columns(2)
                            with p1:
                                st.markdown(
                                    f'<div class="metric-box"><div style="font-size:10px;opacity:0.6;font-family:Space Mono,monospace">BULL PROBABILITY</div>'
                                    f'<div style="font-size:24px;font-weight:700;color:#00c878">{bull_p:.1f}%</div></div>',
                                    unsafe_allow_html=True
                                )
                            with p2:
                                st.markdown(
                                    f'<div class="metric-box"><div style="font-size:10px;opacity:0.6;font-family:Space Mono,monospace">BEAR PROBABILITY</div>'
                                    f'<div style="font-size:24px;font-weight:700;color:#ff4d6d">{bear_p:.1f}%</div></div>',
                                    unsafe_allow_html=True
                                )

                            # Explanation using the helper
                            explanation = explain_ml_prediction(model, feat_df, prob_pos=proba[1])
                            st.markdown('<div class="section-header">Why the model thinks so</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="explanation-box">{explanation}</div>', unsafe_allow_html=True)

                            # Latest feature values
                            st.markdown('<div class="section-header">Current Feature Values</div>', unsafe_allow_html=True)
                            feat_latest = feat_df.iloc[-1]
                            fcols = st.columns(len(feat_latest))
                            feat_colors = {
                                'rsi': '#00c878' if feat_latest.get('rsi', 50) < 50 else '#ff4d6d',
                                'macd_diff': '#00c878' if feat_latest.get('macd_diff', 0) > 0 else '#ff4d6d',
                                'bb_pos': '#00c878' if feat_latest.get('bb_pos', 0.5) < 0.5 else '#ff4d6d',
                            }
                            for i, (fname, fval) in enumerate(feat_latest.items()):
                                fc = feat_colors.get(fname, '#0af')
                                with fcols[i]:
                                    st.markdown(
                                        f'<div class="metric-box"><div style="font-size:9px;opacity:0.6;font-family:Space Mono,monospace">{fname.upper()}</div>'
                                        f'<div style="font-size:16px;font-weight:600;color:{fc}">{fval:.3f}</div></div>',
                                        unsafe_allow_html=True
                                    )

                            # Feature importance
                            st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
                            try:
                                importances = pd.Series(model.feature_importances_, index=feat_df.columns).sort_values(ascending=True)
                                bar_colors = ['#00e5ff' if v == importances.max() else '#0af' for v in importances.values]
                                fig_fi = go.Figure(go.Bar(
                                    x=importances.values, y=importances.index,
                                    orientation='h', marker_color=bar_colors,
                                    text=[f"{v * 100:.1f}%" for v in importances.values],
                                    textposition='outside', textfont=dict(size=11),
                                ))
                                fig_fi.update_layout(
                                    template='plotly_dark', paper_bgcolor='#080d12', plot_bgcolor='#0a1018',
                                    title='<b>Feature Importance</b> (Random Forest Gini)',
                                    height=340, margin=dict(l=110, r=60, t=50, b=40),
                                    xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                                    yaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                                )
                                st.plotly_chart(fig_fi, use_container_width=True)
                            except Exception as e:
                                logger.warning(f"Could not compute feature importances: {e}")

                            # Rolling accuracy
                            st.markdown('<div class="section-header">Rolling 30-Day Accuracy</div>', unsafe_allow_html=True)
                            if hist_2y is not None and len(hist_2y) >= 200:
                                feat_all = build_ml_features(hist_2y, rsi_period=rsi_period, boll_period=boll_period, boll_std=boll_std, atr_period=atr_period)
                                target_all = (hist_2y['Close'].squeeze().shift(-1) > hist_2y['Close'].squeeze()).astype(int).reindex(feat_all.index).dropna()
                                feat_all = feat_all.reindex(target_all.index)
                                try:
                                    X_all_s = scaler.transform(feat_all)
                                    preds_all = model.predict(X_all_s)
                                    correct = pd.Series((preds_all == target_all.values).astype(int), index=target_all.index)
                                    roll_acc = correct.rolling(30).mean() * 100
                                    roll_acc = roll_acc.dropna()
                                    if not roll_acc.empty:
                                        fig_ra = go.Figure()
                                        fig_ra.add_hline(y=50, line_dash='dash',
                                                         line_color='rgba(255,255,255,0.25)',
                                                         annotation_text='50% (random)')
                                        ra_colors = ['#00c878' if v >= 55 else '#ffd700' if v >= 50 else '#ff4d6d'
                                                     for v in roll_acc.values]
                                        for i in range(1, len(roll_acc)):
                                            fig_ra.add_trace(go.Scatter(
                                                x=roll_acc.index[i - 1:i + 1],
                                                y=roll_acc.values[i - 1:i + 1],
                                                mode='lines', line=dict(color=ra_colors[i], width=2),
                                                showlegend=False, hoverinfo='skip'))
                                        fig_ra.update_layout(
                                            template='plotly_dark', paper_bgcolor='#080d12',
                                            plot_bgcolor='#0a1018',
                                            title='<b>30-Day Rolling Accuracy</b>',
                                            yaxis=dict(range=[30, 80], gridcolor='rgba(255,255,255,0.04)'),
                                            xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                                            height=300, margin=dict(l=60, r=20, t=50, b=40),
                                        )
                                        st.plotly_chart(fig_ra, use_container_width=True)
                                except Exception as e:
                                    logger.warning(f"Could not compute rolling accuracy: {e}")

                        else:
                            st.info("Prediction not available.")
                except Exception as e:
                    logger.exception(f"Prediction alignment error: {e}")
                    st.error(f"Prediction alignment failed: {e}")

            st.markdown("""<div class="explanation-box" style="border-left-color:#ffd700">
                ⚠️ <b>Disclaimer:</b> This ML model is for educational/research purposes only.
                Past accuracy does not guarantee future performance. Not financial advice.
                Features: RSI, returns, 20-day vol, Bollinger Band position, ATR, volume ratio, MACD diff.
            </div>""", unsafe_allow_html=True)
